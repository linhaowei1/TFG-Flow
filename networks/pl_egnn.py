import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
from torch_geometric.nn import radius_graph, knn_graph


class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50, fixed_offset=True):
        super(GaussianSmearing, self).__init__()
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians
        if fixed_offset:
            # customized offset
            offset = torch.tensor([0, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10])
        else:
            offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def __repr__(self):
        return f'GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})'

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AngleExpansion(nn.Module):
    def __init__(self, start=1.0, stop=5.0, half_expansion=10):
        super(AngleExpansion, self).__init__()
        l_mul = 1. / torch.linspace(stop, start, half_expansion)
        r_mul = torch.linspace(start, stop, half_expansion)
        coeff = torch.cat([l_mul, r_mul], dim=-1)
        self.register_buffer('coeff', coeff)

    def forward(self, angle):
        return torch.cos(angle.view(-1, 1) * self.coeff.view(1, -1))


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


NONLINEARITIES = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "swish": Swish(),
    'silu': nn.SiLU()
}


class MLP(nn.Module):
    """MLP with the same hidden dim across all layers."""

    def __init__(self, in_dim, out_dim, hidden_dim, num_layer=2, norm=True, act_fn='relu', act_last=False):
        super().__init__()
        layers = []
        for layer_idx in range(num_layer):
            if layer_idx == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif layer_idx == num_layer - 1:
                layers.append(nn.Linear(hidden_dim, out_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            if layer_idx < num_layer - 1 or act_last:
                if norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(NONLINEARITIES[act_fn])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def outer_product(*vectors):
    for index, vector in enumerate(vectors):
        if index == 0:
            out = vector.unsqueeze(-1)
        else:
            out = out * vector.unsqueeze(1)
            out = out.view(out.shape[0], -1).unsqueeze(-1)
    return out.squeeze()


def get_h_dist(dist_metric, hi, hj):
    if dist_metric == 'euclidean':
        h_dist = torch.sum((hi - hj) ** 2, -1, keepdim=True)
        return h_dist
    elif dist_metric == 'cos_sim':
        hi_norm = torch.norm(hi, p=2, dim=-1, keepdim=True)
        hj_norm = torch.norm(hj, p=2, dim=-1, keepdim=True)
        h_dist = torch.sum(hi * hj, -1, keepdim=True) / (hi_norm * hj_norm)
        return h_dist, hj_norm


def get_r_feat(r, r_exp_func, node_type=None, edge_index=None, mode='basic'):
    if mode == 'origin':
        r_feat = r
    elif mode == 'basic':
        r_feat = r_exp_func(r)
    elif mode == 'sparse':
        src, dst = edge_index
        nt_src = node_type[src]  # [n_edges, 8]
        nt_dst = node_type[dst]
        r_exp = r_exp_func(r)
        r_feat = outer_product(nt_src, nt_dst, r_exp)
    else:
        raise ValueError(mode)
    return r_feat


def compose_context(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    # previous version has problems when ligand atom types are fixed
    # (due to sorting randomly in case of same element)

    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    # sort_idx = batch_ctx.argsort()
    sort_idx = torch.sort(batch_ctx, stable=True).indices

    mask_ligand = torch.cat([
        torch.zeros([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx, mask_ligand


def compose_context_prop(h_protein, h_ligand, pos_protein, pos_ligand, batch_protein, batch_ligand):
    batch_ctx = torch.cat([batch_protein, batch_ligand], dim=0)
    sort_idx = batch_ctx.argsort()

    mask_protein = torch.cat([
        torch.ones([batch_protein.size(0)], device=batch_protein.device).bool(),
        torch.zeros([batch_ligand.size(0)], device=batch_ligand.device).bool(),
    ], dim=0)[sort_idx]

    batch_ctx = batch_ctx[sort_idx]
    h_ctx = torch.cat([h_protein, h_ligand], dim=0)[sort_idx]       # (N_protein+N_ligand, H)
    pos_ctx = torch.cat([pos_protein, pos_ligand], dim=0)[sort_idx]  # (N_protein+N_ligand, 3)

    return h_ctx, pos_ctx, batch_ctx


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


def hybrid_edge_connection(ligand_pos, protein_pos, k, ligand_index, protein_index):
    # fully-connected for ligand atoms
    dst = torch.repeat_interleave(ligand_index, len(ligand_index))
    src = ligand_index.repeat(len(ligand_index))
    mask = dst != src
    dst, src = dst[mask], src[mask]
    ll_edge_index = torch.stack([src, dst])

    # knn for ligand-protein edges
    ligand_protein_pos_dist = torch.unsqueeze(ligand_pos, 1) - torch.unsqueeze(protein_pos, 0)
    ligand_protein_pos_dist = torch.norm(ligand_protein_pos_dist, p=2, dim=-1)
    knn_p_idx = torch.topk(ligand_protein_pos_dist, k=k, largest=False, dim=1).indices
    knn_p_idx = protein_index[knn_p_idx]
    knn_l_idx = torch.unsqueeze(ligand_index, 1)
    knn_l_idx = knn_l_idx.repeat(1, k)
    pl_edge_index = torch.stack([knn_p_idx, knn_l_idx], dim=0)
    pl_edge_index = pl_edge_index.view(2, -1)
    return ll_edge_index, pl_edge_index


def batch_hybrid_edge_connection(x, k, mask_ligand, batch, add_p_index=False):
    batch_size = batch.max().item() + 1
    batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index = [], [], []
    with torch.no_grad():
        for i in range(batch_size):
            ligand_index = ((batch == i) & (mask_ligand == 1)).nonzero()[:, 0]
            protein_index = ((batch == i) & (mask_ligand == 0)).nonzero()[:, 0]
            ligand_pos, protein_pos = x[ligand_index], x[protein_index]
            ll_edge_index, pl_edge_index = hybrid_edge_connection(
                ligand_pos, protein_pos, k, ligand_index, protein_index)
            batch_ll_edge_index.append(ll_edge_index)
            batch_pl_edge_index.append(pl_edge_index)
            if add_p_index:
                all_pos = torch.cat([protein_pos, ligand_pos], 0)
                p_edge_index = knn_graph(all_pos, k=k, flow='source_to_target')
                p_edge_index = p_edge_index[:, p_edge_index[1] < len(protein_pos)]
                p_src, p_dst = p_edge_index
                all_index = torch.cat([protein_index, ligand_index], 0)
                p_edge_index = torch.stack([all_index[p_src], all_index[p_dst]], 0)
                batch_p_edge_index.append(p_edge_index)

    if add_p_index:
        edge_index = [torch.cat([ll, pl, p], -1) for ll, pl, p in zip(
            batch_ll_edge_index, batch_pl_edge_index, batch_p_edge_index)]
    else:
        edge_index = [torch.cat([ll, pl], -1) for ll, pl in zip(batch_ll_edge_index, batch_pl_edge_index)]
    edge_index = torch.cat(edge_index, -1)
    return edge_index

class EnBaseLayer(nn.Module):
    def __init__(self, hidden_dim, edge_feat_dim, num_r_gaussian, update_x=True, act_fn='silu', norm=False):
        super().__init__()
        self.r_min = 0.
        self.r_max = 10.
        self.hidden_dim = hidden_dim
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        if num_r_gaussian > 1:
            self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)
        self.edge_mlp = MLP(2 * hidden_dim + edge_feat_dim + num_r_gaussian, hidden_dim, hidden_dim,
                            num_layer=2, norm=norm, act_fn=act_fn, act_last=True)
        self.edge_inf = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        if self.update_x:
            # self.x_mlp = MLP(hidden_dim, 1, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)
            x_mlp = [nn.Linear(hidden_dim, hidden_dim), NONLINEARITIES[act_fn]]
            layer = nn.Linear(hidden_dim, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
            x_mlp.append(layer)
            x_mlp.append(nn.Tanh())
            self.x_mlp = nn.Sequential(*x_mlp)

        self.node_mlp = MLP(2 * hidden_dim, hidden_dim, hidden_dim, num_layer=2, norm=norm, act_fn=act_fn)

    def forward(self, h, x, edge_index, mask_ligand, edge_attr=None):
        src, dst = edge_index
        hi, hj = h[dst], h[src]
        # \phi_e in Eq(3)
        rel_x = x[dst] - x[src]
        d_sq = torch.sum(rel_x ** 2, -1, keepdim=True)
        if self.num_r_gaussian > 1:
            d_feat = self.distance_expansion(torch.sqrt(d_sq + 1e-8))
        else:
            d_feat = d_sq
        if edge_attr is not None:
            edge_feat = torch.cat([d_feat, edge_attr], -1)
        else:
            edge_feat = d_sq

        mij = self.edge_mlp(torch.cat([hi, hj, edge_feat], -1))
        eij = self.edge_inf(mij)
        mi = scatter_sum(mij * eij, dst, dim=0, dim_size=h.shape[0])

        # h update in Eq(6)
        h = h + self.node_mlp(torch.cat([mi, h], -1))
        if self.update_x:
            # x update in Eq(4)
            xi, xj = x[dst], x[src]
            # (xi - xj) / (\|xi - xj\| + C) to make it more stable
            delta_x = scatter_sum((xi - xj) / (torch.sqrt(d_sq + 1e-8) + 1) * self.x_mlp(mij), dst, dim=0)
            x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated

        return h, x


class EGNN(nn.Module):
    def __init__(self, num_layers=9, hidden_dim=128, edge_feat_dim=4, num_r_gaussian=20, k=32, cutoff=10.0, cutoff_mode='knn',
                 update_x=True, act_fn='silu', norm=False):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.update_x = update_x
        self.act_fn = act_fn
        self.norm = norm
        self.k = k
        self.cutoff = cutoff
        self.cutoff_mode = cutoff_mode
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=num_r_gaussian)
        self.net = self._build_network()

    def _build_network(self):
        # Equivariant layers
        layers = []
        for l_idx in range(self.num_layers):
            layer = EnBaseLayer(self.hidden_dim, self.edge_feat_dim, self.num_r_gaussian,
                                update_x=self.update_x, act_fn=self.act_fn, norm=self.norm)
            layers.append(layer)
        return nn.ModuleList(layers)

    # todo: refactor
    def _connect_edge(self, x, mask_ligand, batch):
        # if self.cutoff_mode == 'radius':
        #     edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        if self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    # todo: refactor
    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, mask_ligand, batch):
        # x: (B, N, 3), h: (B, N, H), mask_ligand: (B, N,), batch: (B)
        for l_idx, layer in enumerate(self.net):
            edge_index = self._connect_edge(x, mask_ligand, batch)
            edge_type = self._build_edge_type(edge_index, mask_ligand)
            h, x = layer(h, x, edge_index, mask_ligand, edge_attr=edge_type)
        return h, x
