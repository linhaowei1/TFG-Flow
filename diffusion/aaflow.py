import torch
from torch import nn
from einops import rearrange, repeat
# from networks.egnn import EGNN
from networks.egnn import EGNN
from dataloader.feature_utils import remove_mean_with_mask
from dataloader.constants import *


class AAFlow(nn.Module):

    def __init__(
            self,
            T,
            max_len,
            device,
            h_embed_size=256,
            num_layers=4,
            cls_embed_size=None,
            e_embed_size=None,
            condition_embed_size=None,
            target_property=None
            ):
        super(AAFlow, self).__init__()

        self.max_len = max_len
        self.h_embed_size = h_embed_size
        self.e_embed_size = e_embed_size if e_embed_size is not None else h_embed_size
        self.cls_embed_size = cls_embed_size if cls_embed_size is not None else h_embed_size
        self.device = device
        self.T = T

        self.atom_emb = nn.Embedding(len(ATOMNAMES), self.h_embed_size)
        self.time_emb = nn.Embedding(T, self.h_embed_size // 2)
        
        if target_property is not None and condition_embed_size is not None: 
            if len(target_property) == 1:
                if target_property[0] == 'structure':
                    self.condition_emb = nn.Linear(1024, condition_embed_size)
                else:
                    self.condition_emb = nn.Linear(1, condition_embed_size)
            else:
                self.condition_emb = nn.ModuleList([
                    nn.Linear(1024, condition_embed_size) if target == 'structure' else nn.Linear(1, condition_embed_size) for target in range(len(target_property))
                ])
        else:
            self.condition_emb = None

        self.gnn = EGNN(
            in_node_nf=self.h_embed_size, in_edge_nf=1,
            hidden_nf=self.h_embed_size, device=device, act_fn=torch.nn.SiLU(),
            n_layers=num_layers, attention=True, tanh=True, norm_constant=1,
            inv_sublayers=1, sin_embedding=False,
            normalization_factor=1,
            aggregation_method='sum'
        )
        
        self.atom_classifier = nn.Sequential(
            nn.Linear(self.h_embed_size, self.cls_embed_size),
            nn.ReLU(),
            nn.Linear(self.cls_embed_size, self.cls_embed_size),
            nn.ReLU(),
            nn.Linear(self.cls_embed_size, len(ATOMNAMES))
        )

    def forward(self, coors, atom_types, mask, t_coors, t_atom_types, condition=None):
        
        # embed atom types
        h = self.atom_emb(atom_types)   # (bs, n, h_embed_size)
        h = h * rearrange(mask, 'b n -> b n ()')

        # concatenate time embeddings
        time_coors_emb, time_atom_types_emb = self.time_emb(t_coors), self.time_emb(t_atom_types)
        time_emb = torch.cat([time_coors_emb, time_atom_types_emb], dim=-1)
        time_emb = repeat(time_emb, 'b d -> b n d', n=h.shape[1])
        time_emb = time_emb * rearrange(mask, 'b n -> b n ()')

        h = h + time_emb

        if self.condition_emb is not None:
            if isinstance(self.condition_emb, nn.ModuleList):
                if condition[0].dim() == 1:
                    condition[0] = condition[0].unsqueeze(1)
                    condition[1] = condition[1].unsqueeze(1)
                condition_emb = self.condition_emb[0](condition[0]) + self.condition_emb[1](condition[1])
            else:
                if condition[0].dim() == 1:
                    condition[0] = condition[0].unsqueeze(1)
                condition_emb = self.condition_emb(condition[0])
            condition_emb = repeat(condition_emb, 'b d -> b n d', n=h.shape[1])
            condition_emb = condition_emb * rearrange(mask, 'b n -> b n ()')
            h = h + condition_emb

        h, coors = self.gnn.forward(h, coors, mask.bool())

        h = h * rearrange(mask, 'b n -> b n ()')
        
        predicted_atom = self.atom_classifier(h)

        # zero out coors gravity
        coors = remove_mean_with_mask(coors, rearrange(mask, 'b n -> b n ()'))

        return coors, predicted_atom
    
    def loss(self, coors_1, atom_types_1, coors, atom_types, mask, time_coors, time_atom_types):

        atom_types_1 = rearrange(atom_types_1, 'b n c -> (b n) c')
        atom_types = rearrange(atom_types, 'b n -> (b n)')

        loss_atom = nn.functional.cross_entropy(atom_types_1, atom_types, reduction='none')
        loss_atom = loss_atom * rearrange(mask, 'b n -> (b n)')
        loss_atom = loss_atom.sum() / mask.sum()

        loss_coors = nn.functional.mse_loss(coors_1, coors, reduction='none')
        loss_coors = loss_coors * rearrange(mask, 'b n -> b n ()')
        # loss_coors = loss_coors / rearrange(repeat(1 - time_coors / self.T, 'b -> b n', n=loss_coors.shape[1]), 'b n -> b n ()')
        loss_coors = loss_coors.sum() / mask.sum()

        return loss_coors, loss_atom

    @torch.no_grad()
    def sample_t(self, bs):
        # sample time for coors (bs)

        # uniform
        time_coors = torch.randint(0, self.T, size=(bs,), device=self.device)

        time_atom_types = torch.randint(0, self.T, size=(bs,), device=self.device)

        # round to the smaller integer
        time_coors = time_coors.long()
        time_atom_types = time_atom_types.long()

        return time_coors, time_atom_types

    @torch.no_grad()
    def sample_x_t(self, coors, atom_types, mask, time_coors, time_atom_types):

        mask = rearrange(mask, 'b n -> b n ()')
        time_coors = rearrange(repeat(time_coors, 'b -> b n', n=coors.shape[1]), 'b n -> b n ()')

        # sample x_0
        coors_0 = torch.randn_like(coors) * mask
        coors_0 = remove_mean_with_mask(coors_0, mask)

        # sample x_t
        coors_t = time_coors / self.T * coors + (1 - time_coors / self.T) * coors_0
        coors_t = coors_t * mask

        # mask_atom (bs, n)
        mask_atom = torch.rand(atom_types.shape, device=self.device) > repeat(time_atom_types / self.T, 'b -> b n', n=atom_types.shape[1])

        # turn the masked index to MASK
        # if index i is not [MASK] before, if it's masked by mask_atom, it will be [MASK]
        # if index i is [MASK] before, if it's masked by mask_atom, it will still be [MASK]
        atom_types_t = atom_types * ~mask_atom + ATOMNAME_TO_INDEX['MASK'] * mask_atom

        return coors_t, atom_types_t
    
    @torch.no_grad()
    def compute_next_state(self, coors_t, atom_types_t, mask, t_coors, t_atom_types, delta_time, condition=None):
        
        '''
            coors_t: (bs, n, 3)
            atom_types_t: (bs, n)
            mask: (bs, n)
            t_coors: (bs)
            t_atom_types: (bs)
        '''
        num_atoms = coors_t.shape[1]
        expand_to_every_node = lambda x: repeat(x, 'b -> b n', n=num_atoms)

        # coors_1: (bs, n, 3), atom_type_1: (bs, n, len(ATOMNAMES))
        coors_1, atom_type_1 = self.forward(coors_t, atom_types_t, mask, t_coors, t_atom_types, condition=condition)
        atom_type_1 = torch.nn.functional.softmax(atom_type_1, dim=-1)
        
        # softmax to onehot (2024.9.5: haowei)
        atom_type_1 = torch.distributions.OneHotCategorical(probs=atom_type_1).sample()

        # compute the velocity
        # v = (x_1 - x_t) / (1 - t/T)
        v_coors = (coors_1 - coors_t) / rearrange(expand_to_every_node(1 - t_coors / self.T), 'b n -> b n ()')
        v_coors = v_coors * rearrange(mask, 'b n -> b n ()')
        coors_next = coors_t + v_coors * delta_time / self.T

        # compute the rate matrix
        # mask_mask: (bs, n)
        mask_mask = torch.ones_like(atom_types_t) * ATOMNAME_TO_INDEX['MASK']
        mask_mask = rearrange((mask_mask == atom_types_t).float(), 'b n -> b n ()')

        # if the atom type is not mask, the rate matrix will be 0. else it will be atom_type_1 / (1 - t/T)
        rate_matrix = atom_type_1 * (mask_mask) / rearrange(expand_to_every_node(1 - t_atom_types / self.T), 'b n -> b n ()')

        # rate matrix * delta_t
        prob = rearrange((rate_matrix * delta_time / self.T), 'b n c -> (b n) c')

        # if the atom type is not mask, the prob will be 0. else it will be 1 - prob
        prob[torch.arange(prob.shape[0]), rearrange(atom_types_t, 'b n -> (b n)')] = 1 - prob[torch.arange(prob.shape[0]), rearrange(atom_types_t, 'b n -> (b n)')]

        atom_type_next = torch.distributions.Categorical(prob).sample()
        atom_type_next = rearrange(atom_type_next, '(b n) -> b n', n=atom_types_t.shape[1])

        return coors_next, atom_type_next

    @torch.no_grad()
    def sample_euler(self, coors_0, atom_types_0, mask=None, fix_coors=False, fix_atom_types=False, time_delta=1, condition=None):
        # coors_0: (bs, n, 3), atom_types_0: (bs, n)

        traj = []

        if mask is None:
            mask = torch.ones_like(atom_types_0).long()

        time_coors, time_atom_types = torch.zeros(coors_0.shape[0], device=self.device, dtype=torch.long), torch.zeros(coors_0.shape[0], device=self.device, dtype=torch.long)
        
        if fix_coors:
            time_coors += (self.T - 1)
        if fix_atom_types:
            time_atom_types += (self.T - 1)

        coors, atom_types = coors_0, atom_types_0

        while time_coors[0] < self.T and time_atom_types[0] < self.T:
            
            coors_next, atom_types_next = self.compute_next_state(coors, atom_types, mask, time_coors, time_atom_types, time_delta, condition=condition)
            
            time_coors_next = time_coors + time_delta
            time_atom_types_next = time_atom_types + time_delta

            if fix_atom_types:
                atom_types_next = atom_types
                time_atom_types_next = time_atom_types
            if fix_coors:
                coors_next = coors
                time_coors_next = time_coors
            
            traj.append((coors, atom_types))

            coors, atom_types = coors_next, atom_types_next
            time_coors, time_atom_types = time_coors_next, time_atom_types_next

        return coors, atom_types, traj


    def get_loss_batch(self, batch, args):
        coors = batch['coors'].float().to(args.device)
        atom_types = batch['atom_types'].to(args.device)
        mask = batch['mask'].to(args.device)

        time_coors, time_atom_types = self.sample_t(coors.shape[0])    # (bs), (bs)
        coors_t, atom_types_t = self.sample_x_t(coors, atom_types, mask, time_coors, time_atom_types)

        # forward
        condition = None
        if self.condition_emb is not None:
            condition = [batch[target].float().to(args.device) for target in args.target_property]

        coors_1, atom_types_1 = self(coors_t, atom_types_t, mask, time_coors, time_atom_types, condition=condition)

        # loss
        loss_coors, loss_atom = self.loss(coors_1, atom_types_1, coors, atom_types, mask, time_coors, time_atom_types)

        # backward
        loss = loss_coors + loss_atom
        
        return loss_coors, loss_atom, loss

    def sample(self, lengths, args, condition=None):

        # construct mask according to lengths: (bs, ) instead of sample_length
        mask = torch.stack(
            [torch.cat([torch.ones([length], device=args.device).long(), torch.zeros([args.max_len - length], device=args.device).long()])
            for length in lengths]
        )

        coors_0 = torch.randn([args.sample_bs, args.max_len, 3], device=args.device)
        coors_0 = coors_0 * mask[:, :, None]
        coors_0 = remove_mean_with_mask(coors_0, mask[:, :, None])

        atom_types_0 = torch.ones([args.sample_bs, args.max_len], device=args.device).long() * ATOMNAME_TO_INDEX['MASK']

        coors, atom_types, traj = self.sample_euler(
            coors_0=coors_0,
            atom_types_0=atom_types_0,
            mask=mask,
            condition=condition
        )
        
        # get rid of the padding coors
        coors = coors.cpu()
        atom_types = atom_types.cpu()
        traj = [(coors.cpu(), atom_types.cpu()) for coors, atom_types in traj]
        
        coors = [coors[i, :length, :] for i, length in enumerate(lengths)]
        atom_types = [atom_types[i, :length] for i, length in enumerate(lengths)]

        traj = [
            [(coors[i, :length, :], atom_types[i, :length]) for i, length in enumerate(lengths)]
            for (coors, atom_types) in traj
        ]

        return coors, atom_types, traj