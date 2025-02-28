import torch
from einops import rearrange, repeat
from functools import partial

import torch.nn.functional
from dataloader.constants import *
from .aaflow import AAFlow
from dataloader.feature_utils import remove_mean_with_mask
from dataloader.fingerprints import compute_fingerprint, tanimoto_similarity
def rescale_grad(
    grad: torch.Tensor, mask, clip_scale=1.0
): 

    scale = (grad ** 2).mean(dim=-1)
    scale: torch.Tensor = scale.sum(dim=-1) / mask.float().sum(dim=-1)  # [B]
    clipped_scale = torch.clamp(scale, max=clip_scale)
    co_ef = clipped_scale / scale  # [B]
    grad = grad * co_ef.view(-1, 1, 1)

    return grad


class ConditionalFlow:

    def __init__(self, flow_model, predictor, oracle, dist, k=4, temperature=0.1, enable=True,
                 n_iter=4, n_recur=1, rho=0.1, mu=0.1, gamma=0.0, num_eps=1, guidance_weight=None):

        self.flow = flow_model
        self.predictor = predictor

        self.k = k
        self.temperature = temperature

        self.device = self.flow.device
        self.T = self.flow.T
        self.enable = enable
        self.dist = dist

        self.oracle = oracle
        self.n_iter = n_iter
        self.n_recur = n_recur
        self.rho = rho
        self.mu = mu
        self.gamma = gamma
        self.num_eps = num_eps
        self.guidance_weight = guidance_weight
    
    @torch.no_grad()
    def compute_next_state(self, coors_t, atom_types_t, mask, t_coors, t_atom_types, delta_time, target_value, condition=None):
        
        '''
            coors_t: (bs, n, 3)
            atom_types_t: (bs, n)
            mask: (bs, n)
            t_coors: (bs)
            t_atom_types: (bs)
        '''
        num_atoms = coors_t.shape[1]
        expand_to_every_node = lambda x: repeat(x, 'b -> b n', n=num_atoms)

        @torch.no_grad()
        def compute_rate_mat(atom_type_t, atom_type_1, t_atom_types):
            # atom_type_1: (bs, n)
            # turn atom_type_1 into one-hot
            atom_type_1 = torch.nn.functional.one_hot(atom_type_1, num_classes=len(ATOMNAMES)).float()
            # compute the rate matrix
            # mask_mask: (bs, n)
            mask_mask = torch.ones_like(atom_type_t) * ATOMNAME_TO_INDEX['MASK']
            mask_mask = rearrange((mask_mask == atom_type_t).float(), 'b n -> b n ()')

            # if the atom type is not mask, the rate matrix will be 0. else it will be atom_type_1 / (1 - t/T)
            rate_matrix = atom_type_1 * (mask_mask) / rearrange(expand_to_every_node(1 - t_atom_types / self.T), 'b n -> b n ()')

            return rate_matrix
            # rate matrix * delta_t
        @torch.enable_grad()
        def tilde_f(coors_1, atom_type_1, mask, t_coors):
            
            mask_lgd = repeat(mask, 'b n -> b num_eps n', num_eps=self.num_eps)
            mask_lgd = rearrange(mask_lgd, 'b num_eps n -> (b num_eps) n')

            coors_1_lgd = repeat(coors_1, 'b n c -> b num_eps n c', num_eps=self.num_eps)
            coors_1_lgd = rearrange(coors_1_lgd, 'b num_eps n c -> (b num_eps) n c')
            
            t_coors_lgd = repeat((1-t_coors[:, None, None]/self.T), 'b n c -> b num_eps n c', num_eps=self.num_eps)
            t_coors_lgd = rearrange(t_coors_lgd, 'b num_eps n c -> (b num_eps) n c')
            noise_lgd = remove_mean_with_mask(torch.randn_like(coors_1_lgd) * mask_lgd[:, :, None], mask_lgd[:, :, None])
            noise_lgd = noise_lgd * t_coors_lgd * self.gamma
            coors_1_lgd = coors_1_lgd + noise_lgd
            
            atom_type_1_lgd = repeat(atom_type_1, 'b n -> b num_eps n', num_eps=self.num_eps)
            atom_type_1_lgd = rearrange(atom_type_1_lgd, 'b num_eps n -> (b num_eps) n')
            
            loss_lgd = 0.

            for predictor, target_v, weight in zip(self.predictor, target_value, self.guidance_weight):
                prediction_lgd = predictor(coors_1_lgd, atom_type_1_lgd, mask_lgd).squeeze()
                if prediction_lgd.dim() > 1:
                    prediction_lgd = torch.sigmoid(prediction_lgd)
                else:
                    prediction_lgd = prediction_lgd.unsqueeze(-1)

                if target_v.dim() == 1:
                    target_v = target_v.unsqueeze(1)
                
                target_value_lgd = repeat(target_v, 'b d -> b num_eps d', num_eps=self.num_eps)
                target_value_lgd = rearrange(target_value_lgd, 'b num_eps d -> (b num_eps) d')
                loss_lgd += torch.nn.MSELoss(reduction='none')(prediction_lgd, target_value_lgd) * weight
            
            return loss_lgd

        for n_recur in range(self.n_recur):

            # coors_1: (bs, n, 3), atom_type_1: (bs, n, len(ATOMNAMES))
            coors_1, atom_type_1 = self.flow.forward(coors_t, atom_types_t, mask, t_coors, t_atom_types, condition=condition)
            atom_type_1 = torch.nn.functional.softmax(atom_type_1, dim=-1)
            
            # we need to sample top k "atom_types" from the distribution
            atom_type_1_top_k = torch.distributions.OneHotCategorical(probs=repeat(atom_type_1, 'bs n atomnames -> bs k n atomnames', k=self.k)).sample()

            # turn into index from one-hot
            atom_type_1_top_k = torch.argmax(atom_type_1_top_k, dim=-1)

            # for every dimension, R(a_t, b) = \sum_{k=1}^K f(X_{1|t}, a_{1|t,k} R_{t|1})

            # obtain the top k atom_types' probability
            coors_1_top_k = repeat(coors_1, 'bs n c -> bs k n c', k=self.k)
            mask_top_k = repeat(mask, 'bs n -> bs k n', k=self.k)

            atom_type_1_top_k = rearrange(atom_type_1_top_k, 'bs k n -> (bs k) n')
            coors_1_top_k = rearrange(coors_1_top_k, 'bs k n c -> (bs k) n c')
            mask_top_k = rearrange(mask_top_k, 'bs k n -> (bs k) n')
            
            # the prediction for the target property
            loss = 0.
            for predictor, target_v, weight in zip(self.predictor, target_value, self.guidance_weight):
                prediction = predictor(coors_1_top_k, atom_type_1_top_k, mask_top_k).squeeze()
                
                if prediction.dim() > 1:
                    prediction = torch.sigmoid(prediction)
                else:
                    prediction = prediction.unsqueeze(-1)
                
                if target_v.dim() == 1:
                    target_v = target_v.unsqueeze(1)
                
                loss_ = torch.nn.MSELoss(reduction='none')(prediction, rearrange(repeat(target_v, 'bs d -> bs k d', k=self.k), 'bs k d -> (bs k) d'))
                loss_ = rearrange(loss_, '(bs k) d -> bs k d', bs=coors_t.shape[0])
                loss_ = loss_.mean(-1)
                loss = loss + loss_ * weight

            # f(X, A): [bs, k]
            energy = torch.nn.functional.softmax(-loss / self.temperature, dim=-1)
            atom_type_t_top_k = rearrange(repeat(atom_types_t, 'bs n -> bs k n', k=self.k), 'bs k n -> (bs k) n')
            t_atom_types_top_k = rearrange(repeat(t_atom_types, 'bs -> bs k', k=self.k), 'bs k -> (bs k)')
            rate_matrices = compute_rate_mat(atom_type_t_top_k, atom_type_1_top_k, t_atom_types_top_k)
            rate_matrices = rearrange(rate_matrices, '(bs k) n atomnames -> bs k n atomnames', k=self.k)

            weighted_rate_matrices = torch.sum(rate_matrices * rearrange(energy, 'bs k -> bs k 1 1'), dim=1)

            # sample from the top k atom_types
            idx = torch.distributions.Categorical(energy).sample()
            atom_type_1_top_k = rearrange(atom_type_1_top_k, '(bs k) n -> bs k n', bs=coors_t.shape[0])
            atom_type_1 = atom_type_1_top_k[torch.arange(atom_type_1_top_k.shape[0]), idx]

            # mask
            atom_type_1 = atom_type_1 * mask

            with torch.enable_grad():

                # enable grad
                for _ in range(self.n_iter):
                    coors_t_copy = coors_t.clone().detach().requires_grad_(True)
                    coors_1, _ = self.flow.forward(coors_t_copy, atom_types_t, mask, t_coors, t_atom_types, condition=condition)
                    loss = tilde_f(coors_1, atom_type_1, mask, t_coors)
                    # compute gradient wrt to coors_t
                    grad_t = torch.autograd.grad(loss.sum(), coors_t_copy)[0]
                    grad_t = remove_mean_with_mask(grad_t, rearrange(mask, 'b n -> b n ()'))
                    grad_t = rescale_grad(grad_t, mask, clip_scale=1.0) * self.rho
                    
                    coors_t_may_contain_nan = coors_t - grad_t
                    try:
                        coors_t_may_contain_nan = remove_mean_with_mask(coors_t_may_contain_nan, rearrange(mask, 'b n -> b n ()'))
                        coors_t = coors_t_may_contain_nan
                    except:
                        break
                
                coors_1, _ = self.flow.forward(coors_t_copy, atom_types_t, mask, t_coors, t_atom_types, condition=condition)

                for _ in range(self.n_iter):
                    
                    coors_1 = coors_1.clone().detach().requires_grad_(True)
                    
                    loss_lgd = tilde_f(coors_1, atom_type_1, mask, t_coors)
                    
                    grad_0 = torch.autograd.grad(loss_lgd.sum(), coors_1)[0]
                    grad_0 = remove_mean_with_mask(grad_0, rearrange(mask, 'b n -> b n ()'))
                    grad_0 = rescale_grad(grad_0, mask, clip_scale=1.0) * self.mu
                
                    coors_1_may_contain_nan = coors_1 - grad_0

                    print(loss_lgd.mean(), t_coors[0])
                    try:
                        coors_1_may_contain_nan = remove_mean_with_mask(coors_1_may_contain_nan, rearrange(mask, 'b n -> b n ()'))
                        coors_1 = coors_1_may_contain_nan
                    except:
                        break

            # compute the velocity
            # v = (x_1 - x_t) / (1 - t/T)
            v_coors = (coors_1 - coors_t) / rearrange(expand_to_every_node(1 - t_coors / self.T), 'b n -> b n ()')
            v_coors = v_coors * rearrange(mask, 'b n -> b n ()')

            coors_next = coors_t + v_coors * delta_time / self.T

            prob = rearrange((weighted_rate_matrices * delta_time / self.T), 'b n c -> (b n) c')

            # if the atom type is not mask, the prob will be 0. else it will be 1 - prob
            prob[torch.arange(prob.shape[0]), rearrange(atom_types_t, 'b n -> (b n)')] = 1 - prob[torch.arange(prob.shape[0]), rearrange(atom_types_t, 'b n -> (b n)')]

            atom_type_next = torch.distributions.Categorical(prob).sample()
            atom_type_next = rearrange(atom_type_next, '(b n) -> b n', n=atom_types_t.shape[1])

            if n_recur < self.n_recur - 1:
                coors_noise = remove_mean_with_mask(torch.randn_like(coors_1) * mask[:, :, None], mask[:, :, None])
                coors_t = (coors_next * t_coors[:, None, None] + coors_noise) / (t_coors[:, None, None] + 1)
                # atom_type_t: sample from Cat(delta(ATOMNAMES['mask']) * 1/(t+1) + delta(atom_type_next) * t/(t+1))
                prob = torch.nn.functional.one_hot(atom_type_next, num_classes=len(ATOMNAMES)).float() * (t_coors[:, None, None] / (t_coors[:, None, None] + 1)) + torch.nn.functional.one_hot(torch.ones_like(atom_type_next) * ATOMNAME_TO_INDEX['MASK'], num_classes=len(ATOMNAMES)).float() * (1 / (t_coors[:, None, None] + 1))
                atom_types_t = torch.distributions.Categorical(prob).sample()
        
        torch.cuda.empty_cache()

        return coors_next, atom_type_next

    @torch.no_grad()
    def sample_euler(self, coors_0, atom_types_0, target_value, mask=None, fix_coors=False, fix_atom_types=False, time_delta=1, condition=None):
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
            
            if self.enable:
                coors_next, atom_types_next = self.compute_next_state(coors, atom_types, mask, time_coors, time_atom_types, time_delta, target_value, condition=condition)
            else:
                coors_next, atom_types_next = self.flow.compute_next_state(coors, atom_types, mask, time_coors, time_atom_types, time_delta, condition=condition)
            
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

    def sample(self, lengths, time_delta, args):

        # construct mask according to lengths: (bs, ) instead of sample_length
        mask = torch.stack(
            [torch.cat([torch.ones([length], device=args.device).long(), torch.zeros([args.max_len - length], device=args.device).long()])
            for length in lengths]
        )

        target_value = self.dist.sample_batch([length for length in lengths]).to(args.device)
        if args.target_property[0] == 'structure':
            # keep structure as the first target
            target_value = target_value.split(1024, dim=1)
        else:
            target_value = target_value.split(1, dim=1)

        target_value = [target_value[i].squeeze() for i in range(len(target_value))]

        if args.condition_embed_size is not None and args.condition_embed_size > 0:
            condition = target_value
            # assert args.do_guidance == False
        else:
            condition = None

        coors_0 = torch.randn([args.sample_bs, args.max_len, 3], device=args.device)
        coors_0 = coors_0 * mask[:, :, None]
        coors_0 = remove_mean_with_mask(coors_0, mask[:, :, None])

        atom_types_0 = torch.ones([args.sample_bs, args.max_len], device=args.device).long() * ATOMNAME_TO_INDEX['MASK']

        coors, atom_types, traj = self.sample_euler(
            coors_0=coors_0,
            atom_types_0=atom_types_0,
            mask=mask,
            target_value=target_value,
            time_delta=time_delta,
            condition=condition
        )

        # get prediction for the target property
        mae = []

        if 'structure' in args.target_property:
            _, fingerprint = compute_fingerprint(coors.cpu(), atom_types.cpu(), mask.sum(1))
            fingerprint = torch.stack(fingerprint).cuda()
            mae.append(-tanimoto_similarity(target_value[0], fingerprint))

        for oracle, target_v in zip(self.oracle, target_value):
            prediction = oracle(coors, atom_types, mask).squeeze()
            mae.append(torch.nn.L1Loss(reduction='none')(prediction, target_v))
        
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

        return coors, atom_types, traj, mae