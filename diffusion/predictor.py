import torch
from torch import nn
from einops import rearrange
from networks.egnn import EGNN
from dataloader.constants import *


class Predictor(nn.Module):

    def __init__(
            self,
            max_len,
            device,
            h_embed_size=256,
            num_layers=4,
            cls_embed_size=None,
            e_embed_size=None,
            class_num=1
            ):
        super(Predictor, self).__init__()

        self.max_len = max_len
        self.h_embed_size = h_embed_size
        self.e_embed_size = e_embed_size if e_embed_size is not None else h_embed_size
        self.cls_embed_size = cls_embed_size if cls_embed_size is not None else h_embed_size
        self.device = device
        
        self.atom_emb = nn.Embedding(len(ATOMNAMES), self.h_embed_size)

        self.gnn = EGNN(
            in_node_nf=self.h_embed_size, in_edge_nf=1,
            hidden_nf=self.h_embed_size, device=device, act_fn=torch.nn.SiLU(),
            n_layers=num_layers, attention=True, tanh=True, norm_constant=1,
            inv_sublayers=1, sin_embedding=False,
            normalization_factor=1,
            aggregation_method='sum'
        )
        
        self.class_num = class_num
        self.predictor = nn.Linear(self.h_embed_size, class_num)

    def forward(self, coors, atom_types, mask):
        
        # embed atom types
        h = self.atom_emb(atom_types)   # (bs, n, h_embed_size)
        h = h * rearrange(mask, 'b n -> b n ()')

        h, coors = self.gnn.forward(h, coors, mask.bool())

        h = h * rearrange(mask, 'b n -> b n ()')
        
        # avg pooling
        h = torch.sum(h, dim=1, keepdim=False) / torch.sum(mask, dim=1, keepdim=True)

        prediction = self.predictor(h)

        return prediction
    
    def get_loss_batch(self, batch, target_name, args):
        coors = batch['coors'].float().to(args.device)
        atom_types = batch['atom_types'].to(args.device)
        mask = batch['mask'].to(args.device)

        prediction = self(coors, atom_types, mask)

        # loss
        if self.class_num == 1:
            loss = nn.functional.mse_loss(prediction, batch[target_name].float().to(args.device))
        else:
            prediction = torch.sigmoid(prediction)
            loss = nn.functional.binary_cross_entropy(prediction, batch[target_name].to(args.device))
        
        return loss

    @torch.no_grad()
    def get_prediction(self, batch, args):
        
        # this method returns log(p)
        coors = batch['coors'].float().to(args.device)
        atom_types = batch['atom_types'].to(args.device)
        mask = batch['mask'].to(args.device)

        prediction = self(coors, atom_types, mask)
        if self.class_num > 1:
            prediction = torch.sigmoid(prediction)

        return prediction