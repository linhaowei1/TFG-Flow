import torch
import os
import pickle
import lmdb
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch_geometric.data import Data
from dataloader.constants import *
from einops import rearrange, repeat

class CrossDocekd2020Dataset(Dataset):

    def __init__(self, args, validation=False):
        super().__init__()
        
        db_path = 'storage/datasets/crossdocked2020.lmdb'
        keys_path = 'storage/datasets/crossdocked_split.pt'
        
        self._load_db(db_path)

        self.args = args

        np.random.seed(args.seed)
        np.random.shuffle(self.keys)

        if 'train' in args.partition and not validation:
            self.keys = self.keys[:160000]
            if args.partition == 'train_flow':
                self.keys = self.keys[:int(len(self.keys) // 2)]
            elif args.partition == 'train_classifier':
                self.keys = self.keys[int(len(self.keys) // 2):]
        elif validation:
            self.keys = self.keys[160000:]
        else:
            raise NotImplementedError

    def _load_db(self, path):

        self.db = lmdb.open(
            path,
            map_size=10*(1024*1024*1024),
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False))
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):

        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))

        n_node = data['protein_pos'].size(0) + data['ligand_pos'].size(0)

        protein_mean = data['protein_pos'].mean(0)
        data['protein_pos'] = data['protein_pos'] - protein_mean
        data['ligand_pos'] = data['ligand_pos'] - protein_mean

        x = torch.cat([
            data['protein_pos'],
            data['ligand_pos'],
        ])
        h = torch.cat([
            data['protein_element'],
            data['ligand_element'],
        ])
        ligand_mask = torch.zeros(x.size(0), dtype=torch.bool)
        ligand_mask[data['protein_pos'].size(0):] = True

        t_coors = t_atom_types = None
        t_coors = torch.randint(0, self.args.T, (n_node,), device=x.device)
        t_atom_types = torch.randint(0, self.args.T, (n_node,), device=x.device)

        return Data(
            x=x,
            h=h,
            ligand_mask=ligand_mask,
            t_coors=t_coors,
            t_atom_types=t_atom_types,
        )
        
        