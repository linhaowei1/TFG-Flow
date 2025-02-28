import numpy as np
import os
import pickle
import lmdb
from tqdm import tqdm
from torch.utils.data import Dataset
from rdkit import Chem
from datasets import load_dataset
from dataloader.molecule import construct_mol_conformation, get_valid_mols, save_mols
from dataloader.constants import *
from dataloader.feature_utils import create_empty_np_features, pad_np_features, convert_np_features_to_tensor
from dataloader.fingerprints import compute_fingerprint
import json
import torch

class GEOMDrugDataset(Dataset):

    def __init__(self, partition='train'):
        
        self.base_dir = 'storage/datasets'
        self._load_db()
        self.max_length = 30
        self.keys = list(range(5521399))
        self.node_num_dist = self.get_node_num_dist()
        # random shuffle
        np.random.seed(42)
        np.random.shuffle(self.keys)
        if 'train' in partition:
            self.keys = self.keys[:5000000]
            if 'train_flow' in partition:
                self.keys = self.keys[:5000000//2]
            elif 'train_classifier' in partition:
                self.keys = self.keys[5000000//2:]
        elif partition == 'validation':
            self.keys = self.keys[5000000:5000000+70000]
        else:
            self.keys = self.keys[5000000+70000:5000000+70000+70000]

        self.mols = None
    
    def get_smiles(self):
        return None
        smiles = []
        for key in self.keys:
            data = pickle.loads(self.db.begin().get(str(key).encode()))
            smiles.append(data['smiles'])
        return smiles
        
    def _load_db(self):

        self.db = lmdb.open(
            os.path.join(self.base_dir, 'geom_drug.lmdb'),
            map_size=10*(1024*1024*1024),
            create=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_node_num_dist(self):

        if os.path.exists('storage/datasets/geom_node_list.json'):
            
            with open('storage/datasets/geom_node_list.json', 'r') as f:
                dist_values = json.load(f)

        else:

            dist = {i: 0 for i in range(1, self.max_length+1)}

            for key in self.keys:
                data = pickle.loads(self.db.begin().get(str(key).encode()))
                num_nodes = data['num_atoms'][0]
                dist[num_nodes] += 1
            
            print(dist)

            # normalize
            dist = {k: v / sum(list(dist.values())) for k, v in dist.items()}
            print(dist)
            dist_values = list(dist.values())
            with open('storage/datasets/geom_node_list.json', 'w') as f:
                json.dump(dist_values, f)
            
        dist = torch.distributions.categorical.Categorical(torch.tensor(dist_values))

        # add 1 to the sample to avoid 0
        dist.sample_lengths = lambda x: dist.sample(x) + 1

        return dist

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(str(key).encode()))
        data = convert_np_features_to_tensor(data, 'cpu')
        return data
    
if __name__ == '__main__':
    ds = GEOMDrugDataset()
    import pdb; pdb.set_trace()