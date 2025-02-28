import numpy as np
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from rdkit import Chem
from datasets import load_dataset
from dataloader.molecule import construct_mol_conformation, get_valid_mols, save_mols
from dataloader.constants import *
from dataloader.feature_utils import create_empty_np_features, pad_np_features, convert_np_features_to_tensor
from dataloader.fingerprints import compute_fingerprint
import torch

class QM9Dataset(Dataset):

    def __init__(self, max_length=9, withH=False, filter_length=None, partition='train', fingerprint=False):
        
        assert max_length in [29, 9]

        data = load_dataset('yairschiff/qm9')
        self.data = data['train']
        self.max_length = max_length
        self.withH = max_length == 29

        self.process_dataset()
        self.node_num_dist = self.get_node_num_dist()

        if filter_length is not None:
            self.features = [x for x in self.features if x['mask'].sum() == filter_length]
            print("filtered dataset with length {}, results in {} / {} samples".format(filter_length, len(self.features), len(self.data)))
        
        # random shuffle
        np.random.seed(42)
        np.random.shuffle(self.features)

        if 'train' in partition:
            self.features = self.features[:100000]
            if 'train_flow' in partition:
                self.features = self.features[:50000]
            elif 'train_classifier' in partition:
                self.features = self.features[50000:]
        elif partition == 'validation':
            self.features = self.features[100000:100000+1800]
        elif partition == 'test':
            self.features = self.features[100000+1800:]
        elif partition == 'all':
            self.features = self.features
        else:
            raise NotImplementedError
        
        self.mols = None

        if fingerprint:
            self.update_fingerprints()

    def process_dataset(self):
        
        if os.path.exists('storage/datasets/qm9_features_withH={}.pt'.format(self.withH)):
            self.features = torch.load('storage/datasets/qm9_features_withH={}.pt'.format(self.withH))
            print("loaded qm9 features from storage OK")
            return
        
        self.features = []
        
        print("preprocess qm9 dataset...")

        for idx in tqdm(range(len(self.data)), total=len(self.data)):
            datapoint = self.data[idx]
            
            if not self.withH:
                H_indices = [i for i, x in enumerate(datapoint['atomic_symbols']) if x == 'H']
                num_atoms = len(datapoint['atomic_symbols']) - len(H_indices)
                coors = np.array([datapoint['pos'][i] for i in range(len(datapoint['pos'])) if i not in H_indices])
                coors = coors - np.mean(coors, axis=0, keepdims=True)
                atom_types = np.array([ATOMNAME_TO_INDEX[x] for x in datapoint['atomic_symbols'] if x != 'H'])

            else:
                num_atoms = len(datapoint['atomic_symbols'])
                H_indices = []
                coors = np.array(datapoint['pos'])
                coors = coors - np.mean(coors, axis=0, keepdims=True)
                atom_types = np.array([ATOMNAME_TO_INDEX[x] for x in datapoint['atomic_symbols']])
            
            np_features = create_empty_np_features(num_atoms)
            np_features['atom_types'] = atom_types
            np_features['coors'] = coors

            np_features['mu'] = np.array([datapoint['mu']])
            np_features['alpha'] = np.array([datapoint['alpha']])
            np_features['homo'] = np.array([datapoint['homo']])
            np_features['lumo'] = np.array([datapoint['lumo']])
            np_features['gap'] = np.array([datapoint['gap']])
            np_features['cv'] = np.array([datapoint['cv']])

            np_features = convert_np_features_to_tensor(pad_np_features(np_features, self.max_length), 'cpu')
            self.features.append(np_features)
        
        for idx in tqdm(range(len(self.features)), total=len(self.data)):
            fingerprint_bits, fingerprint_1024 = compute_fingerprint(self.features[idx]['coors'].unsqueeze(0), self.features[idx]['atom_types'].unsqueeze(0), self.features[idx]['num_atoms'].unsqueeze(0))
            self.features[idx]['structure'] = fingerprint_1024[0]
        
        os.makedirs('storage/datasets', exist_ok=True)
        torch.save(self.features, 'storage/datasets/qm9_features_withH={}.pt'.format(self.withH))

    def get_node_num_dist(self):

        if os.path.exists('storage/datasets/qm9_node_num_dist.json'):
            
            with open('storage/datasets/qm9_node_num_dist.json', 'r') as f:
                dist_values = json.load(f)

        else:

            dist = {i: 0 for i in range(1, self.max_length+1)}

            for feature in self.features:
                num_nodes = feature['num_atoms'].item()
                dist[num_nodes] += 1
            
            print(dist)

            # normalize
            dist = {k: v / sum(list(dist.values())) for k, v in dist.items()}
            print(dist)
            dist_values = list(dist.values())
            with open('storage/datasets/qm9_node_num_dist.json', 'w') as f:
                json.dump(dist_values, f)
            
        dist = torch.distributions.categorical.Categorical(torch.tensor(dist_values))

        # add 1 to the sample to avoid 0
        dist.sample_lengths = lambda x: dist.sample(x) + 1

        return dist
    
    def update_fingerprints(self):
        
        if self.features[0].get('structure') is not None:
            return
        
        for idx in range(len(self.features)):
            fingerprint_bits, fingerprint_1024 = compute_fingerprint(self.features[idx]['coors'].unsqueeze(0), self.features[idx]['atom_types'].unsqueeze(0), self.features[idx]['num_atoms'].unsqueeze(0))
            self.features[idx]['structure'] = fingerprint_1024[0]
        
    
    def prepare_mols(self):

        if self.mols is not None: 
            return self.mols
    
        mols = []
        for idx in range(len(self.features)):
            coors, atom_types = self.features[idx]['coors'], self.features[idx]['atom_types']
            try:
                mol = construct_mol_conformation(coors.cpu().numpy(), atom_types.cpu().numpy())
            except:
                continue
            mols.append(mol)
        
        print("validity:", len(mols) / len(self.features))

        self.mols = mols

        return mols

    def get_smiles(self):

        mols = self.prepare_mols()
        return [Chem.MolToSmiles(mol) for mol in mols]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

    def visualization(self):
        
        mols = []
        for i in range(100):
            coors, atom_types = self.features[i]['coors'], self.features[i]['atom_types']
            mol = construct_mol_conformation(coors.cpu().numpy(), atom_types.cpu().numpy())
            mols.append(mol)
        
        valid_mols = get_valid_mols(mols)

        connected_mols = [mol for mol in valid_mols if '.' not in Chem.MolToSmiles(mol)]

        save_mols(mols, f'storage/results/testset.sdf')
            
        validity = len(valid_mols) / len(mols) if len(mols) != 0 else 0
        connectivity = len(connected_mols) / len(valid_mols) if len(valid_mols) != 0 else 0
        
        success = validity * connectivity

        print("validity: {}, connectivity: {}, success: {}".format(validity, connectivity, success))
