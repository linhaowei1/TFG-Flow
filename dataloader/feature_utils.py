import numpy as np
import torch

from dataloader.constants import *

def remove_mean_with_mask(x, node_mask):
    masked_max_abs_value = (x * (1 - node_mask)).abs().sum().item()
    assert masked_max_abs_value < 1e-5, f'Error {masked_max_abs_value} too high'
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def create_empty_np_features(length):
	
	atom_types = np.ones((length)) * ATOMNAME_TO_INDEX['MASK']
	coors = np.zeros((length, 3))

	# Create
	np_features = {
		'atom_types': atom_types.astype(int),
		'coors': coors.astype(float),
		'num_atoms': np.array([length]),
	}

	return np_features

def pad_np_features(np_features, max_n_res):

	np_features['atom_types'] = np.pad(np_features['atom_types'], (0, max_n_res - len(np_features['atom_types'])), 'constant', constant_values=ATOMNAME_TO_INDEX['MASK'])
	np_features['coors'] = np.pad(np_features['coors'], ((0, max_n_res - len(np_features['coors'])), (0, 0)), 'constant', constant_values=0)
	np_features['mask'] = np.array([1] * np_features['num_atoms'][0] + [0] * (max_n_res - np_features['num_atoms'][0]))

	return np_features

def stack_np_features(np_features_list):
	
	# Stack
	np_features = {
		'atom_types': np.stack([np_features['atom_types'] for np_features in np_features_list]).astype(int),
		'coors': np.stack([np_features['coors'] for np_features in np_features_list]).astype(float),
		'num_atoms': np.array([np_features['num_atoms'][0] for np_features in np_features_list]).astype(int),
		'mask': np.stack([np_features['mask'] for np_features in np_features_list]).astype(int),
	}

	return np_features

def convert_np_features_to_tensor(features, device):

	return {
		key: torch.tensor(value, device='cpu') if isinstance(value, np.ndarray) else value
		for key, value in features.items()
	}

def depadding(features):
	    
    # Depad
    features['atom_types'] = features['atom_types'][:features['num_atoms'][0]]
    features['coors'] = features['coors'][:features['num_atoms'][0]]

    return features