from rdkit import Chem
import numpy as np
from dataloader.molecule import construct_mol_conformation
from utils import bond_analyze
from dataloader.constants import *


#### New implementation ####

bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                 Chem.rdchem.BondType.AROMATIC]


class Evaluator(object):
    def __init__(self, dataset_smiles_list=None):
        self.dataset_smiles_list = dataset_smiles_list

    def compute_validity_one_mol(self, c, a):
        """ generated: list of couples (positions, atom_types)"""
        
        largest_mol = None

        try:
            mol = construct_mol_conformation(c.cpu().numpy(), a.cpu().numpy())
            smiles = mol2smiles(mol)
        except:
            mol = None
            smiles = None
            
        if smiles is not None:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smiles = mol2smiles(largest_mol)
        
        return smiles, largest_mol

    def compute_connectivity_one_mol(self, c, a):
        try:
            mol = construct_mol_conformation(c.cpu().numpy(), a.cpu().numpy())
            smiles = mol2smiles(mol)
        except:
            mol = None
            smiles = None

        if smiles is not None:
            return '.' not in smiles
    
        return False

    def compute_stability_one_mol(self, coors, atom_types):

        stable_atoms = []
        atom_elements = [ATOMNAME_TO_INDEX['H'] if atom == ATOMNAME_TO_INDEX['MASK'] else atom for atom in atom_types]
        atom_orders = [0] * len(atom_elements)

        for i in range(len(atom_elements)):
            for j in range(i+1, len(atom_elements)):
                rel_dist = np.linalg.norm(coors[i] - coors[j])
                order = bond_analyze.get_bond_order(ATOMNAMES[atom_elements[i]], ATOMNAMES[atom_elements[j]], rel_dist)
                atom_orders[i] += order
        
        for i in range(len(atom_elements)):
            allowed_bonds_ = bond_analyze.allowed_bonds[ATOMNAMES[atom_elements[i]]]
            if isinstance(allowed_bonds_, list):
                allowed_bonds_ = max(allowed_bonds_)
            if atom_orders[i] <= allowed_bonds_:
                stable_atoms.append(i)

        return len(stable_atoms) / len(atom_elements)

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def evaluate(self, generated):
        
        valid = []
        valid_mols = []
        stability = []
        connectivity = []

        for i, (c, a) in enumerate(generated):
            
            smiles, mol = self.compute_validity_one_mol(c, a)
            stability.append(self.compute_stability_one_mol(c, a))
            connectivity.append(self.compute_connectivity_one_mol(c, a))
            if smiles is not None:
                valid.append(smiles)
                valid_mols.append(mol)

        connectivity = np.mean(connectivity)
        print(f"Connectivity over {len(valid)} valid molecules: {connectivity * 100 :.2f}%")
            
        atom_stability = np.mean(stability)
        print(f"Stability over {len(valid)} valid molecules: {atom_stability * 100 :.2f}%") 
        
        mol_stability = sum([x == 1 for x in stability]) / len(stability)
        print(f"Molecule stability over {len(valid)} valid molecules: {mol_stability * 100 :.2f}%")
        
        validity = len(valid) / len(generated)
        print(f"Validity over {len(generated)} generated molecules: {validity * 100 :.2f}%")

        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = 0.0
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
    
        return {
            'validity': validity,
            'valid_smiles': valid,
            'connectivity': connectivity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'mol_stability': mol_stability,
            'atom_stability': atom_stability,
            'valid_mols': valid_mols
        }


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)

