import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from utils.bond_analyze import get_bond_order

from dataloader.constants import *
from dataloader.feature_utils import create_empty_np_features, pad_np_features, stack_np_features, convert_np_features_to_tensor


def get_conformation(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def parse_smiles(smiles):

    atoms, coords = [], []
    mol = get_conformation(smiles)
    # remove Hs
    mol = Chem.RemoveHs(mol)
    for atom in mol.GetAtoms():
        atom_type = ATOMNAME_TO_INDEX[atom.GetSymbol()]
        atoms.append(atom_type)
        coords.append(mol.GetConformer().GetAtomPosition(atom.GetIdx()))

    return atoms, coords

def create_np_features_from_smiles(smiles):
    # Load
    atom_types, coors = parse_smiles(smiles)
    length = len(atom_types)
    
    # Generate
    np_features = create_empty_np_features(length)

    # Generate
    atom_types = np.array(atom_types)

    atom_positions = np.stack(coors)
    atom_positions = atom_positions - np.mean(atom_positions, axis=0, keepdims=True)

    # Update
    np_features['atom_types'] = atom_types.astype(int)
    np_features['coors'] = atom_positions.astype(float)

    return np_features

def construct_mol_conformation(atom_positions, atom_elements):
    # atom_positions: (n, 3)
    # atom_elements: (n)

    # Create an empty editable molecule
    mol = Chem.RWMol()
    
    # Create a dictionary to store atom indices
    atom_indices = {}

    # prevent error from MASK
    atom_elements = [ATOMNAME_TO_INDEX['H'] if atom == ATOMNAME_TO_INDEX['MASK'] else atom for atom in atom_elements]

    # Add atoms to the molecule
    for i, (element, pos) in enumerate(zip(atom_elements, atom_positions)):
        atom = Chem.Atom(ATOMNAMES[element])
        idx = mol.AddAtom(atom)
        atom_indices[i] = idx
        # Set the 3D coordinates of the atom
    
    conf = Chem.Conformer(len(atom_elements))
    mol.AddConformer(conf)

    for i, (element, pos) in enumerate(zip(atom_elements, atom_positions)):
        mol.GetConformer().SetAtomPosition(i, Point3D(*pos.astype(np.double)))
    
    # Add bonds between atoms
    for i in range(len(atom_elements)):
        for j in range(i + 1, len(atom_elements)):
            rel_dist = np.linalg.norm(atom_positions[i] - atom_positions[j])
            order = get_bond_order(ATOMNAMES[atom_elements[i]], ATOMNAMES[atom_elements[j]], rel_dist, check_exists=True)
            if order == 0:
                continue
            elif order == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif order == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif order == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
            mol.AddBond(atom_indices[i], atom_indices[j], bond_type)
    
    return mol

def save_mols(mols, filepath):
    writer = Chem.SDWriter(filepath)
    for mol in mols:
        writer.write(mol)
    writer.close()

def calculate_rmsd(mol1, mol2):
    
    mol1 = Chem.RemoveAllHs(mol1)
    mol2 = Chem.RemoveAllHs(mol2)

    if mol1.GetNumAtoms() != mol2.GetNumAtoms():
        print(mol1.GetNumAtoms(), mol2.GetNumAtoms())
        raise ValueError("Molecules must have the same number of atoms.")
    
    rmsd = AllChem.GetBestRMS(mol1, mol2)
    
    return rmsd

def get_valid_mols(mols):
    valid_mols = []
    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        valid_mols.append(mol)
    return valid_mols

def node_num_distribution(np_features, max_length):

    dist = {i: 0 for i in range(1, max_length+1)}

    for feature in np_features:
        num_nodes = feature['num_atoms']
        dist[num_nodes] += 1
    
    return dist

