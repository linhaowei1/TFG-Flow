# from dataloader.datasets.qm9 import QM9Dataset
import openbabel as ob
import pybel
from ase.data import atomic_masses
from dataloader.constants import *
import torch
import numpy as np

ATOM2IDX = {
    'C': 6,
    'H': 1,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'MASK': 1
}

def mapping(atoms):
    atomic_symbols = [ATOMNAMES[x.item()] for x in atoms]
    atomic_numbers = torch.tensor([ATOM2IDX[atom] for atom in atomic_symbols])
    return atomic_numbers

def tanimoto_similarity(A, B):
    # Ensure A and B are tensors
    
    # Compute the number of common 1s (intersection)
    intersection = torch.sum(A * B, dim=1)
    
    # Compute the number of 1s in A and B (individual sums)
    sum_A = torch.sum(A, dim=1)
    sum_B = torch.sum(B, dim=1)
    
    # Compute the Tanimoto similarity
    tanimoto = intersection / (sum_A + sum_B - intersection)
    
    return tanimoto  # Return the result as a Python float
    
def compute_fingerprint(poss,numberss,num_atomss):
    fingerprint_1024 = []
    fingerprint_bits = []
    ids = len(num_atomss)
    for i in range(ids):
        pos = poss[i, :]
        numbers = numberss[i, :].squeeze()
        num_atoms = num_atomss[i]

        numbers = numbers[:num_atoms]
        pos = pos[:num_atoms]

        # minius compute mass
        numbers = mapping(numbers)
        m = atomic_masses[numbers]
        com = np.dot(m, pos) / m.sum()
        pos = pos - com

        # order atoms by distance to center of mass
        d = torch.sum(pos ** 2, dim=1)
        center_dists = torch.sqrt(torch.maximum(d, torch.zeros_like(d)))
        idcs_sorted = torch.argsort(center_dists)
        pos = pos[idcs_sorted]
        numbers = numbers[idcs_sorted]

        # Open Babel OBMol representation
        obmol = ob.OBMol()
        obmol.BeginModify()
        # set positions and atomic numbers of all atoms in the molecule
        for p, n in zip(pos, numbers):
            obatom = obmol.NewAtom()
            obatom.SetAtomicNum(int(n))
            obatom.SetVector(*p.tolist())
        # infer bonds and bond order
        obmol.ConnectTheDots()
        obmol.PerceiveBondOrders()
        obmol.EndModify()
        _fp = pybel.Molecule(obmol).calcfp()
        fp_bits = {*_fp.bits}
        fingerprint_bits.append(fp_bits)

        fp_32 = np.array(_fp.fp, dtype=np.uint32)
        # convert fp to 1024bit
        fp_1024 = np.array(fp_32, dtype='<u4')
        fp_1024 = torch.FloatTensor(
            np.unpackbits(fp_1024.view(np.uint8), bitorder='little'))
        fingerprint_1024.append(fp_1024)

    return fingerprint_bits,fingerprint_1024

if __name__ == '__main__':
    dataset = QM9Dataset(max_length=9, partition='train_flow')
    test_data = dataset.features[0]
    compute_fingerprint(test_data['coors'].unsqueeze(0),test_data['atom_types'].unsqueeze(0),test_data['num_atoms'].unsqueeze(0))
