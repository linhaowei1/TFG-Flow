from openbabel import openbabel
from dataloader.constants import *

def unbatch(coors, atom_types, batch):

    batch = batch.to(coors.device)

    bs = batch.batch[-1]

    mols = []
    
    for i in range(bs):
        
        batch_mask = (batch.batch == i).bool()
        ligand_mask_b = batch.ligand_mask[batch_mask]

        atom_types_b = atom_types[batch_mask][ligand_mask_b]
        coors_b = coors[batch_mask][ligand_mask_b]

        mols.append((coors_b, atom_types_b))
    
    return mols

def reconstruct(coors, atom_types, fn):

    mol = openbabel.OBMol()

    coordinates = coors.cpu().numpy().tolist()
    atom_types = atom_types.cpu().numpy().tolist() 

    for atom_type, coord in zip(atom_types, coordinates):
        atom = mol.NewAtom()  
        atom.SetAtomicNum(atom_type)  
        atom.SetVector(*coord)  

    mol.ConnectTheDots()  
    mol.PerceiveBondOrders() 

    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat("mol") 
    obConversion.WriteFile(mol, f"{fn}.mol")
