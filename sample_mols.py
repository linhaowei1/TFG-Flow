import json
import torch
import numpy as np
import os
from dataloader.constants import *
from diffusion.aaflow import AAFlow
from diffusion.predictor import Predictor
from diffusion.guidance import ConditionalFlow
from utils.configs import setup_sampling
from utils.evaluator import Evaluator
from dataloader.datasets.qm9 import QM9Dataset
from dataloader.datasets.geom_drug import GEOMDrugDataset
from dataloader.property import DistributionProperty
from dataloader.molecule import construct_mol_conformation, save_mols
import wandb

args = setup_sampling()

if args.wandb:
    wandb.init(project=args.project, config=args, name=args.log_dir, entity=args.entity)

if os.path.exists(f'{args.log_dir}/results.json'):
    print(f'{args.log_dir}/results.json already exists. Skipping...')
    exit()

os.makedirs(f'{args.log_dir}', exist_ok=True)

# Initialize the model
model = AAFlow(
    T=args.T,
    max_len=args.max_len,
    device=args.device,
    h_embed_size=args.h_embed_size,
    num_layers=args.num_layers,
    cls_embed_size=args.cls_embed_size,
    e_embed_size=args.e_embed_size,
    condition_embed_size=args.condition_embed_size,
    target_property=args.target_property,
)

predictor = []
oracle = []

for i, target in enumerate(args.target_property):
    
    predictor_ = Predictor(max_len=args.max_len, device=args.device, h_embed_size=128, num_layers=6, cls_embed_size=args.cls_embed_size, e_embed_size=128, class_num=1024 if 'structure' in args.target_property else 1)
    
    oracle_ = Predictor(max_len=args.max_len, device=args.device, h_embed_size=128, num_layers=6, cls_embed_size=args.cls_embed_size, e_embed_size=128, class_num=1024 if 'structure' in args.target_property else 1)
    
    predictor_.load_state_dict(torch.load(args.predictor_ckpt[i], map_location='cpu'))
    predictor_.to(args.device)
    predictor_.eval()

    oracle_.load_state_dict(torch.load(args.oracle_ckpt[i], map_location='cpu'))
    oracle_.to(args.device)
    oracle_.eval()

    predictor.append(predictor_)
    oracle.append(oracle_)

ckpts = torch.load(args.flow_ckpt, map_location='cpu')
model.load_state_dict(ckpts)
model.to(args.device)
model.eval()

if args.dataset == 'qm9':
    dataset = QM9Dataset(args.max_len, partition='train_flow' if 'structure' not in args.target_property else 'train_classifier', fingerprint=True if 'structure' in args.target_property else False)
elif args.dataset == 'geom_drug':
    dataset = GEOMDrugDataset(partition='train_flow' if 'structure' not in args.target_property else 'train_classifier')

evaluator = Evaluator(dataset.get_smiles())
dist = DistributionProperty(dataset, properties=args.target_property, num_bins=1000)

conditional_flow = ConditionalFlow(model, predictor, oracle, dist, k=args.k, temperature=args.temperature, enable=args.do_guidance, n_iter=args.n_iter, n_recur=args.n_recur, rho=args.rho, mu=args.mu, gamma=args.gamma, num_eps=args.num_eps, guidance_weight=args.guidance_weight)

global_results = {
    'validity': 0,
    'uniqueness': 0,
    'novelty': 0,
    'mol_stability': 0,
    'atom_stability': 0, 
    'connectivity': 0,
    'mae_1': 0,
    'mae_2': 0,
}
valid_smiles = []

for _ in range(args.sample_tot_num // args.sample_bs):

    with torch.no_grad():

        failed = 0

        lengths = dataset.node_num_dist.sample_lengths(torch.tensor([args.sample_bs]))
        coors, atom_types, traj, mae = conditional_flow.sample(lengths, args.time_delta, args)
        
        results = evaluator.evaluate(list(zip(coors, atom_types)))
        # # save the results
        if args.save_mol:
            for i in range(args.sample_bs):
                mols = []
                for time in range(args.T // args.time_delta):
                    mol = traj[time][i]
                    try:
                        mol = construct_mol_conformation(mol[0].cpu().numpy(), mol[1].cpu().numpy())
                    except:
                        continue
                    mols.append(mol)
                # time = args.T // args.time_delta - 1
                # mol = traj[time][i]
                # try:
                #     mol = construct_mol_conformation(mol[0].cpu().numpy(), mol[1].cpu().numpy())
                # except:
                #     continue
                # mols.append(mol)
                if len(mols) > 0:
                    print(args.log_dir)
                    save_mols(mols, f'{args.log_dir}/{i}.sdf')

        # print the evaluation results: validity, uniqueness, novelty, stability
        print(f"Validity: {results['validity'] * 100 :.2f}%")
        print(f"Uniqueness: {results['uniqueness'] * 100 :.2f}%")
        print(f"Novelty: {results['novelty'] * 100 :.2f}%")
        print(f"Molecule Stability: {results['mol_stability'] * 100 :.2f}%")
        print(f"Atom Stability: {results['atom_stability'] * 100 :.2f}%")
        print(f"Mean absolute Error: {[mae_.mean().item() for mae_ in mae]}")
        print(f"Connectivity: {results['connectivity'] * 100 :.2f}%")

        if args.wandb:
            wandb.log({
                'validity': results['validity'],
                'uniqueness': results['uniqueness'],
                'novelty': results['novelty'],
                'mol_stability': results['mol_stability'],
                'atom_stability': results['atom_stability'],
                'connectivity': results['connectivity'],
                'mae_1': mae[0].mean().item(),
                'mae_2': 0 if len(mae) == 1 else mae[1].mean().item(),
            })
        
        global_results['validity'] += results['validity']
        global_results['uniqueness'] += results['uniqueness']
        global_results['novelty'] += results['novelty']
        global_results['mol_stability'] += results['mol_stability']
        global_results['atom_stability'] += results['atom_stability']
        global_results['connectivity'] += results['connectivity']
        global_results['mae_1'] += mae[0].mean().item()
        global_results['mae_2'] += 0 if len(mae) == 1 else mae[1].mean().item()

        valid_smiles.extend(results['valid_smiles'])

for key in global_results.keys():
    global_results[key] /= (args.sample_tot_num // args.sample_bs)

uniqueness = len(set(valid_smiles)) / len(valid_smiles)
global_results['uniqueness'] = uniqueness

if args.wandb:
    wandb.log(global_results)

print(f"Global Results:")
print(f"Validity: {global_results['validity'] * 100 :.2f}%")
print(f"Uniqueness: {global_results['uniqueness'] * 100 :.2f}%")
print(f"Novelty: {global_results['novelty'] * 100 :.2f}%")
print(f"Molecule Stability: {global_results['mol_stability'] * 100 :.2f}%")
print(f"Atom Stability: {global_results['atom_stability'] * 100 :.2f}%")
print(f"Connectivity: {global_results['connectivity'] * 100 :.2f}%")
print(f"Uniqueness: {uniqueness * 100 :.2f}%")
print(f"Mean absolute Error 1: {global_results['mae_1']}")
print(f"Mean absolute Error 2: {global_results['mae_2']}")

# global_results['search_score'] = -global_results['validity'] * 100 if global_results['validity'] < 0.75 else -global_results['mae']

with open(f'{args.log_dir}/results.json', 'w') as f:
    json.dump(global_results, f)