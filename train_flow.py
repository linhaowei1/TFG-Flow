from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import wandb
from rdkit import Chem
from ema_pytorch import EMA
from torch.optim import AdamW
from dataloader.molecule import save_mols
from dataloader.datasets.qm9 import QM9Dataset
from dataloader.datasets.geom_drug import GEOMDrugDataset
from dataloader.constants import *
from diffusion.aaflow import AAFlow
from utils.configs import setup_flow_training
from utils.evaluator import Evaluator
from dataloader.property import DistributionProperty

# set seed

args = setup_flow_training()

if args.wandb:
    wandb.init(project=args.wandb_project, config=args, entity=args.wandb_entity, name=args.log_dir)

# Load the dataset
if args.dataset == 'geom_drug':
    dataset = GEOMDrugDataset(partition='train_flow')
    testset = GEOMDrugDataset(partition='validation')
elif args.dataset == 'qm9':
    dataset = QM9Dataset(partition='train_flow')
    testset = QM9Dataset(partition='validation')

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, worker_init_fn=np.random.seed(args.seed))
testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

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
).to(args.device)

if args.ema:
    ema = EMA(
        model,
        beta = 0.9999,              # exponential moving average factor
        update_after_step = 100,    # only after this number of .update() calls will it start updating
        update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    )

optimizer = AdamW(model.parameters(), lr=args.lr, amsgrad=True, weight_decay=1e-12)

evaluator = Evaluator(dataset.get_smiles(), )

dist = None
 
if args.target_property is not None:
    dist = DistributionProperty(dataset, properties=args.target_property, num_bins=1000)

best_valid_loss = 1e09

# Train the model
for epoch in range(args.epoch):
    
    model.train()

    total_loss = total_loss_coors = total_loss_atom = 0.

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', total=len(dataloader))
    for batch in progress_bar:
        
        loss_coors, loss_atom, loss = model.get_loss_batch(batch, args)
        
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()

        if args.ema:
            ema.update()

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        if args.wandb:
            wandb.log({'lr': current_lr})

        total_loss += loss.item()
        total_loss_coors += loss_coors.item()
        total_loss_atom += loss_atom.item()

        progress_bar.set_postfix({'loss': loss.item(), 'loss_coors': loss_coors.item(), 'loss_atom': loss_atom.item()})

    total_loss /= len(dataloader)
    total_loss_coors /= len(dataloader)
    total_loss_atom /= len(dataloader)

    print(f'Epoch {epoch} | Loss: {total_loss} | Loss Coors: {total_loss_coors} | Loss Atom: {total_loss_atom}')
    
    if args.wandb:
        wandb.log({'loss': total_loss, 'loss_coors': total_loss_coors, 'loss_atom': total_loss_atom})

    with torch.no_grad():
        if epoch % args.log_epoch_nums == 0:
            
            ema.ema_model.eval()
            model.eval()

            total_eval_loss = 0.
            total_eval_loss_coors = 0.
            total_eval_loss_atom = 0.

            for batch in tqdm(testloader, desc=f'Test Epoch {epoch}', total=len(testloader)):
                
                loss_coors, loss_atom, loss = model.get_loss_batch(batch, args)
                
                total_eval_loss += loss.item()
                total_eval_loss_coors += loss_coors.item()
                total_eval_loss_atom += loss_atom.item()
            
            total_eval_loss /= len(testloader)
            total_eval_loss_coors /= len(testloader)
            total_eval_loss_atom /= len(testloader)

            print(f'Epoch {epoch} | Eval Loss: {total_eval_loss} | Eval Loss Coors: {total_eval_loss_coors} | Eval Loss Atom: {total_eval_loss_atom}')
            if args.wandb:
                wandb.log({'eval_loss': total_eval_loss, 'eval_loss_coors': total_eval_loss_coors, 'eval_loss_atom': total_eval_loss_atom})

            torch.save(model.state_dict(), f'storage/ckpts/{args.log_dir}/epoch={epoch}+loss={total_eval_loss}.pth')

            lengths = dataset.node_num_dist.sample_lengths(torch.tensor([args.sample_bs]))

            if dist is not None:
                target_value = dist.sample_batch([length for length in lengths]).to(args.device)
                if args.target_property[0] == 'structure':
                    # we must keep structure as the first target!
                    target_value = target_value.split(1024, dim=1)
                else:
                    target_value = target_value.split(1, dim=1)
                coors, atom_types, _ = ema.ema_model.sample(lengths, args, target_value)
            else:
                coors, atom_types, _ = ema.ema_model.sample(lengths, args, )

            results = evaluator.evaluate(list(zip(coors, atom_types)))

            save_mols(results['valid_mols'], f'storage/results/{args.log_dir}/{epoch}.sdf')
           
            for key, value in results.items():
                if key not in ['valid_mols', 'valid_smiles']:
                    print(f"{key}: {value * 100 :.2f}%")

            if args.wandb:
                wandb.log({
                    k: v for k, v in results.items() if k not in ['valid_mols', 'valid_smiles']
                })
            
