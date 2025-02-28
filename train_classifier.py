from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np
import wandb
from torch.optim import AdamW
from dataloader.datasets.geom_drug import GEOMDrugDataset
from dataloader.datasets.qm9 import QM9Dataset
from dataloader.constants import *
from diffusion.predictor import Predictor
from utils.configs import setup_classifier_training

# set seed

args = setup_classifier_training()

if args.wandb:
    wandb.init(project=args.wandb_project, config=args, entity=args.wandb_entity, name=args.log_dir)

# Load the dataset
train_split = 'train_flow' if 'guide' in args.log_dir else 'train_classifier'

if args.dataset == 'geom_drug':
    dataset = GEOMDrugDataset(partition=train_split)
    testset = GEOMDrugDataset(partition='validation')
elif args.dataset == 'qm9':
    dataset = QM9Dataset(partition=train_split)
    testset = QM9Dataset(partition='validation')

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, worker_init_fn=np.random.seed(args.seed))

testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Initialize the model
model = Predictor(
    max_len=args.max_len,
    device=args.device,
    h_embed_size=args.h_embed_size,
    num_layers=args.num_layers,
    cls_embed_size=args.cls_embed_size,
    e_embed_size=args.e_embed_size,
    class_num=1024 if 'structure' in args.target_property else 1
).to(args.device)


optimizer = AdamW(model.parameters(), lr=args.lr)

best_valid_loss = 1e09

# Train the model
for epoch in range(args.epoch):
    
    model.train()

    total_loss = total_loss_coors = total_loss_atom = 0.

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', total=len(dataloader))
    for batch in progress_bar:
        
        loss = model.get_loss_batch(batch, args.target_property[0], args)
        
        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        optimizer.zero_grad()

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        if args.wandb:
            wandb.log({'lr': current_lr})

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    total_loss /= len(dataloader)

    print(f'Epoch {epoch} | Loss: {total_loss}')
    
    if args.wandb:
        wandb.log({'loss': total_loss})

    with torch.no_grad():
        if epoch % args.log_epoch_nums == 0:
            
            model.eval()

            mae_total = 0

            for batch in testloader:
                prediction = model.get_prediction(batch, args)
                for target in args.target_property:
                    if target == 'structure':
                        mae = nn.functional.binary_cross_entropy(prediction, batch[target].to(args.device))
                    else:
                        mae = nn.functional.l1_loss(prediction, batch[target].to(args.device))
                mae_total += mae.item()
        
            mae_total /= len(testloader)
            print(f'mae: {mae_total}')
            if args.wandb:
                wandb.log({'mae': mae_total})

            if mae_total < best_valid_loss:
                best_valid_loss = mae_total
                torch.save(model.state_dict(), f'storage/ckpts/{args.log_dir}/best_loss={best_valid_loss}.pth')
                print('Model saved!')

