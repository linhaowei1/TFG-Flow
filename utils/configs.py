import os
import torch
from argparse import Namespace
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Union, List
from utils.env_utils import *
from transformers import HfArgumentParser


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Arguments:

    device: str = field(default='cuda')
    dataset: str = field(default='qm9')
    partition: str = field(default='train_flow')

    # diffusion related
    T: int = field(default=1000)
    max_len: int = field(default=9)

    # GNN related
    cls_embed_size: int = field(default=64)
    e_embed_size: int = field(default=256)
    h_embed_size: int = field(default=256)
    condition_embed_size: int = field(default=None)
    num_layers: int = field(default=12)

    # training related
    batch_size: int = field(default=128)
    lr: float = field(default=1e-4)
    epoch: int = field(default=3000)
    ema: bool = field(default=True)
    seed: int = field(default=42)
    warm_up_epochs: int = field(default=20)
    clip_grad: float = field(default=1.0)

    # sample related
    sample_bs: int = field(default=128)
    flow_ckpt: str = field(default=None)
    sample_tot_num: int = field(default=128)

    # guidance related
    do_guidance: bool = field(default=True)
    n_iter: int = field(default=4)
    n_recur: int = field(default=1)
    rho: float = field(default=0.1)
    mu: float = field(default=0.1)
    gamma: float = field(default=0.1)
    num_eps: int = field(default=4)
    k: int = field(default=256)
    temperature: float = field(default=0.5)
    time_delta: int = field(default=10)
    target_property: List[str] = field(default=None)
    guidance_weight: List[float] = field(default=None)

    # log related
    wandb: bool = field(default=False)
    wandb_project: str = field(default='flow')
    wandb_entity: str = field(default=None)
    log_epoch_nums: int = field(default=20)
    log_dir: str = field(default='storage/logs')
    save_mol: bool = field(default=False)

def setup_flow_training():

    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    
    args.device = torch.device(args.device)

    if args.log_dir is None:
        args.log_dir = f'model=flow+num_layers={args.num_layers}+epoch={args.epoch}+seed={args.seed}'
    
    os.makedirs(f'storage/results/{args.log_dir}', exist_ok=True)
    os.makedirs(f'storage/ckpts/{args.log_dir}', exist_ok=True)

    set_seed(args.seed)

    return args

def setup_classifier_training():

    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]

    args.device = torch.device(args.device)

    if args.log_dir is None:
        args.log_dir = f'model=classifier+num_layers={args.num_layers}+epoch={args.epoch}+seed={args.seed}'
    
    if args.target_property is not None:
        args.log_dir += f'+target_name={args.target_property}'
    
    os.makedirs(f'storage/results/{args.log_dir}', exist_ok=True)
    os.makedirs(f'storage/ckpts/{args.log_dir}', exist_ok=True)

    set_seed(args.seed)

    return args

    

def get_logging_dir(args: dict) -> str:

    expr_id = 'dataset={}+k={}+temperature={}+rho={}+mu={}+n_iter={}+target={}+seed={}'.format(args['dataset'], args['k'], args['temperature'], args['rho'], args['mu'], args['n_iter'], args['target_property'], args['seed'])

    return os.path.join(
        args['log_dir'],
        expr_id
    )

def setup_sampling():

    args = HfArgumentParser([Arguments]).parse_args_into_dataclasses()[0]
    
    args.device = torch.device(args.device)
    
    if args.sample_bs > args.sample_tot_num:
        args.sample_bs = args.sample_tot_num
        print('sample_bs should be less than or equal to sample_tot_num')
    
    set_seed(args.seed)

    args.log_dir = get_logging_dir(vars(args))
    
    if args.target_property is not None:
        args.predictor_ckpt = [predictor_ckpt[args.dataset][k] for k in args.target_property]
        args.oracle_ckpt = [oracle_ckpt[args.dataset][k] for k in args.target_property]

        assert len(args.target_property) == len(args.guidance_weight)

    return args