import torch
import numpy as np
import sys
import yaml
sys.path.append("..")
sys.path.append("../..")
from visd_transformer import VisdTransformer
import torch.nn as nn
import argparse
import time
import pdb

parser = argparse.ArgumentParser('Dense Co-Attention Encoder Test')
parser.add_argument('--n-embd',
                    type=int,
                    default=768,
                    help='Number of the unified embedding dimension.'
                   )
parser.add_argument('--rounds',
                    type=int,
                    default=10,
                    help='Rounds of dialog'
                   )
parser.add_argument('--n-head',
                    type=int,
                    default=12,
                    help='Rounds of dialog'
                   )
parser.add_argument('--n-ctx',
                    type=int,
                    default=20,
                    help='Number of maximum sequence length.'
                   )
parser.add_argument('--n-layer',
                    type=int,
                    default=2,
                    help='Number of stacked transformer layers'
                   )
parser.add_argument('--attn-pdrop',
                    type=float,
                    default=0.5,
                    help='Attention dropout rate.'
                   )
parser.add_argument('--resid-pdrop',
                    type=float,
                    default=0.5,
                    help='Residual dropout rate'
                   )
parser.add_argument('--n-positions',
                    type=int,
                    default=48,
                    help='Number of positions.'
                   )
parser.add_argument('--layer-norm-epsilon',
                    type=float,
                    default=0.00001,
                    help='Layer norm epsilon'
                   )
parser.add_argument('--gpu-ids',
                    nargs='+',
                    type=int,
                    default=0,
                    help='GPU ids to be used.'
                   )

def main(args):
    #pdb.set_trace()
    batch = 32
    rounds = args.rounds
    n_embd = args.n_embd

    config = vars(args)
    if isinstance(args.gpu_ids, int):
       args.gpu_ids = [args.gpu_ids]
    device = (
       torch.device("cuda", args.gpu_ids[0])
       if args.gpu_ids[0] >= 0
       else torch.device("cpu")
    )
    fusioner = VisdTransformer(config=config)
    fusioner = fusioner.to(device)
    if -1 not in args.gpu_ids:
       fusioner = nn.DataParallel(fusioner, args.gpu_ids)
    fusioner.eval()
    torch.manual_seed(231)
    history = torch.randn(batch, rounds, n_embd).to(device) #mimic history content coded by dense-coatt-encoder
    question = torch.randn(batch, rounds, n_embd).to(device) #mimic question

    hist, ques = fusioner(history, question)
    assert hist.size() == history.size()
    assert ques.size() == question.size()
    print("Pass VisdTransformer sanity check for output shape")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
