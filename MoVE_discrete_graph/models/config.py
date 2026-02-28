import argparse
import torch
import os

parser = argparse.ArgumentParser()

# 1. dataset
parser.add_argument(
    "--dataset",
    type=str,
    default="collab_04",
    help="collab, yelp, act, collab_04, collab_06, collab_08",
)
parser.add_argument("--num_nodes", type=int, default=-1, help="num of nodes")
parser.add_argument("--nfeat", type=int, default=128, help="dim of input feature")

# 2. experiments
parser.add_argument("--use_cfg", type=int, default=1, help="if use configs")
parser.add_argument(
    "--max_epoch", type=int, default=200, help="number of epochs to train"
)
parser.add_argument(
    "--min_epoch", type=int, default=50, help="min_epoch"
)
parser.add_argument("--testlength", type=int, default=3, help="length for test")
parser.add_argument("--device", type=str, default="cuda:0", help="training device")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--patience", type=int, default=20, help="patience for early stop")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-7,
    help="weight for L2 loss on basic models",
)
parser.add_argument("--output_folder", type=str, default="", help="need to be modified")
parser.add_argument(
    "--sampling_times", type=int, default=1, help="negative sampling times"
)
parser.add_argument("--log_dir", type=str, default="../logs/")
parser.add_argument(
    "--log_interval", type=int, default=10, help="every n epoches to log"
)
parser.add_argument("--nhid", type=int, default=8, help="dim of hidden embedding")
parser.add_argument(
    "--delta_d", type=int, default=16, help="dimension under each environment"
)
parser.add_argument("--n_layers", type=int, default=1, help="number of hidden layers")
parser.add_argument("--heads", type=int, default=4, help="attention heads")
parser.add_argument("--n_factors", type=int, default=8, help="latent factors")
parser.add_argument("--norm", type=int, default=1, help="normalization")
parser.add_argument("--nbsz", type=int, default=10, help="number of sampling neighbors")
parser.add_argument("--maxiter", type=int, default=4, help="number of iteration")
parser.add_argument("--dropout", type=float, default=0.4, help="dropout rate")
parser.add_argument(
    "--alpha", type=float, default=0.0001, help="parameter of alignment loss"
)
parser.add_argument("--beta", type=float, default=0.1, help="parameter of regularization loss")
parser.add_argument("--gamma_mu", type=float, default=0.1, help="regulate the diversity of pseudo_environments")
parser.add_argument("--gamma_std", type=float, default=0.9, help="regulate the diversity of pseudo_environments")
parser.add_argument("--k", type=int, default=5, help="number of augmentation branches")
parser.add_argument("--env_lr", type=float, default=0.0001, help="learning rate of environment estimator weight")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate of main model weight")
parser.add_argument("--num_runs", type=int, default=3, help="number of runs")
parser.add_argument("--dim", type=int, default=32, help="dim")


args = parser.parse_args()

# set the running device
if torch.cuda.is_available():
    args.device = torch.device("cuda:0")
    print("using gpu:0 to train the model")
else:
    args.device = torch.device("cpu")
    print("using cpu to train the model")


def setargs(args, hp):
    for k, v in hp.items():
        setattr(args, k, v)


if args.use_cfg:
    if args.dataset == "collab":
        hp = {
            "n_factors": 4,
            "delta_d": 8,
            "nbsz": 32,
        }
        setargs(args, hp)
    if args.dataset == "yelp":
        hp = {
             "n_factors": 4,
            "delta_d": 8,
            "nbsz": 32,
        }
        setargs(args, hp)
    elif args.dataset == "act":
        hp = {
            "n_factors": 4,
            "delta_d": 8,
            "nbsz": 32,
        }
        setargs(args, hp)
    elif args.dataset == "collab_04":
        hp = {
            "n_factors": 4,
            "delta_d": 16,
            "nbsz": 32,
            "dim": 64
        }
        setargs(args, hp)
    elif args.dataset == "collab_06":
        hp = {
            "n_factors": 4,
            "delta_d": 16,
            "nbsz": 32,
            "dim": 64
        }
        setargs(args, hp)
    elif args.dataset == "collab_08":
        hp = {
            "n_factors": 4,
            "delta_d": 16,
            "nbsz": 32,
            "dim": 64
        }
        setargs(args, hp)
