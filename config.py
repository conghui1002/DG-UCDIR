import builtins
import warnings
import argparse
import logging
import os
from datetime import datetime
import torchvision.models as models
import torch.distributed as dist
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='comma_separated list of paths to textfiles')
parser.add_argument('--eval-data', metavar='DIR',
                    help='comma_separated list of eval paths to textfiles')
parser.add_argument('--prec-nums', default='1,5,15', type=str,
                    help='the evaluation metric')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--imagenet_pretrained', type=str, default='imagenet',
                    help='Initilaized with imagenet (supervised) pretrained weights, can be empty string for random '
                         'init, a path to a checkpoint file, any other string will load the torchvision weights.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers per domain (default: 1)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=288, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')  # [120, 160]
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

# checkpoints config
parser.add_argument('--clean-model', default=None, type=str, metavar='PATH',
                    help='path to clean model (default: none)')
parser.add_argument('--exp_root', default='experiments', type=str, metavar='PATH',
                    help='path to the root folder for all the experiments (default: none)')
parser.add_argument('--exp_folder_name',
                    default='default',
                    type=str, metavar='PATH',
                    help='The name of the experiment. Supports naming formating by parameter config by using'
                         ' "{PARM_NAME}" in string')
parser.add_argument('--save_n_epochs', default=10, type=int, help='Save every N epochs (default 10)')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.2, type=float,
                    help='softmax temperature (default: 0.07)')

parser.add_argument('--mlp', action='store_true',
                    help='use mlp head') # TODO: Default True
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')  # TODO: Default True
parser.add_argument('--step_scheduler', action='store_false', dest='cos',
                    help='Do not use cosine lr schedule')

parser.add_argument('--multi_q', action='store_true',
                    help='if true would have a Q for every dataset (domain)')

parser.add_argument('--local_rank', default=0, type=int,
                    help='for multi-node DDP, gpu index for torch.distributed.launch run invocation mode')

parser.add_argument('--bn-splits', default=0, type=int,
                    help='simulate multi-gpu behavior of Batchnorm in one gpu')

# Other configs for DG-UCDIR

parser.add_argument('--num_cluster', default=100, type=int,
                    help='number of cluster')

parser.add_argument('--warmup_epoch', default=20, type=int,
                    help='number of warmup epoch')

parser.add_argument('--cluster_loss_w', default=1.0, type=float,
                    help='weight for clustering contrastive loss')

parser.add_argument('--aug_alpha', default=1.0, type=float,
                    help='alpha value for data augmentation')

parser.add_argument('--hpf_range', default=20, type=int,
                    help='the range to apply hpf in image phase')

parser.add_argument('--hpf_alpha', default=0.5, type=float,
                    help='alpha value for hpf data augmentation')

parser.add_argument('--eval_same', action='store_true',
                    help='evaluate while training')

parser.add_argument('--contra_intra_phase', default=1.0, type=float,
                    help='weight for phase intra contrastive loss')

parser.add_argument('--contra_intra_rgb', default=1.0, type=float,
                    help='weight for rgb intra contrastive loss')

parser.add_argument('--contra_cross_phase', default=1.0, type=float,
                    help='weight for phase cross contrastive loss')

parser.add_argument('--contra_cross_rgb', default=1.0, type=float,
                    help='weight for rgb cross contrastive loss')

def setup(args):

    args.rank = dist.get_rank()
    args.is_root = is_root = args.rank == 0

    # fix moco-k in case batch size was manipulated
    real_bs = args.batch_size*2
    args.moco_k -= (args.moco_k % real_bs)
    args.real_bs = real_bs

    # work folder setup
    work_folder = os.path.join(args.exp_root, args.exp_folder_name.format(**args.__dict__))
    os.makedirs(work_folder, exist_ok=True)
    args.work_folder = work_folder

    # logger setup
    if args.is_root:  # TODO: fix logger so the outputs are in stdout, and maybe have several file for each process
        logger = logging.getLogger('DG_UCDIR')
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        log_file_path = os.path.join(work_folder, f'log_{args.rank}.out')
        print('log_file_path: {}'.format(log_file_path))
        try:
            if os.path.isfile(log_file_path) and args.is_root:
                timestamp_str = datetime.now().strftime("%m_%d_%Y.%H.%M.%S")
                bckp_pth = log_file_path.replace(f'log_{args.rank}.out', f'log_{args.rank}_{timestamp_str}.out')
                os.rename(log_file_path, bckp_pth)
        except FileNotFoundError:
            pass
        if args.is_root:
            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(message)s')  # %(asctime)s - %(name)s - %(levelname)s
        if args.is_root:
            fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        if args.is_root:
            logger.addHandler(fh)
        logger.addHandler(ch)
        args.logger = logger

    # suppress printing if not master
    if not args.is_root: # TODO: fix logger to write from all processes
        def print_pass(*args):
            pass
        builtins.print = print_pass
    else:
        def info(*a, **ka):
            args.logger.info(a[0])

        def warn(*a, **ka):
            args.logger.warning(a[0])

        builtins.print = info
        warnings.warn = warn

    return args
