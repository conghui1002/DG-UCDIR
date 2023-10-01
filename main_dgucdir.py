#!/usr/bin/env python
import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import math
import os
import shutil
import time
import warnings
import glob
import faiss
from tqdm import tqdm
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils.torchvision_wrappers as models_wrappers
from utils import domainnet
from utils.data import DataCoLoader, IndexedDataset, ImageNormalize
import moco.loader
from config import parser, setup
from moco.builder import concat_all_gather, concat_all_gather_interlace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

def main(args=None):
    if args is None:
        args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    init_ddp(args.local_rank)

    main_worker(args.local_rank, args)


def init_ddp(local_rank):
    if 'RANK' not in os.environ.keys():
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ.keys():
        os.environ['LOCAL_RANK'] = str(local_rank)
    if 'WORLD_SIZE' not in os.environ.keys():
        os.environ['WORLD_SIZE'] = '1'
    if 'MASTER_PORT' not in os.environ.keys():
        os.environ['MASTER_PORT'] = '55555'
    if 'MASTER_ADDR' not in os.environ.keys():
        os.environ['MASTER_ADDR'] = 'localhost'

    torch.distributed.init_process_group('nccl')


def main_worker(gpu, args):
    args.gpu = gpu
    args = setup(args)  # TODO: Go over setup

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    print("=> creating model '{}'".format(args.arch))
    import moco.builder
    model = moco.builder.MoCo(
        models_wrappers.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,
        args=args
    )

    print(f'=> original args.batch_size={args.batch_size}')

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / dist.get_world_size())
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)

    print(f'=> modified args.batch_size={args.batch_size}')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.clean_model:
        if os.path.isfile(args.clean_model):
            print("=> loading pretrained clean model '{}'".format(args.clean_model))

            if args.gpu is None:
                clean_checkpoint = torch.load(args.clean_model)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                clean_checkpoint = torch.load(args.clean_model, map_location=loc)

            current_state = model.state_dict()
            used_pretrained_state = {}

            for k in current_state:
                k_parts = '.'.join(k.split('.')[2:])
                if 'encoder' in k and current_state[k].shape == clean_checkpoint['state_dict']['module.encoder_q.'+k_parts].shape:
                    used_pretrained_state[k] = clean_checkpoint['state_dict']['module.encoder_q.'+k_parts]
                else:
                    print(k)
            current_state.update(used_pretrained_state)
            model.load_state_dict(current_state)
        else:
            print("=> no clean model found at '{}'".format(args.clean_model))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        pre_augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        pre_augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
        ]
    pre_transform = transforms.Compose(pre_augmentation)

    post_augmentation = [
        transforms.ToTensor(),
        normalize
    ]

    post_transform = transforms.Compose(post_augmentation)

    eval_pre_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224)])

    eval_post_transform = transforms.Compose(
        [transforms.ToTensor(),
         normalize])

    train_loaders = []
    train_samplers = []
    datas = args.data.split(',')
    assert args.batch_size % len(datas) == 0, f'for simplicity, please make sure args.batch_size={args.batch_size} ' \
                                              f'is divisible by len(datasets)={len(datas)}'
    for iData, data in enumerate(datas):
        if os.path.isdir(data):
            print(f'=> Loading a custom dataset: {data}')
            train_dataset = datasets.ImageFolder(data, pre_transform, post_transform)
            print(f'=> loaded {len(train_dataset)} images')
        else:
            train_dataset = domainnet.Dataset(data, root=os.path.dirname(data), pre_transform=pre_transform,
                                              post_transform=post_transform, all_train_image_list=datas,
                                              alpha=args.aug_alpha, hpf_range=args.hpf_range,
                                              hpf_alpha=args.hpf_alpha)

        train_dataset = IndexedDataset(train_dataset, args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_samplers.append(train_sampler)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=int(args.batch_size / len(datas)), shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

        train_loaders.append(train_loader)

    if len(train_loaders) > 1:
        train_loader = DataCoLoader(train_loaders, args)
    else:
        train_loader = train_loaders[0]

    train_eval_loaders = []
    train_eval_samplers = []
    for iData, data in enumerate(datas):
        train_eval_dataset = domainnet.Dataset(data, train_val=True, root=os.path.dirname(data),
                                               pre_transform=eval_pre_transform,
                                               post_transform=eval_post_transform, alpha=args.aug_alpha,
                                               hpf_range=args.hpf_range, hpf_alpha=args.hpf_alpha)

        # we wrap this to get the indices and paths for the batch out
        train_eval_sampler = torch.utils.data.distributed.DistributedSampler(train_eval_dataset, shuffle=False)
        train_eval_samplers.append(train_eval_sampler)

        train_eval_loader = torch.utils.data.DataLoader(
            train_eval_dataset, batch_size=int(args.batch_size / len(datas)), shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_eval_sampler, drop_last=False)

        train_eval_loaders.append(train_eval_loader)

    eval_loaders = []
    eval_samplers = []
    eval_datas = args.eval_data.split(',')
    for iEData, edata in enumerate(eval_datas):
        if edata not in datas:
            eval_dataset = domainnet.Dataset(edata, test=True, root=os.path.dirname(edata),
                                             pre_transform=eval_pre_transform,
                                             post_transform=eval_post_transform, alpha=args.aug_alpha,
                                             hpf_range=args.hpf_range, hpf_alpha=args.hpf_alpha)

            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False)
            eval_samplers.append(eval_sampler)

            eval_loader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=args.batch_size, shuffle=False,
                sampler=eval_sampler, num_workers=args.workers, pin_memory=True, drop_last=False)
            eval_loaders.append(eval_loader)

    for epoch in range(args.epochs):
        for train_sampler in train_samplers:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        print(f'=> Start training epoch #{epoch}')
        cluster_result = None
        if epoch >= args.warmup_epoch:
            cluster_result = cluter_getter(train_eval_loaders, model, args)
        train(train_loader, model, criterion, optimizer, epoch, args, cluster_result=cluster_result)
        print(f'=> Finished training epoch #{epoch}')

        checkpoint_freq = args.save_n_epochs
        if args.is_root and np.mod(epoch, checkpoint_freq) == 0:
            save_dict = {
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'enet_state_dict': [],
                'enet_optimizer': [],
            }
            print(f'=> Start evaluating epoch #{epoch}')
            _ = eval(eval_loaders, model, args)
            print(f'=> Finished evaluating epoch #{epoch}')
            save_checkpoint(save_dict, is_best=False, filename=os.path.join(args.work_folder, 'checkpoint_epoch_{:04d}.pth.tar'.format(epoch)))


def eval(eval_loaders, model, args):
    model.eval()
    prec_nums = args.prec_nums.split(',')
    is_root = dist.get_rank() == 0
    device = next(model.parameters()).device
    features_dim = 2048
    domain_num = len(eval_loaders)

    features_all_domain = []
    labels_all_domain = []
    domain_names = []
    for eval_loader in eval_loaders:
        num_eval_samples = len(eval_loader.dataset)
        domain_names.append(eval_loader.dataset.image_list.split('/')[-1].split('.')[0])
        eval_features = np.zeros((num_eval_samples, features_dim), dtype=np.float16)
        eval_labels = np.zeros(num_eval_samples, dtype=np.int64)

        end_ind = 0
        for ind, (x, y, img_path) in enumerate(eval_loader):
            with torch.no_grad():
                _, extra_outputs = model(x.to(device), is_eval=True)
                features = extra_outputs['q']
                features = concat_all_gather_interlace(features)
                y = concat_all_gather_interlace(y.to(device))
                if not is_root:
                    continue
            y = y.cpu().numpy().astype(np.float16)
            features = features.cpu().numpy().astype(np.float16)
            begin_ind = end_ind
            end_ind = min(begin_ind + len(features), len(eval_features))
            eval_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
            eval_labels[begin_ind:end_ind] = y[:end_ind - begin_ind]
            if end_ind >= num_eval_samples:
                break

        features_all_domain.append(eval_features)
        labels_all_domain.append(eval_labels)

    preck = [int(prec_nums[0]), int(prec_nums[1]), int(prec_nums[2])]
    avg_accuracy = 0.
    for domain_1 in range(domain_num):
        domain_1_name = domain_names[domain_1]
        for domain_2 in range(domain_1+1, domain_num):
            domain_2_name = domain_names[domain_2]
            res_1, res_2 = retrieval_precision_cal(features_all_domain[domain_1],
                                                   labels_all_domain[domain_1],
                                                   features_all_domain[domain_2],
                                                   labels_all_domain[domain_2],
                                                   preck=preck)

            print("Domain {}->{}: P@{}: {}; P@{}: {}; P@{}: {} \n".format(
                domain_1_name, domain_2_name,
                int(prec_nums[0]), res_1[0],
                int(prec_nums[1]), res_1[1],
                int(prec_nums[2]), res_1[2]))

            print("Domain {}->{}: P@{}: {}; P@{}: {}; P@{}: {} \n".format(
                domain_2_name, domain_1_name,
                int(prec_nums[0]), res_2[0],
                int(prec_nums[1]), res_2[1],
                int(prec_nums[2]), res_2[2]))

            avg_accuracy += (res_1[0] + res_2[0])

    avg_accuracy /= (domain_num - 1) * domain_num
    return avg_accuracy


def retrieval_precision_cal(features_A, targets_A, features_B, targets_B, preck=(1, 5, 15)):

    dists = cosine_similarity(features_A, features_B)

    res_A = []
    res_B = []
    for domain_id in ['A', 'B']:
        if domain_id == 'A':
            query_targets = targets_A
            gallery_targets = targets_B

            all_dists = dists

            res = res_A
        else:
            query_targets = targets_B
            gallery_targets = targets_A

            all_dists = dists.transpose()
            res = res_B

        sorted_indices = np.argsort(-all_dists, axis=1)
        sorted_cates = gallery_targets[sorted_indices.flatten()].reshape(sorted_indices.shape)
        correct = (sorted_cates == np.tile(query_targets[:, np.newaxis], sorted_cates.shape[1]))

        for k in preck:
            total_num = 0
            positive_num = 0
            for index in range(all_dists.shape[0]):

                temp_total = min(k, (gallery_targets == query_targets[index]).sum())
                pred = correct[index, :temp_total]

                total_num += temp_total
                positive_num += pred.sum()
            res.append(positive_num / total_num * 100.0)

    return res_A, res_B


def cluter_getter(train_eval_loaders, model, args):

    model.eval()
    is_root = dist.get_rank() == 0
    device = next(model.parameters()).device
    features_dim = 2048

    features_all_domain = []
    for train_eval_loader in train_eval_loaders:
        num_eval_samples = len(train_eval_loader.dataset)
        train_eval_features = np.zeros((num_eval_samples, features_dim), dtype=np.float32)

        end_ind = 0
        for ind, (x, _) in enumerate(train_eval_loader):
            with torch.no_grad():
                _, extra_outputs = model(x.to(device), is_eval=True)
                features = extra_outputs['q']
                features = concat_all_gather_interlace(features)
                if not is_root:
                    continue
            features = features.cpu().numpy().astype(np.float16)
            begin_ind = end_ind
            end_ind = min(begin_ind + len(features), len(train_eval_features))
            train_eval_features[begin_ind:end_ind, :] = features[:end_ind - begin_ind]
            if end_ind >= num_eval_samples:
                break

        features_all_domain.append(train_eval_features)

    # set domain index
    cluster_result = run_kmeans(features_all_domain, args.num_cluster, args.gpu)
    print('gpu: '+ str(args.gpu))
    print('gpu1: ' + str(device))

    return cluster_result


def train(train_loader, model, criterion, optimizer, epoch, args, cluster_result):

    batch_time = AverageMeter('Time', ':6.5f')
    data_time = AverageMeter('Data', ':6.5f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    losses_cluster = AverageMeter('Loss_cluster', ':.4e')
    top1_cluster = AverageMeter('Acc@1_cluster', ':6.2f')
    top5_cluster = AverageMeter('Acc@5_cluster', ':6.2f')

    meters = [batch_time, data_time, losses, top1, top5, losses_cluster, top1_cluster, top5_cluster]

    progress = ProgressMeter(
        len(train_loader), meters, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    device = next(model.parameters()).device

    end = time.time()
    print(f'=> Entering train loop')

    for i, batch in enumerate(train_loader):
        images = None
        domain_label = []
        domain_index = []
        img_src = []

        for ib, b in enumerate(batch):

            b, _, di, isrc = b  # throw away the class labels

            if images is None:
                images = [[] for _ in range(len(b))]
            for j in range(len(b)):
                images[j].append(b[j])
            domain_label.append(ib * torch.ones((b[0].shape[0]), dtype=torch.long))
            domain_index.append(di)
            img_src += isrc

        images = [torch.cat(img, dim=0) for img in images]

        # noinspection PyTypeChecker
        domain_label = torch.cat(domain_label, dim=0).to(device)
        domain_index = torch.cat(domain_index, dim=0).to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        images = [x.view([-1] + list(x.shape[2:])) for x in images]

        # metadata
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            im_q = images[0]
            im_k = images[1]

            im_q_rgb = images[2].cuda(args.gpu, non_blocking=True)
            im_k_rgb = images[3].cuda(args.gpu, non_blocking=True)
        else:
            im_q = images[0]
            im_k = images[1]

            im_q_rgb = images[2]
            im_k_rgb = images[3]

        output, target, extra_outputs, \
        output_cluster, target_cluster, extra_outputs_rgb = model(im_q=im_q, im_k=im_k,
                                                                  q_selector=domain_label, sample_idx=domain_index,
                                                                  sample_pth=img_src, cluster_result=cluster_result,
                                                                  im_q_rgb=im_q_rgb, im_k_rgb=im_k_rgb)

        output_size = output.shape[0]
        size_per_part = output_size // 8

        output_intra_phase = output[0:size_per_part]
        target_intra_phase = target[0:size_per_part]

        output_intra_rgb = output[size_per_part:size_per_part*2]
        target_intra_rgb = target[size_per_part:size_per_part*2]

        output_cross_phase = output[size_per_part*2:size_per_part*3]
        target_cross_phase = target[size_per_part*2:size_per_part*3]

        output_cross_rgb = output[size_per_part*3:size_per_part*4]
        target_cross_rgb = target[size_per_part*3:size_per_part*4]

        loss = args.contra_intra_phase * criterion(output_intra_phase, target_intra_phase) + \
               args.contra_intra_rgb * criterion(output_intra_rgb, target_intra_rgb) + \
               args.contra_cross_phase * criterion(output_cross_phase, target_cross_phase) + \
               args.contra_cross_rgb * criterion(output_cross_rgb, target_cross_rgb)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        if not cluster_result is None:
            loss_cluster = criterion(output_cluster, target_cluster)
            loss = loss + args.cluster_loss_w * loss_cluster

            acc1_cluster, acc5_cluster = accuracy(output_cluster, target_cluster, topk=(1, 5))
            losses_cluster.update(loss_cluster.item(), images[0].size(0))
            top1_cluster.update(acc1_cluster[0], images[0].size(0))
            top5_cluster.update(acc5_cluster[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def run_kmeans(feat_all_domain, num_cluster, gpu):
    print('performing kmeans clustering')
    results = {'im2cluster': []}

    feat = np.concatenate(feat_all_domain, axis=0)
    feat_nums = []
    for domain_id in range(len(feat_all_domain)):
        feat_nums.append(feat_all_domain[domain_id].shape[0])

    # intialize faiss clustering parameters
    d = feat.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 2000
    clus.min_points_per_centroid = 2
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu
    index = faiss.IndexFlatL2(d)

    clus.train(feat, index)
    D, I = index.search(feat, 1)  # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]

    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids_normed = nn.functional.normalize(centroids, p=2, dim=1)
    im2cluster = torch.LongTensor(im2cluster).cuda()

    results['centroids'] = centroids_normed
    prev_num = 0
    for feat_num in feat_nums:
        results['im2cluster'].append(im2cluster[prev_num:prev_num+feat_num])
        prev_num += feat_num

    return results


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            if batch_size > 0:
                res.append(correct_k.mul_(100.0 / batch_size))
            else:
                res.append(correct_k.mul_(0.0))
        return res


if __name__ == '__main__':

    main()
