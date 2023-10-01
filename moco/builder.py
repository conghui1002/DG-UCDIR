import itertools
import torch
import torch.nn.functional as func
import os
import glob
import numpy as np
import utils.torchvision_wrappers as models_wrappers
import torch.distributed as dist
import torch.nn as nn
from functools import partial

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, debug=False, args=None):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.debug = debug
        self.args = args

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        norm_layer = partial(SplitBatchNorm, num_splits=args.bn_splits) if args.bn_splits > 1 else nn.BatchNorm2d

        self.encoder_q = base_encoder(num_classes=dim, pretrained=args.imagenet_pretrained)
        self.encoder_k = base_encoder(num_classes=dim, pretrained=args.imagenet_pretrained, norm_layer=norm_layer)

        if mlp:  # hack: brute-force replacement

            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.qMult = 1
        if self.args.multi_q:
            self.qMult = len(self.args.data.split(','))

            self.register_buffer("queue", torch.randn(2, self.qMult, dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=2)

            self.register_buffer("queue_idx", - torch.ones(2, self.qMult, K))

            self.register_buffer("queue_large", torch.randn(2, self.qMult, 2048, K))
            self.queue_large = nn.functional.normalize(self.queue_large, dim=2)
        else:
            self.register_buffer("queue", torch.randn(2, dim, K))
            self.queue = nn.functional.normalize(self.queue, dim=1)
            self.register_buffer("queue_idx", - torch.ones(2, K))

            self.register_buffer("queue_large", torch.randn(2, 2048, K))
            self.queue_large = nn.functional.normalize(self.queue_large, dim=1)

        self.Q_lim = [None] * self.qMult
        self.register_buffer("queue_ptr", torch.zeros(2, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_large, q_selector=None, sample_idx=None, if_phase=True):

        if if_phase:
            queue_id = 0
        else:
            queue_id = 1

        # gather keys before updating queue
        if not self.debug:
            keys = concat_all_gather(keys)
            keys_large = concat_all_gather(keys_large)
            if q_selector is not None:
                q_selector = concat_all_gather(q_selector)
            if sample_idx is not None:
                sample_idx = concat_all_gather(sample_idx)

        # also for simplicity, here we assume each domain contributes the same number of samples to the batch
        batch_size = keys.shape[0] // self.qMult

        ptr = int(self.queue_ptr[queue_id])
        assert self.K % batch_size == 0, f'self.K={self.K}, batch_size={batch_size}'  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        if q_selector is not None:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    local_ptr = ptr % self.Q_lim[iQ]
                else:
                    local_ptr = ptr

                active_keys = keys[q_selector == iQ]
                self.queue[queue_id, iQ, :, local_ptr:local_ptr + batch_size] = active_keys.T
                self.queue_idx[queue_id, iQ, local_ptr:local_ptr + batch_size] = sample_idx[q_selector == iQ]

                active_keys_large = keys_large[q_selector == iQ]
                self.queue_large[queue_id, iQ, :, local_ptr:local_ptr + batch_size] = active_keys_large.T
        else:
            if self.Q_lim[0] is not None:
                local_ptr = ptr % self.Q_lim[0]
            else:
                local_ptr = ptr

            self.queue[queue_id, :, local_ptr:local_ptr + batch_size] = keys.T
            self.queue_idx[queue_id, local_ptr:local_ptr + batch_size] = sample_idx

            self.queue_large[queue_id, :, local_ptr:local_ptr + batch_size] = keys_large.T

        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[queue_id] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, ix=None):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        if ix is not None:
            ix_gather = concat_all_gather(ix)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        dist.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        ix_ret = None
        if ix is not None:
            ix_ret = ix_gather[idx_this]

        return x_gather[idx_this], ix_ret, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = dist.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def queue_init(self, im_q, im_k, q_selector, sample_idx):

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC

            if isinstance(k, tuple) or isinstance(k, list):  # handle the case features are returned too
                k = k[0]
            k = nn.functional.normalize(k, dim=1)

        self._dequeue_and_enqueue(k, q_selector if self.args.multi_q else None, sample_idx)

    def neg_logit_getter(self, q, q_selector, sample_idx, queue, if_phase=True):

        if if_phase:
            queue_id = 0
        else:
            queue_id = 1

        BIG_NUMBER = 10000.0 * self.T  # taking temp into account for a good measure
        l_neg = (torch.zeros((q.shape[0], self.K)).cuda() - BIG_NUMBER)
        if self.args.multi_q:
            for iQ in range(self.qMult):
                if self.Q_lim[iQ] is not None:
                    qlim = self.Q_lim[queue_id, iQ]
                else:
                    qlim = queue[queue_id, iQ].shape[1]
                ixx = (q_selector == iQ)
                _l_neg = torch.einsum('nc,ck->nk', [q[ixx], queue[queue_id, iQ][:, :qlim].clone().detach()])
                if sample_idx is not None:
                    for ii, indx in enumerate(sample_idx[ixx]):
                        _l_neg[ii, self.queue_idx[queue_id, iQ][:qlim] == indx] = - BIG_NUMBER
                l_neg[ixx, :qlim] = _l_neg
        else:
            if self.Q_lim[0] is not None:
                qlim = self.Q_lim[0]
            else:
                qlim = queue[queue_id].shape[1]
            _l_neg = torch.einsum('nc,ck->nk', [q, queue[queue_id, :, :qlim].clone().detach()])
            if sample_idx is not None:
                for ii in range(q.shape[0]):
                    _l_neg[ii, self.queue_idx[queue_id, :qlim] == sample_idx[ii]] = - BIG_NUMBER
            l_neg[:, :qlim] = _l_neg

        return l_neg


    def cluster_logit_getter(self, cluster_result, q, q_selector, sample_idx):

        BIG_NUMBER = 10000.0
        cluster_num = cluster_result['centroids'].shape[0]

        l_pos_cluster = (torch.zeros((q.shape[0], 1)).cuda() - BIG_NUMBER)
        l_neg_cluster = (torch.zeros((q.shape[0], cluster_num)).cuda() - BIG_NUMBER)
        for iQ in range(len(cluster_result['im2cluster'])):
            ixx = (q_selector == iQ)
            sample_id = sample_idx[ixx]

            sample_cluster = cluster_result['im2cluster'][iQ][sample_id]
            sample_centroid = cluster_result['centroids'][sample_cluster]

            _l_pos = torch.einsum('nc,nc->n', [q[ixx], sample_centroid.clone()]).unsqueeze(-1)
            l_pos_cluster[ixx] = _l_pos

            _l_neg = torch.einsum('nc,ck->nk', [q[ixx], cluster_result['centroids'].clone().t()])

            for ii, cluster_id in enumerate(sample_cluster):
                _l_neg[ii, cluster_id] = - BIG_NUMBER
            l_neg_cluster[ixx] = _l_neg

        logits_cluster = torch.cat([l_pos_cluster, l_neg_cluster], dim=1)

        # logits_cluster /= self.T

        labels_cluster = torch.zeros(logits_cluster.shape[0], dtype=torch.long).cuda()

        return logits_cluster, labels_cluster


    def forward(self, im_q, im_k=None, isedge_q=None, isedge_k=None,
                q_selector=None, sample_idx=None, sample_pth=None,
                is_eval=False, cluster_result=None, is_init=False,
                im_q_rgb=None, im_k_rgb=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            isedge_q: binary index, 0 = image, 1 = edge
            isedge_k: binary index, 0 = image, 1 = edge
        Output:
            logits, targets
        """

        extra_outputs = {}
        extra_outputs_rgb = {}
        # compute query features
        if not im_q_rgb is None:
            concated_im = torch.cat([im_q, im_q_rgb], dim=0)
            concated_im_q = self.encoder_q(concated_im)
            q_shape = im_q.shape[0]
            if isinstance(concated_im_q, tuple) or isinstance(concated_im_q, list):  # handle the case features are returned too
                concated_fm = nn.functional.normalize(concated_im_q[2], dim=1)
                concated_q = nn.functional.normalize(torch.mean(torch.mean(concated_im_q[2], dim=3), dim=2), dim=1)

                extra_outputs['q_fm'] = concated_fm[0:q_shape]
                extra_outputs_rgb['q_fm'] = concated_fm[q_shape:]

                extra_outputs['q'] = concated_q[0:q_shape]
                extra_outputs_rgb['q'] = concated_q[q_shape:]

                concated_im_q = concated_im_q[0]
            concated_im_q = nn.functional.normalize(concated_im_q, dim=1)

            q_phase = concated_im_q[0:q_shape]
            q_rgb = concated_im_q[q_shape:]
            q = concated_im_q
            extra_outputs['q_proj'] = q_phase
            extra_outputs_rgb['q_proj'] = q_rgb

            q_phase_large = extra_outputs['q']
            q_rgb_large = extra_outputs_rgb['q']
        else:
            q = self.encoder_q(im_q)  # queries: NxC
            if isinstance(q, tuple) or isinstance(q, list):  # handle the case features are returned too
                extra_outputs['q_fm'] = nn.functional.normalize(q[2], dim=1)
                extra_outputs['q'] = nn.functional.normalize(torch.mean(torch.mean(q[2], dim=3), dim=2), dim=1)
                q = q[0]
            q = nn.functional.normalize(q, dim=1)
            extra_outputs['q_proj'] = q

            q_large = extra_outputs['q']

        if is_eval:
            return q, extra_outputs

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k_shape = im_k.shape[0]
            # shuffle for making use of BN
            if not self.debug:
                concated_im_k = torch.cat([im_k, im_k_rgb])
                concated_im_k, isedge_k, idx_unshuffle = self._batch_shuffle_ddp(concated_im_k, isedge_k)

            k = self.encoder_k(concated_im_k)  # keys: NxC

            if isinstance(k, tuple) or isinstance(k, list):  # handle the case features are returned too
                k_ = k[0]
                k_large = torch.mean(torch.mean(k[2], dim=3), dim=2)
            k = nn.functional.normalize(k_, dim=1)
            k_large = nn.functional.normalize(k_large, dim=1)

            # undo shuffle
            if not self.debug:
                k = self._batch_unshuffle_ddp(k, idx_unshuffle) # no need to unshuffle the "isedge_k" as they are no longer used beyond this point
                k_large = self._batch_unshuffle_ddp(k_large, idx_unshuffle)

            k_phase = k[0:k_shape]
            k_rgb = k[k_shape:]

            k_phase_large = k_large[0:k_shape]
            k_rgb_large = k_large[k_shape:]

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_phase = torch.einsum('nc,nc->n', [q_phase, k_phase]).unsqueeze(-1)
        l_neg_phase = self.neg_logit_getter(q_phase, q_selector, sample_idx, self.queue, if_phase=True)

        l_pos_rgb = torch.einsum('nc,nc->n', [q_rgb, k_rgb]).unsqueeze(-1)
        l_neg_rgb = self.neg_logit_getter(q_rgb, q_selector, sample_idx, self.queue, if_phase=False)

        l_pos_phase_cross = torch.einsum('nc,nc->n', [q_phase, k_rgb]).unsqueeze(-1)
        l_neg_phase_cross = self.neg_logit_getter(q_phase, q_selector, sample_idx, self.queue, if_phase=True)

        l_pos_rgb_cross = torch.einsum('nc,nc->n', [q_rgb, k_phase]).unsqueeze(-1)
        l_neg_rgb_cross = self.neg_logit_getter(q_rgb, q_selector, sample_idx, self.queue, if_phase=False)

        l_pos_phase_large = torch.einsum('nc,nc->n', [q_phase_large, k_phase_large]).unsqueeze(-1)
        l_neg_phase_large = self.neg_logit_getter(q_phase_large, q_selector, sample_idx, self.queue_large, if_phase=True)

        l_pos_rgb_large = torch.einsum('nc,nc->n', [q_rgb_large, k_rgb_large]).unsqueeze(-1)
        l_neg_rgb_large = self.neg_logit_getter(q_rgb_large, q_selector, sample_idx, self.queue_large, if_phase=False)

        l_pos_phase_large_cross = torch.einsum('nc,nc->n', [q_phase_large, k_rgb_large]).unsqueeze(-1)
        l_neg_phase_large_cross = self.neg_logit_getter(q_phase_large, q_selector, sample_idx, self.queue_large, if_phase=True)

        l_pos_rgb_large_cross = torch.einsum('nc,nc->n', [q_rgb_large, k_phase_large]).unsqueeze(-1)
        l_neg_rgb_large_cross = self.neg_logit_getter(q_rgb_large, q_selector, sample_idx, self.queue_large, if_phase=False)

        l_pos = torch.cat([l_pos_phase, l_pos_rgb, l_pos_phase_cross, l_pos_rgb_cross,
                           l_pos_phase_large, l_pos_rgb_large, l_pos_phase_large_cross, l_pos_rgb_large_cross], dim=0)
        l_neg = torch.cat([l_neg_phase, l_neg_rgb, l_neg_phase_cross, l_neg_rgb_cross,
                           l_neg_phase_large, l_neg_rgb_large, l_neg_phase_large_cross, l_neg_rgb_large_cross], dim=0)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        if not cluster_result is None:
            logits_cluster_phase, labels_cluster_phase = self.cluster_logit_getter(cluster_result, extra_outputs['q'],
                                                                                   q_selector, sample_idx)
            logits_cluster_rgb, labels_cluster_rgb = self.cluster_logit_getter(cluster_result, extra_outputs_rgb['q'],
                                                                               q_selector, sample_idx)

            logits_cluster = torch.cat([logits_cluster_phase, logits_cluster_rgb], dim=0)
            labels_cluster = torch.cat([labels_cluster_phase, labels_cluster_rgb], dim=0)
        else:
            logits_cluster = None
            labels_cluster = None

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_phase, k_phase_large, q_selector if self.args.multi_q else None, sample_idx, if_phase=True)
        self._dequeue_and_enqueue(k_rgb, k_rgb_large, q_selector if self.args.multi_q else None, sample_idx, if_phase=False)

        return logits, labels, extra_outputs, logits_cluster, labels_cluster, extra_outputs_rgb

    def params_lock(self):
        for p in self.parameters():
            p.requires_grad_b_ = p.requires_grad
            p.requires_grad = False

    def params_unlock(self):
        for p in self.parameters():
            p.requires_grad = p.requires_grad_b_


# SplitBatchNorm: simulate multi-gpu behavior of BatchNorm in one gpu by splitting alone the batch dimension
# implementation adapted from https://github.com/davidcpage/cifar10-fast/blob/master/torch_backend.py
class SplitBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits

    def forward(self, input):
        N, C, H, W = input.shape

        if self.training or not self.track_running_stats:
            running_mean_split = self.running_mean.repeat(self.num_splits)
            running_var_split = self.running_var.repeat(self.num_splits)

            outcome = func.batch_norm(
                input.view(-1, C * self.num_splits, H, W), running_mean_split, running_var_split,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
            self.running_mean.data.copy_(running_mean_split.view(self.num_splits, C).mean(dim=0))
            self.running_var.data.copy_(running_var_split.view(self.num_splits, C).mean(dim=0))
            return outcome
        else:
            return func.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias, False, self.momentum, self.eps)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def concat_all_gather_interlace(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    inds = [torch.arange(i, len(g)*len(tensors_gather), len(tensors_gather)) for i, g in enumerate(tensors_gather)]
    inds = torch.cat(inds, dim=0)
    sort_inds = torch.argsort(inds)
    return output[sort_inds]


def concat_all_gather_object_interlace(obj):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    obj_gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(obj_gather, obj)
    if len(obj_gather[0]) > 1:
        return list(filter(None, itertools.chain(*itertools.zip_longest(*obj_gather))))
    else:
        return obj_gather


def load_model(args, return_epoch=False):
    # create model
    print("=> creating model '{}'".format(args.arch))

    model = models_wrappers.__dict__[args.arch](num_classes=args.moco_dim, pretrained=args.imagenet_pretrained)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # load from pre-trained, before DistributedDataParallel constructor
    checkpoint = {}
    if args.resume:
        if not os.path.isfile(args.resume):
            if args.resume != 'not a path':
                print("=> no checkpoint found at '{}'".format(args.resume))

            if os.path.isfile(os.path.join(args.work_folder, 'checkpoint_last.pth.tar')):
                args.resume = os.path.join(args.work_folder, 'checkpoint_last.pth.tar')
            else:
                gpp = os.path.join(args.work_folder, 'checkpoint_*.pth.tar')
                cpts = glob.glob(gpp)
                cpts_ix = [int(x.split('/')[-1].split('_')[1].split('.')[0]) for x in cpts]
                if len(cpts_ix) > 0:
                    mx_ix = np.argmax(cpts_ix)
                    args.resume = cpts[mx_ix]
                else:
                    print('=> no checkpoints found')

        path2load = args.resume

        if os.path.isfile(path2load):
            print("=> loading checkpoint '{}'".format(path2load))
            checkpoint = torch.load(path2load, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q'): # and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            if args.mlp:
                dim_mlp = model.fc.weight.shape[1]
                model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            print(f'=> missing keys: {msg.missing_keys}')

            print("=> loaded pre-trained model '{}'".format(path2load))
        else:
            print("=> no checkpoint found at '{}'".format(path2load))

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)


    model.eval()

    if return_epoch:
        return model, checkpoint.get('epoch', 0)
    else:
        return model, 0

