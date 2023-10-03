#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=1  --master_port 47700 main_dgucdir.py \
--batch-size 64 \
--mlp \
--multi_q \
--aug-plus \
--lr 0.0002 \
--epochs 200 \
--print-freq 50 \
--clean-model moco_v2_800ep_pretrain.pth.tar \
--exp_folder_name domainnet_clipart-sketch \
--save_n_epochs 10 \
--moco-k 5120 \
--data clipart.txt,sketch.txt \
--eval-data clipart.txt,sketch.txt,infograph.txt,real.txt,painting.txt,quickdraw.txt \
--workers 16 \
--imagenet_pretrained '' \
--num_cluster 7 \
--prec-nums '50,100,200' \
--moco-t 0.2 \
--hpf_range 50 \
--hpf_alpha 0.4 \
--aug_alpha 1.0 \
--warmup_epoch 20 \
--cluster_loss_w 0.1 \
--contra_intra_phase 0.5 \
--contra_intra_rgb 0.5 \
--contra_cross_phase 0.5 \
--contra_cross_rgb 0.5 \

