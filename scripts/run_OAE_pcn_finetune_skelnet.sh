#!/usr/bin/env bash

# set -x
# GPUS=$1
# PY_ARGS=${@:2}
ckpts=/home/hyoshida/git/3D-OAE/experiments/Transformer_pcn_trainskel/PCN_models/skelnet_bs6/ckpt-last.pth
skel_ckptpath=/home/hyoshida/git/Point2Skeleton/weights/train-weight_all_and_branches_128/weights-skelpoint.pth
export RANK=2
export WORLD_SIZE=3
export CUDA_VISIBLE_DEVICES="0,1,2"
# export MASTER_ADDR=local_host
# export MASTER_PORT=12355
python main_OAE_pcn.py \
    --config cfgs/PCN_models/Transformer_pcn_trainskel.yaml \
    --finetune_model \
    --ckpts $ckpts  \
    --exp_name skelnet_bs12 \
    --skelnet_ckpt $skel_ckptpath \
    --val_freq 20 \
    --finetune \
    # --resume 
    # --launcher pytorch --sync_bn\
    
 
        # --launcher pytorch \ # for distributed 
