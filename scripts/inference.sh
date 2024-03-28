#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}

oae_ckptpath=./experiments/Transformer_pcn/PCN_models/downloaded/pcd_completion.pth

# skel_ckptpath=/home/hyoshida/git/Point2Skeleton/weights/train-weight_onlybranches_128/weights-skelpoint.pth
skel_ckptpath=/home/hyoshida/git/Point2Skeleton/weights/trainingrecon-weight128/weights-skelpoint.pth

# input_pc=./data/TestShapeNet/shapenet_pc/1000-3branches.npy
# input_pc=./data/TestShapeNet/shapenet_pc/1000-1branch.npy  
input_pc=./data/TestShapeNet/shapenet_pc/1000-branchstructure.npy

### pc_skeletor single
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth \
# --skelnet --pc_skeletor \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy

### skelnet single
CUDA_VISIBLE_DEVICES="1,2" python main_OAE_pcn.py \
--config cfgs/PCN_models/Transformer_pcn.yaml \
--exp_name complete_test \
--ckpts $oae_ckptpath \
--skelnet_ckpt $skel_ckptpath  \
--inference  $input_pc

### skelenet group
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth \
# --groups --skelnet \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy

### normal single
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/downloaded/pcd_completion.pth \
# --inference  $input_pc


### normal groups
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/downloaded/pcd_completion.pth \
# --groups \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy
