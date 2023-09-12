#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}


### pc_skeletor single
CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
--config cfgs/PCN_models/Transformer_pcn.yaml \
--exp_name complete_test \
--ckpts experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth \
--skelnet --pc_skeletor \
--inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy
# --inference  ./data/TestShapeNet/shapenet_pc/1000-3branches.npy
## --inference  ./data/TestShapeNet/shapenet_pc/1000-branchstructure.npy

### skelnet single
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth \
# --skelnet \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-3branches.npy
## --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy
## --inference  ./data/TestShapeNet/shapenet_pc/1000-branchstructure.npy


### skelenet group
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth \
# --groups --skelnet \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy
##--inference  ./data/TestShapeNet/shapenet_pc/1000-3branches.npy

### normal
# CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/downloaded/pcd_completion.pth \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-3branches.npy

# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy


### normal groups
# --config cfgs/PCN_models/Transformer_pcn.yaml \
# --exp_name complete_test \
# --ckpts experiments/Transformer_pcn/PCN_models/downloaded/pcd_completion.pth \
# --groups \
# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy

# --inference  ./data/TestShapeNet/shapenet_pc/1000-branchstructure.npy

# --inference  ./data/TestShapeNet/shapenet_pc/1000-1branch.npy
# --inference  ./data/TestShapeNet/shapenet_pc/1000-branchstructure.npy \
