#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE_pcn.py \
    --config cfgs/PCN_models/Transformer_pcn.yaml \
    --finetune_model \
    --ckpts ./experiments/pcd_completion.pth \
    --exp_name skelnet_1 \
    --finetune