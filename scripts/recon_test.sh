#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=0 python main_OAE_pcn.py \
--test --ckpts ./experiments/pcd_completion.pth \
--config cfgs/PCN_models/Transformer_pcn.yaml \
--exp_name complete_test \