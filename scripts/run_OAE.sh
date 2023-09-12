#!/usr/bin/env bash

set -x
GPUS=$1

PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES="0,1,2" python main_OAE.py ${PY_ARGS}
