#!/usr/bin/python3
import torch
import numpy as np
import logging
from numpysocket import NumpySocket
import socket
from types import SimpleNamespace
# from test import const_model, calc_occu, mesh_export

from tools.runner_OAE_pcn_finetune import inference_net
from utils.config import get_config
# from extensions.pc_skeletor.laplacian import LBC
# import open3d as o3d


logger = logging.getLogger('skeleton server')
logger.setLevel(logging.INFO)

def calc_points(data):
    args = SimpleNamespace(config='cfgs/PCN_models/Transformer_pcn.yaml', launcher='none', local_rank=0, num_workers=4, seed=0, deterministic=False, sync_bn=False, exp_name='complete_test', start_ckpts=None, 
                           ckpts='experiments/Transformer_pcn/PCN_models/skelnet_1/ckpt-best.pth', val_freq=1, resume=False, test=False, 
                           skelnet="/home/hyoshida/git/Point2Skeleton/trainingrecon-weight128/weights-skelpoint.pth", 
                           pc_skeletor=False, groups=False, finetune_model=False, scratch_model=False, mode=None, way=-1, shot=-1, fold=-1,
                           experiment_path='experiments/Transformer_pcn/PCN_models/complete_test', tfboard_path='experiments/Transformer_pcn/PCN_models/TFBoard/complete_test', 
                           log_name='Transformer_pcn', use_gpu=True, distributed=False)
    config = get_config(args, logger = logger)

    ret = inference_net(args, config, data)
    return ret


with NumpySocket() as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', 9999))
    s.listen()

    try:
        while True :
            print('Wait tcp accepting...')
            conn, addr = s.accept()
            with conn:
                logger.info(f"connected: {addr}")
                frame = conn.recv()

                logger.info("array received")
                # logger.info(frame)
                
                ret = calc_points(frame)
                # print(skel.shape, mesh.faces.shape)
                try:
                    conn.sendall(ret)
                except:
                    logger.warn("nothing")

                conn.close()

    except: # KeyboardInterrupt:
        s.close()
        # print('socket is closed')
        logger.warn(f"disconnected: {addr}")
