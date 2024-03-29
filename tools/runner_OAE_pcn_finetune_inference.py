import torch
# import torch.nn as nn
import os
from tools import builder
from utils.misc import scale_trans_and_divide, random_sample, retrans_rescale, export_imgs2
from utils.logger import *

import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from models.SkelPointNet import SkelPointNet

import open3d


def inference_net(args, config, data_path):

    logger = get_logger(args.log_name)
    print_log('inference start ... ', logger = logger)
# 
    if(type(data_path) == str):
        if(".npy" in data_path):
            data = np.load(data_path)
        elif(".ply" in data_path):
            ply = open3d.io.read_point_cloud(data_path)
            data=np.asarray(ply.points)
    else:
        data = data_path # for taking numpy input directly

    # import ipdb; ipdb.set_trace()
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    base_model.to(0)
    base_model.eval()

    if(args.skelnet_ckpt is not None): 
        base_model.use_skelnet = True
        base_model.skelnet = SkelPointNet(config.model.num_group, input_channels=0, use_xyz=True)
        # skelpath = config.skelnet_ckpt
        base_model.skelnet.load_state_dict(torch.load(args.skelnet_ckpt))
        base_model.skelnet.to(0)
        base_model.skelnet.eval()
        
    scaled_data, global_center, global_scale = scale_trans_and_divide(data)
    print("center:", global_center, "scale:", 1/global_scale)
    # scaled_data_torch = torch.tensor(scaled_data).float().cuda(0)
    # scaled_data_torch = scaled_data_torch.unsqueeze(0).reshape(num_group, -1, 3)
    groupsingle = "_group" if args.groups else "_single"
    skel_normal = "_skel" if base_model.use_skelnet else "_normal"
    # groupsingle += "_pcskeletor" # if args.pc_skeletor ### lazy solution
    if(type(data_path) == str):
        save_path = os.path.join("experiments", data_path.split("/")[-1][:-4]+"_"+args.ckpts.split("/")[-2] + groupsingle + skel_normal)
        os.makedirs(save_path, exist_ok=True)

    if(args.groups is False):
        ### single input (128 center points)
        inpc = random_sample(scaled_data, 2048)
        inpc = torch.tensor(inpc).float().cuda(0).unsqueeze(0)

        with torch.no_grad():

            ret = base_model(inpc, return_center=True)
            
            if(type(data_path) == str):
                coarse, dense = export_imgs2(ret[0], ret[1], ret[2], inpc, save_img_path=save_path, idx=0)
            
            coarse = ret[0].squeeze(0).to('cpu').detach().numpy().copy()
            coarse = retrans_rescale(coarse, global_center, global_scale)
            dense = ret[1].squeeze(0).to('cpu').detach().numpy().copy()
            dense = retrans_rescale(dense, global_center, global_scale)

            if(config.model.use_skelnet):
                skel = ret[2].squeeze(0).to('cpu').detach().numpy().copy()
                skel = retrans_rescale(skel, global_center, global_scale)
                return np.array([coarse, dense, skel]) # np.array([coarse, dense, skel])

            else:
                return np.array([coarse, dense]) 

   
    # else:   ### group inputs
    #     group_size = 1024
    #     num_groups = 10
    #     inpc = random_sample(scaled_data, group_size * num_groups)
    #     inpc = torch.tensor(inpc).float().cuda(0).unsqueeze(0)
        
    #     if(args.skelnet):
    #         centers, _, _, groups = skel_net(torch.tensor(scaled_data).float().cuda(0).unsqueeze(0), group=True)
    #         groups = groups.squeeze(0)
    #         centers = centers.squeeze(0)

    #         #     group = groups[begin:end]
    #     else:
    #         ### without skeleton
    #         # num_group = int(scaled_data.shape[0]/group_size)
    #         group_divider = Group(num_group = num_groups, group_size = group_size).float().cuda(0)
    #         groups, centers = group_divider(inpc) #groups, centers = group_divider(scaled_data) ?? ToDO
            
    #     with torch.no_grad():
    #         # for group in groups:
    #         if(args.skelnet):
    #             bs = 12
    #             begin_batch_num = np.arange(0,128,bs)
    #             end_batch_num = np.arange(bs,128,bs)
    #             if(begin_batch_num.shape != end_batch_num.shape):
    #                 end_batch_num = np.append(end_batch_num,groups.shape[0])

    #             for idx, (begin, end) in enumerate(zip (begin_batch_num, end_batch_num)):
    #                 groups_ = groups[begin:end]
    #                 centers_ = centers[begin:end]
    #                 ret = base_model(groups_, skelnet=skel_net, return_center=True)
    #                 coarse_points = ret[0] + centers_.unsqueeze(1).expand(-1, ret[0].shape[1], -1)
    #                 dense_points = ret[1] + centers_.unsqueeze(1).expand(-1, ret[1].shape[1], -1)
    #                 centers_model = ret[2] + centers_.unsqueeze(1).expand(-1, ret[2].shape[1], -1)
    #                 # import ipdb;ipdb.set_trace()
    #                 coarse, dense = export_imgs2(coarse_points, dense_points, centers_model, inpc.squeeze(0), save_img_path=save_path, idx=idx)
    #         else:
    #             ret = base_model(groups.squeeze(0), return_center=True) 
    #             coarse_points = ret[0] + centers.unsqueeze(1).expand(-1, ret[0].shape[1], -1)
    #             dense_points = ret[1] + centers.unsqueeze(1).expand(-1, ret[1].shape[1], -1)
    #             centers_model = ret[2] + centers.unsqueeze(1).expand(-1, ret[2].shape[1], -1)
    #             # import ipdb;ipdb.set_trace()
    #             coarse, dense = export_imgs2(coarse_points, dense_points, centers_model, inpc.squeeze(0), save_img_path=save_path, idx=0)
       
        #         coarse = retrans_rescale(coarse, global_center, global_scale)
        #         dense = retrans_rescale(dense, global_center, global_scale)
        #         save_off_points(data, os.path.join(save_path, "%i_input.off"%idx))
        #         save_off_points(coarse, os.path.join(save_path, "%i_coarse.off"%idx))
        #         save_off_points(dense, os.path.join(save_path, "%i_dense.off"%idx))
        #         print("%i to %i saved in "%(idx, idx), os.path.join(save_path, "%i_dense.off"%idx) )


    