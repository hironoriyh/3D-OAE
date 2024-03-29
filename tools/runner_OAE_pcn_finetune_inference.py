import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from models.util_models import Group
from models.SkelPointNet import SkelPointNet

import open3d
from tqdm import tqdm
from collections import OrderedDict

def export_imgs(partial, coarse_points, dense_points, gt, val_writer=None, idx=0, epoch=0, save_img_path=None):
    # import ipdb; ipdb.set_trace()
    fig = plt.figure()
    fig.set_size_inches(40, 10)
    input_pc = partial.squeeze().detach().cpu().numpy()
    # ax = fig.add_subplot(121, projection='3d')
    ax = fig.add_subplot(141, projection='3d')
    misc.get_ptcloud_img(input_pc, ax, savepath=os.path.join(save_img_path, "%i_input.png"%idx), size=1)
    
    ax = fig.add_subplot(142, projection='3d')
    sparse = coarse_points.squeeze().cpu().numpy()
    sparse_img = misc.get_ptcloud_img(sparse, ax, savepath=os.path.join(save_img_path, "%i_sparse.png"%idx), size=4)

    ax = fig.add_subplot(143, projection='3d')
    dense = dense_points.squeeze().cpu().numpy()
    dense_img = misc.get_ptcloud_img(dense, ax, savepath=os.path.join(save_img_path, "%i_dense.png"%idx))
    
    ax = fig.add_subplot(144, projection='3d')
    gt_ptcloud = gt.squeeze().cpu().numpy()
    gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud, ax, savepath=os.path.join(save_img_path, "%i_gt.png"%idx))
    
    fig.savefig(os.path.join(save_img_path, "%i_.png"%idx))


    if(val_writer):
        val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')
        val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')
        val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
        val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')

    

crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def scale_trans_and_divide(input_pcd):

    if(((input_pcd.max(axis=0)-input_pcd.min(axis=0)).max() > 2) 
       or ((input_pcd.max(axis=0)-input_pcd.min(axis=0)).max() < 1.0)
       or (input_pcd.mean() > 0.5 )
       or (input_pcd.mean() < -0.5) ):
        print("normalizing input point cloud")
        center = input_pcd.mean(axis=0)
        input_pcd_trans = input_pcd - center
        scale = 2*0.95/(input_pcd_trans.max(axis=0) - input_pcd_trans.min(axis=0)).max()
        input_pcd_trans = input_pcd_trans * scale
    else:
        scale = 1
        center = [0,0,0]
        input_pcd_trans = input_pcd

    # if(input_pcd.mean() > 0.5 or input_pcd.mean() <0.5):

    return input_pcd_trans, center, scale

def retrans_rescale(input_pcd, center, scale):
    input_pcd *= 1/scale
    input_pcd += center

    return input_pcd

def vis_points(ax, pcd, title, rangesize=1, ptsize=0.5):
    ax.scatter3D (pcd[:, 0],pcd[:, 1],pcd[:, 2], s=ptsize, zdir='y')
#     ax.axis("off")
    ax.set_title(title)
    ax.set_xlim(-rangesize, rangesize)
    ax.set_ylim(-rangesize, rangesize)
    ax.set_zlim(-rangesize, rangesize)
    return ax

def export_imgs2(coarse_points, dense_points, centers, scaled_data, save_img_path= "experiments", idx=0):

    # coarse_points = coarse_points + centers.expand(-1, coarse_points.shape[1], -1)
    # dense_points = dense_points + centers.expand(-1, dense_points.shape[1], -1)
    # centers_model = centers_model + centers.expand(-1, centers_model.shape[1], -1)

    scaled_data = scaled_data.cpu().numpy().reshape(-1,3)
    centers = centers.cpu().numpy().reshape(-1,3)
    coarse_points = coarse_points.cpu().numpy().reshape(-1,3)
    dense_points = dense_points.cpu().numpy().reshape(-1,3)


    fig = plt.figure()
    fig.set_size_inches(40, 10)
    ax = fig.add_subplot(141, projection='3d')
    ax = vis_points(ax, scaled_data,  "input_" + str(scaled_data.shape[0]))

    ax = fig.add_subplot(142, projection='3d')
    ax = vis_points(ax, centers,  "centers_" + str(centers.shape[0]), ptsize=1)

    ax = fig.add_subplot(143, projection='3d')
    ax = vis_points(ax, coarse_points,  "coarse_" + str(coarse_points.shape[0]))

    ax = fig.add_subplot(144, projection='3d')
    ax = vis_points(ax, dense_points[::10,:],  "dense_" + str(dense_points.shape[0]))

    fig.savefig(save_img_path +  "_%i_.png"%idx)
    print("saved in ", save_img_path +  "_%i_.png"%idx)

    return coarse_points, dense_points

def save_off_points(points, path):
    with open(path, "w") as file:
        file.write("OFF\n")
        file.write(str(int(points.shape[0])) + " 0" + " 0\n")
        for i in range(points.shape[0]):
            file.write(
                str(float(points[i][0])) + " " + str(float(points[i][1])) + " " + str(float(points[i][2])) + "\n")

def random_sample(pc, num):
    permutation = np.arange(pc.shape[0])
    np.random.shuffle(permutation)
    pc = pc[permutation[:num]]
    return pc

def inference_net(args, config, data_path):

    logger = get_logger(args.log_name)
    print_log('inference start ... ', logger = logger)
# 
    if(type(data_path) == str):
        if(".npy" in data_path):
            data = np.load(data_path)
        elif(".ply" in data_path):
            ply = 
    else:
        data = data_path # for taking numpy input directly

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

   
    else:   ### group inputs
        group_size = 1024
        num_groups = 10
        inpc = random_sample(scaled_data, group_size * num_groups)
        inpc = torch.tensor(inpc).float().cuda(0).unsqueeze(0)
        
        if(args.skelnet):
            centers, _, _, groups = skel_net(torch.tensor(scaled_data).float().cuda(0).unsqueeze(0), group=True)
            groups = groups.squeeze(0)
            centers = centers.squeeze(0)

            #     group = groups[begin:end]
        else:
            ### without skeleton
            # num_group = int(scaled_data.shape[0]/group_size)
            group_divider = Group(num_group = num_groups, group_size = group_size).float().cuda(0)
            groups, centers = group_divider(inpc) #groups, centers = group_divider(scaled_data) ?? ToDO
            
        with torch.no_grad():
            # for group in groups:
            if(args.skelnet):
                bs = 12
                begin_batch_num = np.arange(0,128,bs)
                end_batch_num = np.arange(bs,128,bs)
                if(begin_batch_num.shape != end_batch_num.shape):
                    end_batch_num = np.append(end_batch_num,groups.shape[0])

                for idx, (begin, end) in enumerate(zip (begin_batch_num, end_batch_num)):
                    groups_ = groups[begin:end]
                    centers_ = centers[begin:end]
                    ret = base_model(groups_, skelnet=skel_net, return_center=True)
                    coarse_points = ret[0] + centers_.unsqueeze(1).expand(-1, ret[0].shape[1], -1)
                    dense_points = ret[1] + centers_.unsqueeze(1).expand(-1, ret[1].shape[1], -1)
                    centers_model = ret[2] + centers_.unsqueeze(1).expand(-1, ret[2].shape[1], -1)
                    # import ipdb;ipdb.set_trace()
                    coarse, dense = export_imgs2(coarse_points, dense_points, centers_model, inpc.squeeze(0), save_img_path=save_path, idx=idx)
            else:
                ret = base_model(groups.squeeze(0), return_center=True) 
                coarse_points = ret[0] + centers.unsqueeze(1).expand(-1, ret[0].shape[1], -1)
                dense_points = ret[1] + centers.unsqueeze(1).expand(-1, ret[1].shape[1], -1)
                centers_model = ret[2] + centers.unsqueeze(1).expand(-1, ret[2].shape[1], -1)
                # import ipdb;ipdb.set_trace()
                coarse, dense = export_imgs2(coarse_points, dense_points, centers_model, inpc.squeeze(0), save_img_path=save_path, idx=0)
        #         coarse = retrans_rescale(coarse, global_center, global_scale)
        #         dense = retrans_rescale(dense, global_center, global_scale)
        #         save_off_points(data, os.path.join(save_path, "%i_input.off"%idx))
        #         save_off_points(coarse, os.path.join(save_path, "%i_coarse.off"%idx))
        #         save_off_points(dense, os.path.join(save_path, "%i_dense.off"%idx))
        #         print("%i to %i saved in "%(idx, idx), os.path.join(save_path, "%i_dense.off"%idx) )


    