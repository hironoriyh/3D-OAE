import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc
from pointnet2.utils import pointnet2_utils


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)



def seprate_point_cloud(xyz, num_points, crop, fixed_points = None, padding_zeros = False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    _,n,c = xyz.shape

    assert n == num_points
    assert c == 3
    if crop == num_points:
        return xyz, None
        
    INPUT = []
    CROP = []
    for points in xyz:
        if isinstance(crop,list):
            num_crop = random.randint(crop[0],crop[1])
        else:
            num_crop = crop

        points = points.unsqueeze(0)

        if fixed_points is None:       
            center = F.normalize(torch.randn(1,1,3),p=2,dim=-1).cuda()
        else:
            if isinstance(fixed_points,list):
                fixed_point = random.sample(fixed_points,1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1,1,3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1), p =2 ,dim = -1)  # 1 1 2048

        idx = torch.argsort(distance_matrix,dim=-1, descending=False)[0,0] # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] =  input_data[0,idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0, idx[num_crop:]].unsqueeze(0) # 1 N 3

        crop_data =  points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop,list):
            INPUT.append(fps(input_data,2048))
            CROP.append(fps(crop_data,2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT,dim=0)# B N 3
    crop_data = torch.cat(CROP,dim=0)# B M 3

    return input_data.contiguous(), crop_data.contiguous()

def get_ptcloud_img(ptcloud, ax = None, savepath=None, size=0.1):
    # ptcloud = ptcloud[0, :, :]
# try:
    fig = plt.figure(figsize=(10, 10))
    # x, z, y = ptcloud.transpose(1, 0)
    if(ax is None):
        ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(30, 45)
    max, min = np.max(ptcloud), np.min(ptcloud)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter3D(ptcloud[0,:,0],ptcloud[0,:,1], ptcloud[0,:,2], zdir='z', s=size, c=ptcloud[0,:,0], cmap='jet')

    # fig.canvas.draw()
    # if(savepath):
    #     fig.savefig(savepath)
    #     print("saved", savepath)
    # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
    # except:
    #     print("misc.py line 205 error for img", ptcloud.shape)
    #     img = np.zeros((800,800, 3))
    return ax



def visualize_KITTI(path, data_list, titles = ['input','pred'], cmap=['bwr','autumn'], zdir='y', 
                         xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1) ):
    fig = plt.figure(figsize=(6*len(data_list),6))
    cmax = data_list[-1][:,0].max()

    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:,0] /cmax
        ax = fig.add_subplot(1, len(data_list) , i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=color,vmin=-1,vmax=1 ,cmap = cmap[0],s=4,linewidth=0.05, edgecolors = 'black')
        ax.set_title(titles[i])

        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_path = path + '.png'
    fig.savefig(pic_path)

    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)


def random_dropping(pc, e):
    up_num = max(64, 768 // (e//50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1,1))[0,0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim = 1)
    return pc
    

def random_scale(partial, scale_range=[0.8, 1.2]):
    scale = torch.rand(1).cuda() * (scale_range[1] - scale_range[0]) + scale_range[0]
    return partial * scale



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