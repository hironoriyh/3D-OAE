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

import open3d as o3d
# import ipdb
from extensions.pc_skeletor.laplacian import LBC

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader) = builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)

    # build model
    base_model = builder.model_builder(config.model)
    skel_net = SkelPointNet(config.model.num_group, input_channels=0, use_xyz=True)

    if args.use_gpu:
        base_model.to(args.local_rank)
        skel_net.to(0)

    # from IPython import embed; embed()
    skel_net.eval()
    skelpath = "/home/hyoshida/git/Point2Skeleton/trainingrecon-weight128/weights-skelpoint.pth"
    ckpt = torch.load(skelpath)
    skel_net.load_state_dict(ckpt)

    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    else:
        if args.ckpts is not None:
            base_model.base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # import ipdb;ipdb.set_trace()
    metrics = validate(base_model, test_dataloader, 1, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            if(skel_net is not None) :
                ret = base_model(partial)
            else:
                 ret = base_model(partial, skelnet=skel_net)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()
            if config.clip_gradients:
                norm = builder.clip_gradients(base_model, config.clip_grad)

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)
        
        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            try:
                metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)
            except:
                print("failed validation")
            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        if (config.max_epoch - epoch) < 10 or epoch%10 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    train_writer.close()
    val_writer.close()

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

    

    # fig = plt.figure()
    # fig.set_size_inches(10, 30)
    # plt.imgs
    # gt_ptcloud_img


def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # import ipdb; ipdb.set_trace()
            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            # if val_writer is not None and idx % 200 == 0:
            #     import ipdb; ipdb.set_trace()
            #     export_imgs(partial, coarse_points, dense_points, gt, val_writer, idx, epoch, "/home/hyoshida/git/3D-OAE/experiments/Transformer_pcn")
            #     print("export_imgs not working")
            if (idx+1) % 20 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def scale_trans_and_divide(input_pcd):
    #center
    center = input_pcd.mean(axis=0)
    input_pcd -= center

    scale = 1/(input_pcd.max(axis=0) - input_pcd.min(axis=0)).max()
    input_pcd *= scale
    # input_mesh.apply_scale(scale)
    # center = input_mesh.centroid
    # input_mesh.apply_translation(-center)
    # input_pc = torch.tensor(input_mesh.vertices).cuda().float().unsqueeze(axis=0)

    return input_pcd, center, scale

def retrans_rescale(input_pc, center, scale):
    input_pc += center
    input_pc *= 1/scale
    return input_pc

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

    fig.savefig(os.path.join(save_img_path, "%i_.png"%idx))
    print("saved in ", os.path.join(save_img_path, "%i_.png"%idx))

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

    # logger = get_logger(args.log_name)
    # print_log('inference start ... ', logger = logger)
    data = np.load(data_path)

    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.ckpts)
    base_model.to(0)

    base_model.eval()
    if(args.skelnet):
        skel_net = SkelPointNet(config.model.num_group, input_channels=0, use_xyz=True)
        skel_net.to(0)
        skel_net.eval()
        skelpath = "/home/hyoshida/git/Point2Skeleton/trainingrecon-weight128/weights-skelpoint.pth"
        skel_net.load_state_dict(torch.load(skelpath))

    scaled_data, global_center, global_scale = scale_trans_and_divide(data)
    # scaled_data_torch = torch.tensor(scaled_data).float().cuda(0)
    # scaled_data_torch = scaled_data_torch.unsqueeze(0).reshape(num_group, -1, 3)
    groupsingle = "_group" if args.groups else "_single"
    save_path = os.path.join("experiments", data_path.split("/")[-1][:-4]+"_"+args.ckpts.split("/")[-2]+groupsingle)

    os.makedirs(save_path, exist_ok=True)

    if(args.groups is False):
        ### single input (128 center points)
        skel, neighborhood = pc_skeletor(scaled_data) ### ToDo test pc-skeletor
        import ipdb; ipdb.set_trace()
        neighborhood = neighborhood.squeeze(0).to("cpu").detach().numpy().copy() 
        neighborhood = neighborhood[:, ::100, :]
        idxneighb = np.zeros(neighborhood.shape[:2]) 
        for i in range(0, idxneighb.shape[0]): idxneighb[i, :] = i 
        savefig(neighborhood.rehsape(-1, 3), "3_2_pcskel_neighbor.png", color=idxneighb.flatten())

        savefig(scaled_data[::100, :], "0_input.png")
        center, _, _, neighborhood = skel_net(torch.tensor(scaled_data).float().cuda(0).unsqueeze(0), group=True)
        # savefig(center, "1_skelnet_center.png")
        savefig(center.squeeze(0).to("cpu").detach().numpy().copy(), "1_skelnet_center.png")
        savefig(neighborhood.squeeze(0).to("cpu").detach().numpy().copy()[::50, :].reshape(-1,3), "2_skelnet_neighbor.png", size=0.5)


        inpc = random_sample(scaled_data, 2048)
        inpc = torch.tensor(inpc).float().cuda(0).unsqueeze(0)

        with torch.no_grad():
            begin=0
            end=12
            if(args.skelnet):
                ret = base_model(inpc, skelnet=skel_net, return_center=True)
            else:
                ret = base_model(inpc, return_center=True)
            coarse, dense = export_imgs2(ret[0], ret[1], ret[2], inpc, save_img_path=save_path, idx=begin)
            # coarse = retrans_rescale(coarse, global_center, global_scale)
            # dense = retrans_rescale(dense, global_center, global_scale)
            # save_off_points(data, os.path.join(save_path, "%i_input.off"%begin))
            # save_off_points(coarse, os.path.join(save_path, "%i_coarse.off"%begin))
            # save_off_points(dense, os.path.join(save_path, "%i_dense.off"%begin))
            print("%i to %i saved in "%(begin, end), os.path.join(save_path, "%i_dense.off"%begin) )
   
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
            

def pc_skeletor(pcd):
            ### test pc-skeletor
    # npinpc = inpc.squeeze(0).to("cpu").detach().numpy().copy()
    pcd0_o3d = o3d.geometry.PointCloud()
    pcd0_o3d.points = o3d.utility.Vector3dVector(pcd)
    # lbc = LBC(point_cloud=pcd0_o3d, down_sample=0.008)
    # lbc = LBC(point_cloud=pcd0_o3d.scale(100, [0,0,0]), down_sample=0.01)
    lbc = LBC(point_cloud=pcd0_o3d, down_sample=0.01)

    lbc.extract_skeleton()
    lbc.extract_topology(fps_points=128)
    # import ipdb; ipdb.set_trace()
    skel =np.asarray([x[1]["pos"] for x in lbc.skeleton_graph.nodes.data()])

    inpc = torch.tensor(pcd).float().cuda(0).unsqueeze(0)
    skelcuda = torch.tensor(skel).float().cuda(0).unsqueeze(0)
    # import ipdb;ipdb.set_trace()

    from knn_cuda import KNN
    groupsize = int(pcd.shape[0]/skel.shape[0])
    knn = KNN(k=groupsize, transpose_mode=True)
    _, idx = knn(inpc, skelcuda)
    assert idx.size(1) == skel.shape[0] #num_group
    assert idx.size(2) == groupsize #group_size
   
    # _, idx = knn(torch.tensor(pcd).float().cuda(0), torch.tensor(skel).float().cuda(0))
    idx_base = torch.arange(0, 1, device=0).view(-1, 1, 1) * pcd.shape[0]
    idx = idx + idx_base
    neighborhood = inpc.view(1 * pcd.shape[0], -1)[idx, :]

    return skel, neighborhood

    ###

def voxel_grid(pcd):
    pcd0_o3d = o3d.geometry.PointCloud()
    pcd0_o3d.points = o3d.utility.Vector3dVector(pcd)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd0_o3d, voxel_size=0.05)
    indices = np.stack(list(vx.grid_index for vx in voxel_grid.get_voxels()))
    voxels = np.zeros(indices.max(axis=0)+1)
    voxels[indices.transpose()[0], indices.transpose()[1], indices.transpose()[2]] = 1
    return voxels

def savefig(pcd, name, rangesize=0.5, size=1, save=True, color="blue"):

    if torch.is_tensor(pcd):
        print("is tensor")
        pcd=pcd.squeeze(0).to("cpu").detach().numpy().copy()
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.jet
    ax.scatter3D (pcd[:, 0],pcd[:, 1],pcd[:, 2], c=color, s=size, cmap=cmap)
    ax.set_title(name)
    ax.set_xlim(-rangesize, rangesize)
    ax.set_ylim(-rangesize, rangesize)
    ax.set_zlim(-rangesize, rangesize)
    if(save):
        fig.savefig(name)
    else:
        return ax
    

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    # import ipdb; ipdb.set_trace()
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret (partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
                export_imgs(partial, coarse_points, dense_points, gt, None, idx, 0, "/home/hyoshida/git/3D-OAE/experiments/Transformer_pcn/PCN_models/test_complete_test")

                # import ipdb;ipdb.set_trace()

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
