#!/usr/bin/python3
import torch
import numpy as np
import logging
from numpysocket import NumpySocket
import socket
from types import SimpleNamespace
# from test import const_model, calc_occu, mesh_export

from extensions.pc_skeletor.laplacian import LBC
import open3d as o3d


logger = logging.getLogger('skeleton server')
logger.setLevel(logging.INFO)

# def calc_mesh(frame):
#     args = SimpleNamespace(grid_dims=32, 
#                         n_levels=4, outfilename=None, 
#                         pointcloud_filename='plys/20230714/testjointedge_4parts.ply', 
#                         steps=100, threshold=0.2)
#     if frame.shape[-1] == 6:
#         print("normal!")
#         args.normal = True
#         args.model_file='experiment/trees_raysampled_match_normal/model90000_32.pt'

#     else:
#         args.normal = False
#         # args.model_file='experiment/trees_raysampled_match/model50000_32.pt'
#         # args.model_file='experiment/trees_raysampled_match/model30000_32.pt'
#         args.model_file='experiment/trees_raysampled_match/model90000_32.pt'
#         # args.model_file='experiment/trees_raysampled_match/model290000_32.pt'


#     print("normal", args.normal)

#     data = torch.load(args.model_file)
#     state_dict = data["model_state"]
#     device = None
#     if torch.cuda.is_available():
#         device = torch.cuda.current_device()
#     model = const_model(state_dict, args).to(device)
#     model.train(False)

#     # import ipdb; ipdb.set_trace()
#     frame = frame[np.newaxis, :, :]
#     frame = torch.tensor(frame, dtype=torch.float32)
#     if(args.normal):
#         in_points = frame[:,:, :3].to(device)
#         in_normals = frame[:,:, 3:6].to(device)
#         print(in_points.shape, in_normals.shape)
#     else:
#         in_points = frame.to(device)
#         in_normals = None

#     try:
#         prediction, steps, scale, middle = calc_occu(in_points, in_normals, model, device, args)
#         mesh, filename_mesh  = mesh_export(prediction, steps, scale, middle, args)
#         print(mesh)

#     except Exception as e:
#         print(e)
#         mesh = None
    
#     return mesh 

def calc_skeleton(pcd):
    pcd0_o3d = o3d.geometry.PointCloud()
    pcd0_o3d.points = o3d.utility.Vector3dVector(pcd)
    lbc = LBC(point_cloud=pcd0_o3d, down_sample=0.01)

    lbc.extract_skeleton()
    lbc.extract_topology(fps_points=128)

    # import ipdb;ipdb.set_trace()
    skel = np.zeros((len(lbc.skeleton_graph.nodes.data()), 3))

    for data in lbc.skeleton_graph.nodes.data():
        skel[data[0]] = data[1]["pos"]

    # skel =np.asarray([x[1]["pos"] for x in lbc.skeleton_graph.nodes.data()])
    edges =np.asarray([x for x in lbc.skeleton_graph.edges])

    return skel, edges 


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
                
      
                skel, edges = calc_skeleton(frame)
                # print(skel.shape, mesh.faces.shape)
                # conn.sendall([mesh.vertices, mesh.faces])
                if skel is not None:
                    # conn.sendall([skel, edges])
                    conn.sendall(np.array([skel, edges]))
                else:
                    print("none")

                conn.close()

    except: # KeyboardInterrupt:
        s.close()
        print('socket is closed')
        logger.info(f"disconnected: {addr}")
