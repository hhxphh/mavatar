import torch
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from utils.general_utils import safe_state, to_cuda
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args, NetworkParams, OptimizationParams
#from model.avatar_modelrev import  AvatarModel
from model.avatar_model2f import  AvatarModel
import open3d as o3d
import numpy as np
import cv2
def render_sets(model, net, opt, epoch:int):
    with torch.no_grad():
        avatarmodel = AvatarModel(model, net, opt, train=False)
        avatarmodel.training_setup()
        avatarmodel.load(epoch,test=True)
        
        novel_pose_dataset = avatarmodel.getNovelposeDataset()
        novel_pose_loader = torch.utils.data.DataLoader(novel_pose_dataset,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 4,)
        #novel_pose_loader = avatarmodel.getTrainDataloader()
        render_path = os.path.join(avatarmodel.model_path, 'novel_pose', "ours_{}".format(epoch))
        normal_path=os.path.join(avatarmodel.model_path, 'normal', "ours_{}".format(epoch))
        normal_surf_path=os.path.join(avatarmodel.model_path, 'normal_surf', "ours_{}".format(epoch))
        makedirs(render_path, exist_ok=True)
        makedirs(normal_path, exist_ok=True)
        makedirs(normal_surf_path, exist_ok=True)
        for idx, batch_data in enumerate(tqdm(novel_pose_loader, desc="Rendering progress")):
            batch_data = to_cuda(batch_data, device=torch.device('cuda:0'))
            #avatarmodel.render_free_stage(batch_data, 59400,model.train_stage,idx)
            image,normals_images,depth_normals_images,world_view_transform,points = avatarmodel.render_free_stage(batch_data, 59400,model.train_stage,idx)

            #image, = avatarmodel.render_free_stage(batch_data, 59400,model.train_stage,idx)
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            #visualize_normal_map(normals_images, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"),world_view_transform=world_view_transform)
            visualize_normal_map(normals_images, os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"),)
            visualize_normal_map(depth_normals_images, os.path.join(normal_surf_path, '{0:05d}'.format(idx) + ".png"),world_view_transform=world_view_transform)
            normals_to_save = normals_images.float().data.cpu().numpy()  # 替换为你的变量
            np.save(os.path.join(normal_path, '{0:05d}'.format(idx) + ".npy"), normals_to_save)
            depth_normals_to_save = depth_normals_images.float().data.cpu().numpy()  # 替换为你的变量
            np.save(os.path.join(normal_surf_path, '{0:05d}'.format(idx) + ".npy"), depth_normals_to_save)

            # plysavepath=r'/media/hhx/Lenovo/code/GaussianAvatarori/output/dress_my/novel_pose_ply'
            # if idx % 10  == 0:
            #         save_poitns = points.clone().detach().cpu().numpy()
            #         for i in range(save_poitns.shape[0]):
            #             pcd = o3d.geometry.PointCloud()
            #             pcd.points = o3d.utility.Vector3dVector(save_poitns[i])
            #             o3d.io.write_point_cloud(os.path.join(plysavepath,'{0:03d}'.format(idx)+"pred.ply") , pcd)
            #         print('save ply!')
def visualize_normal_map(normal_map, output_path,world_view_transform=None):
    normal_map=normal_map.cpu().numpy()
    normal_map=normal_map.squeeze(0).transpose(1, 2, 0)
    if world_view_transform is not None:
        #print('world_view_transform is not None')
        normal_map = (normal_map @ (world_view_transform[:3,:3].T))
    epsilon = 1e-16
    norms = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (norms + epsilon)
    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    if world_view_transform is not None:
        normal_map = normal_map[:, :, ::-1]  # BGR转RGB
    else:
        normal_map = normal_map[:, :, [2, 1, 0]]
    cv2.imwrite(output_path, normal_map)
    #print(f"可视化结果已保存至: {output_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    print(model.train_geo)
    network = NetworkParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--epoch", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    render_sets(model.extract(args), network.extract(args), op.extract(args), args.epoch,)