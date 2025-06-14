import os
import torch
import numpy as np
from torch.utils.data import Dataset
from os.path import join
from PIL import Image
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
import cv2
import random
import re  # 新增正则表达式模块
import bisect
def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)

# class MultiViewDataset(Dataset):
#     def __init__(self, dataset_params, device=torch.device('cuda:0')):
#         super().__init__()
        
#         # 基础路径（指向包含所有视角的父文件夹）
#         base_data_folder = join(dataset_params.source_path, 'train')
#         # 加载共享的smpl_data（关键修改）
#         if dataset_params.train_stage == 1 or dataset_params.train_stage == 3:
#             smpl_path = join(base_data_folder, 'smpl_parms.pth')  # 假设smpl数据在根目录
#         else:
#             smpl_path = join(base_data_folder, 'smpl_parms_pred.pth')
            
#         print(f'Loading shared SMPL data from {smpl_path}')
#         self.smpl_data = torch.load(smpl_path)  # 整个多视角数据集共享

#         # 获取所有视角子文件夹（假设每个子文件夹是一个视角）
#         self.view_folders = sorted([
#             d for d in os.listdir(base_data_folder)
#             if os.path.isdir(join(base_data_folder, d)) 
#             and re.fullmatch(r'\d+', d)  # 正则匹配纯数字名称
#         ], key=lambda x: int(x))  # 按数字大小排序（而非字符串顺序）
        
#         # 为每个视角创建单视角数据集
#         self.view_datasets = [
#             MonoDataset_train(
#                 dataset_params=dataset_params,
#                 data_folder=join(base_data_folder, vf),  # 指定该视角的数据路径
#                 smpl_data=self.smpl_data,
#                 device=device
#             ) for vf in self.view_folders
#         ]
        
#         # 校验所有视角数据长度一致
#         self.data_length = len(self.view_datasets[0]) if self.view_datasets else 0
#         for ds in self.view_datasets[1:]:
#             assert len(ds) == self.data_length, "All views must have the same number of samples"
        
#         self.num_views = len(self.view_datasets)
#         # 将smpl_data提升到多视角数据集（按用户要求）
#         self.pose_data = self.view_datasets[0].pose_data  # 示例属性
#         self.transl_data = self.view_datasets[0].transl_data  # 示例属性

#     def __len__(self):
#         return self.data_length * self.num_views  # 140 * 4=560

#     def __getitem__(self, idx):
#         view_idx = idx // self.data_length  # 计算视角索引 (0~3)
#         data_idx = idx % self.data_length   # 计算原数据索引 (0~139)
#         return self.view_datasets[view_idx][data_idx]  # 固定按顺序取所有视角
class MultiViewDataset(Dataset):
    def __init__(self, dataset_params, device=torch.device('cuda:0')):
        super().__init__()
        # self.base_data_path = "/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train"
        self.base_data_path = os.path.join(dataset_params.source_path,'train')
        print(self.base_data_path,'self.base_data_path')
        #self.base_data_path = "/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_my"
        self.take_list = dataset_params.take_list
        self.device = device
        self.take_datasets = []
        
        # 全局pose/transl数据存储
        self.global_pose_data = []
        self.global_transl_data = []
        self.take_offsets = [0]  # 每个take在全局数据中的起始索引
        self.global_beta = None
        if dataset_params.smpl_type == 'smplx':
            self.global_left_hand_pose_data = []
            self.global_right_hand_pose_data = []
            self.global_jaw_pose_data = []
            self.global_leye_pose_data = []
            self.global_reye_pose_data = []
            self.global_expression_data = []
        # 加载每个take数据集
        for take_idx, take_name in enumerate(self.take_list):
            take_path = join(self.base_data_path, take_name)
            if dataset_params.smpl_type == 'smpl':
                smpl_path = join(take_path, 'smpl_parms.pth' if dataset_params.train_stage in [1,2,3] 
                            else 'smpl_parms_pred.pth')
            if dataset_params.smpl_type == 'smplx':
                smpl_path = join(take_path, 'smplx_parms.pth' if dataset_params.train_stage in [1,2,3] 
                            else 'smplx_parms_pred.pth')
            try:
                smpl_data = torch.load(smpl_path)  # 使用局部变量避免覆盖
                print(f'✅ 成功加载 {take_name} 的SMPL数据')
                # 只在第一个有效take读取beta
                if take_idx == 0 and 'beta' in smpl_data:
                    self.global_beta = smpl_data['beta']
                    if not torch.is_tensor(self.global_beta):
                        self.global_beta = torch.from_numpy(self.global_beta)
                    self.global_beta = self.global_beta.to(device)
                    print(f'✅ 全局beta参数已加载: {self.global_beta.shape}')
            except Exception as e:
                print(f'❌ 跳过 {take_name}: {str(e)}')
                continue
            
            # 初始化单Take数据集
            take_ds = TakeDataset(
                dataset_params=dataset_params,
                take_path=take_path,
                smpl_data=smpl_data,
                device=device
            )
            self.take_datasets.append(take_ds)
            
            # 收集全局数据
            if hasattr(take_ds, 'pose_data'):
                self.global_pose_data.append(take_ds.pose_data)
                self.global_transl_data.append(take_ds.transl_data)
                self.take_offsets.append(self.take_offsets[-1] + len(take_ds.pose_data))
                if dataset_params.smpl_type == 'smplx':
                    self.global_left_hand_pose_data.append(take_ds.left_hand_pose_data)
                    self.global_right_hand_pose_data.append(take_ds.right_hand_pose_data)
                    self.global_jaw_pose_data.append(take_ds.jaw_pose_data)
                    self.global_leye_pose_data.append(take_ds.leye_pose_data)
                    self.global_reye_pose_data.append(take_ds.reye_pose_data)
                    self.global_expression_data.append(take_ds.expression_data)
        
        # 合并全局数据
        self.global_pose_data = torch.cat(self.global_pose_data, dim=0) if self.global_pose_data else None
        self.global_transl_data = torch.cat(self.global_transl_data, dim=0) if self.global_transl_data else None
        if dataset_params.smpl_type == 'smplx':
                    self.global_left_hand_pose_data = torch.cat(self.global_left_hand_pose_data, dim=0) if self.global_left_hand_pose_data else None
                    self.global_right_hand_pose_data = torch.cat(self.global_right_hand_pose_data, dim=0) if self.global_right_hand_pose_data else None
                    self.global_jaw_pose_data = torch.cat(self.global_jaw_pose_data, dim=0) if self.global_jaw_pose_data else None
                    self.global_leye_pose_data = torch.cat(self.global_leye_pose_data, dim=0) if self.global_leye_pose_data else None
                    self.global_reye_pose_data = torch.cat(self.global_reye_pose_data, dim=0) if self.global_reye_pose_data else None
                    self.global_expression_data = torch.cat(self.global_expression_data, dim=0) if self.global_expression_data else None

        # 构建累积索引
        self.cumulative_counts = [0]
        for ds in self.take_datasets:
            self.cumulative_counts.append(self.cumulative_counts[-1] + len(ds))

    def __len__(self):
        return sum(len(ds) for ds in self.take_datasets)

    def __getitem__(self, idx):
        # 动态索引分配
        take_idx = bisect.bisect_right(self.cumulative_counts, idx) - 1
        local_idx = idx - self.cumulative_counts[take_idx]
        data_item = self.take_datasets[take_idx][local_idx]
        
        # 转换pose_idx为全局索引
        data_item['pose_idx'] += self.take_offsets[take_idx]
        return data_item
    
class TakeDataset(Dataset):
    """处理单个Take下的多视角数据"""
    def __init__(self, dataset_params, take_path, smpl_data, device):
        self.take_path = take_path
        self.smpl_data = smpl_data
        self.device = device

        # 初始化各视角数据集
        self.view_folders = sorted([
            d for d in os.listdir(take_path)
            if os.path.isdir(join(take_path, d)) and re.fullmatch(r'\d+', d)
        ], key=lambda x: int(x))
        
        self.view_datasets = [
            MonoDataset_train(
                dataset_params,
                data_folder=join(take_path, vf),
                smpl_data=self.smpl_data,
                device=device
            ) for vf in self.view_folders
        ]
        
        # 从第一个视角获取共享参数
        self.pose_data = self.view_datasets[0].pose_data
        self.transl_data = self.view_datasets[0].transl_data
        if dataset_params.smpl_type == 'smplx':
            self.left_hand_pose_data = self.view_datasets[0].left_hand_pose
            self.right_hand_pose_data = self.view_datasets[0].right_hand_pose
            self.jaw_pose_data = self.view_datasets[0].jaw_pose
            self.leye_pose_data = self.view_datasets[0].leye_pose
            self.reye_pose_data = self.view_datasets[0].reye_pose
            self.expression_data = self.view_datasets[0].expression

        # 校验数据一致性
        self.data_length = len(self.view_datasets[0]) if self.view_datasets else 0
        for ds in self.view_datasets[1:]:
            assert len(ds) == self.data_length, "所有视角必须包含相同数量的帧"
        self.num_views = len(self.view_datasets)

    def __len__(self):
        return self.data_length * len(self.view_folders)

    def __getitem__(self, idx):
        view_idx = idx // self.data_length
        data_idx = idx % self.data_length
        return self.view_datasets[view_idx][data_idx]
        
class MonoDataset_train(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_params,data_folder=None, smpl_data=None,
                 device = torch.device('cuda:0')):
        super(MonoDataset_train, self).__init__()
        self.smpl_data = smpl_data  # 使用外部传入的共享数据
        self.dataset_params = dataset_params
        self.data_folder = data_folder  #/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/train/take2
        self.device = device 
        self.gender = self.dataset_params.smpl_gender
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0
        self.no_mask = bool(self.dataset_params.no_mask)
        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_params.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        if dataset_params.smpl_type == 'smplx':
            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            #self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            self.left_hand_pose = self.smpl_data['left_hand_pose']
            self.right_hand_pose = self.smpl_data['right_hand_pose']
            self.jaw_pose = self.smpl_data['jaw_pose']
            self.leye_pose = self.smpl_data['leye_pose']
            self.reye_pose = self.smpl_data['reye_pose']
            self.expression = self.smpl_data['expression']

        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
        
        # 转换为Tensor
        if not torch.is_tensor(self.pose_data):
            self.pose_data = torch.from_numpy(self.pose_data)
        if not torch.is_tensor(self.transl_data):
            self.transl_data = torch.from_numpy(self.transl_data)

        if dataset_params.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_params.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)
        data_item = dict()
        if self.dataset_params.train_stage == 2 or self.dataset_params.train_stage == 1 or self.dataset_params.train_stage == 3:
            self.data_folder_inp = os.path.normpath(self.data_folder)
            if os.path.basename(self.data_folder_inp) != "train":
                self.data_folder_inp = os.path.dirname(self.data_folder_inp)
            if self.dataset_params.smpl_type == 'smpl':
                inp_posmap_path = self.data_folder_inp + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
                laplacian_path = self.data_folder_inp + '/laplacian/' +'laplacian_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
            if self.dataset_params.smpl_type == 'smplx':
                inp_posmap_path = self.data_folder_inp + '/inp_map_x/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
                laplacian_path = self.data_folder_inp + '/laplacian_x/' +'laplacian_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
                normal_path=self.data_folder + '/normal/' +'%s.npy'% (str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_params.inp_posmap_size)]
            laplacian_data=np.load(laplacian_path)
            
            #data_item['laplacian_data'] = laplacian_data['laplacian_data']
            data_item['magnitude'] = laplacian_data['magnitude']
            data_item['valid_mask'] = laplacian_data['valid_mask']
            data_item['normal']=np.load(normal_path)

        if not self.dataset_params.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic
        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]
        image = Image.open(image_path)
        width, height = image.size
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        if not self.dataset_params.no_mask:
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) <3:
                mask = mask[...,None]
            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

        if self.dataset_params.train_stage == 2 or self.dataset_params.train_stage == 1 or self.dataset_params.train_stage == 3:
            #print(inp_posmap.shape,'inp_posmapori.shape')
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)
            #print(data_item['inp_pos_map'].shape,'dataset')

        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        # if self.dataset_params.smpl_type == 'smplx':
        #     rest_pose = self.rest_pose_data[pose_idx]
        #     data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_test(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_params,
                 device = torch.device('cuda:0')):
        super(MonoDataset_test, self).__init__()

        self.dataset_params = dataset_params

        self.data_folder = join(dataset_params.source_path, 'test')
        self.device = device
        self.gender = self.dataset_params.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_params.no_mask)

        # if dataset_params.train_stage == 1:
        #     print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        #     self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        # else:
        #     print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
        #     self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))
        print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_params.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_params.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_params.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        

        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        pose_idx, name_idx = self.name_list[index]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)
        
        cam_path = join(self.data_folder, 'cam_parms', name_idx + '.npz')

        if not self.dataset_params.no_mask:
            mask_path = join(self.data_folder, 'masks', name_idx + '.' + self.mask_fix)
        if self.dataset_params.train_stage == 2:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_params.inp_posmap_size)]

        if not self.dataset_params.cam_static:
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extr_npy[:3, 3], np.float32)
            intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        else:
            R = self.R
            T = self.T
            intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        if not self.dataset_params.no_mask:
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) <3:
                mask = mask[...,None]

            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            # color_img = image * mask 
            color_img = image * mask + (1 - mask) * 255
            image = Image.fromarray(np.array(color_img, dtype=np.byte), "RGB")

    
        data_item = dict()

        # data_item['vtransf'] = vtransf
        # data_item['query_pos_map'] = query_posmap.transpose(2,0,1)
        if self.dataset_params.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)


        resized_image = torch.from_numpy(np.array(image)) / 255.0
        if len(resized_image.shape) == 3:
            resized_image =  resized_image.permute(2, 0, 1)
        else:
            resized_image =  resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

        original_image = resized_image.clamp(0.0, 1.0)

        data_item['original_image'] = original_image
        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_params.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_novel_pose(Dataset):
    @torch.no_grad()
    def __init__(self, dataset_params,
                 device = torch.device('cuda:0')):
        super(MonoDataset_novel_pose, self).__init__()
        self.dataset_params = dataset_params
        self.data_folder = dataset_params.test_folder
        self.device = device
        self.gender = self.dataset_params.smpl_gender
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0
        self.no_mask = bool(self.dataset_params.no_mask)
        if dataset_params.smpl_type == 'smpl':
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
        if dataset_params.smpl_type == 'smplx':
            self.smpl_data = torch.load(join(self.data_folder, 'smplx_parms.pth'))
            print('loading smpl data ', join(self.data_folder, 'smplx_parms.pth'))
        self.data_length = self.smpl_data['body_pose'].shape[0]
        print("total pose length", self.data_length )
        if dataset_params.smpl_type == 'smplx':
            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.beta=self.smpl_data['beta'][0,:]
            #self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
            self.left_hand_pose = self.smpl_data['left_hand_pose']
            self.right_hand_pose = self.smpl_data['right_hand_pose']
            self.jaw_pose = self.smpl_data['jaw_pose']
            self.leye_pose = self.smpl_data['leye_pose']
            self.reye_pose = self.smpl_data['reye_pose']
            self.expression = self.smpl_data['expression']
        # if dataset_params.smpl_type == 'smplx':
        #     self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
        #     self.transl_data = self.smpl_data['trans'][:self.data_length,:]
        #     self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose']
            self.transl_data = self.smpl_data['trans']
            self.beta=self.smpl_data['beta'][0,:]
            #print(self.beta)
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        print('novel pose shape', self.pose_data.shape)
        print('novel pose shape', self.transl_data.shape)
        if dataset_params.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
   
    def __len__(self):
        return self.data_length

    def __getitem__(self, index, ignore_list=None):
        return self.getitem(index, ignore_list)

    @torch.no_grad()
    def getitem(self, index, ignore_list):
        #print(self.dataset_params.train_stage,'self.dataset_params.train_stage')
        pose_idx  =  index
        if self.dataset_params.train_stage == 2 or self.dataset_params.train_stage == 1 or self.dataset_params.train_stage == 3:
            if self.dataset_params.smpl_type == 'smpl':
                inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
                inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_params.inp_posmap_size)]
            if self.dataset_params.smpl_type == 'smplx':
                inp_posmap_path = self.data_folder + '/inp_map_x/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
                inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_params.inp_posmap_size)]
        R = self.R
        T = self.T
        intrinsic = self.intrinsic
        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]
        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]
        # width, height = 1024, 1024
        width, height = 940, 1280
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        data_item = dict()
        if self.dataset_params.train_stage == 2 or self.dataset_params.train_stage == 1 or self.dataset_params.train_stage == 3:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)

        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        data_item['beta_data'] = self.beta


        if self.dataset_params.smpl_type == 'smplx':
            # rest_pose = self.rest_pose_data[pose_idx]
            # data_item['rest_pose'] = rest_pose
            data_item['left_hand_pose'] = self.left_hand_pose[pose_idx]
            data_item['right_hand_pose'] = self.right_hand_pose[pose_idx]
            data_item['jaw_pose'] = self.jaw_pose[pose_idx]
            data_item['leye_pose'] = self.leye_pose[pose_idx]
            data_item['reye_pose'] = self.reye_pose[pose_idx]
            data_item['expression'] = self.expression[pose_idx]
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item

class MonoDataset_novel_view(Dataset):
    # these code derive from humannerf(https://github.com/chungyiweng/humannerf), to keep the same view point
    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }
    @torch.no_grad()
    def __init__(self, dataset_params, device = torch.device('cuda:0')):
        super(MonoDataset_novel_view, self).__init__()

        self.dataset_params = dataset_params

        self.data_folder = join(dataset_params.source_path, 'test')
        self.device = device
        self.gender = self.dataset_params.smpl_gender

        self.zfar = 100.0
        self.znear = 0.01
        self.trans = np.array([0.0, 0.0, 0.0]),
        self.scale = 1.0

        self.no_mask = bool(self.dataset_params.no_mask)

        if dataset_params.train_stage == 1:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms.pth'))
        else:
            print('loading smpl data ', join(self.data_folder, 'smpl_parms_pred.pth'))
            self.smpl_data = torch.load(join(self.data_folder, 'smpl_parms_pred.pth'))

        self.data_length = len(os.listdir(join(self.data_folder, 'images')))
        
        self.name_list = []
        for index, img in enumerate(sorted(os.listdir(join(self.data_folder, 'images')))):
            base_name = img.split('.')[0]
            self.name_list.append((index, base_name))
        
        self.image_fix = os.listdir(join(self.data_folder, 'images'))[0].split('.')[-1]
        
        if not dataset_params.no_mask:
            self.mask_fix = os.listdir(join(self.data_folder, 'masks'))[0].split('.')[-1]

        print("total pose length", self.data_length )


        if dataset_params.smpl_type == 'smplx':

            self.pose_data = self.smpl_data['body_pose'][:self.data_length, :66]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]
        else:
            self.pose_data = self.smpl_data['body_pose'][:self.data_length]
            self.transl_data = self.smpl_data['trans'][:self.data_length,:]
            # self.rest_pose_data = self.smpl_data['body_pose'][:self.data_length, 66:]

        if dataset_params.cam_static:
            cam_path = join(self.data_folder, 'cam_parms.npz')
            cam_npy = np.load(cam_path)
            extr_npy = cam_npy['extrinsic']
            intr_npy = cam_npy['intrinsic']
            self.extr_npy = extr_npy
            self.R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            self.T = np.array([extr_npy[:3, 3]], np.float32)
            self.intrinsic = np.array(intr_npy, np.float32).reshape(3, 3)
        
        self.src_type = 'wild'
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index,):
        return self.getitem(index)
    
    def update_smpl(self, pose_idx, frame_num):
        from third_parties.smpl.smpl_numpy import SMPL
        MODEL_DIR = self.dataset_params.project_path + '/third_parties/smpl/models'
        smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)
        _, tpose_joints = smpl_model(np.zeros((1, 72)), self.smpl_data['beta'].squeeze().numpy())

        pelvis_pos = tpose_joints[0].copy()

        Th = pelvis_pos +self.smpl_data['trans'][pose_idx].numpy()
        self.Th = Th

        self.data_length = frame_num
        self.fix_pose_idx = pose_idx

    def get_freeview_camera(self, frame_idx, total_frames, trans):
        E = rotate_camera_by_frame_idx(
                extrinsics= self.extr_npy, 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        return E

    @torch.no_grad()
    def getitem(self, index,):
        pose_idx = self.fix_pose_idx
        _, name_idx = self.name_list[0]

        image_path = join(self.data_folder, 'images' ,name_idx + '.' + self.image_fix)

        if self.dataset_params.train_stage == 2:

            inp_posmap_path = self.data_folder + '/inp_map/' +'inp_posemap_%s_%s.npz'% (str(self.dataset_params.inp_posmap_size), str(pose_idx).zfill(8))
            inp_posmap = np.load(inp_posmap_path)['posmap' + str(self.dataset_params.inp_posmap_size)]

        extr_npy =  self.get_freeview_camera(index, self.data_length, self.Th)

        R = np.array(extr_npy[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array([extr_npy[:3, 3]], np.float32)

        intrinsic = self.intrinsic

        focal_length_x = intrinsic[0, 0]
        focal_length_y = intrinsic[1, 1]

        pose_data = self.pose_data[pose_idx]
        transl_data = self.transl_data[pose_idx]

        image = Image.open(image_path)
        width, height = image.size

        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        data_item = dict()
        if self.dataset_params.train_stage == 2:
            data_item['inp_pos_map'] = inp_posmap.transpose(2,0,1)

        data_item['FovX'] = FovX
        data_item['FovY'] = FovY
        data_item['width'] = width
        data_item['height'] = height
        data_item['pose_idx'] = pose_idx
        data_item['pose_data'] = pose_data
        data_item['transl_data'] = transl_data
        if self.dataset_params.smpl_type == 'smplx':
            rest_pose = self.rest_pose_data[pose_idx]
            data_item['rest_pose'] = rest_pose
  
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FovX, fovY=FovY, K=intrinsic, h=height, w=width).transpose(0,1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        data_item['world_view_transform'] = world_view_transform
        data_item['projection_matrix'] = projection_matrix
        data_item['full_proj_transform'] = full_proj_transform
        data_item['camera_center'] = camera_center
        
        return data_item


