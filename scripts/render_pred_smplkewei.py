import sys
sys.path.append('../')
from submodules import smplx
import torch
import os
from os.path import join
import numpy as np
import trimesh
import cv2
import torch.nn as nn
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
def calculate_fov(K_tensor):
    """
    根据内参矩阵计算垂直视场角
    K_tensor: 3x3相机内参矩阵 (torch.Tensor)
    """
    fy = K_tensor[1, 1].item()
    image_height = 1500  # 根据实际图像高度调整
    return 2 * np.arctan(image_height/(2 * fy)) * 180 / np.pi
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 原始数据加载部分保持不变
smpl_model = smplx.SMPL(model_path='/media/hhx/Lenovo/code/GaussianAvatarori/assets/smpl_files/smpl', 
                      gender='neutral', batch_size=1)
data_path = '/media/hhx/Lenovo/code/GaussianAvatar/gs_data/MVHuman/mytest'
outpath = data_path
beta_smpl_path = join(data_path, 'smpl_parms.pth')
beta_smpl_data = torch.load(beta_smpl_path)
smplx_parms_path = join(data_path, 'smpl_parms.pth')
cam_parms_path  = data_path + '/cam_parms.npz'
image_path = join(data_path, 'images')
ori_render_path = join(outpath, 'pred_smplx_render_kewei')
os.makedirs(ori_render_path, exist_ok=True)
smpl_data = torch.load(smplx_parms_path)
print(smpl_data['body_pose'].shape)
# 颜色配置保持不变
colors_dict = {'white': np.array([1.0, 0.98, 0.94])}
color = colors_dict['white']

# 加载相机参数（修改为PyTorch3D格式）
cam_npy = np.load(cam_parms_path)
extr_npy = cam_npy['extrinsic']
intr_npy = cam_npy['intrinsic'].reshape(3, 3)

# 转换相机参数到PyTorch3D格式
R = torch.from_numpy(extr_npy[:3, :3].astype(np.float32)).to(device)
T = torch.from_numpy(extr_npy[:3, 3].astype(np.float32)).to(device)
K = torch.from_numpy(intr_npy.astype(np.float32))
adjustment_matrix = torch.tensor(
    [
        [1, 0, 0],
        [0, -1, 0],  # 翻转Y轴
        [0, 0, -1]    # 翻转Z轴（OpenGL风格）
    ],
    dtype=torch.float32,  # 显式指定为float32类型
    device=device
)
R = R @ adjustment_matrix  # 现在两个矩阵都是float32类型
# 创建可微渲染器

# 修正后的相机参数设置（替换原来的FoVPerspectiveCameras部分）
# 计算垂直视场角（单位：度）
fx = K[0, 0].item()
fy = K[1, 1].item()
image_height = 1500  # 根据实际尺寸修改
fov_vertical = 2 * np.arctan(image_height/(2 * fy)) * 180 / np.pi


# 创建PyTorch3D相机
cameras = FoVPerspectiveCameras(
    device=device,
    R=R[None].transpose(1,2),  # 添加batch维度
    T=T[None],
    fov=calculate_fov(K),  # 需要提前计算视场角
    znear=0.1,
    zfar=100.0,
    degrees=True
)

raster_settings = RasterizationSettings(
    image_size=(1500, 2048),
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
).to(device)

shader = SoftPhongShader(device=device, cameras=cameras)

# 处理每一帧
for pose_idx, image_name in enumerate(sorted(os.listdir(image_path))):
    print(f'Processing frame {pose_idx}')
    idx_name = image_name.split('.')[0]
    idx_image_path = join(image_path, image_name)
    idx_ori_rend_path = join(ori_render_path, image_name)

    # 生成SMPL网格时强制类型转换
    ori_smpl = smpl_model.forward(
        betas=beta_smpl_data['beta'][0][None].to(torch.float32),  # 确保输入是float32
        global_orient=smpl_data['body_pose'][pose_idx, :3][None].to(torch.float32),
        transl=smpl_data['trans'][pose_idx][None].to(torch.float32),
        body_pose=smpl_data['body_pose'][pose_idx, 3:][None].to(torch.float32)
    )

    # 转换顶点数据
    vertices = ori_smpl.vertices.detach().to(device).to(torch.float32)  # 强制转换为float32
    faces = torch.from_numpy(smpl_model.faces.astype(np.int64)).to(device)[None]

    # 创建纹理（显式指定数据类型）
    color_float32 = torch.tensor(color, dtype=torch.float32, device=device)  # 先转换颜色
    verts_rgb = torch.ones_like(vertices) * color_float32
    textures = TexturesVertex(verts_features=verts_rgb)
    # 创建可微网格
    mesh = Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    )
    
    # 执行可微渲染
    with torch.no_grad():
        fragments = renderer(mesh)
        images = shader(fragments, mesh)
    
    # 后处理
    render_img = (images[0,...,:3].cpu().numpy() * 255).astype(np.uint8)
    render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
    
    # 保存结果
    cv2.imwrite(idx_ori_rend_path, render_img) 