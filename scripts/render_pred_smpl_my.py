import sys
sys.path.append('../')
import os
import torch
import numpy as np
import cv2
import trimesh
import pyrender
from pyrender.constants import RenderFlags
from submodules import smplx

# 配置路径
width = 2048
height = 1500
data_path = '/media/hhx/Lenovo/code/GaussianAvatar/gs_data/MVHuman/train/test'
smpl_model_path = '/media/hhx/Lenovo/code/GaussianAvatarori/assets/smpl_files/smpl'
output_path = os.path.join(data_path, 'pred_smplx_render')

# 初始化SMPL模型
smpl_model = smplx.SMPL(
    model_path=smpl_model_path,
    gender='neutral',
    batch_size=1
)

# 加载参数
smpl_params = torch.load(os.path.join(data_path, 'smpl_parms.pth'))
cam_params = np.load(os.path.join(data_path, 'cam_parms.npz'))

# 解析相机参数
extrinsic = cam_params['extrinsic']
intrinsic = cam_params['intrinsic'].reshape(3, 3)
R = extrinsic[:3, :3].T  # 正确的旋转矩阵转置
T = extrinsic[:3, 3]

# 配置渲染器
renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
scene = pyrender.Scene()

# 设置相机
camera = pyrender.IntrinsicsCamera(
    fx=intrinsic[0, 0],
    fy=intrinsic[1, 1],
    cx=intrinsic[0, 2],
    cy=intrinsic[1, 2],
    znear=0.1,
    zfar=10.0
)
camera_pose = np.eye(4)
camera_pose[:3, :3] = R
camera_pose[:3, 3] = T
scene.add(camera, pose=camera_pose)

# 添加光照
light = pyrender.DirectionalLight(
    color=np.ones(3),
    intensity=5.0
)
light_pose = np.eye(4)
light_pose[:3, 3] = [0, -1, 1]
scene.add(light, pose=light_pose)

# 材质配置
material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.1,
    roughnessFactor=0.3,
    alphaMode='BLEND',
    baseColorFactor=(0.7, 0.7, 0.6, 1.0)
)

# 处理每帧图像
os.makedirs(output_path, exist_ok=True)
image_files = sorted([
    f for f in os.listdir(os.path.join(data_path, 'images')) 
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

for idx, img_file in enumerate(image_files):
    print(f'Processing frame {idx+1}/{len(image_files)}')
    
    # 加载原始图像
    img_path = os.path.join(data_path, 'images', img_file)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    
    # 生成SMPL网格
    with torch.no_grad():
        output = smpl_model(
            betas=smpl_params['beta'],
            global_orient=smpl_params['body_pose'][idx, :3][None],
            body_pose=smpl_params['body_pose'][idx, 3:][None],
            transl=smpl_params['trans'][idx][None]
        )
    
    # 转换坐标系系统
    vertices = output.vertices[0].numpy()
    vertices = vertices @ R.T + T  # 转换到相机坐标系
    
    # 创建网格
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=smpl_model.faces,
        process=False
    )
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    # 渲染设置
    mesh_node = scene.add(mesh)
    color, depth = renderer.render(scene, RenderFlags.RGBA)
    scene.remove_node(mesh_node)
    
    # 图像合成
    alpha = color[:, :, 3:4] / 255.0
    composite = (color[:, :, :3] * alpha + img * (1 - alpha)).astype(np.uint8)
    
    # 保存结果
    output_file = os.path.join(output_path, f"{os.path.splitext(img_file)[0]}.png")
    cv2.imwrite(output_file, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

# 清理资源
renderer.delete()
print("Rendering completed successfully!")