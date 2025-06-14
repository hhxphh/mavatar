import sys
sys.path.append('../')
import torch
import torch.optim as optim
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from submodules import smplx
import numpy as np
import cv2
import os
from os.path import join

# 配置可微渲染器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化SMPL模型
smpl_model = smplx.SMPL(model_path='/media/hhx/Lenovo/code/GaussianAvatarori/assets/smpl_files/smpl', 
                      gender='neutral', batch_size=1).to(device)
width = 2048
height = 1500
data_path = '/media/hhx/Lenovo/code/GaussianAvatar/gs_data/MVHuman/mytest'
smpl_data = torch.load(join(data_path, 'smpl_parms.pth'))
cam_params = np.load(join(data_path, 'cam_parms.npz'))
image_path = join(data_path, 'images')
print(cam_params['extrinsic'])
print(cam_params['intrinsic'])
# 新建调试输出目录
debug_output_path = join(data_path, 'debug_output')
os.makedirs(debug_output_path, exist_ok=True)

# 坐标系转换矩阵
axis_conversion = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=np.float32)

# 处理外参
extrinsic = cam_params['extrinsic']
R = extrinsic[:3, :3] @ axis_conversion.T
T = extrinsic[:3, 3] * np.array([1, 1, -1])

# ================= 新增：初始相机验证 =================
def debug_render(mesh, cameras, name):
    with torch.no_grad():
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=(height, width),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    z_clip_value=0.1  # 设置近裁剪面
                )
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=PointLights(
                    device=device,
                    ambient_color=((0.8, 0.8, 0.8),),
                    diffuse_color=((0.5, 0.5, 0.5),),
                    specular_color=((0.3, 0.3, 0.3),),
                    location=[[0.0, 2.0, 2.0]]  # 提高光源位置
            )
        )
        )
        image = renderer(mesh, cameras=cameras)
        img_np = image[0, ..., :3].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        cv2.imwrite(join(debug_output_path, f'debug_{name}.png'), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
# ====================================================

# 转换为PyTorch张量
R = torch.from_numpy(R).float().to(device).unsqueeze(0)
T = torch.from_numpy(T).float().to(device).unsqueeze(0)

# 处理内参
intrinsic = cam_params['intrinsic']
focal_length = torch.tensor([[intrinsic[0,0], intrinsic[1,1]]], device=device)
principal_point = torch.tensor([[intrinsic[0,2], intrinsic[1,2]]], device=device)

# 配置相机
cameras = PerspectiveCameras(
    device=device,
    focal_length=focal_length,
    principal_point=principal_point,
    R=R,
    T=T,
    image_size=torch.tensor([[height, width]], device=device)
)

# 预加载面数据到设备
faces = torch.from_numpy(smpl_model.faces.astype('int64')).to(device)

# 优化循环
for pose_idx, image_name in enumerate(sorted(os.listdir(image_path))):
    # 加载真实图像
    idx_image_path = join(image_path, image_name)
    img = cv2.imread(idx_image_path)
    true_mask = (img.mean(axis=2) < 245).astype(np.float32)
    cv2.imwrite(join(debug_output_path, f'true_mask_{pose_idx}.png'), true_mask*255)
    
    # 转换为PyTorch张量
    true_mask = torch.from_numpy(true_mask).to(device)
    
    # 初始化参数
    beta = torch.nn.Parameter(smpl_data['beta'][0].clone().to(device))
    body_pose = torch.nn.Parameter(smpl_data['body_pose'][pose_idx].clone().to(device))
    transl = torch.nn.Parameter(smpl_data['trans'][pose_idx].clone().to(device))
    
    # ========== 新增：初始状态渲染验证 ==========
    with torch.no_grad():
        smpl_output = smpl_model(
            betas=beta[None],
            body_pose=body_pose[None, 3:],
            global_orient=body_pose[None, :3],
            transl=transl[None]
        )
        textures = TexturesVertex(verts_features=torch.ones_like(smpl_output.vertices).to(device))
        init_mesh = Meshes(
            verts=smpl_output.vertices,
            faces=faces[None].expand(1, -1, -1),
            textures=textures
        )
        debug_render(init_mesh, cameras, f'init_{pose_idx}')

        # 在初始化后验证模型位置
    with torch.no_grad():
        vertices = smpl_output.vertices[0]
        print(f"模型顶点坐标范围: X({vertices[:,0].min():.2f}-{vertices[:,0].max():.2f}) "
            f"Y({vertices[:,1].min():.2f}-{vertices[:,1].max():.2f}) "
            f"Z({vertices[:,2].min():.2f}-{vertices[:,2].max():.2f})")
        
        # 将顶点投影到屏幕空间
        screen_coords = cameras.transform_points_screen(vertices)
        print(f"投影坐标范围: X({screen_coords[:,0].min():.2f}-{screen_coords[:,0].max():.2f}) "
            f"Y({screen_coords[:,1].min():.2f}-{screen_coords[:,1].max():.2f})")
    # ========================================
    
    optimizer = optim.Adam([beta, body_pose, transl], lr=0.1)  # 调大学习率
    
    for iter in range(100):
        optimizer.zero_grad()
        
        # 生成SMPL网格
        smpl_output = smpl_model(
            betas=beta[None],
            body_pose=body_pose[None, 3:],
            global_orient=body_pose[None, :3],
            transl=transl[None]
        )
        
        # 创建可渲染网格
        textures = TexturesVertex(verts_features=torch.ones_like(smpl_output.vertices).to(device))
        mesh = Meshes(
            verts=smpl_output.vertices,
            faces=faces[None].expand(1, -1, -1),
            textures=textures
        )
        # 在优化循环外正确定义渲染器
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=(height, width),
                    blur_radius=0.0,
                    faces_per_pixel=1
                )
            ),
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=PointLights(device=device, location=[[0.0, 1.0, 5.0]])
            )
        )
        # 可微渲染
        rendered_image = renderer(mesh, cameras=cameras)
        
        # ========== 改进的损失函数 ==========
        # 使用Sigmoid平滑alpha通道
        pred_alpha = torch.sigmoid(5*(rendered_image[..., 3] - 0.5))  # 可微的alpha预测
        
        # 计算加权IoU
        intersection = (pred_alpha * true_mask).sum()
        union = (pred_alpha + true_mask).sum() - intersection
        iou = intersection / (union + 1e-6)
        
        # # 添加轮廓对齐损失
        # pred_edges = F.conv2d(pred_alpha.unsqueeze(0).unsqueeze(0), 
        #                     torch.ones(1,1,3,3,device=device)/9, padding=1)
        # true_edges = F.conv2d(true_mask.unsqueeze(0).unsqueeze(0), 
        #                     torch.ones(1,1,3,3,device=device)/9, padding=1)
        # edge_loss = F.mse_loss(pred_edges, true_edges)
        
        loss = 1 - iou
        # ================================
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([beta, body_pose, transl], 1.0)
        
        optimizer.step()
        
        if iter % 10 == 0:
            with torch.no_grad():
                # 保存中间结果
                vis_img = rendered_image[0, ..., :3].cpu().numpy()
                vis_img = (vis_img * 255).astype(np.uint8)
                cv2.imwrite(join(debug_output_path, f'frame{pose_idx}_iter{iter}.png'), 
                          cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
            print(f"Frame {pose_idx} Iter {iter}: "
                 f"IoU={iou.item():.4f} Loss={loss.item():.4f} "
                 f"Transl={transl.detach().cpu().numpy()}")
    
    # 保存优化后参数
    smpl_data['body_pose'][pose_idx] = body_pose.detach().cpu()
    smpl_data['trans'][pose_idx] = transl.detach().cpu()
    smpl_data['beta'][0] = beta.detach().cpu()

# 最终保存
torch.save(smpl_data, join(data_path, 'optimized_smpl_parms.pth'))