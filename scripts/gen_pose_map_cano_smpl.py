

import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('../')
from submodules  import smplx

from scipy.spatial.transform import Rotation as R
import trimesh
from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori
from arguments import smplx_cpose_param, smpl_cpose_param
from pytorch3d import transforms
import math
# SMPL-X规范姿势参数配置
class SMPLXCanonicalPose:
    def __init__(self):
        # 基础身体姿势参数 (21个关节 * 3轴 = 63维)
        self.body_pose = torch.zeros(1, 63)
        
        # 腿部A字型展开角度 (单位：度)
        #self.leg_angle = 30  
        #self._set_leg_pose()
        
        # 手臂自然下垂姿势
        #self._set_arm_pose()
        
        # 其他部位参数初始化
        self.jaw_pose = torch.zeros(1, 3)
        self.leye_pose = torch.zeros(1, 3)
        self.reye_pose = torch.zeros(1, 3)
        self.expression = torch.zeros(1, 10)
        self.left_hand_pose = torch.zeros(1, 12)  # 12维度手部姿势
        self.right_hand_pose = torch.zeros(1, 12)
        
    def _set_leg_pose(self):
        """设置A字型腿部姿势"""
        # SMPL-X腿部关节索引 (对应身体姿势的3-5号关节)
        left_leg_idx = 5  # 左髋关节外展轴
        right_leg_idx = 8  # 右髋关节外展轴
        
        angle_rad = self.leg_angle / 180 * math.pi
        self.body_pose[:, left_leg_idx] = angle_rad
        self.body_pose[:, right_leg_idx] = -angle_rad

    def _set_arm_pose(self):
        """设置自然下垂手臂姿势"""
        # 左臂关节索引 (肩部)
        arm_joint_idx = 16  # 左肩关节
        
        # 设置手臂旋转 (绕Y轴旋转-90度)
        euler_angles = torch.tensor([[-90.0, 0.0, 0.0]], dtype=torch.float32) / 180 * math.pi
        rotation_matrix = transforms.euler_angles_to_matrix(euler_angles, 'XYZ')
        axis_angle = transforms.matrix_to_axis_angle(rotation_matrix)
        
        # 应用旋转到左肩关节
        self.body_pose[:, arm_joint_idx:arm_joint_idx+3] = axis_angle


def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id

def save_obj(data_path,mode,realgender='female'):
    cano_dir = os.path.join(data_path,)
    if mode=='smpl':
        smpl_data = torch.load( data_path + '/smpl_parms.pth')
        smpl_model = smplx.SMPL(model_path ='../assets/smpl_files/smpl',batch_size = 1,gender=realgender)
        cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                            global_orient=smpl_cpose_param[:, :3],
                            transl = torch.tensor([[0, 0.30, 0]]),
                            body_pose=smpl_cpose_param[:, 3:],
                            )
        ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
        joint_mat = cano_smpl.A
        print(joint_mat.shape)
        torch.save(joint_mat ,join(cano_dir, 'smpl_cano_joint_mat.pth'))
        mesh = trimesh.Trimesh(ori_vertices, smpl_model.faces, process=False)
        mesh.export('%s/%s.obj' % (cano_dir, 'cano_smpl'))
    if mode=='smplx':
        smplx_data = torch.load( data_path + '/smplx_parms.pth')
        smplx_model = smplx.SMPLX(model_path ='../assets/smpl_files/smplx',batch_size = 1,use_pca=True,num_pca_comps=12,gender=realgender)
        cano_pose = SMPLXCanonicalPose()
        leg_angle = 30
        cano_pose.body_pose[:, 2] =  leg_angle / 180 * math.pi
        cano_pose.body_pose[:, 5] =  -leg_angle / 180 * math.pi
        cano_smpl = smplx_model.forward(
            betas=smplx_data['beta'],
            global_orient=torch.zeros(1, 3),  # 无全局旋转
            transl=torch.tensor([[0.0, 1.5, 0.0]]),  # 调整高度
            body_pose=cano_pose.body_pose,
            left_hand_pose=cano_pose.left_hand_pose,
            right_hand_pose=cano_pose.right_hand_pose,
            jaw_pose=cano_pose.jaw_pose,
            leye_pose=cano_pose.leye_pose,
            reye_pose=cano_pose.reye_pose,
            expression=cano_pose.expression
        )
        ori_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
        joint_mat = cano_smpl.A
        print(joint_mat.shape)
        torch.save(joint_mat ,join(cano_dir, 'smplx_cano_joint_mat.pth'))
        mesh = trimesh.Trimesh(ori_vertices, smplx_model.faces, process=False)
        mesh.export('%s/%s.obj' % (cano_dir, 'cano_smplx'))


def save_npz(data_path,mode, res=128,):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    result = {}
    if mode=='smpl':
        body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smpl'), process=False)
    if mode=='smplx':
        body_mesh = trimesh.load('%s/%s.obj'%(data_path, 'cano_smplx'), process=False)
    print(uvs.shape,'uvs.shape')
    print(faces_uvs.shape,'faces_uvs.shape')
    print(body_mesh.vertices.shape,'body_mesh.vertices.shape')
    print(body_mesh.faces.shape,'body_mesh.faces.shape')
    if res==128:
        posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
        result['posmap128'] = posmap128   
    elif res == 256:
    
        posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
        result['posmap256'] = posmap256

    else:
        posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
        result['posmap512'] = posmap512
    if mode=='smpl':
        save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smpl'))
    if mode=='smplx':
        save_fn = join(data_path, 'query_posemap_%s_%s.npz'% (str(res), 'cano_smplx'))
    np.savez(save_fn, **result)



if __name__ == '__main__':
    mode='smplx'
    #n=128
    smplx_parm_path = '/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple/train/take2' 
    #smplx_parm_path = '/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test/take7' 
    mygender='female'#'neutral'#female
    if mode=='smpl':
        uv_template_fn = '../assets/template_mesh_smpl_uv.obj'
    if mode=='smplx':
        uv_template_fn = '../assets/template_mesh_smplx_uv.obj'
    assets_path = ''    # path to the folder that include 'assets'

    print('saving obj...')
    save_obj(smplx_parm_path,mode,realgender=mygender)

    print('saving pose_map 512 ...')
    save_npz(smplx_parm_path,mode, 512)