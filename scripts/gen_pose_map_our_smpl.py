import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('/media/hhx/Lenovo/code/myAvatar/')
#import smplx
from submodules import smplx
from scipy.spatial.transform import Rotation as R
import trimesh
from utils.general_utils import to_cuda
# from utils.general_utils import load_masks, load_barycentric_coords, gen_lbs_weight_from_ori
# from arguments import smplx_cpose_param, smpl_cpose_param

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

left_hand_pose = torch.tensor([0.09001956135034561, 0.1604590266942978, -0.3295670449733734, 0.12445037066936493, -0.11897698789834976, -1.5051144361495972, -0.1194705069065094, -0.16281449794769287, -0.6292539834976196, -0.27713727951049805, 0.035170216113328934, -0.5893177390098572, -0.20759613811969757, 0.07492011040449142, -1.4485805034637451, -0.017797302454710007, -0.12478633224964142, -0.7844052314758301, -0.4157009720802307, -0.5140947103500366, -0.2961726784706116, -0.7421528100967407, -0.11505582183599472, -0.7972996830940247, -0.29345276951789856, -0.18898937106132507, -0.6230823397636414, -0.18764786422252655, -0.2696149945259094, -0.5542467832565308, -0.47717514634132385, -0.12663133442401886, -1.2747308015823364, -0.23940050601959229, -0.1586960405111313, -0.7655659914016724, 0.8745182156562805, 0.5848557353019714, -0.07204405218362808, -0.5052485466003418, 0.1797526329755783, 0.3281439244747162, 0.5276764035224915, -0.008714836090803146, -0.4373648762702942], dtype = torch.float32)
right_hand_pose = torch.tensor([0.034751810133457184, -0.12605343759059906, 0.5510415434837341, 0.19454114139080048, 0.11147838830947876, 1.4676157236099243, -0.14799435436725616, 0.17293521761894226, 0.4679432511329651, -0.3042353689670563, 0.007868679240345955, 0.8570928573608398, -0.1827319711446762, -0.07225851714611053, 1.307037591934204, -0.02989627793431282, 0.1208646297454834, 0.7142824530601501, -0.3403030335903168, 0.5368582606315613, 0.3839572072029114, -0.9722614884376526, 0.17358140647411346, 0.911861002445221, -0.29665058851242065, 0.21779759228229523, 0.7269846796989441, -0.15343312919139862, 0.3083758056163788, 0.7146623730659485, -0.5153037309646606, 0.1721675992012024, 1.2982604503631592, -0.2590428292751312, 0.12812566757202148, 0.7502076029777527, 0.8694817423820496, -0.5263001322746277, 0.06934576481580734, -0.4630220830440521, -0.19237111508846283, -0.25436165928840637, 0.5972414612770081, -0.08250168710947037, 0.5013565421104431], dtype = torch.float32)

def save_obj(data_path, name,mode,realgender=None):
    if mode=='smpl':
        smpl_data = torch.load( data_path + '/' + name)
        frame_num = smpl_data['body_pose'].shape[0]
        print('frame_num', frame_num)
        start_pose = 0
        smpl_model = smplx.SMPL(model_path ='../assets/smpl_files/smpl',  batch_size = 1,gender=realgender)
        norm_obj_dir = os.path.join(data_path, 'norm_obj')
        os.makedirs(norm_obj_dir, exist_ok=True)
        for pose_idx in range(start_pose, frame_num + start_pose):
            image_key = str(pose_idx).zfill(8)

            cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                                    global_orient=smpl_data['body_pose'][pose_idx, :3][None],
                                    transl = smpl_data['trans'][pose_idx, :][None],
                                    body_pose=smpl_data['body_pose'][pose_idx, 3:][None],
                                    )
            norm_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
            mesh = trimesh.Trimesh(norm_vertices, smpl_model.faces, process=False)
            mesh.export('%s/%s.obj' % (norm_obj_dir, str(image_key)))
    if mode=='smplx':
        smplx_data = torch.load( data_path + '/' + name)
        frame_num = smplx_data['body_pose'].shape[0]
        print('frame_num', frame_num)
        start_pose = 0
        smplx_model = smplx.SMPLX(model_path ='../assets/smpl_files/smplx',  batch_size = 1,use_pca=True,num_pca_comps=12,gender=realgender)
        norm_obj_dir = os.path.join(data_path, 'norm_obj_x')
        os.makedirs(norm_obj_dir, exist_ok=True)
        for pose_idx in range(start_pose, frame_num + start_pose):
            image_key = str(pose_idx).zfill(8)
            cano_smplx = smplx_model.forward(betas=smplx_data['beta'],
                                    global_orient=smplx_data['body_pose'][pose_idx, :3][None],
                                    transl = smplx_data['trans'][pose_idx, :][None],
                                    body_pose=smplx_data['body_pose'][pose_idx, 3:][None],
                                    jaw_pose = smplx_data['jaw_pose'][pose_idx, :][None],
                                    leye_pose=smplx_data['leye_pose'][pose_idx, :][None],
                                    reye_pose=smplx_data['reye_pose'][pose_idx, :][None],
                                    left_hand_pose=smplx_data['left_hand_pose'][pose_idx, :][None],
                                    right_hand_pose=smplx_data['right_hand_pose'][pose_idx, :][None],
                                    expression=smplx_data['expression'][pose_idx, :][None])
            norm_vertices = cano_smplx.vertices.detach().cpu().numpy().squeeze()
            mesh = trimesh.Trimesh(norm_vertices, smplx_model.faces, process=False)
            mesh.export('%s/%s.obj' % (norm_obj_dir, str(image_key)))
            #print('save success')

def save_npz(data_path,mode, res=128):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0
    if mode=='smpl':
        norm_obj_dir = os.path.join(data_path, 'norm_obj')
        inp_map_dir = os.path.join(data_path, 'inp_map')
    if mode=='smplx':
        norm_obj_dir = os.path.join(data_path, 'norm_obj_x')
        inp_map_dir = os.path.join(data_path, 'inp_map_x')
    os.makedirs(inp_map_dir, exist_ok=True)
    # os.makedirs(query_map_dir, exist_ok=True)

    norm_obj_length = len(os.listdir(norm_obj_dir))
    result = {}
    for indx in range(start_obj_num, start_obj_num+norm_obj_length):
        image_key = str(indx).zfill(8)
        body_mesh = trimesh.load('%s/%s.obj'%(norm_obj_dir, image_key), process=False)

        if res==128:
            posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
            result['posmap128'] = posmap128   
        elif res == 256:
        
            posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
            result['posmap256'] = posmap256

        else:
            posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
            result['posmap512'] = posmap512

        save_fn = join(inp_map_dir, 'inp_posemap_%s_%s.npz'% (str(res), image_key))
        np.savez(save_fn, **result)
    # print(uvs.shape,'uvs.shape')
    # print(faces_uvs.shape,'faces_uvs.shape')
    # print(body_mesh.vertices.shape,'body_mesh.vertices.shape')
    # print(body_mesh.faces.shape,'body_mesh.faces.shape')
def compute_laplacian(posmap):
    # 创建扩展掩码
    non_zero_mask = torch.any(posmap != 0, dim=2)
    zero_mask = torch.all(posmap == 0, dim=2)
    # 膨胀操作（使用卷积实现）
    zero_mask_float = zero_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    dilation_kernel = torch.ones(1,1,3,3, dtype=torch.float32).cuda()
    dilated_zero_mask = (torch.nn.functional.conv2d(zero_mask_float, dilation_kernel, padding=1) > 0).squeeze()
    valid_mask = non_zero_mask & ~dilated_zero_mask
    # 拉普拉斯卷积核
    laplace_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], 
                                dtype=torch.float32).view(1,1,3,3).cuda()
    # 各通道卷积
    laplacians = torch.zeros_like(posmap)
    for c in range(3):
        channel = posmap[...,c].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        conv = torch.nn.functional.conv2d(channel, laplace_kernel, padding=1)
        laplacians[...,c] = conv.squeeze() * valid_mask.float()
    # 计算幅度
    magnitude = torch.sqrt(torch.sum(laplacians**2, dim=2, keepdim=True))
    return magnitude.cpu().numpy(), valid_mask.cpu().numpy()
def save_laplacian(takepath,n,mode):
    if mode=='smpl':
        name='laplacian'
        norm_obj_dir = os.path.join(takepath, 'norm_obj')
    if mode=='smplx':
        name='laplacian_x'
        norm_obj_dir = os.path.join(takepath, 'norm_obj_x')
    laplacian_dir = os.path.join(takepath, name)
    os.makedirs(laplacian_dir, exist_ok=True)
    if mode=='smpl':
        inp_map_dir = os.path.join(takepath, 'inp_map')
    if mode=='smplx':
        inp_map_dir = os.path.join(takepath, 'inp_map_x')
    for inp_map_idx in range(len(os.listdir(norm_obj_dir))):
        image_key = str(inp_map_idx).zfill(8)
        result = {}
        posmap_path = inp_map_dir +'/inp_posemap_%s_%s.npz'% (str(n), image_key)
        #print(posmap_path)
        posmap_np = np.load(posmap_path)['posmap' + str(n)]
        posmap = torch.from_numpy(posmap_np).float()  # 转为 float32
        posmap = posmap.to('cuda:0')
        magnitude, valid_mask=compute_laplacian(posmap)
        result['magnitude']=magnitude
        result['valid_mask']=valid_mask
        save_fn = join(laplacian_dir, 'laplacian_%s_%s.npz'% (str(n), image_key))
        np.savez(save_fn, **result)
    print('save success'+save_fn)

if __name__ == '__main__':
    #takelist = ['Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9']
    #takelist = ['take2', 'take3', 'take4', 'take5']
    #takelist = ['take7']
    #rootpath=r'/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_more/test'
    rootpath=r'/media/hhx/Lenovo/code/data/GaussianAvatar/gs_data/dress_simple/train'
    takelist = ['take2']
    n=128
    mode='smplx'
    mygender='female'#'neutral'#female
    if mode=='smpl':
        parms_name = 'smpl_parms.pth'
        uv_template_fn = '../assets/template_mesh_smpl_uv.obj' # smplx/smpl
    if mode=='smplx':
        parms_name = 'smplx_parms.pth'
        uv_template_fn = '../assets/template_mesh_smplx_uv.obj' # smplx/smpl
    for take_idx, take_name in enumerate(takelist):
        smpl_parm_path = os.path.join(rootpath,take_name)
        print('saving obj...')
        save_obj(smpl_parm_path,parms_name, mode,realgender=mygender)
        print('saving pose_map ',n)
        save_npz(smpl_parm_path, mode,n)
        print('saving laplacian...')
        save_laplacian(smpl_parm_path,n,mode)
        