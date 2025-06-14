import torch
import os
import numpy as np
from submodules import smplx
import trimesh
from scene.dataset_mono_moretake import MonoDataset_train, MonoDataset_test, MonoDataset_novel_pose, MonoDataset_novel_view,MultiViewDataset
from utils.general_utils import worker_init_fn
from utils.system_utils import mkdir_p
from model.network_new_gs_2f import POP_no_unet
from utils.general_utils import load_masks
from gaussian_renderer import render_batch
from os.path import join
import torch.nn as nn
from model.modules import UnetNoCond5DS
from scipy.ndimage import convolve, binary_dilation
import torch.nn.functional as F
import sys
#sys.path.append('/media/hhx/Lenovo/code/GaussianAvatarori')
from utils.rotations import quaternion_multiply,quaternion_to_matrix,matrix_to_quaternion
import cv2
import gc
from pprint import pprint
class MemoryMonitor:
    def __enter__(self):
        self.begin = torch.cuda.memory_allocated()
        
    def __exit__(self, *args):
        used = (torch.cuda.memory_allocated() - self.begin) / 1024**3
        print(f"本代码块显存增量: {used:.2f} GB")

def get_cuda_memory_usage():
    cuda_tensors = []
    for obj in gc.get_objects():  # 遍历所有Python对象
        if torch.is_tensor(obj) and obj.is_cuda:
            cuda_tensors.append({
                "shape": obj.shape,
                "dtype": obj.dtype,
                "device": obj.device,
                "size_MB": obj.element_size() * obj.nelement() / 1024**2
            })
    
    # 按内存占用排序
    cuda_tensors.sort(key=lambda x: x["size_MB"], reverse=True)
    return cuda_tensors

class TemporaryGaussians(nn.Module):
    def __init__(self, initial_pos):
        super().__init__()
        # 位置参数（保持原有初始化）
        self.pos = nn.Parameter(initial_pos.clone().requires_grad_(True))
        
        # 旋转参数（四元数随机初始化）
        num_points = initial_pos.shape[1]
        rot = torch.randn(1, num_points, 4, device='cuda')  # 初始随机四元数
        self.rot = nn.Parameter(torch.nn.functional.normalize(rot, dim=-1).requires_grad_(True))

class AvatarModel:
    def __init__(self, model_parms, net_parms, opt_parms, load_iteration=None, train=True):
        print(model_parms.train_smpl)
        self.more_take=1
        self.net_pred_all=True#False#
        #self.use_tmp_gs=model_parms.use_tmp_gs,
        self.use_tmp_gs=1
        self.mul=True
        self.cam_num=8
        self.train_o=False
        self.model_parms = model_parms
        self.net_parms = net_parms
        self.opt_parms = opt_parms
        self.model_path = model_parms.model_path
        self.loaded_iter = None
        self.train = train
        self.train_mode = model_parms.train_mode
        self.gender = self.model_parms.smpl_gender
        self.dimension=self.model_parms.dimension
        print(self.gender,'self.gender')
        
        if train:
            self.batch_size = self.model_parms.batch_size
        else:
            self.batch_size = 1
        if train:
            split = 'train'
        else:
            split = 'test'
        if train:
            if self.mul:
                self.train_dataset = MultiViewDataset(model_parms)
            else:
                self.train_dataset = MonoDataset_train(model_parms)
#        self.smpl_data = self.train_dataset.smpl_data
        # partial code derive from POP (https://github.com/qianlim/POP)
        assert model_parms.smpl_type in ['smplx', 'smpl']
        if model_parms.smpl_type == 'smplx':
            self.smpl_model = smplx.SMPLX(model_path=self.model_parms.smplx_model_path, gender = self.gender, use_pca = True, num_pca_comps = 12, flat_hand_mean = True, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smplx')
            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smplx.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smplx_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path, split, 'smplx_cano_joint_mat.pth')
            joint_num = 55
        else:
            self.smpl_model = smplx.SMPL(model_path=self.model_parms.smpl_model_path, gender = self.gender, batch_size = self.batch_size).cuda().eval()
            flist_uv, valid_idx, uv_coord_map = load_masks(model_parms.project_path, self.model_parms.query_posmap_size, body_model='smpl')

            query_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.query_posmap_size))
            inp_map_path = join(model_parms.source_path, split, 'query_posemap_{}_cano_smpl.npz'.format(self.model_parms.inp_posmap_size))

            query_lbs_path =join(model_parms.project_path, 'assets', 'lbs_map_smpl_{}.npy'.format(self.model_parms.query_posmap_size))
            mat_path = join(model_parms.source_path,  split, 'smpl_cano_joint_mat.pth')
            joint_num = 24
        print(query_map_path,'query_map_path')
        self.uv_coord_map = uv_coord_map
        self.valid_idx = valid_idx
        if model_parms.fixed_inp:
            fix_inp_map = torch.from_numpy(np.load(inp_map_path)['posmap' + str(self.model_parms.inp_posmap_size)].transpose(2,0,1)).cuda()
            self.fix_inp_map = fix_inp_map[None].expand(self.batch_size, -1, -1, -1)
        self.tmp_gs=None
        if self.use_tmp_gs == 1:
            self.my_map=torch.from_numpy(np.load(query_map_path)['posmap' + str(self.model_parms.query_posmap_size)]).cuda().contiguous()
            initial_pos = self.my_map.reshape(-1, 3).expand(1, -1, -1)
            self.tmp_gs = TemporaryGaussians(initial_pos)
        ## query_map store the sampled points from the cannonical smpl mesh, shape as [512. 512, 3] 
        query_map = torch.from_numpy(np.load(query_map_path)['posmap' + str(self.model_parms.query_posmap_size)]).reshape(-1,3)
        query_map=query_map.cuda() 
        query_points = query_map[valid_idx, :].cuda().contiguous()

        self.query_points = query_points[None].expand(self.batch_size, -1, -1)
        # we fix the opacity and rots of 3d gs as described in paper 
        self.fix_opacity = torch.ones((self.query_points.shape[1], 1)).cuda()
        rots = torch.zeros((self.query_points.shape[1], 4), device="cuda")
        rots[:, 0] = 1
        self.fix_rotation = rots
        # we save the skinning weights from the cannonical mesh
        query_lbs = torch.from_numpy(np.load(query_lbs_path)).reshape(self.model_parms.query_posmap_size*self.model_parms.query_posmap_size, joint_num)
        query_lbs =query_lbs.cuda()
        #self.myquery_lbs = query_lbs[:, :][None].expand(self.batch_size, -1, -1).cuda().contiguous()
        self.myquery_lbs = query_lbs[:, :][None].expand(1, -1, -1).cuda().contiguous()
        self.query_lbs = query_lbs[valid_idx, :][None].expand(self.batch_size, -1, -1).cuda().contiguous()
        self.inv_mats = torch.linalg.inv(torch.load(mat_path)).expand(self.batch_size, -1, -1, -1).cuda()
        self.myinv_mats = torch.linalg.inv(torch.load(mat_path)).expand(1, -1, -1, -1).cuda()
        print('inv_mat shape: ', self.inv_mats.shape)
        if train:
            if self.mul:
                num_training_frames = len(self.train_dataset)/self.cam_num
            else:
                num_training_frames = len(self.train_dataset)
            param = []
            if self.more_take==1:   
                if not torch.is_tensor(self.train_dataset.global_beta):
                    self.betas = torch.from_numpy(self.train_dataset.global_beta[0])[None].expand(self.batch_size, -1).cuda()
                else:
                    self.betas = self.train_dataset.global_beta[0][None].expand(self.batch_size, -1).cuda()
            else:
                self.smpl_data = self.train_dataset.smpl_data
                if not torch.is_tensor(self.smpl_data['beta']):
                    self.betas = torch.from_numpy(self.smpl_data['beta'][0])[None].expand(self.batch_size, -1).cuda()
                else:
                    self.betas = self.smpl_data['beta'][0][None].expand(self.batch_size, -1).cuda()
            if model_parms.smpl_type == 'smplx':
                if self.more_take==1:
                    self.pose = torch.nn.Embedding(len(self.train_dataset.global_pose_data), 66, _weight=self.train_dataset.global_pose_data, sparse=True).cuda()
                    self.transl = torch.nn.Embedding(len(self.train_dataset.global_transl_data), 3, _weight=self.train_dataset.global_transl_data, sparse=True).cuda()
                    
                    self.jaw_pose = torch.nn.Embedding(len(self.train_dataset.global_jaw_pose_data), 3, _weight=self.train_dataset.global_jaw_pose_data, sparse=True).cuda()
                    self.leye_pose = torch.nn.Embedding(len(self.train_dataset.global_leye_pose_data), 3, _weight=self.train_dataset.global_leye_pose_data, sparse=True).cuda()
                    self.reye_pose = torch.nn.Embedding(len(self.train_dataset.global_reye_pose_data), 3, _weight=self.train_dataset.global_reye_pose_data, sparse=True).cuda()
                    self.left_hand_pose = torch.nn.Embedding(len(self.train_dataset.global_left_hand_pose_data), 12, _weight=self.train_dataset.global_left_hand_pose_data, sparse=True).cuda()
                    self.right_hand_pose = torch.nn.Embedding(len(self.train_dataset.global_right_hand_pose_data), 12, _weight=self.train_dataset.global_right_hand_pose_data, sparse=True).cuda()
                    self.expression = torch.nn.Embedding(len(self.train_dataset.global_expression_data), 10, _weight=self.train_dataset.global_expression_data, sparse=True).cuda()
                
                else:
                    self.pose = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data, sparse=True).cuda()
                    self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
                if self.model_parms.train_smpl != 0:
                    param += list(self.pose.parameters())
                    param += list(self.transl.parameters())
            else:
                if self.more_take==1:
                    self.pose = torch.nn.Embedding(len(self.train_dataset.global_pose_data), 72, _weight=self.train_dataset.global_pose_data, sparse=True).cuda()
                    self.transl = torch.nn.Embedding(len(self.train_dataset.global_transl_data), 3, _weight=self.train_dataset.global_transl_data, sparse=True).cuda()
                else:
                    self.pose = torch.nn.Embedding(num_training_frames, 72, _weight=self.train_dataset.pose_data, sparse=True).cuda()
                    self.transl = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.transl_data, sparse=True).cuda()
                if self.model_parms.train_smpl != 0:
                    param += list(self.pose.parameters())
                    param += list(self.transl.parameters())
            print(num_training_frames)

        if self.model_parms.train_smpl != 0:
            self.optimizer_pose = torch.optim.SparseAdam(param, 5.0e-3)
        bg_color = [1, 1, 1] if model_parms.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.rotation_activation = torch.nn.functional.normalize
        self.sigmoid_activation =  nn.Sigmoid()
        self.net_set(self.model_parms.train_stage)

    def net_set(self, mode):
        assert mode in [1, 2, 3]

        self.net = POP_no_unet(
            c_geom=self.net_parms.c_geom, # channels of the geometric features
            geom_layer_type=self.net_parms.geom_layer_type, # the type of architecture used for smoothing the geometric feature tensor
            nf=self.net_parms.nf, # num filters for the unet
            hsize=self.net_parms.hsize, # hidden layer size of the ShapeDecoder MLP
            up_mode=self.net_parms.up_mode,# upconv or upsample for the upsampling layers in the pose feature UNet
            use_dropout=bool(self.net_parms.use_dropout), # whether use dropout in the pose feature UNet
            uv_feat_dim=2, # input dimension of the uv coordinates
            d=self.model_parms.dimension,
            net_pred_all=self.net_pred_all,
            use_tmp_gs=self.use_tmp_gs,
        ).cuda()
        # if self.model_parms.train_stage ==3:
        #     geo_feature = torch.zeros(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        # else:
        #     geo_feature = torch.ones(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        #64,128,128
        geo_feature = torch.ones(1, self.net_parms.c_geom, self.model_parms.inp_posmap_size, self.model_parms.inp_posmap_size).normal_(mean=0., std=0.01).float().cuda()
        self.geo_feature = nn.Parameter(geo_feature.requires_grad_(True))
        

        #print(self.tmp_gs.shape,'mmmmmmmmmm')
        if self.model_parms.train_stage == 2 or self.model_parms.train_stage == 3:
            self.pose_encoder = UnetNoCond5DS(
                input_nc=3,
                output_nc=self.net_parms.c_pose,
                nf=self.net_parms.nf,
                up_mode=self.net_parms.up_mode,
                use_dropout=False,
            ).cuda()
    def training_setup(self):
        if self.model_parms.train_stage == 1:
            params = [
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net},
                {"params": self.geo_feature, "lr": self.opt_parms.lr_geomfeat}
            ]
        if self.model_parms.train_stage == 2:
            params = [
                {"params": self.net.parameters(), "lr": self.opt_parms.lr_net * 0.1},
                {"params": self.pose_encoder.parameters(), "lr": self.opt_parms.lr_net}
            ]
        if self.model_parms.train_stage == 3:
                params = [
                    {"params": self.net.parameters(), "lr": self.opt_parms.lr_net * 0.1},
                    {"params": self.pose_encoder.parameters(), "lr": self.opt_parms.lr_net},
                    {"params": self.geo_feature, "lr": self.opt_parms.lr_geomfeat}
                ]
        if self.use_tmp_gs == 1:
            #params.append({"params": self.tmp_gs, "lr": self.opt_parms.lr_tmp_gs})
            params.append({"params": self.tmp_gs.parameters(),"lr": self.opt_parms.lr_tmp_gs})
        self.optimizer = torch.optim.Adam(params)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.opt_parms.sched_milestones, gamma=0.1)

    def save(self, iteration):
        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(iteration))
        mkdir_p(net_save_path)
        
        if self.model_parms.train_stage == 1:
            save_dict = {
                "net": self.net.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            if self.use_tmp_gs == 1:
                #save_dict["tmp_gs"] = self.tmp_gs
                save_dict["tmp_gs"] = self.tmp_gs.state_dict()
            torch.save(save_dict, os.path.join(net_save_path, "net.pth"))
        
        if self.model_parms.train_stage == 2:
            save_dict = {
                "pose_encoder": self.pose_encoder.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            if self.use_tmp_gs == 1:
                #save_dict["tmp_gs"] = self.tmp_gs
                save_dict["tmp_gs"] = self.tmp_gs.state_dict()
            torch.save(save_dict, os.path.join(net_save_path, "pose_encoder.pth"))
        
        if self.model_parms.train_stage == 3:
            save_dict = {
                "pose_encoder": self.pose_encoder.state_dict(),
                "geo_feature": self.geo_feature,
                "pose": self.pose.state_dict(),
                "transl": self.transl.state_dict(),
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict()
            }
            if self.use_tmp_gs == 1:
                #save_dict["tmp_gs"] = self.tmp_gs
                save_dict["tmp_gs"] = self.tmp_gs.state_dict()
            torch.save(save_dict, os.path.join(net_save_path, "pose_encoder3.pth"))

    def load(self, epoch, test=False):
        net_save_path = os.path.join(self.model_path, "net/iteration_{}".format(epoch))
        if self.model_parms.train_stage  == 1:
            name="net.pth"
            saved_model_state = torch.load(
                os.path.join(net_save_path, name))
            print('load pth: ', os.path.join(net_save_path, name))
            self.net.load_state_dict(saved_model_state["net"], strict=False)
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
            if self.use_tmp_gs == 1:
                #self.tmp_gs.data[...] = saved_model_state["tmp_gs"].data[...]
                self.tmp_gs.load_state_dict(saved_model_state["tmp_gs"])  # 自动加载所有参数

        if self.model_parms.train_stage  == 2:
            saved_model_state = torch.load(
                os.path.join(net_save_path, "pose_encoder.pth"))
            print('load pth: ', os.path.join(net_save_path, "pose_encoder.pth"))
            self.net.load_state_dict(saved_model_state["net"], strict=False)
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
            if self.use_tmp_gs == 1:
                #self.tmp_gs.data[...] = saved_model_state["tmp_gs"].data[...]
                self.tmp_gs.load_state_dict(saved_model_state["tmp_gs"])  # 自动加载所有参数
            self.pose_encoder.load_state_dict(saved_model_state["pose_encoder"], strict=False)

        if self.model_parms.train_stage  == 3:
            saved_model_state = torch.load(
                os.path.join(net_save_path, "pose_encoder3.pth"))
            print("Saved model state keys:", saved_model_state.keys())
            print('load pth: ', os.path.join(net_save_path, "pose_encoder3.pth"))
            self.net.load_state_dict(saved_model_state["net"], strict=False)
            if not test:
                self.pose.load_state_dict(saved_model_state["pose"], strict=False)
                self.transl.load_state_dict(saved_model_state["transl"], strict=False)
            self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
            if self.use_tmp_gs == 1:
                #self.tmp_gs.data[...] = saved_model_state["tmp_gs"].data[...]
                self.tmp_gs.load_state_dict(saved_model_state["tmp_gs"])  # 自动加载所有参数
            self.pose_encoder.load_state_dict(saved_model_state["pose_encoder"], strict=False)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved_model_state["optimizer"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(saved_model_state["scheduler"])

    def stage_load(self, epoch):
        net_save_path =  os.path.join(self.model_path, "net/iteration_{}".format(epoch))
        print('load pth: ', os.path.join(net_save_path, "net.pth"))
        saved_model_state = torch.load(
            os.path.join(net_save_path, "net.pth"))
        
        self.net.load_state_dict(saved_model_state["net"], strict=False)
        self.pose.load_state_dict(saved_model_state["pose"], strict=False)
        self.transl.load_state_dict(saved_model_state["transl"], strict=False)
        self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
        self.geo_feature = nn.Parameter(self.geo_feature.requires_grad_(True))
        if self.use_tmp_gs == 1:
            self.tmp_gs.data[...] = saved_model_state["tmp_gs"].data[...]
        #self.geo_feature.data[...] = saved_model_state["geo_feature"].data[...]
    # def getTrainDataloader(self,):
    #     return torch.utils.data.DataLoader(self.train_dataset,
    #                                         batch_size = self.batch_size,
    #                                         shuffle = True,#shuffle = False,
    #                                         num_workers = 4,
    #                                         worker_init_fn = worker_init_fn,
    #                                         drop_last = True)    
    def getTrainDataloader(self):
        # 创建两个独立DataLoader（参考之前的实现）
        def create_loader(generator):
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=4,
                worker_init_fn=worker_init_fn,
                drop_last=True,
                generator=generator
            )
        seed1 = torch.randint(0, 2**32, (1,)).item()
        seed2 = torch.randint(0, 2**32, (1,)).item()
        loader1 = create_loader(torch.Generator().manual_seed(seed1))
        loader2 = create_loader(torch.Generator().manual_seed(seed2))
        self.current_epoch_loader_length = len(loader1)  # 保存长度供外部访问
        return zip(loader1, loader2)

    def getTestDataset(self,):
        self.test_dataset = MonoDataset_test(self.model_parms)
        return self.test_dataset
    
    def getNovelposeDataset(self,):
        self.novel_pose_dataset = MonoDataset_novel_pose(self.model_parms)
        return self.novel_pose_dataset

    def getNovelviewDataset(self,):
        self.novel_view_dataset = MonoDataset_novel_view(self.model_parms)
        return self.novel_view_dataset

    def zero_grad(self, epoch):
        self.optimizer.zero_grad()
        if self.model_parms.train_smpl != 0:
            if self.model_parms.train_stage  == 1:
                if epoch > self.opt_parms.pose_op_start_iter:
                    self.optimizer_pose.zero_grad()

    def step(self, epoch):
        self.optimizer.step()
        self.scheduler.step()
        if self.model_parms.train_smpl != 0:
            if self.model_parms.train_stage  == 1 :
                if epoch > self.opt_parms.pose_op_start_iter:
                    self.optimizer_pose.step()
    def compute_laplacian(self,posmap,mask=None):
        #torch.Size([1280, 940, 3]) 
        if mask is None:
            # 创建扩展掩码
            non_zero_mask = torch.any(posmap != 0, dim=2)
            zero_mask = torch.all(posmap == 0, dim=2)
            # 膨胀操作（使用卷积实现）
            zero_mask_float = zero_mask.float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            dilation_kernel = torch.ones(1,1,3,3, dtype=torch.float32).cuda()
            dilated_zero_mask = (torch.nn.functional.conv2d(zero_mask_float, dilation_kernel, padding=1) > 0).squeeze()
            valid_mask = non_zero_mask & ~dilated_zero_mask
        else:
            valid_mask=mask
        # 拉普拉斯卷积核
        laplace_kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], 
                                    dtype=torch.float32).view(1,1,3,3).cuda()
        # 各通道卷积
        laplacians = torch.zeros_like(posmap)
        for c in range(3):
            channel = posmap[...,c].unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            conv = torch.nn.functional.conv2d(channel, laplace_kernel, padding=1)
            laplacians[...,c] = conv.squeeze() * valid_mask.float()
        #magnitude = torch.sqrt(torch.sum(laplacians**2, dim=2, keepdim=True))
        # return magnitude, valid_mask
        return laplacians, valid_mask
    def save_tensor_as_image(self, tensor, file_path):
        if tensor.dim() != 3:
            raise ValueError(f"张量形状应为 [H, W, C] 其中 C=1 或 3，但实际为 {tensor.shape}")
        if tensor.is_cuda:
            tensor = tensor.cpu()
        numpy_array = tensor.detach().numpy()
        # 对于单通道输入，复制为三通道
        if numpy_array.shape[2] == 1:
            numpy_array = np.repeat(numpy_array, 3, axis=2)
        min_val = numpy_array.min()
        max_val = numpy_array.max()
        if max_val > min_val:  # 避免除以零
            numpy_array = 255 * (numpy_array - min_val) / (max_val - min_val)
        else:
            numpy_array = numpy_array * 255
        numpy_array = np.clip(numpy_array, 0, 255).astype(np.uint8)
        cv2.imwrite(file_path, cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR))
        print(f"已保存图像至: {file_path}")
        print(f"图像尺寸: {numpy_array.shape}")
        print(f"数据类型: {numpy_array.dtype}")
        print(f"值范围: {numpy_array.min()} - {numpy_array.max()}")
    def twist(self, batch_data1, batch_data2, iteration=0,use_cal=False):
        if use_cal:
            # # 在twist函数中添加：
            # print(f"梯度要求检查: batch_data1.requires_grad={batch_data1.requires_grad}, "
            #     f"batch_data2.requires_grad={batch_data2.requires_grad}")
            gt_normal=batch_data1
            pred_normal=batch_data2
            laplacian2, mask2 = self.compute_laplacian(pred_normal)
            laplacian1, mask1 = self.compute_laplacian(gt_normal,mask2)
            
            # self.printzero(laplacian1,'laplacian1')
            # self.printzero(laplacian2,'laplacian2')

            # self.save_tensor_as_image(laplacian1, '/media/hhx/Lenovo/code/myAvatar/model/1.png')
            # self.save_tensor_as_image(laplacian2, '/media/hhx/Lenovo/code/myAvatar/model/2.png')
            # return 
            common_mask = mask1 & mask2
            #diff_p = torch.abs(laplacian1 - laplacian2) * common_mask[..., None]  # 自动应用mask
            diff_p = torch.mean(torch.abs(laplacian1 - laplacian2))
            #diff_p = torch.abs(batch_data2 - batch_data1) * common_mask[..., None]  # 自动应用mask
            normal_lpls_loss=diff_p
            num_valid = common_mask.sum()

            #normal_lpls_loss=torch.sum(diff_p) / num_valid 
            return normal_lpls_loss
        else:
            laplacian1, mask1 = batch_data1['magnitude'],batch_data1['valid_mask']
            laplacian2, mask2 = batch_data2['magnitude'],batch_data2['valid_mask']
            common_mask = mask1 & mask2
            diff_p = torch.abs(laplacian1 - laplacian2) * common_mask[..., None]  # 自动应用mask
            return diff_p, common_mask  # 返回共同有效区域掩码

    def diff_p_f(self, batch_data1, batch_data2, feature1, feature2, iteration=0):
        diff_p, common_mask = self.twist(batch_data1, batch_data2)  # 获取diff_p和共同mask
        diff_f_single = (feature1 - feature2) ** 2
        diff_f = torch.sum(diff_f_single, dim=2) * common_mask.float()  # 应用mask到diff_f
        
        # 仅针对有效区域进行归一化
        valid_diff_p = diff_p[common_mask[..., None]]  # 提取有效区域的diff_p
        valid_diff_f = diff_f[common_mask]             # 提取有效区域的diff_f
        
        if valid_diff_p.numel() == 0 or valid_diff_f.numel() == 0:
            return torch.tensor(0.0, device=diff_p.device)  # 无有效区域时返回0
        
        # 标准化处理 (均值+标准差)
        mean_p, std_p = valid_diff_p.mean(), valid_diff_p.std() + 1e-8
        mean_f, std_f = valid_diff_f.mean(), valid_diff_f.std() + 1e-8
        
        # 归一化后重新应用mask确保无效区域为0
        diff_p_normalized = (diff_p - mean_p) / std_p * common_mask[..., None].float()
        diff_f_normalized = (diff_f - mean_f) / std_f * common_mask.float()
        
        # 计算归一化后的乘积损失
        product = diff_p_normalized * diff_f_normalized.unsqueeze(-1)
        num_valid = common_mask.sum() + 1e-8  # 有效像素数
        diff_loss = -torch.sum(product) / num_valid  # 仅对有效区域取平均
        
        return diff_loss*0.003#0.1 L1 0.0017 SSIM 0.0023 diff -0.0967
       
    def train_stage(self, batch_data, iteration,stage,batchname=1):
        rendered_images = []
        idx = batch_data['pose_idx']
        pose_batch = self.pose(idx)
        transl_batch = self.transl(idx)
        if self.model_parms.smpl_type == 'smplx':
            #rest_pose = batch_data['rest_pose']
            jaw_pose=self.jaw_pose(idx)
            leye_pose=self.leye_pose(idx)
            reye_pose=self.reye_pose(idx)
            left_hand_pose= self.left_hand_pose(idx)
            right_hand_pose= self.right_hand_pose(idx)
            expression=self.expression(idx)
            live_smpl = self.smpl_model.forward(betas = self.betas,
                                                global_orient = pose_batch[:, :3],
                                                transl = transl_batch,
                                                body_pose = pose_batch[:, 3:66],
                                                jaw_pose = jaw_pose,
                                                leye_pose=leye_pose,
                                                reye_pose=reye_pose,
                                                left_hand_pose=left_hand_pose,
                                                right_hand_pose=right_hand_pose,
                                                expression=expression)
        else:
            live_smpl = self.smpl_model.forward(betas=self.betas,
                                global_orient=pose_batch[:, :3],
                                transl = transl_batch,
                                body_pose=pose_batch[:, 3:])
        
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        #print(geom_featmap.shape)#torch.Size([1, 64, 128, 128])
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        #print(uv_coord_map.shape)#torch.Size([1, 262144, 2])
        new_tmp_gs = None
        if self.use_tmp_gs == 1:
            mycano2live_jnt_mats = torch.matmul(live_smpl.A, self.myinv_mats)
            mypt_mats = torch.einsum('bnj,bjxy->bnxy', self.myquery_lbs, mycano2live_jnt_mats)
            new_tmp_gs = torch.einsum('bnxy,bny->bnx', mypt_mats[..., :3, :3], self.tmp_gs.pos) + mypt_mats[..., :3, 3]
            # new_tmp_gs = torch.einsum('bnxy,bny->bnx', mypt_mats[..., :3, :3], self.tmp_gs) + mypt_mats[..., :3, 3]
            new_tmp_gs = new_tmp_gs.reshape(self.batch_size, 3,512, 512)
        pose_featmap = None
        if stage == 2 or stage == 1 or stage == 3:
            inp_posmap = batch_data['inp_pos_map']
            #pose_featmap = inp_posmap
            pose_featmap = self.pose_encoder(inp_posmap)
            if batchname==2:
                # print('batchname=2 now!')
                # print(pose_featmap[0].permute(1,2,0).shape,'pose_featmap[0].permute(1,2,0)')
                return pose_featmap[0].permute(1,2,0)
        if self.net_pred_all:
            pred_res,pred_scales, pred_shs,pred_rotations, pred_opacitys = self.net.forward(pose_featmap=pose_featmap,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map,
                                                        tmp_gs=new_tmp_gs)
        else:
            pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map,
                                                        tmp_gs=new_tmp_gs)
        #print(pred_res.shape)#torch.Size([1, 3, 262144])
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()
        cano_deform_point = pred_point_res + self.query_points
        #print(cano_deform_point.shape)#torch.Size([1, 202738, 3])
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]
        #print(torch.cuda.memory_allocated() / 1024**3, "GB used")
        if self.net_pred_all:
            pred_rotations = pred_rotations.permute([0,2,1])
            if self.use_tmp_gs==1:
                final_rotations=torch.nn.functional.normalize(pred_rotations+self.tmp_gs.rot, dim=-1)
                final_rotations = final_rotations[:, self.valid_idx, ...].contiguous()
                rotation_mats = quaternion_to_matrix(final_rotations)
            else:
                pred_rotations = pred_rotations[:, self.valid_idx, ...].contiguous()
                rotation_mats = quaternion_to_matrix(pred_rotations)
            full_pred_rotation_mats = torch.einsum('bnxy,bnxy->bnxy', pt_mats[..., :3, :3], rotation_mats)
            full_pred_rotation = matrix_to_quaternion(full_pred_rotation_mats)

            # pred_opacitys=pred_opacitys.permute([0,2,1])
            # pred_opacitys = pred_opacitys[:, self.valid_idx, ...].contiguous() 
            pred_opacitys = None

        # if (stage == 1 or stage == 3) and iteration < 90:
        #     pred_scales = pred_scales.permute([0,2,1]) * 1e-2 * iteration 
        # else:
        #     pred_scales = pred_scales.permute([0,2,1])
        pred_scales = pred_scales.permute([0,2,1])* 1e-2
        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        if not self.net_pred_all:
            pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs.permute([0,2,1])
        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()

        offset_loss = torch.mean(pred_res ** 2)
        scale_loss = torch.mean(pred_scales)
        gs_loss=torch.tensor(0,device='cuda:0')
        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            if self.dimension==2:
                scales = pred_scales[batch_index][:,:2]
            if self.dimension==3:
                scales = pred_scales[batch_index]
            if self.net_pred_all:
                #opacity = pred_opacitys[batch_index]
                opacity=self.fix_opacity
                rotations = full_pred_rotation[batch_index]
            else:
                rotations=self.fix_rotation
                opacity=self.fix_opacity
            # 调用函数并打印结果
            # memory_usage = get_cuda_memory_usage()
            # pprint(memory_usage[:5])  # 显示占用最高的前5个张量
            # print('!!!!!!!!!!!!!!33333333333333333')
            # with MemoryMonitor():
            render_pkg = render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=rotations,
                    scales=scales,
                    opacity=opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
        if self.model_parms.dimension==2:
            rendered_images.append(render_pkg["render"])
            rend_dist = render_pkg["rend_dist"]
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['surf_normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = self.opt_parms.lambda_normal * (normal_error).mean()
            dist_loss = self.opt_parms.lambda_dist * (rend_dist).mean()
            gs_loss=gs_loss+normal_loss+dist_loss
        else:
            rendered_images.append(render_pkg)
        if stage == 1:
            geo_loss = torch.mean(self.geo_feature**2)
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, gs_loss,scale_loss
        if stage == 2:
            pose_loss = torch.mean(pose_featmap ** 2)
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, pose_loss, gs_loss
        if stage == 3:
            geo_loss = torch.mean(self.geo_feature**2)
            pose_loss = torch.mean(pose_featmap ** 2)
            normal_lpls_loss=self.twist(batch_data["normal"].squeeze(0),rend_normal.squeeze(0).permute(1, 2, 0),use_cal=True)
            return torch.stack(rendered_images, dim=0), full_pred, offset_loss, geo_loss, pose_loss, gs_loss,scale_loss,normal_lpls_loss,pose_featmap[0].permute(1,2,0)
        return
    
    def render_free_stage(self, batch_data, iteration,stage,idx):
        # savedir=r'/media/hhx/Lenovo/code/GaussianAvatarori/output/dress_my/novel_pose_smpl/obj'
        # os.makedirs(savedir, exist_ok=True)
        image_key = str(idx).zfill(8)
        rendered_images = []
        normals_images = []
        depth_normals_images = []
        pose_data = batch_data['pose_data']
        transl_data = batch_data['transl_data']
        beta_data = batch_data['beta_data']
        if self.model_parms.smpl_type == 'smplx':
            jaw_pose=batch_data['jaw_pose']
            leye_pose=batch_data['leye_pose']
            reye_pose=batch_data['reye_pose']
            left_hand_pose=batch_data['left_hand_pose']
            right_hand_pose=batch_data['right_hand_pose']
            expression=batch_data['expression']
        
            # live_smpl = self.smpl_model.forward(betas = self.betas,
            #                                     global_orient = pose_data[:, :3],
            #                                     transl = transl_data,
            #                                     body_pose = pose_data[:, 3:66],
            #                                     jaw_pose = rest_pose[:, :3],
            #                                     leye_pose=rest_pose[:, 3:6],
            #                                     reye_pose=rest_pose[:, 6:9],
            #                                     left_hand_pose= rest_pose[:, 9:54],
            #                                     right_hand_pose= rest_pose[:, 54:])
            live_smpl = self.smpl_model.forward(betas = beta_data,
                                                global_orient = pose_data[:, :3],
                                                transl = transl_data,
                                                body_pose = pose_data[:, 3:66],
                                                jaw_pose = jaw_pose,
                                                leye_pose=leye_pose,
                                                reye_pose=reye_pose,
                                                left_hand_pose=left_hand_pose,
                                                right_hand_pose=right_hand_pose,
                                                expression=expression)
        else:
            live_smpl = self.smpl_model.forward(betas=beta_data,#self.betas,
                                global_orient=pose_data[:, :3],
                                transl = transl_data,
                                body_pose=pose_data[:, 3:])
        cano2live_jnt_mats = torch.matmul(live_smpl.A, self.inv_mats)
        geom_featmap = self.geo_feature.expand(self.batch_size, -1, -1, -1).contiguous()
        uv_coord_map = self.uv_coord_map.expand(self.batch_size, -1, -1).contiguous()
        pose_featmap=None
        new_tmp_gs = None
        if self.use_tmp_gs == 1:
            mypt_mats = torch.einsum('bnj,bjxy->bnxy', self.myquery_lbs, cano2live_jnt_mats)
            # print(mypt_mats[..., :3, :3].shape,'mypt_mats[..., :3, :3].shape')#torch.Size([1, 262144, 3, 3])
            # print(self.tmp_gs.shape,'self.tmp_gs.shape')

            #new_tmp_gs = torch.einsum('bnxy,bny->bnx', mypt_mats[..., :3, :3], self.tmp_gs) + mypt_mats[..., :3, 3]
            new_tmp_gs = torch.einsum('bnxy,bny->bnx', mypt_mats[..., :3, :3], self.tmp_gs.pos) + mypt_mats[..., :3, 3]
            new_tmp_gs = new_tmp_gs.reshape(self.batch_size, 3,512, 512)

        if stage == 2 or stage == 1 or stage == 3:
            inp_posmap = batch_data['inp_pos_map']
            #pose_featmap = inp_posmap
            pose_featmap = self.pose_encoder(inp_posmap)
        if self.net_pred_all:
            pred_res,pred_scales, pred_shs,pred_rotations, pred_opacitys = self.net.forward(pose_featmap=pose_featmap,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map,
                                                        tmp_gs=new_tmp_gs)
        else:
            pred_res,pred_scales, pred_shs, = self.net.forward(pose_featmap=pose_featmap,
                                                        geom_featmap=geom_featmap,
                                                        uv_loc=uv_coord_map,
                                                        tmp_gs=new_tmp_gs)
        pred_res = pred_res.permute([0,2,1]) * 0.02  #(B, H, W ,3)
        pred_point_res = pred_res[:, self.valid_idx, ...].contiguous()
        cano_deform_point = pred_point_res + self.query_points 

        pt_mats = torch.einsum('bnj,bjxy->bnxy', self.query_lbs, cano2live_jnt_mats)
        full_pred = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]
        if self.net_pred_all:
            pred_rotations = pred_rotations.permute([0,2,1])
            if self.use_tmp_gs==1:
                final_rotations=torch.nn.functional.normalize(pred_rotations+self.tmp_gs.rot, dim=-1)
                final_rotations = final_rotations[:, self.valid_idx, ...].contiguous()
                rotation_mats = quaternion_to_matrix(final_rotations)
            else:
                pred_rotations = pred_rotations[:, self.valid_idx, ...].contiguous()
                rotation_mats = quaternion_to_matrix(pred_rotations)
            full_pred_rotation_mats = torch.einsum('bnxy,bnxy->bnxy', pt_mats[..., :3, :3], rotation_mats)
            full_pred_rotation = matrix_to_quaternion(full_pred_rotation_mats)

        pred_scales = pred_scales.permute([0,2,1])*1e-2
        pred_scales = pred_scales[:, self.valid_idx, ...].contiguous()
        if not self.net_pred_all:
            pred_scales = pred_scales.repeat(1,1,3)

        pred_shs = pred_shs.permute([0,2,1])
        pred_shs = pred_shs[:, self.valid_idx, ...].contiguous()
        #print(pred_shs)
        for batch_index in range(self.batch_size):
            FovX = batch_data['FovX'][batch_index]
            FovY = batch_data['FovY'][batch_index]
            height = batch_data['height'][batch_index]
            width = batch_data['width'][batch_index]
            world_view_transform = batch_data['world_view_transform'][batch_index]
            full_proj_transform = batch_data['full_proj_transform'][batch_index]
            camera_center = batch_data['camera_center'][batch_index]
        
            points = full_pred[batch_index]
            colors = pred_shs[batch_index]
            if self.dimension==2:
                scales = pred_scales[batch_index][:,:2]
            if self.dimension==3:
                scales = pred_scales[batch_index]
            if self.net_pred_all:
                if self.train_o:
                    opacity = pred_opacitys[batch_index]
                else:
                    opacity=self.fix_opacity
                rotations = full_pred_rotation[batch_index]
            else:
                rotations=self.fix_rotation
                opacity=self.fix_opacity
            render_pkg = render_batch(
                    points=points,
                    shs=None,
                    colors_precomp=colors,
                    rotations=rotations,
                    scales=scales,
                    opacity=opacity,
                    FovX=FovX,
                    FovY=FovY,
                    height=height,
                    width=width,
                    bg_color=self.background,
                    world_view_transform=world_view_transform,
                    full_proj_transform=full_proj_transform,
                    active_sh_degree=0,
                    camera_center=camera_center
                )
            if self.model_parms.dimension==2:
                rendered_images.append(render_pkg["render"])
                #print(render_pkg["render"].shape)
                normals_images.append(render_pkg['rend_normal'])
                depth_normals_images.append(render_pkg['surf_normal'])
                return torch.stack(rendered_images, dim=0),torch.stack(normals_images, dim=0),torch.stack(depth_normals_images, dim=0),world_view_transform.cpu().numpy(), full_pred

            else:
                rendered_images.append(render_pkg)
                return torch.stack(rendered_images, dim=0), full_pred
    