import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import uv_to_grid
from model.modulesless import UnetNoCond5DS, GeomConvLayers, GeomConvBottleneckLayers, ShapeDecoder
from model.modules import ShapeDecoder as ShapeDecoder_more
class POP_no_unet(nn.Module):
    def __init__(
            self,
            c_geom=64,  # 几何特征的通道数
            geom_layer_type='conv',  # 用于平滑几何特征张量的架构类型
            nf=64,  # unet的过滤器数量
            hsize=256,  # ShapeDecoder MLP的隐藏层大小
            up_mode='upconv',  # 用于姿态特征UNet的上采样层的上采样方式
            use_dropout=False,  # 是否在姿态特征UNet中使用dropout
            uv_feat_dim=2,  # uv坐标的输入维度
            d=3,
            net_pred_all=True,
            add_dim=64,
            use_tmp_gs=0,
    ):
        super().__init__()
        c_geom=c_geom+add_dim
        self.net_pred_all=net_pred_all
        self.geom_layer_type = geom_layer_type
        self.use_tmp_gs=use_tmp_gs
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout),  # 使用unet
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False),  # 使用3个可训练的卷积层
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False),  # 使用3个可训练的瓶颈卷积层
        }
        # 可选层,用于空间平滑几何特征张量
        if geom_layer_type is not None :
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
        # 不同服装类型共享的形状解码器
        if self.net_pred_all:
            self.decoder = ShapeDecoder_more(in_size=uv_feat_dim + c_geom,d=d,
                                    hsize=hsize, actv_fn='softplus',)
        else:
            self.decoder = ShapeDecoder(in_size=uv_feat_dim + c_geom,d=d,
                                    hsize=hsize, actv_fn='softplus',)

    def forward(self, pose_featmap, geom_featmap, uv_loc,tmp_gs):
        # if pose_featmap is None:
        #     pix_feature = geom_featmap
        # else:
        #     pix_feature = torch.cat([pose_featmap , geom_featmap], 1)
        # if self.geom_layer_type is not None and geom_featmap is not None :
        #     geom_featmap = self.geom_proc_layers(pix_feature)
        # feat_res = geom_featmap.shape[2]  # 输入姿态和几何特征的空间分辨率
        # uv_res = int(uv_loc.shape[1] ** 0.5)  # 查询uv图的空间分辨率
        # print(uv_res,'uv_res')
        # # 空间双线性上采样特征以匹配查询分辨率
        # if feat_res != uv_res:
        #     query_grid = uv_to_grid(uv_loc, uv_res)
        #     pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)
        feat_res = geom_featmap.shape[2]  # 输入姿态和几何特征的空间分辨率
        uv_res = int(uv_loc.shape[1] ** 0.5)  # 查询uv图的空间分辨率 512
        # 空间双线性上采样特征以匹配查询分辨率
        if feat_res != uv_res:
            query_grid = uv_to_grid(uv_loc, uv_res)
            geom_featmap = F.grid_sample(geom_featmap, query_grid, mode='bilinear', align_corners=False)
            if pose_featmap is not None:
                pose_featmap = F.grid_sample(pose_featmap, query_grid, mode='bilinear', align_corners=False)
                # if self.use_tmp_gs == 1:
                #     pose_featmap = pose_featmap - tmp_gs.pos
        if pose_featmap is None:
            pix_feature = geom_featmap
        else:
            pix_feature = torch.cat([pose_featmap , geom_featmap], 1)
        if self.geom_layer_type is not None and geom_featmap is not None :
            pix_feature = self.geom_proc_layers(pix_feature)
        B, C, H, W = pix_feature.shape
        N_subsample = 1  # 继承SCALE代码的自定义,但现在每个像素只采样一个点
        uv_feat_dim = uv_loc.size()[-1]
        uv_loc = uv_loc.expand(N_subsample, -1, -1, -1).permute([1, 2, 0, 3])
        # uv和像素特征在每个patch中对所有点共享
        pix_feature = pix_feature.view(B, C, -1).expand(N_subsample, -1, -1, -1).permute([1, 2, 3, 0])  # [B, C, N_pix, N_sample_perpix]
        pix_feature = pix_feature.reshape(B, C, -1)
        uv_loc = uv_loc.reshape(B, -1, uv_feat_dim).transpose(1, 2)
        
        if self.net_pred_all:
            residuals, scales, shs ,pred_rotations, pred_opacitys= self.decoder(torch.cat([pix_feature, uv_loc], 1))
            return residuals, scales, shs,pred_rotations, pred_opacitys
        else:
            residuals, scales, shs, = self.decoder(torch.cat([pix_feature, uv_loc], 1))
            return residuals, scales, shs,

