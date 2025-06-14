import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import uv_to_grid
from model.modulesless import UnetNoCond5DS, GeomConvLayers, GeomConvBottleneckLayers, ShapeDecoder
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
            add_dim=3,
    ):
        super().__init__()
        c_geom=c_geom+add_dim
        self.net_pred_all=net_pred_all
        self.geom_layer_type = geom_layer_type
        geom_proc_layers = {
            'unet': UnetNoCond5DS(c_geom, c_geom, nf, up_mode, use_dropout),  # 使用unet
            'conv': GeomConvLayers(c_geom, c_geom, c_geom, use_relu=False),  # 使用3个可训练的卷积层
            'bottleneck': GeomConvBottleneckLayers(c_geom, c_geom, c_geom, use_relu=False),  # 使用3个可训练的瓶颈卷积层
        }
        # 可选层,用于空间平滑几何特征张量
        if geom_layer_type is not None :
            self.geom_proc_layers = geom_proc_layers[geom_layer_type]
        # 不同服装类型共享的形状解码器
        self.decoder = ShapeDecoder(in_size=uv_feat_dim + c_geom,d=d,
                                    hsize=hsize, actv_fn='softplus',)

    def forward(self, pose_featmap, geom_featmap, uv_loc):
        # #现在相当于把3d位置直接和可优化特征向量连接送入unet
        # # 几何特征张量
        # if self.geom_layer_type is not None and geom_featmap is not None :
        #     geom_featmap = self.geom_proc_layers(geom_featmap)

        if pose_featmap is None:
            # 姿态和几何特征被连接以形成每个点的特征
            pix_feature = geom_featmap
        elif geom_featmap is None:
            # 姿态和几何特征被连接以形成每个点的特征
            pix_feature = pose_featmap  
        else:
            #pix_feature = pose_featmap + geom_featmap
            #print('is not none')
            #print(pose_featmap.shape,geom_featmap.shape,'pose_featmap.shape,geom_featmap.shape')
            pix_feature = torch.cat([pose_featmap , geom_featmap], 1)
#先插值再平滑？
        # if self.geom_layer_type is not None and geom_featmap is not None :
        #     pix_feature = self.geom_proc_layers(pix_feature)
        # if self.geom_layer_type is not None and geom_featmap is not None :
        #     geom_featmap = self.geom_proc_layers(geom_featmap)
        feat_res = pix_feature.shape[2]  # 输入姿态和几何特征的空间分辨率
        uv_res = int(uv_loc.shape[1] ** 0.5)  # 查询uv图的空间分辨率
        # 空间双线性上采样特征以匹配查询分辨率
        if feat_res != uv_res:
            #print('需要插值！')
            query_grid = uv_to_grid(uv_loc, uv_res)
            pix_feature = F.grid_sample(pix_feature, query_grid, mode='bilinear', align_corners=False)
        else:
            pass
            #print('不需要插值！')
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

