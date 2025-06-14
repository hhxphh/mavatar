#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
#from scene.gaussian_model import GaussianModel
#from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
class ViewpointCamera:
    def __init__(self, world_view_transform, height, width, full_proj_transform):
        self.world_view_transform = world_view_transform
        self.image_height = height
        self.image_width = width
        self.full_proj_transform = full_proj_transform


def render_batch(points, shs, colors_precomp, rotations, scales, opacity, FovX, FovY, height, width, bg_color,
                 world_view_transform, full_proj_transform, active_sh_degree, camera_center):
    depth_ratio = 0.0    
    # 创建 ViewpointCamera 实例
    viewpoint_camera = ViewpointCamera(
        world_view_transform=world_view_transform,
        height=height,
        width=width,
        full_proj_transform=full_proj_transform
    )
    screenspace_points = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(FovX * 0.5)
    tanfovy = math.tan(FovY * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier= 1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        #antialiasing=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    cov3D_precomp = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap = rasterizer(
        means3D = points,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }

    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    #print(render_normal.shape,'shapeeeeeeeeeee')
    render_normal = (render_normal.permute(1,2,0) @ (world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets