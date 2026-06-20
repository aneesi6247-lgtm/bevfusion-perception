# ============================================================================
#  BEVFusion Inference Configuration from NumPy Arrays
#
#  This configuration enables inference for BEVFusion using multi-camera images
#  and LiDAR point clouds provided as in-memory NumPy arrays. It uses a Swin
#  Transformer Tiny backbone for image feature extraction, applies deterministic
#  resizing, and fuses BEV features from both camera and LiDAR modalities.
#  Designed for efficient memory usage and suitable for both ROS 2 and offline
#  inference scenarios.
# ============================================================================

from mmengine.config import read_base
import os

# ============================================================================
#  Inherit Base Config
# ============================================================================

# Base configuration for training BEVFusion on nuScenes dataset
_base_ = ['./bevfusion_lidar_voxel0075_second_secfpn_8xb4_cyclic_20e_nus_3d.py']

# ============================================================================
#  Sensor & Range Settings
# ============================================================================

# 3D region of interest for filtering LiDAR points
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# Sensors used in this configuration
input_modality = dict(use_lidar=True, use_camera=True)

# Disable custom backend arguments
backend_args = None

# ============================================================================
#  BEVFusion Model Definition
# ============================================================================

model = dict(
    type='BEVFusion',

    # --------------------------------------------
    # Image Normalization (Preprocessing)
    # --------------------------------------------
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],   # ImageNet mean
        std=[58.395, 57.12, 57.375],      # ImageNet std
        bgr_to_rgb=False
    ),

    # --------------------------------------------
    # Image Backbone: Swin Transformer Tiny
    # --------------------------------------------
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],  # Extract 3 levels of features
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
        )
    ),

    # --------------------------------------------
    # Image Neck: Feature Pyramid Network
    # --------------------------------------------
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],  # From Swin Tiny out_indices
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)
    ),

    # --------------------------------------------
    # View Transformer (Depth LSS)
    # --------------------------------------------
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=[256, 704],     # Input image size
        feature_size=[32, 88],     # Matches FPN downsample
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 60.0, 0.5],   # 0.5m depth bins
        downsample=2
    ),

    # --------------------------------------------
    # Fusion Layer (Camera BEV + LiDAR BEV)
    # --------------------------------------------
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[80, 256],  # From view_transform and voxel backbone
        out_channels=256
    )
)

# ============================================================================
#  Test Pipeline (for ROS 2 or offline with NumPy arrays)
# ============================================================================

test_pipeline = [
    # Load LiDAR points from in-memory array
    dict(
        type='LoadPointsFromArray',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4
    ),

    # Load camera images from in-memory array
    dict(
        type='LoadMultiViewImageFromArray',
        to_float32=True,
        color_type='color',
        limit_views=4  # Use 4 cameras for this setup
    ),

    # Deterministic image resize (no data augmentation)
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],  # Fixed resize ratio
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),

    # Remove LiDAR points outside the defined 3D box
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),

    # Pack all data into format expected by BEVFusion model
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ]
    )
]

# ============================================================================
#  Dataloader Definition (Used for Testing with Arrays)
# ============================================================================

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        box_type_3d='LiDAR',
        test_mode=True
    )
)

# Mirror the same configuration for `test_dataloader`
test_dataloader = val_dataloader

# ============================================================================
#  Visualizer Settings
# ============================================================================

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[],  # No saving backend (pure visualization)
    name='visualizer'
)
