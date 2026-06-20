# =====================================================================
#                      BEVFusion Config - Array Input (Single Camera)
#  Adapted for low GPU usage: SwinTiny → ResNet18, 1 camera, reduced FPN
#  and DepthLSS, 2x downsampled, 16 channels.
#  Uses LoadPointsFromArray and LoadMultiViewImageFromArray for arrays.
#  Designed for offline arrays or ROS 2, not file-based inputs.
#  Use only for debugging or low-resource scenarios.
#  Note: This config is not suitable for production or high-performance tasks.
# =====================================================================

from mmengine.config import read_base

# ==============================================================
# Base Config
# ==============================================================

# Inherit default BEVFusion config (NuScenes, voxel-based, cyclic schedule)
_base_ = ['./bevfusion_lidar_voxel0075_second_secfpn_8xb4_cyclic_20e_nus_3d.py']

# ==============================================================
# Point Cloud Range
# ==============================================================

# Define the 3D region of interest for LiDAR points
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

# ==============================================================
# Input Modalities
# ==============================================================

# Specify which sensor modalities are active
input_modality = dict(use_lidar=True, use_camera=True)

# No image backend (e.g., from URLs or servers)
backend_args = None

# ==============================================================
# Model Definition
# ==============================================================

model = dict(
    type='BEVFusion',

    # ------------------------------------------
    # Data Preprocessor
    # ------------------------------------------
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],  # Normalization mean (ImageNet)
        std=[58.395, 57.12, 57.375],     # Normalization std (ImageNet)
        bgr_to_rgb=False                 # Images are assumed already in RGB
    ),

    # ------------------------------------------
    # Image Backbone (Reduced: ResNet-18)
    # ------------------------------------------
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2),  # Use first 3 feature maps (64, 128, 256)
        frozen_stages=0,        # Trainable from stage 0
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'
    ),

    # ------------------------------------------
    # Image Neck (FPN)
    # ------------------------------------------
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 256],  # Must match out_indices of backbone
        out_channels=64,             # Downscaled for efficiency
        num_outs=3
    ),

    # ------------------------------------------
    # View Transformation Module (DepthLSS)
    # ------------------------------------------
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=64,        # Matches FPN output channels
        out_channels=16,       # Reduced to save GPU memory
        image_size=[256, 704], # Input image size (resized by pipeline)
        feature_size=[16, 44], # Spatial size of features (downsampled)
        xbound=[-54.0, 54.0, 0.3],
        ybound=[-54.0, 54.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[1.0, 30.0, 1.0],  # Depth bins (1m resolution)
        downsample=2             # 2x downsampling in spatial domain
    ),

    # ------------------------------------------
    # Fusion Layer (Camera + LiDAR BEV features)
    # ------------------------------------------
    fusion_layer=dict(
        type='ConvFuser',
        in_channels=[32, 256],  # 32 from camera BEV, 256 from LiDAR BEV
        out_channels=256
    )
)

# ==============================================================
# Test Pipeline (For Offline Arrays or ROS 2)
# ==============================================================

test_pipeline = [
    # Load point cloud from array (not from file)
    dict(
        type='LoadPointsFromArray',
        coord_type='LIDAR',
        load_dim=5,     # Original point dimension (x, y, z, intensity, etc.)
        use_dim=4       # Use first 4 (typically x, y, z, intensity)
    ),

    # Load multiview images from array (1 or more cameras)
    dict(
        type='LoadMultiViewImageFromArray',
        to_float32=True,
        color_type='color'
    ),

    # Resize and pad image, no random flip (inference mode)
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],  # Fixed resize ratio (no augmentation)
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False
    ),

    # Filter LiDAR points outside defined 3D range
    dict(
        type='PointsRangeFilter',
        point_cloud_range=point_cloud_range
    ),

    # Package inputs into the expected 3D detection format
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

# ==============================================================
# Validation Dataloader (for testing or offline inference)
# ==============================================================

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

# ==============================================================
# Visualizer Settings
# ==============================================================

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)
