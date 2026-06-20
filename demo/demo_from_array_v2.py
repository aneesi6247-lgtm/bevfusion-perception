# ============================================================================
#  demo_from_array_v2.py
#
#  Launcher script to run BEVFusion inference using image + LiDAR arrays.
#  Used for testing arrays extracted from ROS 2 or rosbags.
#  Initializes model, loads visualizer, runs inference, and displays results.
# ============================================================================

import os
import sys
import torch
import pickle

# === [System path setup for custom imports] ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import projects.BEVFusion.custom_transforms.load_from_array  # Registers custom transforms

# === [MMDetection3D imports] ===
from mmengine import Config
from mmdet3d.apis import init_model
from mmdet3d.registry import VISUALIZERS

# === [Import BEVFusion inference function] ===
from projects.BEVFusion.demo.inference_from_array import run_inference_from_array

# ============================================================================
#  Configuration Paths
# ============================================================================

config_path = 'projects/BEVFusion/configs/bevfusion_from_array.py'
checkpoint_path = 'checkpoints/bevfusion_latest.pth'
pkl_file = 'rosbag_sample_reduced1.pkl'
device = 'cuda:0'

# ============================================================================
#  Run Inference from Arrays
# ============================================================================

result = run_inference_from_array(config_path, checkpoint_path, pkl_file, device)
print("✅ Inference completed.")

# ============================================================================
#  Initialize Visualizer
# ============================================================================

cfg = Config.fromfile(config_path)
model = init_model(cfg, checkpoint_path, device=device)

# Update save directory for visualizer (if enabled)
if cfg.visualizer.get('vis_backends', None) and len(cfg.visualizer.vis_backends) > 0:
    cfg.visualizer.vis_backends[0].save_dir = os.path.abspath('./vis_output')

visualizer = VISUALIZERS.build(cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

# ============================================================================
#  Load Input Data Again (optional, for display only)
# ============================================================================

with open(pkl_file, 'rb') as f:
    sample = pickle.load(f)

img = sample['img']
points = sample['points']

# ============================================================================
#  Normalize Image Shape (Expected: [3, H, W])
# ============================================================================

if isinstance(img, torch.Tensor):
    if img.dim() == 5:
        img = img.squeeze(0).squeeze(0)
    elif img.dim() == 4:
        img = img.squeeze(0)
    elif img.dim() != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}")

# ============================================================================
#  Prepare Data Input for Visualization
# ============================================================================

data_input = dict(points=points, img=img)

# ============================================================================
#  Show Visualization (OpenCV Window)
# ============================================================================

visualizer.add_datasample(
    name='bevfusion_result',
    data_input=data_input,
    data_sample=result,
    show=True,
    wait_time=0,
    draw_gt=False,
    pred_score_thr=0.3,
    vis_task='multi-modality_det'
)
