# ============================================================================
#  BEVFusion Inference Utility: Run Inference from NumPy Arrays
#
#  This script provides a function to run BEVFusion inference using multi-view
#  images and LiDAR point clouds provided as in-memory NumPy arrays. It handles
#  model initialization, data pipeline composition, calibration, and visualization.
# ============================================================================

# -------------------------------
# Imports and Calibration Matrices
# -------------------------------
import os
import sys
import torch
import numpy as np
import traceback

from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import VISUALIZERS
from mmdet3d.apis import init_model
from mmdet3d.structures import Box3DMode, LiDARInstance3DBoxes, Det3DDataSample

# Register custom transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import projects.BEVFusion.custom_transforms.load_from_array

# Fixed calibration matrices for camera and LiDAR (example values)
fixed_cam2img = np.array([
    [700.048429, 0.0, 952.491618],
    [0.0, 700.048429, 552.083765],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

fixed_lidar2cam = np.array([
    [0.999947, 0.010241, 0.002329, 0.27],
    [-0.010306, 0.999748, 0.019912, 0.0],
    [-0.002125, -0.019935, 0.999797, 1.75],
    [0.0, 0.0, 0.0, 1.0]
], dtype=np.float32)

fixed_cam2lidar = np.linalg.inv(fixed_lidar2cam)
fixed_lidar2img = fixed_cam2img @ fixed_lidar2cam[:3, :]

# -------------------------------
# Utility: Print GPU Memory Usage
# -------------------------------
def print_mem(stage=""):
    """Utility to print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[📜 {stage}] GPU Memory - Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

# -------------------------------
# Main Inference Function
# -------------------------------
def run_inference_from_array(
    config_path=None,
    checkpoint_path=None,
    image=None,
    points=None,
    device='cuda:0',
    model=None,
    use_dummy_calib=False,
    visualize=True
):
    # -------------------------------
    # Input Validation and Model Setup
    # -------------------------------
    # print(f"[DEBUG] image type: {type(image)}, len: {len(image) if isinstance(image, (list, tuple)) else 'N/A'}")
    if isinstance(image, (list, tuple)):
        for idx, img in enumerate(image):
            # print(f"[DEBUG] Input image {idx}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
            pass
    # Load model if not provided
    if model is None:
        assert config_path and checkpoint_path, "Must provide config_path and checkpoint_path if model is None"
        cfg = Config.fromfile(config_path)
        model = init_model(cfg, checkpoint_path, device=device)
    else:
        cfg = model.cfg

    model.eval()
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    print_mem("📦 After Compose pipeline")

    # --- Print pipeline steps for debugging ---
    # print("[DEBUG] Pipeline steps:")
    # for t in cfg.test_dataloader.dataset.pipeline:
    #     print(t)

    # -------------------------------
    # Data Pipeline Preparation
    # -------------------------------
    # --- Prepare image and points input ---
    if not isinstance(image, (list, tuple)):
        image = [image]

    assert len(image) > 0, "No images provided"
    assert points is not None, "No LiDAR points provided"

    # Print input image and LiDAR info
    # for i, img in enumerate(image):
    #     print(f"[INFO] Image {i}: shape {img.shape}, dtype {img.dtype}, min {img.min():.4f}, max {img.max():.4f}")
    # print(f"[INFO] LiDAR points shape: {points.shape}, dtype: {points.dtype}")

    # --- Ensure images are uint8 in [0,255] for PIL transforms ---
    for idx in range(len(image)):
        img = image[idx]
        if (img.dtype == np.float32 or img.dtype == np.float64) and img.max() <= 1.0 and img.min() >= 0.0:
            image[idx] = (img * 255).clip(0, 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            image[idx] = img.clip(0, 255).astype(np.uint8)

    # -------------------------------
    # Calibration Matrix Preparation
    # -------------------------------
    # --- Prepare calibration matrices for each view ---
    num_views = len(image)
    if use_dummy_calib:
        cam2img = [np.eye(3)] * num_views
        lidar2cam = [np.eye(4)] * num_views
        cam2lidar = [np.eye(4)] * num_views
        lidar2img = [np.eye(4)] * num_views
    else:
        cam2img = [fixed_cam2img] * num_views
        lidar2cam = [fixed_lidar2cam] * num_views
        cam2lidar = [fixed_cam2lidar] * num_views
        lidar2img = [fixed_lidar2img] * num_views

    # -------------------------------
    # Pipeline Execution (Transforms)
    # -------------------------------
    # --- Prepare input dictionary for the pipeline ---
    input_dict = dict(
        img=image,
        points=points,
        box_type_3d=LiDARInstance3DBoxes,
        box_mode_3d=Box3DMode.LIDAR,
        lidar2img=lidar2img,
        cam2img=cam2img,
        lidar2cam=lidar2cam,
        cam2lidar=cam2lidar,
    )

    # --- Run data pipeline (preprocessing, transforms, etc.) ---
    data = pipeline(input_dict)
    # Print pipeline output image(s) for debugging
    # imgs_after = data['inputs']['img']
    # print(f"[DEBUG] imgs_after type: {type(imgs_after)}, shape: {imgs_after.shape if hasattr(imgs_after, 'shape') else 'N/A'}")
    # if isinstance(imgs_after, (list, tuple)):
    #     for idx, img in enumerate(imgs_after):
    #         print(f"[DEBUG] Pipeline output img {idx}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    # img_after = data['inputs']['img']
    # if isinstance(img_after, torch.Tensor):
    #     img_after_np = img_after.cpu().numpy()
    # else:
    #     img_after_np = np.array(img_after)
    # print(f"[DEBUG] img after pipeline: dtype={img_after_np.dtype}, min={img_after_np.min()}, max={img_after_np.max()}, shape={img_after_np.shape}")
    print_mem("🧪 After applying pipeline")

    # -------------------------------
    # Batch Collation and Dimension Fixes
    # -------------------------------
    # --- Collate data for model input (batch dimension) ---
    data = pseudo_collate([data])
    print_mem("📦 After pseudo_collate")

    # --- Remove batch dimension if present ---
    # if isinstance(data['inputs']['img'], torch.Tensor) and data['inputs']['img'].ndim == 5:
    #     print(f"[DEBUG] Removing batch dimension from data['inputs']['img']: shape before={data['inputs']['img'].shape}")
    #     data['inputs']['img'] = data['inputs']['img'][0]
    #     print(f"[DEBUG] Shape after batch removal: {data['inputs']['img'].shape}")
    # if isinstance(data['inputs']['points'], torch.Tensor) and data['inputs']['points'].ndim == 3:
    #     data['inputs']['points'] = data['inputs']['points'][0]

    # -------------------------------
    # Model Inference
    # -------------------------------
    with torch.no_grad():
        print_mem("🚀 Before model.test_step")
        result = model.test_step(data)[0]
        print_mem("✅ After model.test_step")

    # -------------------------------
    # Post-processing and Visualization
    # -------------------------------
    pred_instances_3d = result.pred_instances_3d
    if hasattr(pred_instances_3d, 'scores_3d') and hasattr(pred_instances_3d, 'labels_3d'):
        if torch.is_tensor(pred_instances_3d.scores_3d):
            scores = pred_instances_3d.scores_3d.cpu().numpy()
        else:
            scores = np.array(pred_instances_3d.scores_3d)

        if torch.is_tensor(pred_instances_3d.labels_3d):
            labels = pred_instances_3d.labels_3d.cpu().numpy()
        else:
            labels = np.array(pred_instances_3d.labels_3d)

        conf_thresh = 0.05  # Lowered threshold for debug/visualization
        mask = scores > conf_thresh
        filtered_boxes = pred_instances_3d.bboxes_3d[mask]
        filtered_scores = scores[mask]
        filtered_labels = labels[mask]
        print(f"[DEBUG] Number of filtered boxes: {len(filtered_boxes)} (conf_thresh={conf_thresh})")

        # Prepare data_input for image and lidar
        data_input = dict(
            points=data['inputs']['points'],
            img=data['inputs']['img']
        )
        # --- PATCH: Do not permute batch tensor here, handle in per-image loop below ---
        if isinstance(data_input['points'], torch.Tensor):
            data_input['points'] = data_input['points'].cpu()

        # --- PATCH: Ensure filtered_boxes is a box object, not a tensor ---
        if isinstance(filtered_boxes, torch.Tensor):
            # print(f"[DEBUG] Wrapping filtered_boxes tensor as LiDARInstance3DBoxes: shape={filtered_boxes.shape}")
            filtered_boxes = LiDARInstance3DBoxes(filtered_boxes)
        elif isinstance(filtered_boxes, np.ndarray):
            # print(f"[DEBUG] Wrapping filtered_boxes ndarray as LiDARInstance3DBoxes: shape={filtered_boxes.shape}")
            filtered_boxes = LiDARInstance3DBoxes(torch.from_numpy(filtered_boxes))

        # --- Fused camera images with projected 3D boxes ---
        imgs = data_input['img']
        # print(f"[DEBUG] data_input['img'] type: {type(imgs)}, shape: {getattr(imgs, 'shape', None)}")
        # Robustly handle all cases: tensor, list of tensors, or list of length 1 containing a tensor batch
        if isinstance(imgs, torch.Tensor) and imgs.ndim == 4:
            print(f"[DEBUG] imgs tensor shape: {imgs.shape}")
            if imgs.shape[1] == 3:
                imgs_list = [imgs[i].permute(1, 2, 0).cpu().numpy() for i in range(imgs.shape[0])]
            else:
                imgs_list = [imgs[i].cpu().numpy() for i in range(imgs.shape[0])]
            print(f"[DEBUG] imgs_list length after split: {len(imgs_list)}")
        elif isinstance(imgs, (list, tuple)):
            print(f"[DEBUG] imgs is list/tuple, len={len(imgs)}")
            # If list of length 1 and the only element is a tensor batch, flatten it
            if len(imgs) == 1 and isinstance(imgs[0], torch.Tensor) and imgs[0].ndim == 4:
                batch = imgs[0]
                print(f"[DEBUG] imgs[0] is tensor batch, shape: {batch.shape}")
                if batch.shape[1] == 3:
                    imgs_list = [batch[i].permute(1, 2, 0).cpu().numpy() for i in range(batch.shape[0])]
                else:
                    imgs_list = [batch[i].cpu().numpy() for i in range(batch.shape[0])]
                print(f"[DEBUG] imgs_list length after flatten: {len(imgs_list)}")
            else:
                imgs_list = list(imgs)
        else:
            print(f"[DEBUG] imgs is neither tensor nor list/tuple, using as single image")
            imgs_list = [imgs]
        fused_images = []
        IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
        vis_imgs = []
        for i, img in enumerate(imgs_list):
            img_vis = img.copy() if isinstance(img, np.ndarray) else img.cpu().numpy().copy()
            if img_vis.ndim == 4:
                img_vis = img_vis[0]
            if img_vis.ndim == 3 and img_vis.shape[0] == 3:
                img_vis = np.transpose(img_vis, (1, 2, 0))
            # print(f"[DEBUG] img_vis raw {i}: dtype={img_vis.dtype}, min={img_vis.min()}, max={img_vis.max()}, shape={img_vis.shape}")
            # Add a unique sum debug to check if images are different
            # print(f"[DEBUG] img_vis[{i}] unique sum: {np.sum(np.unique(img_vis))}, shape: {img_vis.shape}")
            if img_vis.dtype == np.float32 or img_vis.dtype == np.float64:
                if img_vis.max() < 10.0 and img_vis.min() < 0.0:
                    img_vis = (img_vis * IMAGENET_STD[None, None, :]) + IMAGENET_MEAN[None, None, :]
                elif img_vis.max() <= 1.0 and img_vis.min() >= 0.0:
                    img_vis = img_vis * 255.0
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
            # print(f"[DEBUG] Fused image {i} before boxes: dtype={img_vis.dtype}, min={img_vis.min()}, max={img_vis.max()}, shape={img_vis.shape}")
            lidar2cam_mat = lidar2cam[i] if isinstance(lidar2cam, (list, tuple)) else lidar2cam
            cam2img_mat = cam2img[i] if isinstance(cam2img, (list, tuple)) else cam2img
            box_count = 0
            for box in filtered_boxes:
                # print(f"[DEBUG] Box {box_count}: type={type(box)}, has corners={hasattr(box, 'corners')}")
                corners_3d = None
                try:
                    if hasattr(box, 'corners'):
                        corners_3d = box.corners().cpu().numpy()
                        # print(f"[DEBUG] Box {box_count} corners_3d: {corners_3d}")
                    else:
                        # print(f"[DEBUG] Box {box_count} has no 'corners' method!")
                        pass
                except Exception as e:
                    print(f"[DEBUG] Exception calling box.corners(): {e}")
                if corners_3d is not None:
                    corners_hom = np.concatenate([corners_3d, np.ones((8,1))], axis=1).T
                    corners_cam = lidar2cam_mat @ corners_hom
                    # print(f"[DEBUG] Box {box_count} corners_cam[2,:] (Z): {corners_cam[2, :]}")
                    # print(f"[DEBUG] Box {box_count} corners_cam (all): {corners_cam}")
                    mask_in_front = corners_cam[2, :] > 0.1
                    if not np.any(mask_in_front):
                        # print(f"[DEBUG] Box {box_count} skipped: all corners behind camera.")
                        box_count += 1
                        continue
                    corners_img = cam2img_mat @ corners_cam[:3, :]
                    corners_img = corners_img / corners_img[2:3, :]
                    # print(f"[DEBUG] Box {box_count} projected corners_img: min=({corners_img[0,:].min()}, {corners_img[1,:].min()}), max=({corners_img[0,:].max()}, {corners_img[1,:].max()})")
                    # print(f"[DEBUG] Projected corners_img for box {box_count}: {corners_img}")
                    # Print all projected 2D points for this box
                    # for idx_pt in range(corners_img.shape[1]):
                    #     print(f"[DEBUG] Box {box_count} corner {idx_pt}: ({corners_img[0, idx_pt]}, {corners_img[1, idx_pt]})")
                    edges = [
                        [0,1],[1,2],[2,3],[3,0],
                        [4,5],[5,6],[6,7],[7,4],
                        [0,4],[1,5],[2,6],[3,7]
                    ]
                    edge_drawn = False
                    for e in edges:
                        x0, y0 = int(corners_img[0, e[0]]), int(corners_img[1, e[0]])
                        x1, y1 = int(corners_img[0, e[1]]), int(corners_img[1, e[1]])
                        # print(f"[DEBUG] Box {box_count} edge {e}: ({x0},{y0}) -> ({x1},{y1})")
                        if 0 <= x0 < img_vis.shape[1] and 0 <= y0 < img_vis.shape[0] and 0 <= x1 < img_vis.shape[1] and 0 <= y1 < img_vis.shape[0]:
                            try:
                                import cv2
                                cv2.line(img_vis, (x0, y0), (x1, y1), (0,255,0), 2)
                                edge_drawn = True
                            except Exception:
                                pass
                    if not edge_drawn:
                        # print(f"[DEBUG] Box {box_count} projected, but all edges out of image bounds.")
                        pass
                box_count += 1
            # print(f"[DEBUG] Fused image {i} after boxes: dtype={img_vis.dtype}, min={img_vis.min()}, max={img_vis.max()}, shape={img_vis.shape}, boxes drawn: {box_count}")
            fused_images.append(img_vis)
            vis_imgs.append(img_vis)
        # After loop, print summary of vis_imgs
        print(f"[DEBUG] vis_imgs count: {len(vis_imgs)}")
        for idx, vimg in enumerate(vis_imgs):
            print(f"[DEBUG] vis_imgs[{idx}] unique sum: {np.sum(np.unique(vimg))}, shape: {vimg.shape}")
        if len(vis_imgs) == 1:
            print("[WARN] Only one image in vis_imgs! Check pipeline and input batch handling.")
        # Arrange all camera images in a grid (2x2 if 4 cameras, 1xN if N cameras)
        grid_img = None
        try:
            import cv2
            n = len(vis_imgs)
            if n == 1:
                grid_img = vis_imgs[0]
            elif n == 2:
                grid_img = np.hstack(vis_imgs)
            elif n == 3:
                h1 = np.hstack(vis_imgs[:2])
                h2 = vis_imgs[2]
                pad = np.zeros_like(h2)
                h2 = np.hstack([h2, pad])
                grid_img = np.vstack([h1, h2])
            elif n == 4:
                h1 = np.hstack(vis_imgs[:2])
                h2 = np.hstack(vis_imgs[2:])
                grid_img = np.vstack([h1, h2])
            else:
                # For >4, stack horizontally
                grid_img = np.hstack(vis_imgs)
            print(f"[DEBUG] Fused camera grid image created: shape={grid_img.shape}, dtype={grid_img.dtype}")
            cv2.imshow("Fused Cameras (All Views)", grid_img)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[WARN] Could not display fused camera grid: {e}")
        # Attach the grid image to the result for downstream use
        result.fused_grid_img = grid_img

        # --- LiDAR BEV image with boxes (simple topdown view) ---
        pts = data_input['points']
        if isinstance(pts, (list, tuple)):
            if all(isinstance(p, (np.ndarray, torch.Tensor)) for p in pts):
                if isinstance(pts[0], torch.Tensor):
                    pts = torch.cat([p.cpu() if isinstance(p, torch.Tensor) else torch.tensor(p) for p in pts], dim=0).numpy()
                else:
                    pts = np.concatenate([p if isinstance(p, np.ndarray) else np.array(p) for p in pts], axis=0)
            else:
                pts = np.array(pts)
        elif isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        # Subsample for speed if too many points
        max_pts = 10000
        if pts.shape[0] > max_pts:
            idx = np.random.choice(pts.shape[0], max_pts, replace=False)
            pts_vis = pts[idx]
        else:
            pts_vis = pts
        # Create a simple BEV image
        try:
            import cv2
            bev_img = np.zeros((512, 512, 3), dtype=np.uint8)
            # Map x/y to image coordinates
            x_range = (-50, 50)
            y_range = (-50, 50)
            x_img = ((pts_vis[:,0] - x_range[0]) / (x_range[1] - x_range[0]) * bev_img.shape[1]).astype(np.int32)
            y_img = ((pts_vis[:,1] - y_range[0]) / (y_range[1] - y_range[0]) * bev_img.shape[0]).astype(np.int32)
            mask = (x_img >= 0) & (x_img < bev_img.shape[1]) & (y_img >= 0) & (y_img < bev_img.shape[0])
            bev_img[y_img[mask], x_img[mask]] = (255,255,255)
            # Draw 3D boxes in BEV
            for box in filtered_boxes:
                corners = box.corners().cpu().numpy() if hasattr(box, 'corners') else None
                if corners is not None:
                    # Only use x/y for BEV
                    poly = np.stack([
                        corners[[0,1,2,3,0],0],
                        corners[[0,1,2,3,0],1]
                    ], axis=1)
                    poly_img = np.zeros_like(poly)
                    poly_img[:,0] = ((poly[:,0] - x_range[0]) / (x_range[1] - x_range[0]) * bev_img.shape[1]).astype(np.int32)
                    poly_img[:,1] = ((poly[:,1] - y_range[0]) / (y_range[1] - y_range[0]) * bev_img.shape[0]).astype(np.int32)
                    cv2.polylines(bev_img, [poly_img], isClosed=True, color=(0,255,0), thickness=2)
        except Exception:
            bev_img = None

        # --- BEV feature map (if available) ---
        bev_feat = None
        bev_feat_img = None
        if hasattr(result, 'feature_map'):
            fmap = result.feature_map
            if isinstance(fmap, torch.Tensor):
                fmap = fmap.detach().cpu().numpy()
            # fmap shape: (B, C, H, W) or (C, H, W)
            if fmap.ndim == 4:
                bev_feat = fmap[0]  # Take first batch
            elif fmap.ndim == 3:
                bev_feat = fmap
            else:
                bev_feat = fmap
            # Visualize BEV feature map using PCA or channel mean
            try:
                import cv2
                from sklearn.decomposition import PCA
                feat = bev_feat
                # feat: (C, H, W)
                if feat.shape[0] > 3:
                    # Flatten spatial dims
                    C, H, W = feat.shape
                    feat_flat = feat.reshape(C, -1).T  # (H*W, C)
                    pca = PCA(n_components=3)
                    feat_pca = pca.fit_transform(feat_flat)
                    feat_img = feat_pca.reshape(H, W, 3)
                    # Normalize to [0,255]
                    feat_img = feat_img - feat_img.min()
                    if feat_img.max() > 0:
                        feat_img = feat_img / feat_img.max()
                    feat_img = (feat_img * 255).astype(np.uint8)
                else:
                    # If C <= 3, just stack or repeat
                    feat_img = np.transpose(feat, (1, 2, 0))
                    if feat_img.shape[2] == 1:
                        feat_img = np.repeat(feat_img, 3, axis=2)
                    elif feat_img.shape[2] == 2:
                        feat_img = np.concatenate([feat_img, np.zeros((feat_img.shape[0], feat_img.shape[1], 1), dtype=feat_img.dtype)], axis=2)
                    # Normalize
                    feat_img = feat_img - feat_img.min()
                    if feat_img.max() > 0:
                        feat_img = feat_img / feat_img.max()
                    feat_img = (feat_img * 255).astype(np.uint8)
                bev_feat_img = feat_img
                print(f"[DEBUG] BEV feature map visualized: dtype={bev_feat_img.dtype}, min={bev_feat_img.min()}, max={bev_feat_img.max()}, shape={bev_feat_img.shape}")
            except Exception as e:
                print(f"[WARN] Could not visualize BEV feature map: {e}")
                bev_feat_img = None

        # Attach outputs to result for downstream use
        result.fused_images = fused_images
        result.lidar_bev_img = bev_img
        result.bev_feat = bev_feat
        result.bev_feat_img = bev_feat_img
    else:
        # If no predictions, attach empty outputs
        result.fused_images = []
        result.lidar_bev_img = None
        result.bev_feat = None
    return result