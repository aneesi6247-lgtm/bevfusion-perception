# ============================================================================
#  Custom Transforms: Load LiDAR Points and Multi-View Images from NumPy Arrays
#
#  This module provides custom MMDetection3D transforms for loading LiDAR point
#  clouds and multi-view camera images directly from in-memory NumPy arrays.
#  Useful for inference pipelines where data is not read from disk.
#  It includes:
#  - LoadPointsFromArray: Loads LiDAR points from a numpy array and wraps them
#    as LiDARPoints.
#  - LoadMultiViewImageFromArray: Loads multi-view images from numpy arrays,
#    processes them, and prepares them for the pipeline.    
# ============================================================================

import numpy as np
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import LiDARPoints 

@TRANSFORMS.register_module()
class LoadPointsFromArray:
    """Load LiDAR points from a numpy array and wrap as LiDARPoints."""
    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=4):
        # Store configuration for coordinate type and dimensions
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim

    def __call__(self, results):
        # Retrieve points from results dict
        points = results.get('points', None)
        if points is None or not isinstance(points, np.ndarray):
            raise ValueError("No points found or not a numpy array")

        # Adjust dimensions: pad or truncate as needed
        if points.shape[1] < self.load_dim:
            pad_width = self.load_dim - points.shape[1]
            points = np.pad(points, ((0, 0), (0, pad_width)), mode='constant')
        elif points.shape[1] > self.load_dim:
            points = points[:, :self.load_dim]

        # print("Loaded points shape before wrapping:", points.shape)  # Debug print

        # Wrap as LiDARPoints object
        lidar_points = LiDARPoints(points, points_dim=self.load_dim)

        # Assign expected fields in results dict
        results['points'] = lidar_points
        results['lidar_points'] = {'lidar_path': None, 'points': lidar_points}
        results['pts_filename'] = None
        # print(f"✅ lidar_points keys after transform: {results['lidar_points'].keys()}")  # Debug print

        return results


@TRANSFORMS.register_module()
class LoadMultiViewImageFromArray:
    """Load multi-view images from numpy arrays and process for pipeline."""
    def __init__(self, to_float32=True, color_type='color', limit_views=None):
        # Store configuration for image conversion and view limiting
        self.to_float32 = to_float32
        self.color_type = color_type
        self.limit_views = limit_views

    def __call__(self, results):
        # Retrieve images from results dict
        imgs = results.get('img', None)
        if imgs is None:
            raise ValueError("No images found in results")

        # Normalize input to a list of images
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise TypeError(f"Expected list or ndarray, got {type(imgs)}")

        # Optionally limit number of views
        if self.limit_views is not None and len(imgs) > self.limit_views:
            # print(f"⚠️ Limiting views to {self.limit_views} (from {len(imgs)} views)")  # Debug print
            imgs = imgs[:self.limit_views]

        processed_imgs = []
        for i, img in enumerate(imgs):
            # Convert to float32 if required
            if self.to_float32:
                img = img.astype(np.float32)
            # Check image shape
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected image with shape (H, W, 3), got {img.shape}")
            processed_imgs.append(img)  # Keep as numpy array

        # Store processed images and shapes in results dict
        results['img'] = processed_imgs
        results['filename'] = None

        h, w = processed_imgs[0].shape[0:2]
        results['ori_shape'] = (h, w)
        results['img_shape'] = (h, w)

        return results
