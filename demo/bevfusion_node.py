# ============================================================================
#  BEVFusion Real-Time ROS 2 Node for 4-Camera Input + LiDAR
#
#  This script:
#   - Subscribes to 4 camera image topics (CompressedImage)
#   - Subscribes to 1 LiDAR point cloud topic (PointCloud2)
#   - Stacks multi-view images + filters LiDAR points
#   - Runs BEVFusion inference
#   - Displays real-time visualizations (OpenCV)
# ============================================================================

# === [Imports] ===
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, PointCloud2
from cv_bridge import CvBridge
import ros2_numpy
import numpy as np
import cv2
import torch
import os
import sys
import time
import gc
import traceback

# === [Load BEVFusion utilities] ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from mmengine import Config
from mmdet3d.apis import init_model
from projects.BEVFusion.demo.inference_from_array import run_inference_from_array
import projects.BEVFusion.custom_transforms.load_from_array


# ============================================================================
#  BEVFusion ROS 2 Node Class
# ============================================================================

class BEVFusionNode(Node):
    def __init__(self):
        super().__init__('bevfusion_node')

        # === [Runtime setup] ===
        self.bridge = CvBridge()
        self.image_buffer = {}
        self.latest_points = None
        self.last_inference_time = 0.0
        self.inference_interval = 1.0  # [s] Delay between inferences

        # === [Config & Model Paths] ===
        self.config_path = 'projects/BEVFusion/configs/bevfusion_from_array.py'
        self.checkpoint_path = 'checkpoints/bevfusion_latest.pth'
        self.device = 'cuda:0'
        self.point_cloud_range = [-50, -50, -5, 50, 50, 3]
        self.resize_shape = (704, 256)

        # === [Initialize BEVFusion model] ===
        self.cfg = Config.fromfile(self.config_path)
        torch.cuda.empty_cache()
        gc.collect()
        self.model = init_model(self.cfg, self.checkpoint_path, device=self.device)

        # === [Camera subscriptions] ===
        for i in range(1, 5):
            topic = f'/axis_camera_node_{i}/raw_data/compressed'
            self.create_subscription(
                CompressedImage,
                topic,
                lambda msg, cam_id=i - 1: self.image_callback(msg, cam_id),
                10
            )

        # === [LiDAR subscription] ===
        self.create_subscription(PointCloud2, '/rslidar_sdk/rslidar_points', self.lidar_callback, 10)

        self.get_logger().info("✅ BEVFusion real-time node with 4 cameras is ready.")

    # ============================================================================
    #  Image Callback
    # ============================================================================
    def image_callback(self, msg, cam_id):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized = cv2.resize(cv_image, self.resize_shape).astype(np.float32) / 255.0
            self.image_buffer[cam_id] = resized
            # DEBUG: self.get_logger().info(f"Received image cam {cam_id} shape={resized.shape}")
        except Exception as e:
            self.get_logger().error(f"❌ Error converting image from camera {cam_id}: {e}")
        self.try_infer()

    # ============================================================================
    #  LiDAR Callback
    # ============================================================================
    def lidar_callback(self, msg):
        try:
            pc_struct = ros2_numpy.numpify(msg)
            if 'xyz' in pc_struct and 'intensity' in pc_struct:
                xyz = pc_struct['xyz']
                intensity = pc_struct['intensity']
                points = np.hstack((xyz, intensity.reshape(-1, 1))).astype(np.float32)
            else:
                raise ValueError("Unknown point cloud format received.")

            x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
            mask = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
            )
            self.latest_points = points[mask]
            # DEBUG: self.get_logger().info(f"LiDAR points: {self.latest_points.shape}")
        except Exception as e:
            self.get_logger().error(f"❌ Error converting point cloud: {e}")
        self.try_infer()

    # ============================================================================
    #  Run Inference
    # ============================================================================
    def try_infer(self):
        now = time.time()
        if len(self.image_buffer) == 4 and self.latest_points is not None:
            if now - self.last_inference_time < self.inference_interval:
                return

            self.last_inference_time = now
            self.get_logger().info("🚀 Running inference...")

            img_list = [self.image_buffer[i] for i in sorted(self.image_buffer.keys())]
            self.image_buffer.clear()

            try:
                result = run_inference_from_array(
                    image=img_list,
                    points=self.latest_points,
                    model=self.model,
                    device=self.device,
                    use_dummy_calib=False,
                    visualize=False
                )
                self.get_logger().info("✅ Inference completed.")
                self.visualize_bev_feature(getattr(result, 'bev_feat', None))

                # === [Fused Image Grid] ===
                fused_grid_img = getattr(result, 'fused_grid_img', None)
                if fused_grid_img is not None:
                    try:
                        if hasattr(fused_grid_img, 'cpu'):
                            fused_grid_img = fused_grid_img.cpu().numpy()
                        if not isinstance(fused_grid_img, np.ndarray):
                            fused_grid_img = np.array(fused_grid_img)
                        if fused_grid_img.dtype != np.uint8:
                            if fused_grid_img.max() <= 1.0:
                                fused_grid_img = (fused_grid_img * 255).clip(0, 255).astype(np.uint8)
                            else:
                                fused_grid_img = fused_grid_img.clip(0, 255).astype(np.uint8)
                        fused_grid_img = np.ascontiguousarray(fused_grid_img)
                        if fused_grid_img.ndim != 3 or fused_grid_img.shape[2] != 3:
                            raise ValueError(f'Grid image shape must be (H,W,3), got {fused_grid_img.shape}')
                        cv2.imshow("Fused Cameras (All Views)", fused_grid_img)
                    except Exception as e:
                        self.get_logger().warn(f"⚠️ Fused camera grid visualization failed: {e}")
                else:
                    self.get_logger().warn("⚠️ No fused camera grid image found.")

                # === [LiDAR BEV Image] ===
                lidar_bev_img = getattr(result, 'lidar_bev_img', None)
                if lidar_bev_img is not None:
                    try:
                        cv2.imshow("LiDAR BEV (topdown)", lidar_bev_img)
                    except Exception as e:
                        self.get_logger().warn(f"⚠️ LiDAR BEV visualization failed: {e}")
                else:
                    self.get_logger().warn("⚠️ No LiDAR BEV image found.")

                # === [Simple BEV LiDAR Point Projection] ===
                try:
                    pts = self.latest_points
                    if pts is not None and pts.shape[0] > 0:
                        x_range = (-50, 50)
                        y_range = (-50, 50)
                        img = np.zeros((512, 512, 3), dtype=np.uint8)
                        x_img = ((pts[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * img.shape[1]).astype(np.int32)
                        y_img = ((pts[:, 1] - y_range[0]) / (y_range[1] - y_range[0]) * img.shape[0]).astype(np.int32)
                        mask = (x_img >= 0) & (x_img < img.shape[1]) & (y_img >= 0) & (y_img < img.shape[0])
                        img[y_img[mask], x_img[mask]] = (255, 255, 255)
                        cv2.imshow("LiDAR Points (BEV)", img)
                except Exception as e:
                    self.get_logger().warn(f"⚠️ LiDAR points visualization failed: {e}")

                cv2.waitKey(1)

            except Exception as e:
                self.get_logger().error(f"❌ Inference failed: {e}")
                traceback.print_exc()

            try:
                torch.cuda.empty_cache()
                self.get_logger().info("🧹 CUDA cache cleared.")
            except Exception as e:
                self.get_logger().warn(f"⚠️ Could not clear CUDA cache: {e}")

    # ============================================================================
    # Visualize BEV Feature Map
    # ============================================================================
    def visualize_bev_feature(self, bev_feat_tensor):
        if bev_feat_tensor is None:
            self.get_logger().warn("⚠️ BEV feature tensor is None.")
            return
        try:
            bev_img = bev_feat_tensor.mean(dim=0).cpu().numpy()
            bev_img = (255 * (bev_img - bev_img.min()) / (bev_img.max() - bev_img.min() + 1e-6)).astype('uint8')
            bev_img = cv2.resize(bev_img, (700, 700))
            cv2.imshow("BEV Fusion Map", bev_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"⚠️ BEV visualization failed: {e}")


# ============================================================================
# Main ROS 2 Loop
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = BEVFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Shutting down BEVFusion node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
