# ============================================================================
#  ros_listener.py
#
#  ROS 2 node to extract a single camera image and LiDAR point cloud from a rosbag or live topics.
#  The extracted data is resized and filtered to match BEVFusion input requirements,
#  then saved as a `.pkl` file containing the keys: 'img', 'points'.
# ============================================================================

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, PointCloud2
from cv_bridge import CvBridge
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import pickle
import os


class RosbagExtractor(Node):
    def __init__(self):
        super().__init__('rosbag_extractor')
        self.bridge = CvBridge()
        self.image = None
        self.points = None

        # Subscribe to camera and LiDAR topics
        self.create_subscription(
            CompressedImage,
            '/axis_camera_node_1/raw_data/compressed',
            self.image_callback,
            10
        )
        self.create_subscription(
            PointCloud2,
            '/rslidar_sdk/rslidar_points',
            self.lidar_callback,
            10
        )

    def image_callback(self, msg):
        """Receive and store a single camera image."""
        if self.image is None:
            self.image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info('Image received.')

    def lidar_callback(self, msg):
        """Receive and store a single point cloud."""
        if self.points is None:
            cloud_points = list(pc2.read_points(msg, skip_nans=True))
            self.points = np.array([[p[0], p[1], p[2], p[3]] for p in cloud_points], dtype=np.float32)
            self.get_logger().info('Point cloud received.')


def main():
    rclpy.init()
    node = RosbagExtractor()

    print("Waiting for image and point cloud...")

    timeout = 10.0  # seconds
    start = node.get_clock().now().nanoseconds / 1e9

    # Wait until both image and point cloud are received or timeout is reached
    while (node.image is None or node.points is None) and \
          (node.get_clock().now().nanoseconds / 1e9 - start < timeout):
        rclpy.spin_once(node, timeout_sec=0.1)

    os.makedirs("rosbag_output", exist_ok=True)

    if node.image is not None and node.points is not None:
        # Resize image to the expected BEVFusion shape (704x256)
        resized_img = cv2.resize(node.image, (704, 256))

        # Filter points within BEVFusion range
        mask = (
            (node.points[:, 0] >= -50) & (node.points[:, 0] <= 50) &
            (node.points[:, 1] >= -50) & (node.points[:, 1] <= 50) &
            (node.points[:, 2] >= -5)  & (node.points[:, 2] <= 3)
        )
        filtered_points = node.points[mask]

        # Save processed sample
        sample = {
            'img': resized_img,
            'points': filtered_points
        }

        output_path = 'rosbag_output/rosbag_sample_reduced.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(sample, f)

        print(f"Saved reduced sample to: {output_path}")
        print(f"Image shape: {resized_img.shape}")
        print(f"Filtered point count: {filtered_points.shape}")
    else:
        print("Timeout: Image or point cloud not received.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
