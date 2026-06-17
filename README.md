# BEVFusion Perception

Multi-modal 3D object detection fusing LiDAR point clouds and RGB camera data in Bird's Eye View (BEV) space, with deployment on NVIDIA Jetson for real-time inference.

> Developed during a Student Research Assistant position at CARISSMA C-IAD, Technische Hochschule Ingolstadt (THI). This repository contains a cleaned-up/educational version of the work — see notes below.

## Overview

- **Architecture:** BEVFusion — projects LiDAR point clouds and camera features into a shared Bird's Eye View representation for joint 3D object detection
- **Sensor pipeline:** intrinsic/extrinsic calibration, time synchronization, and preprocessing across LiDAR + camera streams
- **Deployment:** optimized and deployed on NVIDIA Jetson, achieving real-time inference at 30 FPS
- **Validation:** evaluated against ground-truth annotations on multi-sensor datasets

## Stack

PyTorch · ROS/ROS2 · NVIDIA Jetson · LiDAR/Camera sensor fusion
