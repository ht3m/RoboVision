"""
d405_camera.py — Intel RealSense D405 相机拍照模块
===================================================
提供 capture_photos() 函数：
  - 自动扫描已有编号，继续往后递增
  - Temporal Filter 预热 + 多帧中值滤波
  - 保存彩色图 (.jpg) 和深度图 (.png) 到指定目录
"""

import os
import numpy as np
import cv2
import pyrealsense2 as rs


def _find_next_index(save_folder: str) -> int:
    """扫描照片文件夹，找到下一个可用编号（允许已有文件不连续）。"""
    if not os.path.exists(save_folder):
        return 0

    max_idx = -1
    for f in os.listdir(save_folder):
        if f.startswith("d405_color_") and f.endswith(".jpg"):
            try:
                num_str = f.replace("d405_color_", "").replace(".jpg", "")
                idx = int(num_str)
                max_idx = max(max_idx, idx)
            except ValueError:
                continue
    return max_idx + 1


def capture_photos(save_folder: str = "photos",
                   warmup_frames: int = 30,
                   collect_frames: int = 10) -> int:
    """使用 D405 相机拍照（Temporal + 中值滤波）。

    Args:
        save_folder:     照片保存目录
        warmup_frames:   Temporal Filter 预热帧数
        collect_frames:  中值滤波收集帧数

    Returns:
        本次拍照的编号 (int)
    """
    os.makedirs(save_folder, exist_ok=True)
    img_idx = _find_next_index(save_folder)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)

    temporal_filter = rs.temporal_filter()

    print(f"\n{'=' * 60}")
    print(f"  D405 拍照中: 预热 {warmup_frames} 帧 → 收集 {collect_frames} 帧")
    print(f"  编号: {img_idx:04d}")
    print(f"{'=' * 60}")

    depth_buffer: list[np.ndarray] = []
    color_to_save: np.ndarray | None = None

    try:
        # 预热阶段
        for i in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            temporal_filter.process(depth_frame)
            if i % 10 == 0:
                print(f"  预热... {i + 1}/{warmup_frames}")

        print(f"  预热完成，开始收集 {collect_frames} 帧...")

        # 收集阶段
        for i in range(collect_frames):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            filtered_depth = temporal_filter.process(depth_frame)
            depth_image = np.asanyarray(filtered_depth.get_data())
            depth_buffer.append(depth_image.copy())

            if color_to_save is None:
                color_to_save = np.asanyarray(color_frame.get_data()).copy()

            if i % 3 == 0:
                print(f"  收集... {i + 1}/{collect_frames}")

        # 中值滤波
        depth_stack = np.stack(depth_buffer, axis=0)
        depth_median = np.median(depth_stack, axis=0).astype(np.uint16)

        # 保存
        color_path = os.path.join(save_folder, f"d405_color_{img_idx:04d}.jpg")
        depth_path = os.path.join(save_folder, f"d405_depth_{img_idx:04d}.png")
        cv2.imwrite(color_path, color_to_save)
        cv2.imwrite(depth_path, depth_median)

        print(f"  ✓ 已保存: {color_path}")
        print(f"  ✓ 已保存: {depth_path}")
        print(f"  照片编号: {img_idx:04d}\n")

    finally:
        pipeline.stop()

    return img_idx
