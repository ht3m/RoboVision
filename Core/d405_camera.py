"""
d405_camera.py - Intel RealSense D405 capture module.

Provides capture_photos():
  - scans existing image IDs and increments automatically
  - optionally aligns depth frames to color frames
  - applies RealSense temporal filtering and multi-frame median filtering
  - saves color JPG and depth PNG to the requested folder
"""

import os

import cv2
import numpy as np
import pyrealsense2 as rs

try:
    from config import D405_ALIGN_DEPTH_TO_COLOR
except Exception:
    D405_ALIGN_DEPTH_TO_COLOR = True


def _find_next_index(save_folder: str) -> int:
    """Return the next available d405_color_XXXX.jpg index."""
    if not os.path.exists(save_folder):
        return 0

    max_idx = -1
    for filename in os.listdir(save_folder):
        if not filename.startswith("d405_color_") or not filename.endswith(".jpg"):
            continue
        try:
            num_str = filename.replace("d405_color_", "").replace(".jpg", "")
            max_idx = max(max_idx, int(num_str))
        except ValueError:
            continue
    return max_idx + 1


def capture_photos(save_folder: str = "photos",
                   warmup_frames: int = 30,
                   collect_frames: int = 10) -> int:
    """Capture one D405 color/depth pair and return its numeric image ID."""
    os.makedirs(save_folder, exist_ok=True)
    img_idx = _find_next_index(save_folder)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(rs_config)

    align = rs.align(rs.stream.color) if D405_ALIGN_DEPTH_TO_COLOR else None
    temporal_filter = rs.temporal_filter()

    align_text = "aligned to color" if align is not None else "native depth"
    print(f"\n{'=' * 60}")
    print(f"  D405 capture: warmup {warmup_frames} frames -> collect {collect_frames} frames")
    print(f"  depth mode: {align_text}")
    print(f"  image id: {img_idx:04d}")
    print(f"{'=' * 60}")

    depth_buffer: list[np.ndarray] = []
    color_to_save: np.ndarray | None = None

    try:
        for i in range(warmup_frames):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                temporal_filter.process(depth_frame)
            if i % 10 == 0:
                print(f"  warmup... {i + 1}/{warmup_frames}")

        print(f"  warmup done, collecting {collect_frames} frames...")

        for i in range(collect_frames):
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            filtered_depth = temporal_filter.process(depth_frame)
            depth_image = np.asanyarray(filtered_depth.get_data())
            depth_buffer.append(depth_image.copy())

            if color_to_save is None:
                color_to_save = np.asanyarray(color_frame.get_data()).copy()

            if i % 3 == 0:
                print(f"  collect... {i + 1}/{collect_frames}")

        if not depth_buffer or color_to_save is None:
            raise RuntimeError("D405 did not provide valid color/depth frames")

        depth_stack = np.stack(depth_buffer, axis=0)
        depth_median = np.median(depth_stack, axis=0).astype(np.uint16)

        color_path = os.path.join(save_folder, f"d405_color_{img_idx:04d}.jpg")
        depth_path = os.path.join(save_folder, f"d405_depth_{img_idx:04d}.png")
        cv2.imwrite(color_path, color_to_save)
        cv2.imwrite(depth_path, depth_median)

        print(f"  saved: {color_path}")
        print(f"  saved: {depth_path}")
        print(f"  image id: {img_idx:04d}\n")

    finally:
        pipeline.stop()

    return img_idx
