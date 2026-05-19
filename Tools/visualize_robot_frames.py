#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize robot base, TCP, and camera coordinate frames.

Default transform convention:
    T_base_camera = T_base_tcp @ HAND_EYE_MATRIX

where HAND_EYE_MATRIX is expected to map camera-frame points into TCP/end frame:
    P_tcp = HAND_EYE_MATRIX @ P_camera
"""

import argparse
import os
import sys
from typing import Iterable

import numpy as np
import open3d as o3d

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
CORE_DIR = os.path.join(PROJECT_ROOT, "Core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from config import HAND_EYE_MATRIX, ROBOT_IP, ROBOT_PORT  # noqa: E402
from Core.robot_arm import get_current_pose, pose_to_matrix  # noqa: E402


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform."""
    inv = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def make_frame(T: np.ndarray, size: float) -> o3d.geometry.TriangleMesh:
    """Create an Open3D coordinate frame transformed by T."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    frame.transform(T)
    return frame


def make_sphere(position: Iterable[float], radius: float, color: Iterable[float]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(np.asarray(position, dtype=np.float64))
    sphere.paint_uniform_color(color)
    return sphere


def make_connection(points: list[np.ndarray]) -> o3d.geometry.LineSet:
    """Create a yellow polyline connecting frame origins."""
    pts = np.asarray(points, dtype=np.float64)
    lines = [[i, i + 1] for i in range(len(points) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.85, 0.05] for _ in lines])
    return line_set


def make_camera_frustum(T_base_camera: np.ndarray, scale: float = 0.06) -> o3d.geometry.LineSet:
    """Draw a small camera frustum in the camera optical frame."""
    z = scale
    x = scale * 0.7
    y = scale * 0.45
    local = np.array([
        [0.0, 0.0, 0.0, 1.0],
        [-x, -y, z, 1.0],
        [x, -y, z, 1.0],
        [x, y, z, 1.0],
        [-x, y, z, 1.0],
    ], dtype=np.float64)
    world = (T_base_camera @ local.T).T[:, :3]
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    frustum = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    frustum.colors = o3d.utility.Vector3dVector([[0.05, 0.8, 1.0] for _ in lines])
    return frustum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize current robot base, TCP/end, and camera coordinate frames."
    )
    parser.add_argument("--ip", default=ROBOT_IP, help=f"UR robot IP, default: {ROBOT_IP}")
    parser.add_argument("--port", type=int, default=ROBOT_PORT, help=f"UR realtime port, default: {ROBOT_PORT}")
    parser.add_argument("--frame-size", type=float, default=0.12, help="Coordinate frame axis size in meters.")
    parser.add_argument(
        "--inverse-handeye",
        action="store_true",
        help="Use inverse(HAND_EYE_MATRIX) if your matrix is T_camera_tcp instead of T_tcp_camera.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading current TCP pose from {args.ip}:{args.port} ...")
    pose = get_current_pose(args.ip, args.port)
    T_base_tcp = pose_to_matrix(pose)

    T_tcp_camera = invert_transform(HAND_EYE_MATRIX) if args.inverse_handeye else HAND_EYE_MATRIX
    T_base_camera = T_base_tcp @ T_tcp_camera

    print("\nTCP pose [x, y, z, rx, ry, rz]:")
    print(np.array2string(pose, precision=6, suppress_small=False))
    print("\nT_base_tcp:")
    print(np.array2string(T_base_tcp, precision=6, suppress_small=False))
    print("\nT_tcp_camera used:")
    print(np.array2string(T_tcp_camera, precision=6, suppress_small=False))
    print("\nT_base_camera:")
    print(np.array2string(T_base_camera, precision=6, suppress_small=False))
    print("\nOpen3D axis colors: X=red, Y=green, Z=blue")
    print("Frames: base at world origin, TCP from robot pose, camera from T_base_tcp @ T_tcp_camera")

    base_origin = np.zeros(3, dtype=np.float64)
    tcp_origin = T_base_tcp[:3, 3]
    camera_origin = T_base_camera[:3, 3]

    geometries = [
        make_frame(np.eye(4), args.frame_size * 1.4),
        make_frame(T_base_tcp, args.frame_size),
        make_frame(T_base_camera, args.frame_size * 0.75),
        make_connection([base_origin, tcp_origin, camera_origin]),
        make_camera_frustum(T_base_camera, scale=args.frame_size * 0.65),
        make_sphere(base_origin, args.frame_size * 0.05, [1.0, 1.0, 1.0]),
        make_sphere(tcp_origin, args.frame_size * 0.05, [1.0, 0.85, 0.05]),
        make_sphere(camera_origin, args.frame_size * 0.05, [0.05, 0.8, 1.0]),
    ]

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Robot Frames: base / TCP / camera",
        width=1280,
        height=900,
    )


if __name__ == "__main__":
    main()
