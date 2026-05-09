"""
point_cloud_reconstruct.py — 原始深度点云重建、滤波与重心计算
================================================================
基于原始深度图 + SAM2 mask → 提取目标点云 → 滤波 → 重心

与 BioRobo_Vision/Core/raw_point_cloud_reconstruct.py 逻辑一致：
  - 体素降采样 → 统计滤波 → (可选) DBSCAN / 半径滤波
  - 计算 OBB 方向的过滤重心
  - 调试模式下显示 open3d 可视化窗口
"""

import os
import numpy as np
import cv2
import open3d as o3d
from numpy.typing import NDArray

from config import (
    RAW_PC_DEPTH_SCALE,
    RAW_PC_VOXEL_SIZE,
    RAW_PC_STAT_NB_NEIGHBORS,
    RAW_PC_STAT_STD_RATIO,
    RAW_PC_FINE_FILTER_MODE,
    RAW_PC_FINE_FILTER_RATIO,
    RAW_PC_RADIUS_FILTER_MIN_NEIGHBORS,
    RAW_PC_DBSCAN_MIN_POINTS,
    RAW_PC_DBSCAN_KEEP_TOP2,
    RAW_PC_DBSCAN_TOP2_RATIO,
    D405_FX, D405_FY, D405_CX, D405_CY,
    MODE,
)


def depth_to_point_cloud(depth: NDArray,
                         mask: NDArray | None = None) -> o3d.geometry.PointCloud:
    """深度图 → 原始点云 (可选 mask 裁剪)。

    Args:
        depth: (H, W) uint16 深度图 (单位: 0.1mm)
        mask:  (H, W) bool, True 表示保留点

    Returns:
        原始 open3d 点云
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    z = depth.astype(np.float64) / RAW_PC_DEPTH_SCALE  # → 米

    valid = z > 0
    if mask is not None:
        valid = valid & mask

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    x = (u_valid - D405_CX) * z_valid / D405_FX
    y = (v_valid - D405_CY) * z_valid / D405_FY

    pts = np.stack([x, y, z_valid], axis=-1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def filter_point_cloud(pcd: o3d.geometry.PointCloud,
                       ) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """完整滤波流水线。

    Returns:
        (pcd_voxel, pcd_fine): 体素降采样后的点云 和 最终精细滤波后的点云
    """
    # 1. 体素降采样
    if RAW_PC_VOXEL_SIZE > 0 and len(pcd.points) > 0:
        pcd_voxel = pcd.voxel_down_sample(RAW_PC_VOXEL_SIZE)
    else:
        pcd_voxel = pcd

    if len(pcd_voxel.points) < 10:
        return pcd_voxel, pcd_voxel

    # 2. 统计滤波
    cl, ind = pcd_voxel.remove_statistical_outlier(
        nb_neighbors=RAW_PC_STAT_NB_NEIGHBORS,
        std_ratio=RAW_PC_STAT_STD_RATIO,
    )
    pcd_clean = pcd_voxel.select_by_index(ind)

    if len(pcd_clean.points) < 10:
        return pcd_voxel, pcd_clean

    # 3. 精细滤波
    if RAW_PC_FINE_FILTER_MODE == "none":
        return pcd_voxel, pcd_clean

    return pcd_voxel, _fine_filter(pcd_clean)


def _fine_filter(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """精细滤波: radius 或 dbscan。"""
    if len(pcd.points) < 5:
        return pcd

    if RAW_PC_FINE_FILTER_MODE == "radius":
        obb = pcd.get_oriented_bounding_box()
        extent = np.array(obb.extent)
        r = RAW_PC_FINE_FILTER_RATIO * max(extent.min(), 0.001)
        cl, ind = pcd.remove_radius_outlier(
            nb_points=RAW_PC_RADIUS_FILTER_MIN_NEIGHBORS,
            radius=r,
        )
        return pcd.select_by_index(ind)

    elif RAW_PC_FINE_FILTER_MODE == "dbscan":
        obb = pcd.get_oriented_bounding_box()
        extent = np.array(obb.extent)
        r = RAW_PC_FINE_FILTER_RATIO * max(extent.min(), 0.001)

        labels = np.array(pcd.cluster_dbscan(
            eps=r,
            min_points=RAW_PC_DBSCAN_MIN_POINTS,
        ))

        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) == 0:
            return pcd

        sorted_idx = unique[np.argsort(-counts)]
        keep_clusters = [sorted_idx[0]]
        if RAW_PC_DBSCAN_KEEP_TOP2 and len(sorted_idx) > 1:
            if counts[1] >= counts[0] * RAW_PC_DBSCAN_TOP2_RATIO:
                keep_clusters.append(sorted_idx[1])

        keep_mask = np.isin(labels, keep_clusters)
        return pcd.select_by_index(np.where(keep_mask)[0])

    return pcd


def compute_centroid(pcd: o3d.geometry.PointCloud) -> NDArray:
    """计算 OBB 方向过滤后的加权重心。

    流程:
      1. 计算 OBB
      2. 将点变换到 OBB 坐标系，去掉最长轴两端 outlier
      3. 返回过滤后点的重心 (相机坐标系, 米)
    """
    points = np.asarray(pcd.points).copy()
    if len(points) < 3:
        return np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    obb = pcd.get_oriented_bounding_box()
    R = np.array(obb.R)
    center = np.array(obb.center)
    extent = np.array(obb.extent)

    pts_local = (points - center) @ R

    # 按 OBB 主轴过滤: 保留主轴方向 [0.1, 0.9] 分位内的点
    filtered_mask = np.ones(len(pts_local), dtype=bool)
    H0 = extent[0] * 0.35
    filtered_mask &= np.abs(pts_local[:, 0]) <= H0

    H1 = extent[1] * 0.45
    filtered_mask &= np.abs(pts_local[:, 1]) <= H1

    H2 = extent[2] * 0.45
    filtered_mask &= np.abs(pts_local[:, 2]) <= H2

    if filtered_mask.sum() < 3:
        return np.mean(points, axis=0)

    centroid = np.mean(points[filtered_mask], axis=0)
    return centroid


def visualize_point_cloud(pcd_before: o3d.geometry.PointCloud,
                          pcd_after: o3d.geometry.PointCloud,
                          window_name: str = "点云滤波对比"):
    """调试模式: 显示滤波前后点云对比。"""
    if MODE != "debug":
        return

    pcd_before.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_after.paint_uniform_color([0.0, 0.8, 0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=900, height=500)
    vis.add_geometry(pcd_before)
    vis.add_geometry(pcd_after)

    # 坐标轴
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.15, 0.15, 0.15])

    vis.run()
    vis.destroy_window()


def process_mask_point_cloud(depth: NDArray, mask: NDArray,
                             display_name: str = "") -> tuple[NDArray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """完整处理单个 mask 的点云。

    Args:
        depth: (H, W) uint16
        mask:  (H, W) bool
        display_name: 用于可视化标题

    Returns:
        (centroid, pcd_voxel, pcd_fine):
            centroid: 目标重心 (相机坐标系, 米)
            pcd_voxel: 体素滤波后点云
            pcd_fine:  精细滤波后点云
    """
    # 生成原始点云
    pcd_raw = depth_to_point_cloud(depth, mask)

    if len(pcd_raw.points) < 5:
        print(f"  [警告] {display_name} 点云点数过少 ({len(pcd_raw.points)}), 跳过")
        return np.zeros(3), pcd_raw, pcd_raw

    # 滤波
    pcd_voxel, pcd_fine = filter_point_cloud(pcd_raw)

    if len(pcd_fine.points) < 3:
        print(f"  [警告] {display_name} 滤波后点数过少, 使用 voxel 点云重心")
        pts = np.asarray(pcd_voxel.points)
        centroid = np.mean(pts, axis=0) if len(pts) > 0 else np.zeros(3)
    else:
        centroid = compute_centroid(pcd_fine)

    print(f"  {display_name}: 原始 {len(pcd_raw.points)} 点 → "
          f"体素 {len(pcd_voxel.points)} 点 → 精细 {len(pcd_fine.points)} 点")

    # 调试可视化
    if MODE == "debug":
        visualize_point_cloud(pcd_voxel, pcd_fine,
                              window_name=f"{display_name} - 滤波对比")

    return centroid, pcd_voxel, pcd_fine


def save_point_cloud_screenshot(pcd_voxel: o3d.geometry.PointCloud,
                                pcd_fine: o3d.geometry.PointCloud,
                                centroid: NDArray,
                                save_path: str):
    """保存点云截图（用于报告）。"""
    if MODE != "debug":
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pcd_v = o3d.geometry.PointCloud(pcd_voxel)
    pcd_f = o3d.geometry.PointCloud(pcd_fine)
    pcd_v.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_f.paint_uniform_color([0.0, 0.8, 0.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=640, height=480)
    vis.add_geometry(pcd_v)
    vis.add_geometry(pcd_f)

    # 标记重心
    if np.any(centroid):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(centroid)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        vis.add_geometry(sphere)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.15, 0.15, 0.15])

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()
