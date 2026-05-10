"""
point_cloud_reconstruct.py — 原始深度点云重建、滤波与重心计算
================================================================
基于原始深度图 + SAM2 mask → 提取目标点云 → 滤波 → 重心

可视化逻辑 (与 BioRobo_Vision/Core/raw_point_cloud_reconstruct.py 一致):
  - 窗口 1: 预滤波点云 (仅体素降采样) — 所有物体 + 背景 + OBB
  - 关闭窗口 1 → 窗口 2: 滤波后点云 — 所有物体 + 背景 + OBB + 被滤掉的点(灰色)
  - 关闭窗口 2 → 截图保存用于报告
"""

import os
import numpy as np
import cv2
import open3d as o3d
from numpy.typing import NDArray
from typing import Optional

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
    MODE, CLOUD_POINT_DIR,
    VIS_PALETTE,
)

_OBJ_COLORS = [(r / 255.0, g / 255.0, b / 255.0) for (r, g, b) in VIS_PALETTE]
REMOVED_GRAY = [0.45, 0.45, 0.45]


def depth_to_point_cloud(depth: NDArray,
                         mask: NDArray | None = None) -> NDArray:
    """深度图 → (N, 3) 点云数组 (相机坐标系, 米)。

    Args:
        depth: (H, W) uint16 深度图 (单位: 0.1mm)
        mask:  (H, W) bool, True 表示保留点

    Returns:
        (N, 3) float64 numpy 数组
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float64)
    v = v.astype(np.float64)
    z = depth.astype(np.float64) / RAW_PC_DEPTH_SCALE

    valid = z > 0
    if mask is not None:
        valid = valid & mask

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = z[valid]

    x = (u_valid - D405_CX) * z_valid / D405_FX
    y = (v_valid - D405_CY) * z_valid / D405_FY

    return np.column_stack([x, y, z_valid])


def voxel_downsample(points: NDArray, voxel_size: float | None = None) -> NDArray:
    """体素降采样 (纯 numpy 实现)。"""
    if voxel_size is None:
        voxel_size = RAW_PC_VOXEL_SIZE
    if voxel_size <= 0 or len(points) == 0:
        return points
    idx = np.floor(points / voxel_size).astype(np.int64)
    u, inv, cnt = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    acc = np.zeros((len(u), 3), dtype=np.float64)
    np.add.at(acc, inv, points.astype(np.float64))
    return acc / cnt[:, None]


def statistical_outlier_filter(points: NDArray) -> tuple[NDArray, NDArray]:
    """统计滤波：保留点, 被剔除的点。"""
    from scipy.spatial import cKDTree
    nb = RAW_PC_STAT_NB_NEIGHBORS
    std = RAW_PC_STAT_STD_RATIO
    if len(points) < nb:
        return points, np.empty((0, 3))
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    thresh = mean_dists.mean() + std * mean_dists.std()
    keep = mean_dists <= thresh
    return points[keep], points[~keep]


def get_obb_shortest_dim(points: NDArray) -> tuple[Optional[NDArray], float]:
    """PCA → OBB 三条边长 (m) 及最短边长度。"""
    if len(points) < 3:
        return None, 0.0
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, eig_vecs = np.linalg.eigh(np.cov(centered.T))
    eig_vecs = eig_vecs[:, ::-1]
    proj = centered @ eig_vecs
    lengths = proj.max(axis=0) - proj.min(axis=0)
    return lengths, lengths.min()


def _dbscan_cluster(points: NDArray, eps: float) -> tuple[NDArray, int]:
    """纯 scipy DBSCAN → 保留最大簇(可选第二大) 的掩码 + 簇数。"""
    from scipy.spatial import cKDTree
    n = len(points)
    min_pts = RAW_PC_DBSCAN_MIN_POINTS
    if n < min_pts:
        return np.ones(n, dtype=bool), 1

    tree = cKDTree(points)
    neighbors_list = tree.query_ball_point(points, r=eps)
    is_core = np.array([len(nei) >= min_pts for nei in neighbors_list])

    parent = np.arange(n)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if not is_core[i]:
            continue
        for nb in neighbors_list[i]:
            if is_core[nb]:
                union(i, nb)

    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    root_to_label = {}
    for i in range(n):
        if not is_core[i]:
            continue
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = cluster_id
            cluster_id += 1
        labels[i] = root_to_label[root]

    for i in range(n):
        if is_core[i]:
            continue
        for nb in neighbors_list[i]:
            if is_core[nb]:
                labels[i] = labels[nb]
                break

    valid = labels >= 0
    if not valid.any():
        return np.ones(n, dtype=bool), 1

    vlabels = labels[valid]
    unique_labels, counts = np.unique(vlabels, return_counts=True)
    n_clusters = len(unique_labels)

    max_label = unique_labels[counts.argmax()]
    keep_mask = labels == max_label

    if RAW_PC_DBSCAN_KEEP_TOP2 and n_clusters >= 2:
        sorted_indices = counts.argsort()[::-1]
        second_count = counts[sorted_indices[1]]
        if second_count >= counts[sorted_indices[0]] * RAW_PC_DBSCAN_TOP2_RATIO:
            second_label = unique_labels[sorted_indices[1]]
            keep_mask |= (labels == second_label)

    return keep_mask, n_clusters


def _mark_removed(input_pts: NDArray, kept_pts: NDArray) -> NDArray:
    """找出 input_pts 中不在 kept_pts 里的点。"""
    from scipy.spatial import cKDTree
    if len(kept_pts) == 0 or len(input_pts) == 0:
        return input_pts.copy()
    tree = cKDTree(kept_pts)
    dists, _ = tree.query(input_pts, k=1, distance_upper_bound=0.001)
    return input_pts[dists > 0.0005]


def filter_single_object(pts_ds: NDArray) -> tuple[NDArray, NDArray]:
    """对单个物体的体素降采样后点云进行滤波。

    Returns:
        (kept_points, all_removed_points)
    """
    from scipy.spatial import cKDTree
    removed_parts = []

    if len(pts_ds) == 0:
        return pts_ds, np.empty((0, 3))

    # ① 统计滤波
    n1 = len(pts_ds)
    pts_clean, removed = statistical_outlier_filter(pts_ds)
    n2 = len(pts_clean)
    removed_parts.append(removed)
    print(f"    统计滤波: {n1} → {n2} ({100*(n1-n2)/max(n1,1):.1f}% 剔除)")

    if len(pts_clean) < 3:
        all_removed = np.vstack(removed_parts) if removed_parts else np.empty((0, 3))
        return pts_clean, all_removed

    # ② PCA → 取 OBB 最短边
    obb_lengths, shortest_dim = get_obb_shortest_dim(pts_clean)
    if obb_lengths is None or shortest_dim <= 0:
        all_removed = np.vstack(removed_parts) if removed_parts else np.empty((0, 3))
        return pts_clean, all_removed

    s1, s2, s3 = sorted(obb_lengths * 1000, reverse=True)
    print(f"    PCA: OBB {s1:.0f}×{s2:.0f}×{s3:.0f} mm  |  最短边={shortest_dim*1000:.1f} mm")

    # ③ 自适应精滤波
    mode = RAW_PC_FINE_FILTER_MODE
    if mode == "none":
        all_removed = np.vstack(removed_parts) if removed_parts else np.empty((0, 3))
        return pts_clean, all_removed

    adaptive_r = shortest_dim * RAW_PC_FINE_FILTER_RATIO
    r_mm = adaptive_r * 1000

    if mode == "radius":
        n3 = len(pts_clean)
        tree = cKDTree(pts_clean)
        cnt = tree.query_ball_point(pts_clean, r=adaptive_r, return_length=True)
        keep_radius = np.array(cnt) >= RAW_PC_RADIUS_FILTER_MIN_NEIGHBORS
        removed_parts.append(pts_clean[~keep_radius])
        pts_clean = pts_clean[keep_radius]
        n4 = len(pts_clean)
        print(f"    自适应半径滤波 (r={r_mm:.1f}mm): {n3} → {n4} ({100*(n3-n4)/max(n3,1):.1f}% 剔除)")

    elif mode == "dbscan":
        n3 = len(pts_clean)
        keep_mask, n_clusters = _dbscan_cluster(pts_clean, eps=adaptive_r)
        removed_parts.append(pts_clean[~keep_mask])
        pts_clean = pts_clean[keep_mask]
        n4 = len(pts_clean)
        print(f"    自适应 DBSCAN (eps={r_mm:.1f}mm): "
              f"{n_clusters} 簇 → {n4} pts ({100*(n3-n4)/max(n3,1):.1f}% 剔除)")

    all_removed = np.vstack(removed_parts) if removed_parts else np.empty((0, 3))
    return pts_clean, all_removed


def pca_obb(points: NDArray) -> Optional[dict]:
    """PCA 主成分分析 → 方向包围盒 OBB。

    Returns:
        dict: {center, centroid, R, extent, dims_mm} 或 None
    """
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, eig_vecs = np.linalg.eigh(np.cov(centered.T))
    eig_vecs = eig_vecs[:, ::-1]
    if np.linalg.det(eig_vecs) < 0:
        eig_vecs[:, 2] *= -1
    proj = centered @ eig_vecs
    mins, maxs = proj.min(axis=0), proj.max(axis=0)
    lengths = maxs - mins
    center = centroid + ((mins + maxs) / 2) @ eig_vecs.T
    return {
        "center": center,
        "centroid": centroid,
        "R": eig_vecs,
        "extent": lengths / 2,
        "dims_mm": lengths * 1000,
    }


def compute_centroid(points: NDArray) -> NDArray:
    """计算 OBB 方向过滤后的加权重心 (相机坐标系, 米)。

    去掉 OBB 主轴两端 outlier，取剩余点均值。
    """
    if len(points) < 3:
        return np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)

    obb = pca_obb(points)
    if obb is None:
        return np.mean(points, axis=0)

    R = obb["R"]
    center = obb["center"]
    extent = obb["extent"]

    pts_local = (points - center) @ R

    filtered = np.ones(len(pts_local), dtype=bool)
    filtered &= np.abs(pts_local[:, 0]) <= extent[0] * 0.7
    filtered &= np.abs(pts_local[:, 1]) <= extent[1] * 0.9
    filtered &= np.abs(pts_local[:, 2]) <= extent[2] * 0.9

    if filtered.sum() < 3:
        return np.mean(points, axis=0)

    return np.mean(points[filtered], axis=0)


# ============================================================================
# 流水线核心：一次性处理所有物体
# ============================================================================

def compute_all_point_clouds(depth_img: NDArray,
                             boxes: list[dict]) -> dict:
    """处理所有检测框的点云：背景 + 各物体预滤波&滤波 + 重心。

    Args:
        depth_img: (H, W) uint16 深度图
        boxes:     detect2sam 输出的 boxes 列表，每个含 name, mask (bool)

    Returns:
        data dict:
        {
            "pre_pcs":     list[NDArray],   # 预滤波各物体点云 (仅体素降采样)
            "pre_obbs":    list[dict],       # 预滤波各物体 OBB
            "pre_names":   list[str],        # 预滤波各物体名称
            "obj_pcs":     list[NDArray],    # 滤波后各物体点云
            "obbs":        list[dict],        # 滤波后各物体 OBB
            "names":       list[str],         # 滤波后各物体名称
            "removed_pcs": list[NDArray],     # 各物体被滤掉的点
            "bg_pcd":      o3d.geometry.PointCloud,  # 背景点云
            "centroids":   dict[str, NDArray],        # {name: centroid_3d}
        }
    """
    print(f"\n{'=' * 70}")
    print(f"  原始深度点云重建 (所有物体)")
    print(f"{'=' * 70}")

    h, w = depth_img.shape[:2]
    print(f"  深度图尺寸: {w}×{h}")
    print(f"  检测到物体数: {len(boxes)}")

    # ── 构建物体区域并集 ──
    union_obj_mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        mask = box.get("mask")
        if mask is not None:
            union_obj_mask |= mask
    n_obj = union_obj_mask.sum()
    print(f"  物体像素 (并集): {n_obj}  ({100*n_obj/(w*h):.1f}%)")

    # ── 背景点云 = 全图有效像素 - 物体区域 ──
    print(f"\n  生成背景点云 (灰色)...")
    bg_mask = (depth_img > 0) & (~union_obj_mask)
    bg_pts = depth_to_point_cloud(depth_img, bg_mask)
    print(f"    背景像素: {bg_mask.sum()} → {len(bg_pts)} pts")
    bg_pts = voxel_downsample(bg_pts)
    print(f"    体素降采样 ({RAW_PC_VOXEL_SIZE*1000:.0f}mm): {len(bg_pts)} pts")

    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(bg_pts)
    bg_pcd.paint_uniform_color([0.35, 0.35, 0.38])

    # ── 逐个物体处理 ──
    obj_data = []  # [(name, pts_raw, pts_ds)]

    for i, box in enumerate(boxes):
        name = box.get("name", f"object_{i}")
        mask = box.get("mask")
        if mask is None:
            print(f"\n  [{i+1}/{len(boxes)}] {name} — 无 mask, 跳过")
            continue

        print(f"\n  [{i+1}/{len(boxes)}] {name}")
        pts_raw = depth_to_point_cloud(depth_img, mask)
        print(f"    原始点: {len(pts_raw)} pts")

        if len(pts_raw) < 10:
            print("    [!] 原始点过少, 跳过")
            continue

        pts_ds = voxel_downsample(pts_raw)
        if len(pts_ds) != len(pts_raw):
            print(f"    体素降采样: {len(pts_raw)} → {len(pts_ds)} pts")

        if len(pts_ds) < 10:
            print("    [!] 降采样后点过少, 跳过")
            continue

        obj_data.append((name, pts_raw, pts_ds))

    if not obj_data:
        print("\n  [X] 无有效物体点云")
        return {
            "pre_pcs": [], "pre_obbs": [], "pre_names": [],
            "obj_pcs": [], "obbs": [], "names": [], "removed_pcs": [],
            "bg_pcd": bg_pcd, "centroids": {},
        }

    # ── 预滤波阶段：OBB ──
    pre_pcs, pre_obbs, pre_names = [], [], []
    for name, pts_raw, pts_ds in obj_data:
        obb = pca_obb(pts_ds)
        if obb is None:
            continue
        pre_pcs.append(pts_ds)
        pre_obbs.append(obb)
        pre_names.append(name)

    # ── 滤波阶段 ──
    print(f"\n{'─' * 70}")
    print(f"  开始精细滤波...")
    print(f"{'─' * 70}")

    obj_pcs, obbs, names, removed_pcs, centroids = [], [], [], [], {}

    for name, pts_raw, pts_ds in obj_data:
        print(f"\n  [{len(obj_pcs)+1}/{len(obj_data)}] {name} (滤波中)")

        pts_filtered, all_removed = filter_single_object(pts_ds)
        if len(pts_filtered) < 10:
            print("    [!] 滤波后点过少, 跳过")
            continue
        print(f"    最终: {len(pts_filtered)} pts  |  剔除: {len(all_removed)} pts")

        obb = pca_obb(pts_filtered)
        if obb is None:
            continue

        centroid = compute_centroid(pts_filtered)
        centroids[name] = centroid

        obj_pcs.append(pts_filtered)
        obbs.append(obb)
        names.append(name)
        removed_pcs.append(all_removed)

    # ── 汇总 ──
    if obj_pcs:
        print(f"\n{'=' * 70}")
        print(f"  汇总")
        print(f"{'=' * 70}")
        print(f"  {'物体':<24s}{'点数':>8s}  {'重心 (m)':>32s}  {'尺寸 (mm)':>26s}")
        print(f"  {'─' * 24}{'─' * 8}  {'─' * 32}  {'─' * 26}")
        for nm, pts, obb in zip(names, obj_pcs, obbs):
            cx, cy, cz = obb["centroid"]
            dw, dh, dd = obb["dims_mm"]
            print(f"  {nm:<24s}{len(pts):>8d}  "
                  f"({cx:8.3f}, {cy:7.3f}, {cz:7.3f})   "
                  f"{dw:7.0f} ×{dh:6.0f} ×{dd:6.0f}")
        print()

    return {
        "pre_pcs": pre_pcs, "pre_obbs": pre_obbs, "pre_names": pre_names,
        "obj_pcs": obj_pcs, "obbs": obbs, "names": names, "removed_pcs": removed_pcs,
        "bg_pcd": bg_pcd, "centroids": centroids,
    }


# ============================================================================
# Open3D 可视化窗口 (debug 模式)
# ============================================================================

def _create_coordinate_frame(size: float = 0.15) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0, 0, 0])


def show_prefilter_window(pre_pcs: list[NDArray],
                          pre_obbs: list[dict],
                          pre_names: list[str],
                          bg_pcd: o3d.geometry.PointCloud,
                          number_str: str):
    """窗口 1: 预滤波点云 (仅体素降采样) — 所有物体 + 背景 + OBB。"""
    if MODE != "debug":
        return
    if not pre_pcs:
        return

    print(f"\n{'─' * 70}")
    print(f"  窗口 1: 预滤波点云 (仅体素降采样) — 请查看后关闭")
    print(f"{'─' * 70}")

    geometries = []
    if bg_pcd is not None and len(bg_pcd.points) > 0:
        geometries.append(bg_pcd)
    geometries.append(_create_coordinate_frame(0.15))

    for i, (pts, obb, name) in enumerate(zip(pre_pcs, pre_obbs, pre_names)):
        color = _OBJ_COLORS[i % len(_OBJ_COLORS)]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        obb_box = o3d.geometry.OrientedBoundingBox(
            center=obb["center"], R=obb["R"], extent=obb["extent"] * 2)
        obb_box.color = color
        geometries.append(obb_box)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(obb["centroid"])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(sphere)

        cx, cy, cz = obb["centroid"]
        dw, dh, dd = obb["dims_mm"]
        rgb = tuple(int(c * 255) for c in color)
        print(f"    {name}: {len(pts)} pts  |  RGB{rgb}")
        print(f"      重心: ({cx:.4f}, {cy:.4f}, {cz:.4f}) m  |  "
              f"尺寸: {dw:.1f}×{dh:.1f}×{dd:.1f} mm")

    _run_visualizer(geometries, title=f"PRE-FILTER: 点云+OBB #{number_str}",
                    save_path=None)


def show_filtered_window(obj_pcs: list[NDArray],
                         obbs: list[dict],
                         names: list[str],
                         removed_pcs: list[NDArray],
                         bg_pcd: o3d.geometry.PointCloud,
                         number_str: str,
                         save_screenshot: bool = True):
    """窗口 2: 滤波后点云 — 所有物体 + 背景 + OBB + 被滤掉的点 (灰色)。"""
    if MODE != "debug":
        return

    print(f"\n{'─' * 70}")
    print(f"  窗口 2: 滤波后点云 — 请查看后关闭")
    print(f"{'─' * 70}")

    geometries = []
    if bg_pcd is not None and len(bg_pcd.points) > 0:
        geometries.append(bg_pcd)
    geometries.append(_create_coordinate_frame(0.15))

    for i, (pts, obb, name) in enumerate(zip(obj_pcs, obbs, names)):
        color = _OBJ_COLORS[i % len(_OBJ_COLORS)]

        # 被滤掉的点 (灰色)
        if i < len(removed_pcs) and len(removed_pcs[i]) > 0:
            rem_pcd = o3d.geometry.PointCloud()
            rem_pcd.points = o3d.utility.Vector3dVector(removed_pcs[i])
            rem_pcd.paint_uniform_color(REMOVED_GRAY)
            geometries.append(rem_pcd)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        obb_box = o3d.geometry.OrientedBoundingBox(
            center=obb["center"], R=obb["R"], extent=obb["extent"] * 2)
        obb_box.color = color
        geometries.append(obb_box)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(obb["centroid"])
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        geometries.append(sphere)

        cx, cy, cz = obb["centroid"]
        dw, dh, dd = obb["dims_mm"]
        rgb = tuple(int(c * 255) for c in color)
        print(f"    {name}: {len(pts)} pts  |  RGB{rgb}")
        print(f"      重心: ({cx:.4f}, {cy:.4f}, {cz:.4f}) m  |  "
              f"尺寸: {dw:.1f}×{dh:.1f}×{dd:.1f} mm")

    # 截图路径
    screenshot_path = None
    if save_screenshot:
        os.makedirs(CLOUD_POINT_DIR, exist_ok=True)
        screenshot_path = os.path.join(CLOUD_POINT_DIR, f"pc3d_{number_str}.png")

    _run_visualizer(geometries, title=f"滤波后点云+OBB #{number_str}",
                    save_path=screenshot_path)


def _run_visualizer(geometries: list,
                    title: str,
                    save_path: str | None = None):
    """通用 Open3D 可视化运行器。"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1280, height=900)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.08])
    opt.point_size = 2.0
    opt.show_coordinate_frame = True

    vc = vis.get_view_control()
    vc.set_front([0.2, -0.6, -0.8])
    vc.set_lookat([0, 0, 1.5])
    vc.set_up([0, -1, 0])
    vc.set_zoom(0.6)

    vis.run()

    if save_path:
        vis.capture_screen_image(save_path, do_render=True)
        print(f"\n  点云截图已保存: {save_path}")

    vis.destroy_window()


def show_point_clouds(data: dict, number_str: str):
    """展示两个连续的可视化窗口 (仅 debug 模式)。

    窗口 1: 预滤波 (仅体素降采样)
    窗口 2: 滤波后 → 截图保存
    """
    if MODE != "debug":
        return

    # 窗口 1: 预滤波
    if data.get("pre_pcs"):
        show_prefilter_window(
            data["pre_pcs"], data["pre_obbs"], data["pre_names"],
            data["bg_pcd"], number_str)
    elif data.get("obj_pcs"):
        pass  # 仅有滤波后，跳过预滤波窗口

    # 窗口 2: 滤波后
    if data.get("obj_pcs"):
        show_filtered_window(
            data["obj_pcs"], data["obbs"], data["names"],
            data["removed_pcs"], data["bg_pcd"],
            number_str, save_screenshot=True)
    elif data["bg_pcd"] is not None and len(data["bg_pcd"].points) > 0:
        # 无物体时仅显示背景全景
        print(f"\n{'─' * 70}")
        print(f"  窗口: 全景 (无物体)")
        print(f"{'─' * 70}")
        geometries = [data["bg_pcd"], _create_coordinate_frame(0.15)]
        screenshot_path = os.path.join(CLOUD_POINT_DIR, f"pc3d_{number_str}.png")
        _run_visualizer(geometries, title=f"全景 #{number_str}",
                        save_path=screenshot_path)
