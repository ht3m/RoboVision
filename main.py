"""
main.py — Sys_Vision 主程序
============================
机械臂搭载 D405 深度相机，实现：
  输入 'S' → 拍照 → VL+SAM 检测分割 → 点云重建 → 重心计算 → 坐标变换到基坐标系

用法:
  python main.py
  终端输入 S 启动一次流程

数据流程:
  photos/         ← D405 拍摄的彩色图和深度图
  procedure/VLM/  ← VL 检测框图
  procedure/SAM/  ← SAM2 分割 mask
  procedure/PC/   ← 点云截图 (debug 模式)
  report/         ← 2x2 报告图 (debug 模式)
"""

import os
import sys
import time
import glob
import traceback
import numpy as np
import cv2
from PIL import Image

# 确保项目根目录在 sys.path
_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "Core"))

from config import (
    MODE, PHOTO_DIR, OUTPUT_DIR, SAM_OUTPUT_DIR, CLOUD_POINT_DIR, REPORT_DIR,
    HAND_EYE_MATRIX,
    ROBOT_IP, ROBOT_PORT,
    D405_FX, D405_FY, D405_CX, D405_CY,
    VIS_PALETTE,
)


# ============================================================================
# 文件编号管理
# ============================================================================

def get_next_number() -> int:
    """扫描 photos 目录，返回下一个可用编号 (从 0 开始递增)"""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    existing = glob.glob(os.path.join(PHOTO_DIR, "d405_color_*.jpg"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        basename = os.path.basename(f)
        # d405_color_XXXX.jpg
        try:
            num_str = basename.replace("d405_color_", "").replace(".jpg", "")
            nums.append(int(num_str))
        except ValueError:
            continue
    return max(nums) + 1 if nums else 0


# ============================================================================
# 坐标变换
# ============================================================================

def transform_to_base(centroid_cam: np.ndarray,
                      T_handeye: np.ndarray,
                      T_endeffector_base: np.ndarray) -> np.ndarray:
    """将相机坐标系下的重心变换到机器人基坐标系

    变换链: P_base = T_end_base * T_cam_end * P_cam

    Args:
        centroid_cam: 相机坐标系下的三维点 (3,)
        T_handeye:    相机相对于机械臂末端的齐次矩阵 (4,4)  (T_cam_end)
        T_endeffector_base: 末端相对于基坐标系的齐次矩阵 (4,4)  (T_end_base)

    Returns:
        基坐标系下的三维点 (3,)
    """
    p_cam = np.append(centroid_cam, 1.0)  # 齐次坐标
    p_end = T_handeye @ p_cam              # 相机 → 末端
    p_base = T_endeffector_base @ p_end    # 末端 → 基坐标
    return p_base[:3]


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(number: int):
    """生成 2x2 报告图 (仅 debug 模式)

    布局:
      (a) 原始彩色图 | (b) SAM2 叠加图
      (c) 深度图 JET | (d) 点云截图
    """
    from config import MODE
    if MODE != "debug":
        return

    number_str = f"{number:04d}"
    color_path = os.path.join(PHOTO_DIR, f"d405_color_{number_str}.jpg")
    overlay_path = os.path.join(SAM_OUTPUT_DIR, f"d405_color_{number_str}_overlay.jpg")
    depth_path = os.path.join(PHOTO_DIR, f"d405_depth_{number_str}.png")

    # 查找滤波后点云截图
    pc_path = os.path.join(CLOUD_POINT_DIR, f"pc3d_{number_str}.png")

    print(f"\n{'─' * 60}")
    print(f"  生成 2×2 报告图...")
    print(f"{'─' * 60}")

    img_color = cv2.imread(color_path)
    if img_color is None:
        print(f"  [警告] 找不到彩色图 {color_path}")
        return

    img_overlay = cv2.imread(overlay_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    img_pc = cv2.imread(pc_path) if pc_path else None

    # 缺图兜底
    if img_overlay is None:
        print(f"  [警告] 找不到叠加图 {overlay_path}，用原图代替")
        img_overlay = img_color.copy()
    if depth_raw is not None:
        depth_uint8 = cv2.convertScaleAbs(depth_raw, alpha=0.03)
        img_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    else:
        print(f"  [警告] 找不到深度图 {depth_path}")
        img_depth = np.zeros_like(img_color)
    if img_pc is None:
        print(f"  [警告] 找不到点云截图")
        img_pc = np.zeros_like(img_color)

    # 统一缩放到相同大小
    TW, TH = 640, 360
    img_color = cv2.resize(img_color, (TW, TH))
    img_overlay = cv2.resize(img_overlay, (TW, TH))
    img_depth = cv2.resize(img_depth, (TW, TH))
    img_pc = cv2.resize(img_pc, (TW, TH))

    # 贴标签
    labels = ["(a) Original Color", "(b) SAM2 Overlay",
              "(c) Raw Depth (JET)", "(d) Point Cloud (filtered)"]
    for img, label in zip([img_color, img_overlay, img_depth, img_pc], labels):
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # 拼成 2×2
    top_row = np.hstack([img_color, img_overlay])
    bottom_row = np.hstack([img_depth, img_pc])
    report = np.vstack([top_row, bottom_row])

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f"report_{number_str}.jpg")
    cv2.imwrite(report_path, report, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  报告图已保存: {report_path}")


# ============================================================================
# 主流程
# ============================================================================

def run_pipeline():
    """执行一次完整的检测流水线"""
    # 1. 拍照
    from Core.d405_camera import capture_photos

    print(f"\n{'=' * 60}")
    print(f"  [Step 1/4] 拍照")
    print(f"{'=' * 60}")

    number = capture_photos(PHOTO_DIR)
    number_str = f"{number:04d}"
    color_path = os.path.join(PHOTO_DIR, f"d405_color_{number_str}.jpg")
    depth_path = os.path.join(PHOTO_DIR, f"d405_depth_{number_str}.png")
    print(f"  彩色图: {color_path}")
    print(f"  深度图: {depth_path}")

    # 2. VL + SAM2
    print(f"\n{'=' * 60}")
    print(f"  [Step 2/4] VL 检测 + SAM2 分割")
    print(f"{'=' * 60}")

    from Core.detect2sam import run_vl_sam
    boxes = run_vl_sam(number)

    if boxes is None:
        print("\n  [错误] VL+SAM 流程失败，终止")
        return
    if not boxes:
        print("\n  [警告] 未检测到任何物品，终止")
        return

    # 3. 点云重建 + 重心计算
    print(f"\n{'=' * 60}")
    print(f"  [Step 3/4] 点云重建 + 重心计算")
    print(f"{'=' * 60}")

    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"  [错误] 无法读取深度图: {depth_path}")
        return

    from Core.point_cloud_reconstruct import (
        compute_all_point_clouds, show_point_clouds,
    )

    # 一次性处理所有物体点云（背景 + 各物体）
    pc_data = compute_all_point_clouds(depth_img, boxes)

    centroids_cam = pc_data["centroids"]

    if not centroids_cam:
        print("\n  [警告] 所有物品点云重建均失败")

    # 可视化窗口 (debug 模式)
    # 窗口 1: 预滤波 → 关闭 → 窗口 2: 滤波后 → 关闭 → 截图
    show_point_clouds(pc_data, number_str)

    # 输出重心
    for name, centroid in centroids_cam.items():
        print(f"  {name} 重心 (相机坐标系): "
              f"({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}) m")

    # 4. 坐标变换
    print(f"\n{'=' * 60}")
    print(f"  [Step 4/4] 坐标变换: 相机 → 机器人基坐标系")
    print(f"{'=' * 60}")

    # 获取机械臂当前位姿
    from Core.robot_arm import get_current_pose, pose_to_matrix

    print(f"  连接机械臂: {ROBOT_IP}:{ROBOT_PORT} ...")
    try:
        pose = get_current_pose(ROBOT_IP, ROBOT_PORT)
        T_end_base = pose_to_matrix(pose)
        print(f"  末端位姿: [{pose[0]:.4f}, {pose[1]:.4f}, {pose[2]:.4f}, "
              f"{pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f}]")
        print(f"\n  末端坐标 (ecef):")
        print(f"    x={pose[0]:.4f}  y={pose[1]:.4f}  z={pose[2]:.4f}")
        print(f"    rx={pose[3]:.4f}  ry={pose[4]:.4f}  rz={pose[5]:.4f}")
    except Exception as e:
        print(f"  [错误] 无法获取机械臂位姿: {e}")
        print("  [错误] 无法继续输出机器人基坐标系，请检查网线、ROBOT_IP 和 ROBOT_PORT")
        return

    print(f"\n  手眼矩阵 (cam → end):")
    print(f"  {HAND_EYE_MATRIX}")

    # 变换每个重心
    centroids_base = {}
    for name, c_cam in centroids_cam.items():
        c_base = transform_to_base(c_cam, HAND_EYE_MATRIX, T_end_base)
        centroids_base[name] = c_base
        print(f"\n  {name}:")
        print(f"    相机坐标系:  ({c_cam[0]:.4f}, {c_cam[1]:.4f}, {c_cam[2]:.4f}) m")
        print(f"    基坐标系:    ({c_base[0]:.4f}, {c_base[1]:.4f}, {c_base[2]:.4f}) m")

    # 5. 生成报告 (debug 模式)
    if MODE == "debug":
        generate_report(number)

    # 6. 最终汇总
    print(f"\n\n{'=' * 60}")
    print(f"  ✅  处理完成!  编号: {number_str}")
    print(f"{'=' * 60}")
    print(f"  模式: {'调试 (debug)' if MODE == 'debug' else '实验 (experiment)'}")
    print(f"  检测到物品数: {len(centroids_base)}")
    print(f"\n  {'─' * 55}")
    print(f"  {'物品名称':<20s}  {'基坐标系重心 (x, y, z) [m]'}")
    print(f"  {'─' * 55}")
    for name, c_base in centroids_base.items():
        print(f"  {name:<20s}  ({c_base[0]:.4f}, {c_base[1]:.4f}, {c_base[2]:.4f})")
    print(f"  {'─' * 55}")

    print(f"\n  输出文件:")
    print(f"    照片:    {PHOTO_DIR}/")
    print(f"    VL 框图: {OUTPUT_DIR}/")
    print(f"    SAM mask: {SAM_OUTPUT_DIR}/")
    if MODE == "debug":
        print(f"    点云截图: {CLOUD_POINT_DIR}/")
        print(f"    报告图:   {REPORT_DIR}/")
    print(f"{'=' * 60}\n")


# ============================================================================
# 终端交互
# ============================================================================

def main():
    """主入口: 等待终端输入 'S' 启动流程, 输入 'Q' 退出"""
    print("=" * 60)
    print("  Sys_Vision — 机械臂视觉定位系统")
    print(f"  模式: {'调试 (debug)' if MODE == 'debug' else '实验 (experiment)'}")
    print("=" * 60)
    print()
    print("  指令:")
    print("    S  — 拍照并执行检测定位")
    print("    Q  — 退出程序")
    print()

    while True:
        try:
            cmd = input(">>> ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n退出程序")
            break

        if cmd == "S":
            t_start = time.time()
            try:
                run_pipeline()
            except Exception as e:
                print(f"\n{'!' * 60}")
                print(f"  [严重错误] 流水线执行失败: {e}")
                traceback.print_exc()
                print(f"{'!' * 60}")
            t_elapsed = time.time() - t_start
            print(f"\n  ⏱ 总耗时: {t_elapsed:.1f}s")
            print("\n  等待下一条指令 (S/Q)...\n")

        elif cmd == "Q":
            print("退出程序")
            break

        else:
            print(f"  未知指令: '{cmd}'，请输入 S 或 Q")


if __name__ == "__main__":
    main()
