import pyrealsense2 as rs
import numpy as np
import cv2
import os

# ===================== 文件夹与自动编号配置 =====================
save_folder = "photos"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"已自动创建保存目录: {save_folder}")

img_counter = 0

# ===================== D405 官方标准配置 =====================
pipeline = rs.pipeline()
config = rs.config()

# D405 推荐分辨率（官方最优）：彩色1280x720，深度1280x720 30fps
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# 启动相机
pipeline.start(config)

# ===================== 参数配置 =====================
WARMUP_FRAMES = 30      # 预热帧数（让Temporal Filter稳定）
COLLECT_FRAMES = 10     # 收集帧数（用于中值滤波）

# ===================== 状态变量 =====================
capturing = False
warmup_count = 0
collect_count = 0
depth_buffer = []       # 收集的深度图列表
color_to_save = None    # 要保存的彩色图

# Temporal Filter 实例（每次按s会重新创建）
temporal_filter = None

try:
    print(f"相机已启动")
    print(f"  按 s → 预热{WARMUP_FRAMES}帧 + 收集{COLLECT_FRAMES}帧(Temporal+中值滤波)后保存")
    print(f"  按 q → 退出")

    while True:
        # 等待获取一帧数据
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())

        if capturing:
            # ---------- 采集模式：应用 Temporal Filter ----------
            filtered_depth = temporal_filter.process(depth_frame)
            depth_image = np.asanyarray(filtered_depth.get_data())

            if warmup_count < WARMUP_FRAMES:
                # 预热阶段：只跑 Temporal Filter，丢弃帧
                warmup_count += 1
                status_text = f"Warming up... {warmup_count}/{WARMUP_FRAMES}"
                cv2.putText(color_image, status_text,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if warmup_count >= WARMUP_FRAMES:
                    print(f"  预热完成，开始收集 {COLLECT_FRAMES} 帧...")
            else:
                # 收集阶段
                depth_buffer.append(depth_image.copy())
                collect_count += 1
                if color_to_save is None:
                    color_to_save = color_image.copy()
                status_text = f"Collecting... {collect_count}/{COLLECT_FRAMES}"
                cv2.putText(color_image, status_text,
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                if collect_count >= COLLECT_FRAMES:
                    # 收集完毕，对深度做中值滤波
                    print(f"  收集完成，正在计算中值滤波...")
                    depth_stack = np.stack(depth_buffer, axis=0)   # shape: (10, H, W)
                    depth_median = np.median(depth_stack, axis=0).astype(np.uint16)

                    # 保存图片
                    color_filename = os.path.join(save_folder, f"d405_color_{img_counter:04d}.jpg")
                    depth_filename = os.path.join(save_folder, f"d405_depth_{img_counter:04d}.png")

                    cv2.imwrite(color_filename, color_to_save)
                    cv2.imwrite(depth_filename, depth_median)

                    print(f"  ✓ 已保存: 第 {img_counter} 组 (Temporal 30帧预热 + 10帧中值滤波)")
                    img_counter += 1

                    # 重置状态
                    capturing = False
                    warmup_count = 0
                    collect_count = 0
                    depth_buffer = []
                    color_to_save = None
                    temporal_filter = None
        else:
            # ---------- 普通预览模式：不应用滤波器（实时帧率优先） ----------
            depth_image = np.asanyarray(depth_frame.get_data())

        # 深度图着色显示
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # 拼接显示
        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('D405 实时画面', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('D405 实时画面', 1280, 360)
        cv2.imshow('D405 实时画面', images)

        # 按键操作
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and not capturing:
            # 按下 s：进入采集模式，重新创建 Temporal Filter（清除旧历史）
            capturing = True
            warmup_count = 0
            collect_count = 0
            depth_buffer = []
            color_to_save = None
            temporal_filter = rs.temporal_filter()
            print(f"\n>>> 开始采集: 预热 {WARMUP_FRAMES} 帧 → 收集 {COLLECT_FRAMES} 帧 (Temporal + 中值滤波)")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()