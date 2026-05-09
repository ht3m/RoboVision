import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

print("=" * 55)
print("D405 相机内参 (1280x720 @ 30fps)")
print("=" * 55)

for name, stream_type in [("Color", rs.stream.color), ("Depth", rs.stream.depth)]:
    s = profile.get_stream(stream_type).as_video_stream_profile()
    i = s.get_intrinsics()
    print(f"\n--- {name} 内参 ---")
    print(f"  分辨率:    {i.width} x {i.height}")
    print(f"  fx:        {i.fx:.6f}")
    print(f"  fy:        {i.fy:.6f}")
    print(f"  ppx(cx):   {i.ppx:.6f}")
    print(f"  ppy(cy):   {i.ppy:.6f}")
    print(f"  畸变模型:   {i.model.name}  (distortion.{i.model.name.lower()})")
    print(f"  畸变系数:   {list(i.coeffs)}")

# 深度 scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"\n--- 深度缩放 ---")
print(f"  depth_scale: {depth_scale:.6f} m/unit")

pipeline.stop()