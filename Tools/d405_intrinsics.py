import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

print("=" * 55)
print("D405 intrinsics (1280x720 @ 30fps)")
print("=" * 55)

intrinsics_by_name = {}
for name, stream_type in [("Color", rs.stream.color), ("Depth", rs.stream.depth)]:
    stream_profile = profile.get_stream(stream_type).as_video_stream_profile()
    intr = stream_profile.get_intrinsics()
    intrinsics_by_name[name] = intr
    print(f"\n--- {name} intrinsics ---")
    print(f"  resolution: {intr.width} x {intr.height}")
    print(f"  fx:         {intr.fx:.6f}")
    print(f"  fy:         {intr.fy:.6f}")
    print(f"  ppx(cx):    {intr.ppx:.6f}")
    print(f"  ppy(cy):    {intr.ppy:.6f}")
    print(f"  distortion: {intr.model.name}")
    print(f"  coeffs:     {list(intr.coeffs)}")

color_intr = intrinsics_by_name["Color"]
print("\n--- Config values for depth aligned to color ---")
print("D405_ALIGN_DEPTH_TO_COLOR = True")
print(f"D405_ALIGNED_FX = {color_intr.fx:.6f}")
print(f"D405_ALIGNED_FY = {color_intr.fy:.6f}")
print(f"D405_ALIGNED_CX = {color_intr.ppx:.6f}")
print(f"D405_ALIGNED_CY = {color_intr.ppy:.6f}")
print(f"D405_ALIGNED_DIST_COEFFS = {list(color_intr.coeffs)}")

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("\n--- Depth scale ---")
print(f"  depth_scale: {depth_scale:.6f} m/unit")

pipeline.stop()
