# Sys_Vision：UR5-D405 机器人视觉定位系统

## 摘要

`Sys_Vision` 是一个面向机器人实验台操作的视觉定位系统。系统使用 Intel RealSense D405 获取 RGB-D 数据，通过 Qwen3-VL 完成开放词汇目标检测，通过 SAM2.1 生成目标级二值掩膜，再结合深度图重建目标点云、滤波并估计三维重心。最后，系统读取 UR5 当前 TCP 位姿，并使用手眼标定矩阵将目标重心从相机坐标系转换到机器人基坐标系。

当前主流程为一次性执行式流水线：运行 `python main.py` 后完成一次拍照、检测、分割、点云重建、坐标变换与结果写出。调试模式下还会弹出 Open3D 点云窗口并生成 2x2 报告图。

---

## 1. 系统目标与坐标约定

系统输出每个被识别目标在机器人基坐标系下的三维重心：

```text
[object_name]: [x, y, z]
```

核心坐标链为：

```text
P_base = T_end_to_base @ T_cam_to_end @ P_cam
```

其中：

- `P_cam`：由 D405 深度图、相机内参和 SAM mask 重建得到的目标重心。
- `T_cam_to_end`：`config.py` 中的 `HAND_EYE_MATRIX`，表示相机坐标系到机械臂末端坐标系的手眼标定矩阵。
- `T_end_to_base`：由 UR5 实时端口读取 TCP 位姿后转换得到的齐次矩阵。
- `P_base`：最终写入 `result/vision_result.txt` 的机器人基坐标系坐标。

---

## 2. 方法流程

### 2.1 RGB-D 采集

D405 采集模块位于 `Core/d405_camera.py`。采集流程包括：

1. 启动 1280 x 720、30 FPS 的 color 和 depth stream。
2. 根据 `D405_ALIGN_DEPTH_TO_COLOR` 决定是否将深度帧对齐到彩色帧。
3. 使用 RealSense `temporal_filter` 对深度帧做时间滤波。
4. 预热 `30` 帧，使 temporal filter 稳定。
5. 连续收集 `10` 帧深度图，并逐像素取中值，得到更稳定的 `uint16` 深度 PNG。
6. 保存彩色图与深度图到 `photos/`。

配置中对应参数为：

```python
D405_ALIGN_DEPTH_TO_COLOR = True
D405_WARMUP_FRAMES = 30
D405_COLLECT_FRAMES = 10
```

当前主流程调用 `capture_photos(PHOTO_DIR)`，函数默认值与上述配置一致。若后续修改预热或采集帧数，应同步保证主流程传参或函数默认值一致。

### 2.2 VLM 检测与 SAM2 分割

`Core/detect2sam.py` 负责从彩色图中得到目标 mask：

1. Qwen3-VL 读取彩色图并输出目标名称与归一化矩形框。
2. 解析 VLM 输出，将 0-1000 坐标映射到原始图像像素坐标。
3. 使用 SAM2.1 对每个矩形框生成目标 mask。
4. 对 mask 执行开运算、闭运算、小连通域去除和高斯平滑。
5. 保存检测框图、彩色 overlay 图和每个目标的单独 mask。

VLM 支持两种模式：

- `VL_MODE = "api"`：调用远程 API，使用 `.env` 中的 `PARATERA_API_KEY`。
- `VL_MODE = "local"`：加载本地 Qwen3-VL 模型。

Prompt 来源由 `VL_PROMPT_SOURCE` 控制：

- `fixed`：使用代码内置 prompt。
- `terminal`：程序启动后从终端读取本次运行的 prompt。

### 2.3 点云重建、滤波与重心估计

`Core/point_cloud_reconstruct.py` 使用深度图和 mask 重建每个目标的三维点云。深度值转换公式为：

```text
Z = depth_uint16 / RAW_PC_DEPTH_SCALE * DEPTH_Z_SCALE
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

其中：

- `RAW_PC_DEPTH_SCALE = 10000.0`，表示原始深度单位为 0.1 mm，转换到米时除以 10000。
- `DEPTH_Z_SCALE = 1.005`，表示来自 `F:\Project_V\Major\Handeye` 手眼标定优化结果的深度 Z 方向补偿系数。
- 当 `D405_ALIGN_DEPTH_TO_COLOR = True` 时，点云重建使用 `D405_ALIGNED_FX/FY/CX/CY`。
- 当 `D405_ALIGN_DEPTH_TO_COLOR = False` 时，使用原生深度相机内参 `D405_FX/FY/CX/CY`。

目标点云处理步骤：

1. 根据 SAM mask 提取目标区域有效深度像素。
2. 使用相机内参和深度补偿系数生成相机坐标系点云。
3. 使用体素降采样减少点数量。
4. 使用统计滤波剔除离群点。
5. 基于 PCA 估计目标 OBB，并以 OBB 最短边构造自适应滤波尺度。
6. 根据 `RAW_PC_FINE_FILTER_MODE` 使用 radius 或 DBSCAN 精滤。
7. 对滤波后的点云计算 OBB 和重心。
8. 将每个目标点云保存为 PLY 到 `cloud_point/<image_id>/`。

### 2.4 机器人坐标转换

`Core/robot_arm.py` 通过 UR 实时数据端口读取当前 TCP 位姿：

```python
ROBOT_IP = "169.168.1.100"
ROBOT_PORT = 30003
```

读取结果为 `[x, y, z, rx, ry, rz]`，单位为米和弧度。旋转向量通过 Rodrigues 公式转换为旋转矩阵，再组成齐次变换矩阵。

---

## 3. 项目结构

```text
Sys_Vision/
├── main.py                         # 主流程入口
├── config.py                       # 全局配置
├── Core/
│   ├── d405_camera.py              # D405 采集、深度对齐、temporal + median 滤波
│   ├── detect2sam.py               # Qwen3-VL 检测 + SAM2.1 分割
│   ├── point_cloud_reconstruct.py  # 点云重建、滤波、OBB 与重心估计
│   └── robot_arm.py                # UR5 TCP 位姿读取与矩阵转换
├── Tools/
│   ├── d405_intrinsics.py          # 读取 D405 对齐后内参
│   └── point_cloud_section_circle.py
│                                      # 实验性脚本，当前主流程未调用
├── photos/                         # 运行时生成：彩色图和深度图
├── procedure/
│   ├── VLM/                        # 运行时生成：VLM 检测框图
│   ├── SAM/                        # 运行时生成：SAM mask 和 overlay 图
│   └── point_cloud/                # 运行时生成：Open3D 点云截图
├── cloud_point/                    # 运行时生成：每个目标的 PLY 点云
├── report/                         # 运行时生成：2x2 调试报告图
├── result/
│   └── vision_result.txt           # 每次运行覆盖写出的最终结果
└── README.md
```

运行时目录通常不应提交到 Git，除非明确需要保留实验样例。

---

## 4. 配置说明

### 4.1 运行模式与输出

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `MODE` | `"debug"` | 调试模式会显示 Open3D 点云窗口并生成报告图；`experiment` 更适合批量实验 |
| `RESULT_DIR` | `result` | 最终结果目录 |
| `RESULT_FILE` | `vision_result.txt` | 每次运行覆盖写出的结果文件 |
| `VL_PROMPT_SOURCE` | `"fixed"` | 使用内置 prompt；设为 `terminal` 时从终端读取 |

### 4.2 手眼标定与机器人连接

| 参数 | 含义 |
| --- | --- |
| `HAND_EYE_MATRIX` | 相机坐标系到机械臂末端坐标系的 4x4 齐次矩阵 |
| `ROBOT_IP` | UR5 控制器 IP |
| `ROBOT_PORT` | UR 实时数据端口，默认 `30003` |

### 4.3 D405 相机与深度对齐

| 参数 | 含义 |
| --- | --- |
| `D405_ALIGN_DEPTH_TO_COLOR` | 是否把 depth stream 对齐到 color stream |
| `D405_FX/FY/CX/CY` | 原生深度/相机内参 |
| `D405_ALIGNED_FX/FY/CX/CY` | depth-to-color 对齐后的点云重建内参 |
| `D405_DIST_COEFFS` | 相机畸变参数 |
| `D405_ALIGNED_DIST_COEFFS` | 对齐模式下的畸变参数 |
| `D405_WARMUP_FRAMES` | temporal filter 预热帧数 |
| `D405_COLLECT_FRAMES` | 中值滤波收集帧数 |

对齐模式很重要：如果保存的是对齐到彩色图的深度图，SAM mask 与深度像素位于同一图像坐标系，点云重建也必须使用对齐后的内参。

### 4.4 深度尺度与点云滤波

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `RAW_PC_DEPTH_SCALE` | `10000.0` | 原始深度单位到米的比例，0.1 mm/unit |
| `DEPTH_Z_SCALE` | `1.005` | 手眼标定程序优化得到的 Z 方向深度补偿系数 |
| `RAW_PC_BUILD_BACKGROUND` | `True` | 是否在 debug 视图中构建背景点云 |
| `RAW_PC_VOXEL_SIZE` | `0.001` | 体素降采样尺寸，单位 m |
| `RAW_PC_STAT_NB_NEIGHBORS` | `20` | 统计滤波邻居数 |
| `RAW_PC_STAT_STD_RATIO` | `1.5` | 统计滤波标准差倍数 |
| `RAW_PC_FINE_FILTER_MODE` | `"dbscan"` | 精滤模式：`none`、`radius` 或 `dbscan` |
| `RAW_PC_FINE_FILTER_RATIO` | `0.1` | 精滤半径与 OBB 最短边的比例 |
| `RAW_PC_DBSCAN_MIN_POINTS` | `5` | DBSCAN 最小聚类点数 |
| `RAW_PC_DBSCAN_KEEP_TOP2` | `True` | 第二大簇足够大时是否保留 |
| `RAW_PC_DBSCAN_TOP2_RATIO` | `0.5` | 第二大簇保留阈值 |

`DEPTH_Z_SCALE` 的来源是 `F:\Project_V\Major\Handeye\results\eye_on_hand\depth_scale.txt`。它并不替代 D405 的物理 depth scale，而是在点云重建阶段对 Z 轴做实验标定补偿。

### 4.5 未纳入主流程的配置

`config.py` 中仍保留了 `POINT_CLOUD_SECTION_*`、`POINT_CLOUD_CIRCLE_*` 和 `POINT_CLOUD_KNOWN_CIRCLE_DIAMETER` 等参数。这些参数服务于 `Tools/point_cloud_section_circle.py` 的点云截面圆实验脚本，当前 `main.py` 主流程不调用该工具，也不依赖这些参数完成定位输出。

---

## 5. 依赖环境

建议使用项目专用 Python 环境。主要依赖包括：

| 依赖 | 用途 |
| --- | --- |
| `numpy` | 数值计算 |
| `opencv-python` | 图像读写、mask 后处理、报告图 |
| `Pillow` | 图像绘制和中文标签 |
| `scipy` | KDTree、空间查询和滤波 |
| `open3d` | 点云保存、显示和截图 |
| `torch` / `torchvision` | SAM2 和本地 VLM 推理 |
| `sam2` | SAM2.1 分割 |
| `openai` | OpenAI-compatible VL API 调用 |
| `modelscope` / `transformers` | 本地 Qwen3-VL 加载 |
| `pyrealsense2` | D405 相机访问 |
| `matplotlib` | 部分可视化能力 |

常用安装命令：

```powershell
pip install numpy opencv-python Pillow scipy open3d torch torchvision openai modelscope transformers accelerate pyrealsense2 matplotlib
pip install git+https://github.com/facebookresearch/sam2.git
```

如只使用远程 VLM API，可以不安装本地 Qwen3-VL 相关依赖。

---

## 6. API Key 与模型路径

远程 VLM API Key 通过 `.env` 提供：

```text
PARATERA_API_KEY=your_actual_api_key_here
```

项目启动时会自动读取 `.env`。当 `VL_MODE = "api"` 且 API Key 为空时，程序会报错退出。若某些离线工具只需要读取配置，可设置：

```powershell
$env:SYS_VISION_SKIP_VL_CONFIG_CHECK = "1"
```

本地模型路径在 `config.py` 中配置：

```python
VL_LOCAL_MODEL_PATH = r"F:\Models\Qwen3-vl"
SAM2_CKPT = r"F:\Models\SAM2\sam2.1_hiera_tiny.pt"
```

---

## 7. 运行方式

1. 确认 D405 已连接，并且 UR5 控制器网络可达。
2. 确认 `.env` 中已配置 API Key，或将 `VL_MODE` 设为可用的本地模型模式。
3. 检查 `config.py` 中的 `HAND_EYE_MATRIX`、`ROBOT_IP`、D405 内参和 `DEPTH_Z_SCALE`。
4. 运行主程序：

```powershell
python main.py
```

程序每次运行会先清空 `result/vision_result.txt`，随后执行一次完整流水线。如果成功定位目标，结果文件按如下格式写出：

```text
[透明试管瓶盖_1]: [0.1234, -0.0456, 0.2789]
```

如果未检测到目标、点云重建失败或机器人位姿读取失败，结果文件可能为空或只保留中间产物。

---

## 8. 输出文件

一次编号为 `0000` 的运行通常生成：

| 路径 | 内容 |
| --- | --- |
| `photos/d405_color_0000.jpg` | D405 彩色图 |
| `photos/d405_depth_0000.png` | temporal + median 滤波后的深度图 |
| `procedure/VLM/d405_color_0000_detected.jpg` | VLM 检测框图 |
| `procedure/SAM/d405_color_0000_overlay.jpg` | SAM mask 叠加图 |
| `procedure/SAM/d405_color_0000_XX_name_mask.png` | 单目标 mask |
| `cloud_point/0000/*.ply` | 每个目标滤波后的点云 |
| `procedure/point_cloud/pc3d_0000.png` | Open3D 点云截图 |
| `report/report_0000.jpg` | 2x2 调试报告图 |
| `result/vision_result.txt` | 最终基坐标系重心 |

---

## 9. 典型问题

### 9.1 深度图与 mask 错位

优先检查：

- `D405_ALIGN_DEPTH_TO_COLOR` 是否与采集时保存的深度图一致。
- 点云重建是否使用了对应的 `D405_ALIGNED_*` 或 `D405_*` 内参。
- 彩色图和深度图是否来自同一次编号的采集。

### 9.2 Z 方向存在系统偏差

检查：

- `RAW_PC_DEPTH_SCALE` 是否匹配 D405 保存的深度单位。
- `DEPTH_Z_SCALE` 是否来自当前手眼标定结果。
- `HAND_EYE_MATRIX` 是否与当前相机安装状态一致。

### 9.3 目标点云过少

可能原因：

- VLM 框没有覆盖目标有效深度区域。
- SAM mask 过小或被后处理过滤。
- 透明物体区域深度缺失严重。
- `RAW_PC_VOXEL_SIZE` 或 DBSCAN 参数过强。

### 9.4 无法输出基坐标系结果

检查：

- UR5 IP 和端口是否正确。
- 控制器实时端口 `30003` 是否可访问。
- 当前网络是否与机器人控制器处于同一网段。

---

## 10. 实验复现建议

为了保证结果可复现，每组实验应记录：

- Git commit hash。
- `config.py` 中的 `HAND_EYE_MATRIX`、`DEPTH_Z_SCALE` 和 D405 内参。
- D405 是否启用 depth-to-color 对齐。
- `photos/` 中对应编号的 RGB-D 原始输入。
- `procedure/SAM/` 中对应编号的 mask。
- `cloud_point/<image_id>/` 中保存的 PLY 点云。
- UR5 当前 TCP 位姿。
- `result/vision_result.txt` 输出。

其中 `DEPTH_Z_SCALE` 与 `HAND_EYE_MATRIX` 应被视为同一标定批次的结果；如果重新安装相机、调整焦距、改变深度对齐策略或重新标定手眼矩阵，应同步更新二者。
