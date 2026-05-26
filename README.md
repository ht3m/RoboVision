# Sys_Vision — 机械臂视觉定位系统

机械臂搭载 Intel RealSense D405 深度相机，利用 **VLM (Qwen3-VL) + SAM2.1** 检测并分割目标物体，通过点云重建计算物体在相机坐标系下的重心，最终结合**手眼标定**和**机械臂当前位姿**将重心坐标转换到**机器人基坐标系**。

---

## 目录结构

```
Sys_Vision/
├── main.py                        # 主程序入口
├── config.py                      # 全局配置 (模式、路径、参数、手眼矩阵)
├── Core/
│   ├── __init__.py
│   ├── d405_camera.py             # D405 相机拍照
│   ├── detect2sam.py              # VL 检测 + SAM2 分割
│   ├── point_cloud_reconstruct.py # 深度图→点云重建 + 滤波 + 重心计算
│   └── robot_arm.py               # 机械臂 TCP 通信 (获取末端位姿)
├── photos/                        # [运行时] 拍摄的彩色图 + 深度图
├── procedure/                     # [运行时] 中间产物
│   ├── VLM/                       #   VL 检测框图
│   ├── SAM/                       #   SAM2 分割 mask + 叠加图
│   └── point_cloud/               #   点云截图 (仅 debug 模式)
├── report/                        # [运行时] 2×2 报告图 (仅 debug 模式)
├── README.md
└── .gitignore
```

> `photos/`、`procedure/`、`report/` 均为运行时自动生成，已加入 `.gitignore`。

---

## 依赖环境

### Python 版本
- Python 3.10+

### 核心依赖

| 包名                    | 用途                           |
| ----------------------- | ------------------------------ |
| `numpy`                 | 数值计算                       |
| `opencv-python` (cv2)   | 图像处理                       |
| `Pillow`                | 图片读写                       |
| `scipy`                 | KDTree、空间查询               |
| `open3d`                | 点云处理与可视化               |
| `torch` + `torchvision` | SAM2 模型推理                  |
| `sam2`                  | SAM2.1 分割模型                |
| `openai`                | VL 远程 API 调用               |
| `modelscope`            | 本地 Qwen3-VL 模型加载         |
| `transformers`          | 本地 VL 模型 (BitsBytesConfig) |
| `pyrealsense2`          | Intel RealSense D405 SDK       |
| `matplotlib`            | 点云可视化 (debug 模式)        |

### 安装命令

满血（本地VLM模型）
```bash
pip install numpy opencv-python Pillow scipy open3d torch torchvision openai modelscope transformers accelerate pyrealsense2 matplotlib
```

半血（无本地VLM）
```bash
pip install numpy opencv-python Pillow scipy open3d torch torchvision openai pyrealsense2 matplotlib
```

SAM2 需要单独安装（可能安装失败）：

```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

---

## 配置说明 (`config.py`)

| 参数                      | 说明                                                                   |
| ------------------------- | ---------------------------------------------------------------------- |
| `MODE`                    | `"debug"` 调试模式（弹出窗口+存点云图+报告） / `"experiment"` 实验模式 |
| `VL_MODE`                 | `"api"` 远程 API / `"local"` 本地模型                                  |
| `HAND_EYE_MATRIX`         | **手眼标定齐次矩阵**（相机 → 机械臂末端），需要用户实测填入            |
| `ROBOT_IP` / `ROBOT_PORT` | 机械臂控制器 IP 和端口                                                 |
| `D405_*`                  | 相机内参 (fx, fy, cx, cy)                                              |
| `*OUTPUT_DIR`             | 各阶段输出目录                                                         |

> ⚠️ **重要**: 使用前必须将 `HAND_EYE_MATRIX` 替换为实际标定值，`ROBOT_IP` 改为实际机械臂 IP。

---

## 环境变量配置

VL 模型 API Key 通过环境变量传入。项目使用 `.env` 文件管理密钥，已加入 `.gitignore` 防止泄露。

1. 复制模板文件：

```bash
copy .env.example .env
```

2. 编辑 `.env`，填入真实 API Key：

```
PARATERA_API_KEY=your_actual_api_key_here
```

3. 启动程序时会自动校验：若 API Key 为空，程序报错并退出。

> 也可直接在终端设置（临时）：`set PARATERA_API_KEY=你的密钥`

---

## 使用方法

1. 激活 Python 环境
2. 确保 D405 相机已连接，机械臂控制器可达
3. 配置 `.env` 文件中的 API Key（参考上方"环境变量配置"）
4. 修改 `config.py` 中的 `HAND_EYE_MATRIX` 和 `ROBOT_IP`
5. 运行主程序：

```bash
python main.py
```

5. 终端提示 `>>>` 后输入 `S` 执行一次完整流水线，输入 `Q` 退出。

---

## 数据流程

```
终端输入 'S'
    │
    ▼
[Step 1] D405 拍照 ──→ photos/d405_color_XXXX.jpg
                 ──→ photos/d405_depth_XXXX.png
    │
    ▼
[Step 2] VLM 检测 ──→ procedure/VLM/d405_color_XXXX_detected.jpg
         SAM2 分割 ──→ procedure/SAM/d405_color_XXXX_overlay.jpg
                    ──→ procedure/SAM/d405_color_XXXX_XX_name_mask.png
    │
    ▼
[Step 3] 深度图 + mask → 点云重建 + 滤波 + PCA 重心
         (debug 模式: 弹出 Open3D 窗口 + 保存点云截图到 procedure/point_cloud/)
    │
    ▼
[Step 4] 读取机械臂当前末端位姿 (TCP)
         手眼矩阵: 相机重心 → 末端坐标
         末端矩阵: 末端坐标 → 基坐标
    │
    ▼
[输出] 终端打印每个物体的基坐标系重心
        (debug 模式: 生成 2×2 报告图 → report/report_XXXX.jpg)
```

---

## 坐标系变换链

```
P_base = T_end→base  ×  T_cam→end  ×  P_cam
          ^                  ^            ^
     机械臂末端位姿      手眼标定矩阵    相机坐标系重心
     (robot_arm.py)   (config.py)     (point_cloud_reconstruct.py)
```

---

## 注意事项

1. **编号递增**: 每次拍照自动检测已有文件，编号持续递增，不会覆盖旧文件。
2. **机械臂连接失败**: 如果无法连接机械臂获取末端位姿，程序会打印警告并回退到**仅输出相机坐标系重心**。
3. **模型文件**: 本地 Qwen3-VL 模型和 SAM2.1 权重需自行下载，并在 `config.py` 中指定正确路径。
4. **字体**: Linux 系统可能需要安装中文字体才能正常显示标签。

---

## 开发适配说明

本项目由 `BioRobo_Vision` 重构而来，核心差异：
- 相机从独立拍照变为搭载在机械臂上
- 增加了**坐标变换**模块（手眼标定矩阵 + 机械臂实时位姿）
- 输出目标从相机坐标系重心变为**基坐标系重心**
- 新增 `config.py` 模式开关控制可视化行为
- 目录结构与文件命名规范化
# 当前运行方式补充

- `python main.py` 会直接执行一次完整流程，不再需要输入 `S` 启动或 `Q` 退出。
- `config.py` 中 `VL_PROMPT_SOURCE = "fixed"` 时使用代码内置 prompt。
- `config.py` 中 `VL_PROMPT_SOURCE = "terminal"` 时，程序启动后会在终端提示输入本次 VL prompt。
- 每次运行都会覆写 `result/vision_result.txt`。成功定位的每行格式为 `[物品名字及编号]：[x, y, z]`；如果没有可用检测结果，文件保持空白。
