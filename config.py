"""
=============================================================================
  Sys_Vision 全局配置文件
  ============================================================================
  机械臂视觉引导系统 — 相机拍照 → VL检测 → SAM2分割 → 点云重建 → 坐标变换

  按流水线顺序组织:
    Step 0 → 运行模式 & 手眼标定 & 机器人连接
    Step 1 → VL 检测 (Qwen3-VL)
    Step 2 → SAM2 分割
    Step 3 → 点云重建与滤波 (原始深度)
    Step 4 → 可视化 & 报告

  注意: 移除了 Beta/ 文件夹中 ClearGrasp 深度补全的专属参数。
=============================================================================
"""

import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_project_env() -> str:
    """Load project .env before reading environment-backed config."""
    env_path = os.path.join(PROJECT_ROOT, ".env")

    try:
        from dotenv import load_dotenv
    except ImportError:
        if not os.path.exists(env_path):
            return "missing"

        with open(env_path, "r", encoding="utf-8-sig") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        return "fallback"

    load_dotenv(env_path)
    return "python-dotenv"


_ENV_LOAD_MODE = _load_project_env()

# ============================================================================
# 0. 运行模式
# ============================================================================
# MODE = "debug":      显示点云可视化窗口 (滤波前+滤波后) + 生成 2x2 报告图
# MODE = "experiment": 无可视化窗口，无报告图，仅终端输出重心坐标
MODE = "debug"

# ============================================================================
# 1. 手眼标定 — 相机相对于机器人末端的齐次变换矩阵 (4x4)
# --------------------------------------------------------------------------
# 矩阵含义: P_end = T_cam_to_end @ P_cam
#   即: 将相机坐标系下的点变换到机械臂末端坐标系
#   ！！！请根据实际标定结果填入！！！
# ============================================================================
HAND_EYE_MATRIX = np.array([
    [0.351230, -0.714777, 0.604677, -0.052565],
    [-0.053744, 0.627255, 0.776953, -0.036955],
    [-0.934741, -0.305386, 0.181896, 0.067569],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# ============================================================================
# 2. 机器人连接 (UR5 实时端口)
# ============================================================================
ROBOT_IP = "169.168.1.100"     # 机器人静态 IP（请根据实际修改）
ROBOT_PORT = 30003              # UR 实时数据端口

# ============================================================================
# 3. 目录路径
# ============================================================================
PHOTO_DIR = os.path.join(PROJECT_ROOT, "photos")                    # 拍照保存
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "procedure", "VLM")         # VL 检测中间产物
SAM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "procedure", "SAM")     # SAM2 分割中间产物
CLOUD_POINT_DIR = os.path.join(PROJECT_ROOT, "procedure", "point_cloud")  # 点云截图中间产物
OBJECT_POINT_CLOUD_DIR = os.path.join(PROJECT_ROOT, "cloud_point")  # Per-shot object PLY point clouds.
REPORT_DIR = os.path.join(PROJECT_ROOT, "report")                   # 2x2 报告图

# ============================================================================
# 3.1 Object point-cloud section circle tool
# ============================================================================
POINT_CLOUD_SECTION_LOW_RATIO = 0.01
POINT_CLOUD_SECTION_HIGH_RATIO = 0.99
POINT_CLOUD_CIRCLE_COVER_RATIO = 0.90

# Diameter in meters. Set <= 0 to estimate diameter from POINT_CLOUD_CIRCLE_COVER_RATIO.
POINT_CLOUD_KNOWN_CIRCLE_DIAMETER = 0.0

# Limits affect visualization/candidate search only; circle scoring still uses all points.
POINT_CLOUD_SECTION_POINT_SAMPLE_LIMIT = 25000
POINT_CLOUD_FIXED_CIRCLE_PAIR_SAMPLE_LIMIT = 2500

# ============================================================================
# 4. VL 模型 (Qwen3-VL) 配置
# ============================================================================
# VL_MODE: "api" 使用远程 API (30B), "local" 使用本地模型 (8B)
VL_MODE = "api"

# ── API 模式 ──
VL_API_KEY = os.environ.get("PARATERA_API_KEY")
VL_API_URL = "https://llmapi.paratera.com/v1"
VL_MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"

# 启动校验: API 模式下密钥为空则报错退出
if (
    VL_MODE == "api"
    and (VL_API_KEY is None or VL_API_KEY.strip() == "")
    and os.environ.get("SYS_VISION_SKIP_VL_CONFIG_CHECK") != "1"
):
    print("=" * 60)
    print("  [致命错误] PARATERA_API_KEY 环境变量未设置！")
    print("  请创建 .env 文件并设置密钥，或执行:")
    print("    set PARATERA_API_KEY=你的密钥")
    print("  程序退出。")
    print("=" * 60)
    if _ENV_LOAD_MODE == "missing":
        print("  Project .env file was not found.")
    elif _ENV_LOAD_MODE == "fallback":
        print("  python-dotenv is not installed; built-in .env fallback was used but no valid key was found.")
    exit(1)

# ── 本地模式 ──
VL_LOCAL_MODEL_PATH = r"F:\Models\Qwen3-vl"
VL_LOCAL_DEVICE = "cuda"
VL_LOCAL_TORCH_DTYPE = "auto"        # "float16" / "bfloat16" / "auto"
VL_LOCAL_MAX_NEW_TOKENS = 4096
# flash_attention_2 加速推理并节省显存，多图和视频场景尤其推荐
# 可选: "flash_attention_2" / "sdpa" / "eager" / None(默认)
VL_LOCAL_ATTN_IMPLEMENTATION = "sdpa"
# 4-bit 量化 (bitsandbytes NF4)，大幅降低显存占用并加速推理
# 设为 True 启用，设为 False 关闭
VL_LOCAL_LOAD_IN_4BIT = True

# ── 精简模式 ──
# True: 只打框，不做场景描述和物品清单对话，减少 token 消耗
# False: 完整对话模式，包含场景描述 + 物品清单 + 矩形框
VL_LEAN_MODE = True

# ── 通用参数 ──
VL_TEMPERATURE = 0.1
VL_MAX_TOKENS = 4096

# ============================================================================
# 5. VL 检测参数
# ============================================================================
VL_MIN_AREA = 3000

# ============================================================================
# 6. SAM2 分割模型配置
# ============================================================================
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
SAM2_CKPT = r"F:\Models\SAM2\sam2.1_hiera_tiny.pt"
SAM2_DEVICE = "cuda" if os.environ.get("CPU_ONLY") is None else "cpu"

# ============================================================================
# 7. SAM2 mask 后处理参数
# ============================================================================
SAM_MASK_OPEN_ITERS = 2
SAM_MASK_CLOSE_ITERS = 1
SAM_MASK_SMOOTH_SIGMA = 3.0
SAM_MASK_MIN_COMPONENT = 500

# ============================================================================
# 8. D405 相机内参
# ============================================================================
D405_FX = 663.6542
D405_FY = 663.6542
D405_CX = 619.5816
D405_CY = 363.5091
D405_DIST_COEFFS = [0.0, 0.0, 0.0, 0.0, 0.0]

# ============================================================================
# 9. D405 拍照参数
# ============================================================================
D405_WARMUP_FRAMES = 30      # Temporal Filter 预热帧数
D405_COLLECT_FRAMES = 10     # 中值滤波收集帧数

# ============================================================================
# 10. 原始深度点云重建与滤波参数
# --------------------------------------------------------------------------
# 原始深度中透明物体区域噪声大、深度值大量丢失，需要保守的滤波策略。
# ============================================================================
RAW_PC_DEPTH_SCALE = 10000.0             # 0.1 mm → m (depth_scale=0.0001 m/unit)
RAW_PC_VOXEL_SIZE = 0.001                # 体素降采样 (m), 8mm
RAW_PC_STAT_NB_NEIGHBORS = 20            # 统计滤波邻域点数
RAW_PC_STAT_STD_RATIO = 1.5              # 统计滤波标准差倍数
RAW_PC_FINE_FILTER_MODE = "dbscan"       # 精滤波: "none" / "radius" / "dbscan"
RAW_PC_FINE_FILTER_RATIO = 0.1           # 滤波半径 = OBB最短边 × 此比例
RAW_PC_RADIUS_FILTER_MIN_NEIGHBORS = 2   # 半径滤波最小邻点数
RAW_PC_DBSCAN_MIN_POINTS = 5             # DBSCAN 最小聚类点数
RAW_PC_DBSCAN_KEEP_TOP2 = True           # 第二大簇点数达标时也保留
RAW_PC_DBSCAN_TOP2_RATIO = 0.5           # 第二大簇保留阈值比例

# ============================================================================
# 11. 可视化调色板
# ============================================================================
VIS_PALETTE = [
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
    (255, 180, 0), (255, 80, 255), (0, 220, 220),
    (220, 220, 0), (180, 80, 255), (80, 255, 180),
    (255, 140, 80),
]

# ============================================================================
# 12. 深度渲染参数
# ============================================================================
DEPTH_ALPHA = 0.03
DEPTH_COLORMAP = None

# ============================================================================
# 13. 中文显示字体
# ============================================================================
FONT_PATHS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/arial.ttf",
]

# Final result output.
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
RESULT_FILE = "vision_result.txt"

# "fixed" uses the prompt defined in code.
# "terminal" reads the prompt from stdin after main.py starts.
VL_PROMPT_SOURCE = "fixed"
