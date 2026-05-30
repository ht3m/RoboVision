"""
detect2sam.py — VL 检测 + SAM2 分割模块
=======================================
适配 Sys_Vision 项目，输出目录改为 procedure/VLM 和 procedure/SAM。
与 BioRobo_Vision/Core/detect2sam.py 功能完全一致：
  - VL 模型 (Qwen3-VL) → 识别物品 + 输出矩形框
  - SAM2.1 Tiny → 对每个矩形框生成精细化分割 mask
"""

import os
import re
import time
import base64
import traceback
import numpy as np
from io import BytesIO
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 确保项目根目录在 sys.path 中
import sys as _sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

from config import (
    VL_MODE, VL_LEAN_MODE,
    VL_API_KEY, VL_API_URL, VL_MODEL_NAME,
    VL_LOCAL_MODEL_PATH, VL_LOCAL_DEVICE, VL_LOCAL_TORCH_DTYPE, VL_LOCAL_MAX_NEW_TOKENS,
    VL_LOCAL_ATTN_IMPLEMENTATION, VL_LOCAL_LOAD_IN_4BIT,
    SAM2_CONFIG, SAM2_CKPT, SAM2_DEVICE as DEVICE,
    OUTPUT_DIR, SAM_OUTPUT_DIR,
    VL_MIN_AREA, VL_TEMPERATURE, VL_MAX_TOKENS,
    SAM_MASK_OPEN_ITERS, SAM_MASK_CLOSE_ITERS,
    SAM_MASK_SMOOTH_SIGMA, SAM_MASK_MIN_COMPONENT,
    VIS_PALETTE as COLOR_PALETTE, FONT_PATHS,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAM_OUTPUT_DIR, exist_ok=True)

# API 客户端 (仅 api 模式使用)
if VL_MODE == "api":
    client = OpenAI(api_key=VL_API_KEY, base_url=VL_API_URL)

SYSTEM_PROMPT = (
    "你是一个机器人视觉助手，正在一个用于生物实验的实验室工作台前。"
    "你的任务是仔细观察桌面上的物品，并用中文描述你看到了什么。"
)

USER_PROMPT_TEMPLATE = (
    "请仔细观察这张图片。这是一个机器人进行生物实验的实验室桌面。\n\n"
    "你只需要关注以下三类物品：\n"
    "- 透明试管/管子（可能装有液体）\n"
    "- 透明瓶子/试剂瓶\n"
    "- 培养皿\n\n"
    "请分两部分回答：\n\n"
    "【第一部分：物品清单】\n"
    "列出你在这张图片中看到了哪些透明试管、透明瓶子和培养皿，包括数量和位置。\n\n"
    "【第二部分：矩形框坐标】\n"
    "请为图片中每一个你看到的透明试管、透明瓶子、培养皿分别标注矩形框坐标，每个物品单独一行。\n"
    "  注意：必须使用 0 到 1000 的归一化坐标（千分比），输出格式必须严格为：\n"
    "  物品名称: [xmin, ymin, xmax, ymax]\n"
    "  其中 xmin 是框的左侧，ymin 是框的顶部，xmax 是框的右侧，ymax 是框的底部。\n"
    "  多个同类物品必须逐个分别标注，不要合并。\n\n"
    "请用中文回答。"
)

LEAN_USER_PROMPT_TEMPLATE = (
    "请为图片中每一个透明试管、透明瓶子和培养皿标注矩形框坐标。\n\n"
    "要求：\n"
    "- 使用 0 到 1000 的归一化坐标（千分比）\n"
    "- 格式：物品名称 [xmin, ymin, xmax, ymax]（不要冒号）\n"
    "- 每个物品单独一行\n"
    "- 不要添加任何其他描述文字\n\n"
    "示例输出格式：\n"
    "透明试剂瓶 [200, 150, 400, 500]\n"
    "培养皿 [500, 600, 800, 900]"
)


# ============================================================================
# VL 模型检测
# ============================================================================

_local_vl_model = None
_local_vl_processor = None


def _load_local_vl_model():
    """懒加载本地 Qwen3-VL 模型（首次调用时加载）"""
    global _local_vl_model, _local_vl_processor
    if _local_vl_model is not None:
        return

    from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

    attn_kwargs = {}
    if VL_LOCAL_ATTN_IMPLEMENTATION:
        attn_kwargs["attn_implementation"] = VL_LOCAL_ATTN_IMPLEMENTATION

    if VL_LOCAL_LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print(f"\n{'=' * 60}")
    print(f"  加载本地 VL 模型: {VL_LOCAL_MODEL_PATH}")
    print(f"    设备: {VL_LOCAL_DEVICE}")
    print(f"    dtype: {VL_LOCAL_TORCH_DTYPE}")
    print(f"    4-bit 量化: {'✅ 已启用' if VL_LOCAL_LOAD_IN_4BIT else '❌ 未启用'}")
    if VL_LOCAL_ATTN_IMPLEMENTATION:
        print(f"    attn_implementation: {VL_LOCAL_ATTN_IMPLEMENTATION}")
    print(f"{'=' * 60}")

    if VL_LOCAL_LOAD_IN_4BIT:
        _local_vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            VL_LOCAL_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            **attn_kwargs,
        )
    else:
        if VL_LOCAL_TORCH_DTYPE == "float16":
            torch_dtype = torch.float16
        elif VL_LOCAL_TORCH_DTYPE == "bfloat16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = "auto"

        _local_vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
            VL_LOCAL_MODEL_PATH,
            dtype=torch_dtype,
            device_map=VL_LOCAL_DEVICE if VL_LOCAL_DEVICE != "cpu" else None,
            **attn_kwargs,
        )
        if VL_LOCAL_DEVICE == "cpu":
            _local_vl_model = _local_vl_model.to("cpu")

    _local_vl_processor = AutoProcessor.from_pretrained(VL_LOCAL_MODEL_PATH)
    print("  本地 VL 模型加载完成!\n")


def encode_image(image: Image.Image) -> str:
    """编码图片为 base64 data URI"""
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image(img_path: str, runtime_prompt: str | None = None) -> str:
    """读取图片并调用 VL 模型"""
    print(f"  加载图片: {img_path}")
    img = Image.open(img_path).convert("RGB")
    print(f"  原始尺寸: {img.size[0]} x {img.size[1]}")
    print(f"  精简模式: {'开启' if VL_LEAN_MODE else '关闭'}")

    user_prompt = runtime_prompt if runtime_prompt is not None else (
        LEAN_USER_PROMPT_TEMPLATE if VL_LEAN_MODE else USER_PROMPT_TEMPLATE
    )

    if VL_MODE == "local":
        return _describe_image_local(img, user_prompt)
    else:
        return _describe_image_api(img, user_prompt)


def _describe_image_api(img: Image.Image, user_prompt: str) -> str:
    """通过远程 API 调用 VL 模型"""
    img_b64 = encode_image(img)
    print(f"  调用 API 模型: {VL_MODEL_NAME} ...")
    resp = client.chat.completions.create(
        model=VL_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        temperature=VL_TEMPERATURE,
        max_tokens=VL_MAX_TOKENS,
    )
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    if usage:
        print(f"  Token 用量: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
    return text


def _describe_image_local(img: Image.Image, user_prompt: str) -> str:
    """通过本地 Qwen3-VL 8B 模型推理"""
    _load_local_vl_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    print(f"  调用本地模型: {VL_LOCAL_MODEL_PATH} ...")
    inputs = _local_vl_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(_local_vl_model.device)

    with torch.no_grad():
        generated_ids = _local_vl_model.generate(
            **inputs,
            max_new_tokens=VL_LOCAL_MAX_NEW_TOKENS,
            temperature=VL_TEMPERATURE,
            do_sample=True if VL_TEMPERATURE > 0 else False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = _local_vl_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    prompt_tokens = inputs.input_ids.shape[1]
    completion_tokens = generated_ids_trimmed[0].shape[0]
    print(f"  Token 用量: prompt≈{prompt_tokens}, completion≈{completion_tokens}")
    return output_text


# ============================================================================
# 矩形框解析
# ============================================================================

_LEAN_BOX_RE = re.compile(r'^(.+?)\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]')


def parse_boxes(text: str, orig_w: int, orig_h: int, min_area: int = VL_MIN_AREA) -> List[Dict]:
    """从 VL 模型输出中解析矩形框坐标"""
    boxes = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        m = _LEAN_BOX_RE.match(line)
        if m:
            name = m.group(1).strip()
            name = re.sub(r'^[\d\.\-\s]+', '', name)
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = map(int, m.groups()[1:])
        elif ':' in line or '：' in line:
            parts = re.split(r'[:：]', line, maxsplit=1)
            name = parts[0].strip()
            name = re.sub(r'^[\d\.\-\s]+', '', name)
            coords_str = parts[1]
            nums = re.findall(r'\d+', coords_str)
            if len(nums) < 4:
                continue
            xmin_norm, ymin_norm, xmax_norm, ymax_norm = map(int, nums[:4])
        else:
            continue

        if not name:
            continue

        if xmax_norm > 1000 or ymax_norm > 1000:
            print(f"    [警告] {name} 输出了非归一化坐标，跳过")
            continue

        x1 = int((xmin_norm / 1000.0) * orig_w)
        y1 = int((ymin_norm / 1000.0) * orig_h)
        x2 = int((xmax_norm / 1000.0) * orig_w)
        y2 = int((ymax_norm / 1000.0) * orig_h)

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < min_area:
            print(f"    [过滤] {name} 面积 {area}px < {min_area}px，跳过")
            continue

        boxes.append({"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        print(f"    解析到 {name}: [{xmin_norm},{ymin_norm},{xmax_norm},{ymax_norm}] "
              f"-> ({x1},{y1})-({x2},{y2})  面积={area}px")

    return boxes


# ============================================================================
# SAM2 分割
# ============================================================================

class SAM2Segmenter:
    """SAM2.1 分割器"""

    def __init__(self, config_path: str, ckpt_path: str, device: str = "cuda"):
        print(f"\n{'=' * 60}")
        print(f"  加载 SAM2.1 模型...")
        print(f"    配置: {config_path}")
        print(f"    权重: {ckpt_path}")
        print(f"    设备: {device}")
        print(f"{'=' * 60}")

        self.device = device
        self.sam_model = build_sam2(config_path, ckpt_path, device=device)
        self.predictor = SAM2ImagePredictor(self.sam_model)
        print("  SAM2.1 模型加载完成!\n")

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)

    def segment_box(self, box: Dict, multimask_output: bool = False) -> Tuple[np.ndarray, float]:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, logits = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=multimask_output,
        )
        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])

    def segment_all_boxes(self, boxes: List[Dict]) -> List[Dict]:
        results = []
        for i, box in enumerate(boxes):
            name = box["name"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            print(f"    SAM2 分割 [{i+1}/{len(boxes)}] {name} ({x1},{y1})-({x2},{y2})", end="")
            try:
                mask, score = self.segment_box(box)
                print(f" → score={score:.3f}")
                mask = clean_mask(mask)
                box["mask"] = mask
                box["score"] = float(score)
            except Exception as e:
                print(f" → 失败: {e}")
                box["mask"] = None
                box["score"] = 0.0
            results.append(box)
        return results


# ============================================================================
# Mask 后处理
# ============================================================================

import cv2

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """对 SAM2 输出的二值 mask 做后处理"""
    mask_u8 = mask.astype(np.uint8)
    original_px = mask_u8.sum()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=SAM_MASK_OPEN_ITERS)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=SAM_MASK_CLOSE_ITERS)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for label_id in range(1, n_labels):
        if stats[label_id, cv2.CC_STAT_AREA] < SAM_MASK_MIN_COMPONENT:
            mask_u8[labels == label_id] = 0

    if SAM_MASK_SMOOTH_SIGMA > 0 and mask_u8.sum() > 0:
        mask_f32 = mask_u8.astype(np.float32)
        mask_f32 = cv2.GaussianBlur(mask_f32, (0, 0), sigmaX=SAM_MASK_SMOOTH_SIGMA,
                                     borderType=cv2.BORDER_REPLICATE)
        mask_u8 = (mask_f32 > 0.5).astype(np.uint8)

    cleaned_px = mask_u8.sum()
    print(f"    [后处理] mask 像素: {original_px} → {cleaned_px} "
          f"(去除 {original_px - cleaned_px} px)")

    return mask_u8.astype(bool)


# ============================================================================
# 可视化
# ============================================================================

def _color_to_rgb255(color: Tuple[int, int, int] | Tuple[float, float, float]) -> tuple[int, int, int]:
    """Accept either 0-255 RGB or 0-1 RGB and return integer RGB."""
    values = np.asarray(color, dtype=np.float32)
    if float(np.max(values)) <= 1.0:
        values = values * 255.0
    values = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return int(values[0]), int(values[1]), int(values[2])


def _color_to_hex(color: Tuple[int, int, int] | Tuple[float, float, float]) -> str:
    r, g, b = _color_to_rgb255(color)
    return f"#{r:02X}{g:02X}{b:02X}"


def apply_colored_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
                       alpha: float = 0.5) -> np.ndarray:
    """在图片上叠加彩色半透明 mask"""
    overlayed = image.copy().astype(np.float32)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    color_arr = np.array(_color_to_rgb255(color), dtype=np.float32)
    overlayed = overlayed * (1.0 - alpha * mask_3ch) + color_arr[None, None, :] * alpha * mask_3ch
    return np.clip(overlayed, 0, 255).astype(np.uint8)


def save_results(image_np: np.ndarray, boxes: List[Dict], name_prefix: str):
    """保存分割结果到 SAM_OUTPUT_DIR"""
    h, w = image_np.shape[:2]

    # 叠加图
    overlay = image_np.copy()
    for i, box in enumerate(boxes):
        if box.get("mask") is None:
            continue
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        overlay = apply_colored_mask(overlay, box["mask"], color, alpha=0.45)

    overlay_pil = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay_pil)

    font = None
    font_size = max(18, w // 55)
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue

    line_w = max(2, w // 350)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        color_rgb = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        color_hex = _color_to_hex(color_rgb)
        draw.rectangle([x1, y1, x2, y2], outline=color_hex, width=line_w)

        score = box.get("score", 0)
        label = f"{box['name']} ({score:.2f})"
        if font:
            tb = draw.textbbox((0, 0), label, font=font)
        else:
            tb = draw.textbbox((0, 0), label)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        lx, ly = x1, y1 - th - 6
        if ly < 0:
            ly = y1 + 4
        draw.rectangle([lx, ly, lx + tw + 8, ly + th + 6], fill=color_hex)
        kwargs = {"xy": (lx + 4, ly + 2), "text": label, "fill": "white"}
        if font:
            kwargs["font"] = font
        draw.text(**kwargs)

    overlay_path = os.path.join(SAM_OUTPUT_DIR, f"{name_prefix}_overlay.jpg")
    overlay_pil.save(overlay_path, "JPEG", quality=92)
    print(f"    已保存叠加图: {overlay_path}")

    # 单独 mask
    for i, box in enumerate(boxes):
        if box.get("mask") is None:
            continue
        safe_name = re.sub(r'[\\/:*?"<>|]', '_', box["name"])
        mask_img = Image.fromarray((box["mask"] * 255).astype(np.uint8), mode="L")
        mask_path = os.path.join(SAM_OUTPUT_DIR, f"{name_prefix}_{i:02d}_{safe_name}_mask.png")
        mask_img.save(mask_path, "PNG")
        print(f"    已保存 mask: {mask_path}")


# ============================================================================
# 一体化处理函数 (供 main.py 调用)
# ============================================================================

def run_vl_sam(photo_number: int | str, runtime_prompt: str | None = None) -> Optional[List[Dict]]:
    """对单张照片执行 VL 检测 + SAM2 分割的完整流程。

    Args:
        photo_number: 照片编号 (如 "0000" 或 0)

    Returns:
        带 mask 的 boxes 列表，失败返回 None
    """
    from config import PHOTO_DIR

    number_str = f"{int(photo_number):04d}"
    img_file = f"d405_color_{number_str}.jpg"
    img_path = os.path.join(PHOTO_DIR, img_file)

    if not os.path.exists(img_path):
        print(f"\n  [错误] 图片不存在: {img_path}")
        return None

    print(f"\n{'=' * 60}")
    print(f"  [VL 检测 + SAM2 分割] → {img_file}")
    print(f"{'=' * 60}")

    # Step 1: VL 检测
    t_vl_start = time.time()
    try:
        vl_result = describe_image(img_path, runtime_prompt=runtime_prompt)
    except Exception as e:
        print(f"  [错误] VL 调用失败: {e}")
        traceback.print_exc()
        return None

    print(f"\n  {'─' * 50}")
    print(f"  模型输出:")
    print(f"  {'─' * 50}")
    for line in vl_result.strip().split('\n'):
        print(f"  {line}")
    print(f"  {'─' * 50}")

    orig_img = Image.open(img_path).convert("RGB")
    w, h = orig_img.size
    boxes = parse_boxes(vl_result, w, h)
    t_vl = time.time() - t_vl_start

    if not boxes:
        print("\n  [警告] 未能解析到矩形框，跳过 SAM2 分割")
        return []

    print(f"\n  检测到 {len(boxes)} 个物品 ({t_vl:.1f}s)，开始 SAM2 分割...")

    # 保存 VL 检测框图
    draw_img = orig_img.copy()
    draw_det = ImageDraw.Draw(draw_img)

    font_det = None
    font_size_det = max(18, w // 55)
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                font_det = ImageFont.truetype(fp, font_size_det)
                break
            except Exception:
                continue

    line_w_det = max(2, w // 350)
    for i, box in enumerate(boxes):
        color_rgb = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        color_hex = _color_to_hex(color_rgb)
        draw_det.rectangle([box["x1"], box["y1"], box["x2"], box["y2"]],
                           outline=color_hex, width=line_w_det)
        if font_det:
            draw_det.text((box["x1"], box["y1"] - font_size_det - 4),
                          box["name"], fill=color_hex, font=font_det)
        else:
            draw_det.text((box["x1"], box["y1"] - 14), box["name"], fill=color_hex)

    name_prefix = f"d405_color_{number_str}"
    det_path = os.path.join(OUTPUT_DIR, f"{name_prefix}_detected.jpg")
    draw_img.save(det_path, "JPEG", quality=92)
    print(f"  已保存检测框图: {det_path}")

    # Step 2: 加载 SAM2
    segmenter = SAM2Segmenter(SAM2_CONFIG, SAM2_CKPT, DEVICE)

    # Step 3: SAM2 分割
    image_np = np.array(orig_img)
    segmenter.set_image(image_np)
    t_sam_start = time.time()
    boxes_with_masks = segmenter.segment_all_boxes(boxes)
    t_sam = time.time() - t_sam_start

    # Step 4: 保存结果
    print(f"\n  保存分割结果...")
    save_results(image_np, boxes_with_masks, name_prefix)

    # 摘要
    print(f"\n  {'─' * 50}")
    print(f"  {img_file} 分割摘要:")
    for i, box in enumerate(boxes_with_masks):
        name = box["name"]
        score = box.get("score", 0)
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        status = "✓" if box.get("mask") is not None else "✗"
        print(f"  {status} [{i}] {name}: box=({x1},{y1})-({x2},{y2})  score={score:.3f}")
    print(f"  ⏱ VL: {t_vl:.1f}s | SAM2: {t_sam:.1f}s ({len(boxes)} 框)")
    print(f"  {'─' * 50}")

    return boxes_with_masks
