"""
Read saved object point clouds by image ID, project a Z section to Z=0,
and draw a 2D circle on the projected points.

Usage:
  python Tools/point_cloud_section_circle.py 0000
  python Tools/point_cloud_section_circle.py 0000 --known-diameter 0.045
  python Tools/point_cloud_section_circle.py 0000 --known-diameter-mm 45
  python Tools/point_cloud_section_circle.py
"""

import argparse
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("SYS_VISION_SKIP_VL_CONFIG_CHECK", "1")

try:
    from config import (
        FONT_PATHS,
        OBJECT_POINT_CLOUD_DIR,
        POINT_CLOUD_CIRCLE_COVER_RATIO,
        POINT_CLOUD_FIXED_CIRCLE_PAIR_SAMPLE_LIMIT,
        POINT_CLOUD_KNOWN_CIRCLE_DIAMETER,
        POINT_CLOUD_SECTION_HIGH_RATIO,
        POINT_CLOUD_SECTION_LOW_RATIO,
        POINT_CLOUD_SECTION_POINT_SAMPLE_LIMIT,
    )
except BaseException:
    FONT_PATHS = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    OBJECT_POINT_CLOUD_DIR = "cloud_point"
    POINT_CLOUD_SECTION_LOW_RATIO = 0.01
    POINT_CLOUD_SECTION_HIGH_RATIO = 0.99
    POINT_CLOUD_CIRCLE_COVER_RATIO = 0.90
    POINT_CLOUD_KNOWN_CIRCLE_DIAMETER = 0.0
    POINT_CLOUD_SECTION_POINT_SAMPLE_LIMIT = 25000
    POINT_CLOUD_FIXED_CIRCLE_PAIR_SAMPLE_LIMIT = 2500

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


TILE_WIDTH = 820
TILE_HEIGHT = 560
GRID_COLUMNS = 2
FIXED_CIRCLE_CANDIDATE_LIMIT = 300000

POINT_CLOUD_PATH = os.path.join(PROJECT_DIR, OBJECT_POINT_CLOUD_DIR)


@dataclass
class ObjectResult:
    number: str
    path: str
    name: str
    index: int
    total_points: int
    selected_points: np.ndarray
    z_min: float
    z_max: float
    z_low: float
    z_high: float
    circle_center: np.ndarray
    circle_radius: float
    circle_cover_count: int
    circle_mode: str


def normalize_number(value: str) -> str:
    value = value.strip()
    return value.zfill(4) if value.isdigit() else value


def validate_args(z_low_ratio: float,
                  z_high_ratio: float,
                  cover_ratio: float,
                  known_diameter: float) -> None:
    if not 0.0 <= z_low_ratio < z_high_ratio <= 1.0:
        raise ValueError("Z section ratios must satisfy 0 <= low < high <= 1")
    if not 0.0 < cover_ratio <= 1.0:
        raise ValueError("Circle cover ratio must satisfy 0 < cover <= 1")
    if known_diameter < 0:
        raise ValueError("Known circle diameter must be >= 0")


def find_point_cloud_files(number: str) -> list[str]:
    if not os.path.isdir(POINT_CLOUD_PATH):
        return []

    shot_dir = os.path.join(POINT_CLOUD_PATH, number)
    files = []
    for directory in [shot_dir, POINT_CLOUD_PATH]:
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            lower = name.lower()
            if not lower.endswith(".ply"):
                continue
            if directory == shot_dir:
                files.append(os.path.join(directory, name))
            elif (
                name.startswith(f"raw_pc_{number}_object_")
                or name.startswith(f"object_pc_{number}_object_")
            ):
                files.append(os.path.join(directory, name))

    def sort_key(path: str):
        name = os.path.basename(path)
        match = re.search(r"_object_(\d+)_", name)
        return int(match.group(1)) if match else 999999, name

    return sorted(set(files), key=sort_key)


def parse_object_name(path: str) -> tuple[int, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"_object_(\d+)_(.+)$", stem)
    if not match:
        return 0, stem
    return int(match.group(1)), match.group(2)


def geometric_median(points: np.ndarray,
                     max_iter: int = 200,
                     eps: float = 1e-7) -> np.ndarray:
    center = np.median(points, axis=0).astype(np.float64)
    for _ in range(max_iter):
        diff = points - center
        dist = np.linalg.norm(diff, axis=1)
        nonzero = dist > eps
        if not np.any(nonzero):
            return center
        weights = 1.0 / dist[nonzero]
        next_center = (points[nonzero] * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(next_center - center) < eps:
            return next_center
        center = next_center
    return center


def circle_covering_ratio(points_xy: np.ndarray,
                          cover_ratio: float) -> tuple[np.ndarray, float, int]:
    center = geometric_median(points_xy)
    distances = np.linalg.norm(points_xy - center, axis=1)
    kth = max(0, math.ceil(len(distances) * cover_ratio) - 1)
    radius = float(np.partition(distances, kth)[kth])
    cover_count = int(np.count_nonzero(distances <= radius + 1e-12))
    return center, radius, cover_count


def count_in_radius(points_xy: np.ndarray,
                    center: np.ndarray,
                    radius: float,
                    tree) -> int:
    if tree is not None:
        return int(tree.query_ball_point(center, radius + 1e-12, return_length=True))
    dist = np.linalg.norm(points_xy - center, axis=1)
    return int(np.count_nonzero(dist <= radius + 1e-12))


def fixed_radius_best_center(points_xy: np.ndarray,
                             radius: float,
                             sample_limit: int) -> tuple[np.ndarray, float, int]:
    if radius <= 0:
        raise ValueError("Known circle diameter must be > 0 for fixed-circle mode")

    all_tree = cKDTree(points_xy) if cKDTree is not None else None
    if len(points_xy) > sample_limit > 0:
        sample_idx = np.linspace(0, len(points_xy) - 1, sample_limit).astype(np.int64)
        source = points_xy[sample_idx]
    else:
        source = points_xy

    best_center = geometric_median(points_xy)
    best_count = count_in_radius(points_xy, best_center, radius, all_tree)

    source_tree = cKDTree(source) if cKDTree is not None else None
    candidate_count = 0

    for point in source:
        count = count_in_radius(points_xy, point, radius, all_tree)
        if count > best_count:
            best_center = point.copy()
            best_count = count

    if source_tree is not None:
        neighbor_lists = source_tree.query_ball_point(source, radius * 2.0)
        for i, neighbors in enumerate(neighbor_lists):
            p = source[i]
            for j in neighbors:
                if j <= i:
                    continue
                candidate_count += 1
                if candidate_count > FIXED_CIRCLE_CANDIDATE_LIMIT:
                    break
                q = source[j]
                for center in pair_circle_centers(p, q, radius):
                    count = count_in_radius(points_xy, center, radius, all_tree)
                    if count > best_count:
                        best_center = center
                        best_count = count
            if candidate_count > FIXED_CIRCLE_CANDIDATE_LIMIT:
                break
    else:
        n = len(source)
        for i in range(n):
            p = source[i]
            for j in range(i + 1, n):
                candidate_count += 1
                if candidate_count > FIXED_CIRCLE_CANDIDATE_LIMIT:
                    break
                q = source[j]
                if np.linalg.norm(q - p) > radius * 2.0:
                    continue
                for center in pair_circle_centers(p, q, radius):
                    count = count_in_radius(points_xy, center, radius, all_tree)
                    if count > best_count:
                        best_center = center
                        best_count = count
            if candidate_count > FIXED_CIRCLE_CANDIDATE_LIMIT:
                break

    return best_center, radius, best_count


def pair_circle_centers(p: np.ndarray,
                        q: np.ndarray,
                        radius: float) -> list[np.ndarray]:
    diff = q - p
    dist = float(np.linalg.norm(diff))
    if dist <= 1e-12 or dist > radius * 2.0 + 1e-12:
        return []
    mid = (p + q) / 2.0
    half = dist / 2.0
    height_sq = max(radius * radius - half * half, 0.0)
    height = math.sqrt(height_sq)
    perp = np.array([-diff[1], diff[0]], dtype=np.float64) / dist
    if height <= 1e-12:
        return [mid]
    return [mid + perp * height, mid - perp * height]


def process_cloud(number: str,
                  path: str,
                  z_low_ratio: float,
                  z_high_ratio: float,
                  cover_ratio: float,
                  known_diameter: float,
                  sample_limit: int) -> ObjectResult | None:
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        print(f"  [!] Empty point cloud skipped: {path}")
        return None

    z = points[:, 2]
    z_min = float(np.min(z))
    z_max = float(np.max(z))
    z_range = z_max - z_min
    if z_range <= 1e-12:
        z_low = z_min
        z_high = z_max
        section_mask = np.ones(len(points), dtype=bool)
    else:
        z_low = z_min + z_range * z_low_ratio
        z_high = z_min + z_range * z_high_ratio
        section_mask = (z >= z_low) & (z <= z_high)

    selected = points[section_mask]
    if len(selected) < 3:
        print(f"  [!] Too few section points skipped: {path} ({len(selected)} pts)")
        return None

    projected_xy = selected[:, :2]
    if known_diameter > 0:
        center, radius, cover_count = fixed_radius_best_center(
            projected_xy, known_diameter / 2.0, sample_limit
        )
        mode = "fixed diameter"
    else:
        center, radius, cover_count = circle_covering_ratio(projected_xy, cover_ratio)
        mode = "cover ratio"

    index, name = parse_object_name(path)
    return ObjectResult(
        number=number,
        path=path,
        name=name,
        index=index,
        total_points=len(points),
        selected_points=projected_xy,
        z_min=z_min,
        z_max=z_max,
        z_low=z_low,
        z_high=z_high,
        circle_center=center,
        circle_radius=radius,
        circle_cover_count=cover_count,
        circle_mode=mode,
    )


def load_font(size: int) -> ImageFont.ImageFont:
    for font_path in FONT_PATHS:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_text(image: np.ndarray,
              text: str,
              xy: tuple[int, int],
              font_size: int = 22,
              color: tuple[int, int, int] = (235, 235, 235)) -> None:
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = load_font(font_size)
    rgb = (color[2], color[1], color[0])
    draw.text(xy, text, fill=rgb, font=font)
    image[:, :, :] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def world_to_pixel(xy: np.ndarray,
                   center_xy: np.ndarray,
                   scale: float,
                   plot_origin: tuple[int, int],
                   plot_size: tuple[int, int]) -> np.ndarray:
    ox, oy = plot_origin
    width, height = plot_size
    pixels = np.empty_like(xy, dtype=np.int32)
    pixels[:, 0] = np.round(ox + width / 2 + (xy[:, 0] - center_xy[0]) * scale).astype(np.int32)
    pixels[:, 1] = np.round(oy + height / 2 - (xy[:, 1] - center_xy[1]) * scale).astype(np.int32)
    return pixels


def draw_object_tile(result: ObjectResult, cover_ratio: float) -> np.ndarray:
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), (28, 30, 34), dtype=np.uint8)
    plot_origin = (36, 78)
    plot_size = (500, 430)
    info_x = 560

    cv2.rectangle(
        tile,
        plot_origin,
        (plot_origin[0] + plot_size[0], plot_origin[1] + plot_size[1]),
        (55, 60, 68),
        1,
    )

    xy = result.selected_points
    circle_min = result.circle_center - result.circle_radius
    circle_max = result.circle_center + result.circle_radius
    bounds_min = np.minimum(np.min(xy, axis=0), circle_min)
    bounds_max = np.maximum(np.max(xy, axis=0), circle_max)
    span = np.maximum(bounds_max - bounds_min, 1e-6)
    data_center = (bounds_min + bounds_max) / 2
    scale = min(plot_size[0] / span[0], plot_size[1] / span[1]) * 0.88

    display_xy = xy
    if len(display_xy) > POINT_CLOUD_SECTION_POINT_SAMPLE_LIMIT:
        idx = np.linspace(0, len(display_xy) - 1, POINT_CLOUD_SECTION_POINT_SAMPLE_LIMIT).astype(np.int64)
        display_xy = display_xy[idx]

    pixels = world_to_pixel(display_xy, data_center, scale, plot_origin, plot_size)
    valid = (
        (pixels[:, 0] >= plot_origin[0])
        & (pixels[:, 0] < plot_origin[0] + plot_size[0])
        & (pixels[:, 1] >= plot_origin[1])
        & (pixels[:, 1] < plot_origin[1] + plot_size[1])
    )
    pixels = pixels[valid]
    tile[pixels[:, 1], pixels[:, 0]] = (92, 205, 255)

    center_px = world_to_pixel(
        result.circle_center.reshape(1, 2), data_center, scale, plot_origin, plot_size
    )[0]
    radius_px = max(1, int(round(result.circle_radius * scale)))
    cv2.circle(tile, tuple(center_px), radius_px, (55, 235, 115), 2, cv2.LINE_AA)
    cv2.circle(tile, tuple(center_px), 3, (40, 80, 255), -1, cv2.LINE_AA)

    diameter_mm = result.circle_radius * 2 * 1000.0
    covered_ratio = result.circle_cover_count / len(result.selected_points)
    z_text = f"{result.z_low * 1000:.1f}~{result.z_high * 1000:.1f} mm"
    cx, cy = result.circle_center

    draw_text(tile, f"#{result.index:02d} {result.name}", (28, 24), 24, (245, 245, 245))
    draw_text(tile, "Diameter", (info_x, 82), 20, (210, 210, 210))
    draw_text(tile, f"{diameter_mm:.1f} mm", (info_x, 110), 25, (80, 245, 130))
    draw_text(tile, "Center XY global", (info_x, 158), 19, (190, 190, 190))
    draw_text(tile, f"x={cx:.4f} m", (info_x, 184), 18, (230, 230, 230))
    draw_text(tile, f"y={cy:.4f} m", (info_x, 210), 18, (230, 230, 230))
    draw_text(tile, "Z section", (info_x, 252), 19, (190, 190, 190))
    draw_text(tile, z_text, (info_x, 278), 18, (230, 230, 230))
    draw_text(tile, "Points", (info_x, 320), 19, (190, 190, 190))
    draw_text(tile, f"{len(result.selected_points)}/{result.total_points}", (info_x, 346), 18, (230, 230, 230))
    draw_text(tile, "Circle cover", (info_x, 388), 19, (190, 190, 190))
    draw_text(tile, f"{covered_ratio * 100:.1f}% / {cover_ratio * 100:.0f}%", (info_x, 414), 18, (230, 230, 230))
    draw_text(tile, result.circle_mode, (info_x, 458), 17, (170, 200, 245))

    scale_bar_m = choose_scale_bar_length(span)
    if scale_bar_m > 0:
        bar_px = int(scale_bar_m * scale)
        x0 = plot_origin[0] + 24
        y0 = plot_origin[1] + plot_size[1] - 26
        cv2.line(tile, (x0, y0), (x0 + bar_px, y0), (235, 235, 235), 2, cv2.LINE_AA)
        draw_text(tile, f"{scale_bar_m * 1000:.0f} mm", (x0, y0 - 30), 17, (235, 235, 235))

    return tile


def choose_scale_bar_length(span_xy: np.ndarray) -> float:
    target = float(max(span_xy) * 0.22)
    if target <= 0:
        return 0.0
    nice_mm = np.array([5, 10, 20, 50, 100, 200, 500, 1000], dtype=np.float64)
    nice_m = nice_mm / 1000.0
    return float(nice_m[np.argmin(np.abs(nice_m - target))])


def compose_result_image(number: str,
                         results: list[ObjectResult],
                         cover_ratio: float) -> np.ndarray:
    tiles = [draw_object_tile(result, cover_ratio) for result in results]
    columns = min(GRID_COLUMNS, max(1, len(tiles)))
    rows = math.ceil(len(tiles) / columns)
    header_h = 70
    canvas = np.full(
        (header_h + rows * TILE_HEIGHT, columns * TILE_WIDTH, 3),
        (18, 20, 24),
        dtype=np.uint8,
    )
    draw_text(
        canvas,
        f"Point cloud Z0 projection circles - {number}",
        (24, 18),
        28,
        (245, 245, 245),
    )

    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        y0 = header_h + row * TILE_HEIGHT
        x0 = col * TILE_WIDTH
        canvas[y0:y0 + TILE_HEIGHT, x0:x0 + TILE_WIDTH] = tile

    return canvas


def process_number(number: str,
                   z_low_ratio: float,
                   z_high_ratio: float,
                   cover_ratio: float,
                   known_diameter: float,
                   sample_limit: int) -> list[ObjectResult]:
    number = normalize_number(number)
    files = find_point_cloud_files(number)
    if not files:
        print(f"[{number}] No point clouds found in {POINT_CLOUD_PATH}")
        print(f"Expected folder: {os.path.join(POINT_CLOUD_PATH, number)}")
        return []

    print(f"\n[{number}] Found {len(files)} point cloud file(s)")
    results = []
    for path in files:
        result = process_cloud(
            number, path, z_low_ratio, z_high_ratio, cover_ratio,
            known_diameter, sample_limit
        )
        if result is None:
            continue
        results.append(result)
        diameter_mm = result.circle_radius * 2 * 1000.0
        center = result.circle_center
        covered_ratio = result.circle_cover_count / len(result.selected_points)
        print(
            f"  #{result.index:02d} {result.name}: "
            f"section {len(result.selected_points)}/{result.total_points} pts, "
            f"diameter={diameter_mm:.2f} mm, "
            f"center_xy=({center[0]:.5f}, {center[1]:.5f}) m, "
            f"cover={covered_ratio * 100:.1f}%"
        )
    return results


def show_results(number: str,
                 results: list[ObjectResult],
                 cover_ratio: float) -> bool:
    if not results:
        return True
    image = compose_result_image(number, results, cover_ratio)
    window_name = f"Point cloud section circles - {number}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyWindow(window_name)
    return key not in (27, ord("q"), ord("Q"))


def iter_interactive_numbers() -> Iterable[str]:
    print("=" * 70)
    print("Point cloud Z0 projection circle tool")
    print("=" * 70)
    print("Input image ID, for example 0000. Input q to quit.")
    while True:
        value = input("\nID: ").strip()
        if value.lower() == "q":
            break
        if value:
            yield value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project object point clouds to Z=0 and draw section circles."
    )
    parser.add_argument("numbers", nargs="*", help="Image IDs, e.g. 0000 0003")
    parser.add_argument("--z-low", type=float, default=POINT_CLOUD_SECTION_LOW_RATIO,
                        help="Lower Z section ratio.")
    parser.add_argument("--z-high", type=float, default=POINT_CLOUD_SECTION_HIGH_RATIO,
                        help="Upper Z section ratio.")
    parser.add_argument("--cover", type=float, default=POINT_CLOUD_CIRCLE_COVER_RATIO,
                        help="Point ratio covered by estimated circle.")
    parser.add_argument("--known-diameter", type=float,
                        default=POINT_CLOUD_KNOWN_CIRCLE_DIAMETER,
                        help="Known circle diameter in meters. <= 0 disables fixed mode.")
    parser.add_argument("--known-diameter-mm", type=float, default=None,
                        help="Known circle diameter in millimeters.")
    parser.add_argument("--sample-limit", type=int,
                        default=POINT_CLOUD_FIXED_CIRCLE_PAIR_SAMPLE_LIMIT,
                        help="Pair-search sample limit for fixed-diameter center search.")
    parser.add_argument("--no-window", action="store_true",
                        help="Only print results; do not open OpenCV windows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    known_diameter = args.known_diameter
    if args.known_diameter_mm is not None:
        known_diameter = args.known_diameter_mm / 1000.0

    try:
        validate_args(args.z_low, args.z_high, args.cover, known_diameter)
    except ValueError as exc:
        print(f"[X] {exc}")
        return 2

    numbers = args.numbers or list(iter_interactive_numbers())
    for raw_number in numbers:
        number = normalize_number(raw_number)
        results = process_number(
            number, args.z_low, args.z_high, args.cover,
            known_diameter, args.sample_limit
        )
        if args.no_window:
            continue
        if not show_results(number, results, args.cover):
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
