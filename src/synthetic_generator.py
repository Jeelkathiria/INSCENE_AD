"""
Synthetic test video generator for replicating the 5 test scenarios
outlined in Slide 10 of the project presentation:
1. Static Object
2. Dynamic Object In Motion
3. Centrally Occluded
4. Partially Occluded
5. Visual Clutter / Multiple Simultaneous Objects
"""

import os
import math
import cv2
import numpy as np
from typing import Dict
from src.config import SAMPLES_DIR


def create_gradient_background(w: int, h: int) -> np.ndarray:
    """Creates a realistic natural sky & ground gradient background."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky gradient (top half)
    for y in range(h // 2):
        ratio = y / (h // 2)
        b = int(240 - 40 * ratio)
        g = int(210 - 30 * ratio)
        r = int(180 - 20 * ratio)
        img[y, :] = (b, g, r)

    # Ground gradient (bottom half)
    for y in range(h // 2, h):
        ratio = (y - h // 2) / (h // 2)
        b = int(100 - 30 * ratio)
        g = int(130 - 30 * ratio)
        r = int(90 - 20 * ratio)
        img[y, :] = (b, g, r)

    return img


def draw_synthetic_billboard(
    canvas: np.ndarray,
    quad: np.ndarray,
    text: str = "BILLBOARD SPACE",
    bg_color=(235, 235, 235)
):
    """Draws a synthetic planar billboard with support poles and inner canvas."""
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    # Draw pole first (behind billboard)
    bottom_center = (int((quad[2][0] + quad[3][0]) / 2), int((quad[2][1] + quad[3][1]) / 2))
    cv2.line(canvas, bottom_center, (bottom_center[0], canvas.shape[0]), (70, 70, 70), 10)

    # Frame/border
    cv2.fillPoly(canvas, [pts], (60, 60, 60))

    # Inner canvas (slightly inset)
    center = np.mean(quad, axis=0)
    inset_quad = quad + (center - quad) * 0.08
    inset_pts = inset_quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [inset_pts], bg_color)
    cv2.polylines(canvas, [inset_pts], True, (40, 40, 40), 2, cv2.LINE_AA)

    # Text inside billboard
    cx, cy = int(center[0]), int(center[1])
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(
        canvas, text, (cx - tw // 2, cy + th // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA
    )


def generate_all_test_videos(output_dir: str = SAMPLES_DIR, fps: int = 30, duration_sec: int = 4) -> Dict[str, str]:
    """
    Generates all 5 test videos corresponding to the project presentation test cases.
    """
    os.makedirs(output_dir, exist_ok=True)
    w, h = 640, 360
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    created_files = {}

    # Case 1: Static Object
    path_1 = os.path.join(output_dir, "test_1_static_object.mp4")
    out = cv2.VideoWriter(path_1, fourcc, fps, (w, h))
    base_quad = np.array([
        [180, 80], [460, 80], [460, 220], [180, 220]
    ], dtype=np.float32)

    for f in range(total_frames):
        frame = create_gradient_background(w, h)
        draw_synthetic_billboard(frame, base_quad, "CITY AD SPACE (STATIC)")
        out.write(frame)
    out.release()
    created_files["case_1_static"] = path_1

    # Case 2: Dynamic Object in Motion (Translating horizontally)
    path_2 = os.path.join(output_dir, "test_2_dynamic_motion.mp4")
    out = cv2.VideoWriter(path_2, fourcc, fps, (w, h))
    for f in range(total_frames):
        frame = create_gradient_background(w, h)
        dx = int(80 * math.sin(2 * math.pi * f / total_frames))
        dyn_quad = base_quad + np.array([dx, 0], dtype=np.float32)
        draw_synthetic_billboard(frame, dyn_quad, "MOVING PANEL")
        out.write(frame)
    out.release()
    created_files["case_2_dynamic"] = path_2

    # Case 3: Centrally Occluded (A person walks directly in front across the center)
    path_3 = os.path.join(output_dir, "test_3_central_occlusion.mp4")
    out = cv2.VideoWriter(path_3, fourcc, fps, (w, h))
    for f in range(total_frames):
        frame = create_gradient_background(w, h)
        draw_synthetic_billboard(frame, base_quad, "OCCLUSION TEST")
        # Moving foreground occluder (pedestrian silhouette)
        ox = int(140 + (w - 200) * (f / total_frames))
        oy = 150
        # Draw occluder box/pedestrian
        cv2.rectangle(frame, (ox, oy), (ox + 50, oy + 120), (30, 30, 160), -1)
        cv2.circle(frame, (ox + 25, oy - 15), 18, (30, 30, 160), -1)
        out.write(frame)
    out.release()
    created_files["case_3_central_occlusion"] = path_3

    # Case 4: Partially Occluded (A persistent pole/tree branch occluding the right 15%)
    path_4 = os.path.join(output_dir, "test_4_partial_occlusion.mp4")
    out = cv2.VideoWriter(path_4, fourcc, fps, (w, h))
    for f in range(total_frames):
        frame = create_gradient_background(w, h)
        draw_synthetic_billboard(frame, base_quad, "PARTIAL OCCLUSION")
        # Persistent foreground vertical pole on the right
        cv2.rectangle(frame, (420, 60), (450, h), (40, 60, 40), -1)
        out.write(frame)
    out.release()
    created_files["case_4_partial_occlusion"] = path_4

    # Case 5: Visual Clutter / Multiple Simultaneous Surfaces
    path_5 = os.path.join(output_dir, "test_5_visual_clutter.mp4")
    out = cv2.VideoWriter(path_5, fourcc, fps, (w, h))
    # Surface A: Optimal (Centered, large, upright)
    quad_a = np.array([[220, 90], [420, 90], [420, 210], [220, 210]], dtype=np.float32)
    # Surface B: Small, acute perspective skew
    quad_b = np.array([[60, 120], [150, 140], [140, 220], [70, 210]], dtype=np.float32)
    # Surface C: Corner small poster
    quad_c = np.array([[480, 130], [580, 130], [580, 190], [480, 190]], dtype=np.float32)

    for f in range(total_frames):
        frame = create_gradient_background(w, h)
        draw_synthetic_billboard(frame, quad_a, "OPTIMAL SPOT", bg_color=(240, 240, 240))
        draw_synthetic_billboard(frame, quad_b, "SKEWED", bg_color=(200, 200, 200))
        draw_synthetic_billboard(frame, quad_c, "SMALL", bg_color=(210, 210, 210))
        out.write(frame)
    out.release()
    created_files["case_5_clutter"] = path_5

    return created_files

