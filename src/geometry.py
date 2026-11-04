"""
Geometric utilities for planar ad space detection.
Handles 4-point quadrilateral extraction, sorting, angle analysis,
convexity verification, and homography computation.
"""

import math
import cv2
import numpy as np
from typing import List, Tuple, Optional


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 points in canonical clockwise order:
    [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    # Sum of coordinates: Top-Left has smallest sum, Bottom-Right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference of coordinates (y - x): Top-Right has smallest diff, Bottom-Left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def extract_quad_from_contour(
    contour: np.ndarray,
    epsilon_ratio: float = 0.03
) -> Optional[np.ndarray]:
    """
    Extracts a 4-point convex quadrilateral from an arbitrary contour.
    First attempts Douglas-Peucker approximation; if that does not produce 4 points,
    falls back to minimum area bounding box (cv2.minAreaRect).
    """
    if contour is None or len(contour) < 4:
        return None

    peri = cv2.arcLength(contour, True)
    # Try dynamic epsilon values around the target ratio to find a true 4-vertex quad
    for factor in [epsilon_ratio, epsilon_ratio * 0.7, epsilon_ratio * 1.4, epsilon_ratio * 2.0]:
        approx = cv2.approxPolyDP(contour, factor * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2)
            return order_points(pts)

    # Fallback to rotated minimum bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.asarray(box, dtype=np.float32)
    return order_points(box)


def compute_quad_angles(quad: np.ndarray) -> List[float]:
    """
    Computes the 4 interior angles (in degrees) of a quadrilateral.
    Returns list of 4 angles corresponding to vertices [TL, TR, BR, BL].
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    angles = []

    for i in range(4):
        p_prev = quad[(i - 1) % 4]
        p_curr = quad[i]
        p_next = quad[(i + 1) % 4]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            angles.append(0.0)
            continue

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        angles.append(angle_deg)

    return angles


def is_valid_planar_quad(
    quad: np.ndarray,
    frame_width: int,
    frame_height: int,
    min_area_fraction: float = 0.015,
    max_area_fraction: float = 0.65,
    min_angle: float = 35.0,
    max_angle: float = 145.0
) -> Tuple[bool, str]:
    """
    Validates if a 4-point quad is suitable as a planar ad candidate.
    Checks:
    1. Convexity
    2. Minimum and maximum relative area
    3. Non-extreme interior angles (avoids sharp grazing perspective)
    """
    if quad is None or len(quad) != 4:
        return False, "Invalid vertex count (expected 4)"

    pts_int = quad.astype(np.int32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(pts_int):
        return False, "Non-convex quadrilateral"

    area = cv2.contourArea(pts_int)
    frame_area = frame_width * frame_height
    area_frac = area / max(1.0, frame_area)

    if area_frac < min_area_fraction:
        return False, f"Area fraction {area_frac:.3f} too small (min {min_area_fraction})"
    if area_frac > max_area_fraction:
        return False, f"Area fraction {area_frac:.3f} too large (max {max_area_fraction})"

    angles = compute_quad_angles(quad)
    for a in angles:
        if a < min_angle or a > max_angle:
            return False, f"Interior angle {a:.1f}° violates limits [{min_angle}°, {max_angle}°]"

    return True, "Valid planar surface"


def compute_polygon_intersection_area(
    poly1: np.ndarray,
    poly2: np.ndarray,
    canvas_shape: Tuple[int, int]
) -> float:
    """
    Computes exact intersection area between two 2D polygons using binary mask rasterization.
    canvas_shape: (height, width)
    """
    h, w = canvas_shape
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    pts1 = poly1.astype(np.int32).reshape((-1, 1, 2))
    pts2 = poly2.astype(np.int32).reshape((-1, 1, 2))

    cv2.fillPoly(mask1, [pts1], 255)
    cv2.fillPoly(mask2, [pts2], 255)

    intersection = cv2.bitwise_and(mask1, mask2)
    return float(np.count_nonzero(intersection))


def compute_occlusion_ratio(
    candidate_quad: np.ndarray,
    occluder_polys_or_boxes: List[np.ndarray],
    canvas_shape: Tuple[int, int]
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Calculates the occlusion ratio: Area(Candidate ∩ Occluders) / Area(Candidate).
    Returns (ratio, combined_occlusion_mask).
    """
    if not occluder_polys_or_boxes:
        return 0.0, None

    h, w = canvas_shape
    cand_mask = np.zeros((h, w), dtype=np.uint8)
    occ_mask = np.zeros((h, w), dtype=np.uint8)

    cand_pts = candidate_quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(cand_mask, [cand_pts], 255)

    for occ in occluder_polys_or_boxes:
        if occ is None or len(occ) == 0:
            continue
        occ_pts = occ.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(occ_mask, [occ_pts], 255)

    cand_area = np.count_nonzero(cand_mask)
    if cand_area == 0:
        return 0.0, None

    intersection_mask = cv2.bitwise_and(cand_mask, occ_mask)
    intersect_pixels = np.count_nonzero(intersection_mask)

    ratio = min(1.0, intersect_pixels / float(cand_area))
    return ratio, intersection_mask


def compute_homography_quality(quad: np.ndarray) -> float:
    """
    Measures homography conditioning and orthogonality.
    Scores from 0 to 100 based on deviation of interior angles from 90 degrees.
    """
    angles = compute_quad_angles(quad)
    # Sum of absolute angular deviations from 90°
    dev = sum(abs(a - 90.0) for a in angles)
    # Perfect rectangle: dev = 0 -> score 100.
    # Worst case (e.g. total deviation 180°): score 0.
    score = max(0.0, 100.0 - (dev / 180.0) * 100.0)
    return float(score)

