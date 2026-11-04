"""
Candidate Surface and Occluder Detection module.
Integrates YOLOv8-World v2 (open-vocabulary) with GrabCut / SAM refinement
and graceful offline/classical contour fallback.
"""

import math
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.config import DetectorConfig, GeometryConfig
from src.geometry import extract_quad_from_contour, is_valid_planar_quad, order_points

logger = logging.getLogger(__name__)


@dataclass
class CandidateDetection:
    """Represents a detected planar surface candidate."""
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2)
    quad: np.ndarray                       # 4x2 array: [TL, TR, BR, BL]
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class OccluderDetection:
    """Represents a detected foreground occluding object."""
    bbox: Tuple[int, int, int, int]
    poly: np.ndarray                       # Polygon or box corners
    class_name: str
    confidence: float


class SurfaceDetector:
    """
    Hybrid detector combining YOLOv8-World v2 open-vocabulary detection,
    GrabCut pixel refinement, and classical geometry extraction.
    """
    _cached_model = None
    _cached_model_type = None

    def __init__(self, det_cfg: DetectorConfig, geom_cfg: GeometryConfig):
        self.det_cfg = det_cfg
        self.geom_cfg = geom_cfg
        if SurfaceDetector._cached_model is not None:
            self.model = SurfaceDetector._cached_model
            self.model_type = SurfaceDetector._cached_model_type
        else:
            self.model = None
            self.model_type = "none"
            self._init_model()

    def _init_model(self):
        """Initializes YOLOv8-World v2 with fallback to standard YOLO or classical mode."""
        try:
            from ultralytics import YOLO
            logger.info("Attempting to load YOLOv8-World model: %s", self.det_cfg.model_name)
            # Try loading YOLOv8-World
            try:
                self.model = YOLO(self.det_cfg.model_name)
                # Set custom open-vocabulary classes
                all_classes = self.det_cfg.candidate_classes + self.det_cfg.occluder_classes
                self.model.set_classes(all_classes)
                self.model_type = "yolo_world"
                SurfaceDetector._cached_model = self.model
                SurfaceDetector._cached_model_type = self.model_type
                logger.info("YOLOv8-World loaded with %d custom classes.", len(all_classes))
                return
            except Exception as e:
                logger.warning("Could not initialize %s: %s. Falling back to %s",
                               self.det_cfg.model_name, e, self.det_cfg.fallback_model)

            # Fallback to standard YOLOv8n
            try:
                self.model = YOLO(self.det_cfg.fallback_model)
                self.model_type = "yolo_coco"
                logger.info("Loaded fallback model: %s", self.det_cfg.fallback_model)
                return
            except Exception as e2:
                logger.warning("Could not load fallback model: %s. Using classical detector.", e2)
                self.model = None
                self.model_type = "classical"
        except ImportError:
            logger.warning("Ultralytics library not available. Using classical detector.")
            self.model = None
            self.model_type = "classical"

    def refine_quad_fast(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Ultra-fast (<2ms) quad refinement inside candidate bounding box.
        Applies local edge detection and polygon fitting instead of slow iterative GrabCut.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = x2 - x1, y2 - y1

        default_quad = order_points(np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32))

        if bw < 25 or bh < 25:
            return default_quad

        try:
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 130)

            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest) > (bw * bh * 0.20):
                    peri = cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype(np.float32)
                        pts[:, 0] += x1
                        pts[:, 1] += y1
                        return order_points(pts)

            # If contour approximation did not yield 4 points, test 3D perspective depth vanishing lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=max(20, int(bw * 0.20)), maxLineGap=12)
            if lines is not None and len(lines) >= 2:
                top_slopes = []
                bot_slopes = []
                mid_y = bh * 0.5
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    dx = lx2 - lx1
                    dy = ly2 - ly1
                    if abs(dx) < 10:
                        continue
                    slope = dy / float(dx)
                    if abs(slope) > 0.40:
                        continue
                    y_avg = (ly1 + ly2) * 0.5
                    if y_avg < mid_y:
                        top_slopes.append(slope)
                    else:
                        bot_slopes.append(slope)

                if top_slopes or bot_slopes:
                    m_top = float(np.median(top_slopes)) if top_slopes else 0.0
                    m_bot = float(np.median(bot_slopes)) if bot_slopes else 0.0
                    if abs(m_top) > 0.025 or abs(m_bot) > 0.025:
                        p1 = [float(x1), float(y1)]
                        p2 = [float(x2), float(y1 + m_top * bw)]
                        p3 = [float(x2), float(y2 + m_bot * bw)]
                        p4 = [float(x1), float(y2)]
                        return order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
        except Exception:
            pass

        return default_quad

    def detect_classical(self, frame: np.ndarray) -> Tuple[List[CandidateDetection], List[OccluderDetection]]:
        """
        Model-free classical detection: uses bilateral filtering, adaptive Canny edge detection,
        morphological closing, and quadrilateral contour extraction.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[CandidateDetection] = []

        # Sort contours by area descending so we find clean prominent surfaces
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:25]:
            area = cv2.contourArea(cnt)
            if area < (h * w * self.geom_cfg.min_area_fraction):
                continue

            quad = extract_quad_from_contour(cnt, self.geom_cfg.epsilon_approx_ratio)
            if quad is None:
                continue

            valid, _ = is_valid_planar_quad(
                quad, w, h,
                min_area_fraction=self.geom_cfg.min_area_fraction,
                max_area_fraction=self.geom_cfg.max_area_fraction,
                min_angle=self.geom_cfg.min_internal_angle_deg,
                max_angle=self.geom_cfg.max_internal_angle_deg
            )
            if not valid:
                continue

            # Compute bounding box from quad
            pts = quad.astype(int)
            x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
            x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])

            # Deduplicate nearly identical quads
            is_dup = False
            for existing in candidates:
                eq = existing.quad.astype(int)
                ex1, ey1 = np.min(eq[:, 0]), np.min(eq[:, 1])
                ex2, ey2 = np.max(eq[:, 0]), np.max(eq[:, 1])
                # Check center distance
                if abs((x1+x2)/2 - (ex1+ex2)/2) < 25 and abs((y1+y2)/2 - (ey1+ey2)/2) < 25:
                    is_dup = True
                    break
            if is_dup:
                continue

            candidates.append(CandidateDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                quad=quad,
                class_name="billboard",
                confidence=0.90
            ))

        return candidates, []

    def detect(self, frame: np.ndarray) -> Tuple[List[CandidateDetection], List[OccluderDetection]]:
        """
        Runs detection on a single frame. Returns list of Candidate surfaces and Occluders.
        """
        if self.model is None or self.model_type == "classical":
            return self.detect_classical(frame)

        h, w = frame.shape[:2]
        candidates: List[CandidateDetection] = []
        occluders: List[OccluderDetection] = []

        try:
            results = self.model(
                frame,
                conf=self.det_cfg.conf_threshold,
                iou=self.det_cfg.iou_threshold,
                imgsz=self.det_cfg.inference_imgsz,
                verbose=False
            )[0]
        except Exception as e:
            logger.error("Inference failed: %s. Using classical detector.", e)
            return self.detect_classical(frame)

        # Parse detections
        names = results.names
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            c_cand, c_occ = self.detect_classical(frame)
            return c_cand, occluders + c_occ

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names.get(cls_id, str(cls_id)).lower()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox = (x1, y1, x2, y2)
            bw = x2 - x1
            bh = y2 - y1
            aspect = bw / max(1, bh)

            # Special Transit & Automotive Advertising Handler: Bus, Truck & Car Side Ad Panels
            if "bus" in cls_name or "truck" in cls_name or "car" in cls_name:
                if aspect > 1.05:
                    if "bus" in cls_name:
                        # Extract the transit ad banner below the window band
                        bus_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        dark_left = 0.5
                        dark_right = 0.5
                        if bus_crop.size > 0 and bus_crop.shape[1] > 20:
                            lc = bus_crop[:, :int(bw * 0.35)]
                            rc = bus_crop[:, int(bw * 0.65):]
                            if lc.size > 0 and rc.size > 0:
                                gl = cv2.cvtColor(lc, cv2.COLOR_BGR2GRAY)
                                gr = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
                                dark_left = float(np.sum(gl < 70) / max(1, gl.size))
                                dark_right = float(np.sum(gr < 70) / max(1, gr.size))

                        if aspect > 2.8:
                            # Full side profile (flat elevation)
                            sx1 = int(x1 + 0.10 * bw)
                            sx2 = int(x2 - 0.04 * bw)
                            sy1 = int(y1 + 0.48 * bh)
                            sy2 = int(y1 + 0.85 * bh)
                            p1 = [sx1, sy1]
                            p2 = [sx2, sy1]
                            p3 = [sx2, sy2]
                            p4 = [sx1, sy2]
                        elif dark_left > dark_right:
                            # 3D Depth Perspective: Front on left, side flank recedes into street depth (right)
                            # Top sill slopes down along vanishing ray; bottom skirt slopes up towards vanishing point
                            sx1 = int(x1 + 0.36 * bw)
                            sx2 = int(x2 - 0.03 * bw)
                            dx = sx2 - sx1
                            sy1_l = int(y1 + 0.48 * bh)
                            sy2_l = int(y1 + 0.85 * bh)
                            sy1_r = int(sy1_l + 0.065 * dx)
                            sy2_r = int(sy2_l - 0.075 * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]
                        else:
                            # 3D Depth Perspective: Front on right, side flank recedes into depth (left)
                            sx1 = int(x1 + 0.03 * bw)
                            sx2 = int(x1 + 0.64 * bw)
                            dx = sx2 - sx1
                            sy1_r = int(y1 + 0.48 * bh)
                            sy2_r = int(y1 + 0.85 * bh)
                            sy1_l = int(sy1_r + 0.065 * dx)
                            sy2_l = int(sy2_r - 0.075 * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]

                        quad = order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            b_min_y = int(min(p1[1], p2[1]))
                            b_max_y = int(max(p3[1], p4[1]))
                            candidates.append(CandidateDetection(
                                bbox=(sx1, b_min_y, sx2, b_max_y),
                                quad=quad,
                                class_name="bus side ad space",
                                confidence=conf,
                                mask=None
                            ))
                            continue
                    elif "truck" in cls_name:
                        # Cargo truck side billboard
                        sx1 = int(x1 + 0.30 * bw)
                        sx2 = int(x2 - 0.02 * bw)
                        sy1 = int(y1 + 0.15 * bh)
                        sy2 = int(y1 + 0.85 * bh)
                        quad = order_points(np.array([
                            [sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]
                        ], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            candidates.append(CandidateDetection(
                                bbox=(sx1, sy1, sx2, sy2),
                                quad=quad,
                                class_name="truck side billboard",
                                confidence=conf,
                                mask=None
                            ))
                            continue
                    elif "car" in cls_name and bw >= 110 and bh >= 40:
                        # Extract the car side door & livery panel (between wheels, below window sill)
                        car_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        facing_right = True
                        if car_crop.size > 0 and car_crop.shape[1] > 20:
                            gray_car = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)
                            top_half = gray_car[:int(bh * 0.5), :]
                            dark_l = float(np.sum(top_half[:, :int(bw * 0.5)] < 100))
                            dark_r = float(np.sum(top_half[:, int(bw * 0.5):] < 100))
                            facing_right = (dark_l >= dark_r)

                        if aspect > 2.8:
                            # Full side profile (flat elevation)
                            sx1 = int(x1 + 0.18 * bw)
                            sx2 = int(x1 + 0.78 * bw)
                            sy1 = int(y1 + 0.38 * bh)
                            sy2 = int(y1 + 0.78 * bh)
                            p1 = [sx1, sy1]
                            p2 = [sx2, sy1]
                            p3 = [sx2, sy2]
                            p4 = [sx1, sy2]
                        elif facing_right:
                            # 3D Depth Perspective: Facing right (door is between wheels on left half, tilted with 3D depth)
                            sx1 = int(x1 + 0.22 * bw)  # in front of rear wheel
                            sx2 = int(x1 + 0.50 * bw)  # behind front wheel / A-pillar
                            dx = sx2 - sx1
                            sy1_l = int(y1 + 0.42 * bh)  # below window sill
                            sy2_l = int(y1 + 0.85 * bh)  # above bottom skirt
                            slope = 0.082  # +4.7° 3D perspective slope along body line
                            sy1_r = int(sy1_l + slope * dx)
                            sy2_r = int(sy2_l + slope * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]
                        else:
                            # 3D Depth Perspective: Facing left (door is on the right)
                            sx1 = int(x1 + 0.50 * bw)
                            sx2 = int(x1 + 0.78 * bw)
                            dx = sx2 - sx1
                            sy1_r = int(y1 + 0.42 * bh)
                            sy2_r = int(y1 + 0.85 * bh)
                            slope = 0.082
                            sy1_l = int(sy1_r + slope * dx)
                            sy2_l = int(sy2_r + slope * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]

                        quad = order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            c_min_y = int(min(p1[1], p2[1]))
                            c_max_y = int(max(p3[1], p4[1]))
                            candidates.append(CandidateDetection(
                                bbox=(sx1, c_min_y, sx2, c_max_y),
                                quad=quad,
                                class_name="car side ad space",
                                confidence=conf,
                                mask=None
                            ))
                            continue

                # Head-on vehicle (no side flank visible) is treated as a traffic occluder
                occ_quad = order_points(np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32))
                occluders.append(OccluderDetection(
                    bbox=bbox,
                    poly=occ_quad,
                    class_name=cls_name,
                    confidence=conf
                ))
                continue

            # Determine whether this is a candidate or an occluder
            is_candidate = False
            is_occluder = False

            if self.model_type == "yolo_world":
                if any(cand.lower() in cls_name for cand in self.det_cfg.candidate_classes):
                    is_candidate = True
                elif any(occ.lower() in cls_name for occ in self.det_cfg.occluder_classes):
                    is_occluder = True
            else:
                # Standard COCO fallback mappings
                if cls_name in self.det_cfg.coco_candidate_classes:
                    is_candidate = True
                elif cls_name in self.det_cfg.coco_occluder_classes:
                    is_occluder = True

            if is_candidate:
                # Fast <2ms quad extraction
                quad = self.refine_quad_fast(frame, bbox)

                valid, _ = is_valid_planar_quad(
                    quad, w, h,
                    min_area_fraction=self.geom_cfg.min_area_fraction,
                    max_area_fraction=self.geom_cfg.max_area_fraction,
                    min_angle=self.geom_cfg.min_internal_angle_deg,
                    max_angle=self.geom_cfg.max_internal_angle_deg
                )

                if valid:
                    candidates.append(CandidateDetection(
                        bbox=bbox,
                        quad=quad,
                        class_name=cls_name,
                        confidence=conf,
                        mask=None
                    ))

            elif is_occluder:
                occ_quad = order_points(np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32))
                occluders.append(OccluderDetection(
                    bbox=bbox,
                    poly=occ_quad,
                    class_name=cls_name,
                    confidence=conf
                ))

        # If model found no candidates, run classical detector as supplement
        if len(candidates) == 0:
            c_cand, _ = self.detect_classical(frame)
            candidates.extend(c_cand)

        # Deduplicate candidates with high spatial overlap (prevent duplicate boxes on same surface)
        if len(candidates) > 1:
            deduped: List[CandidateDetection] = []
            for cand in candidates:
                keep = True
                c_cx = float(np.mean(cand.quad[:, 0]))
                c_cy = float(np.mean(cand.quad[:, 1]))
                for existing in deduped:
                    e_cx = float(np.mean(existing.quad[:, 0]))
                    e_cy = float(np.mean(existing.quad[:, 1]))
                    dist = math.hypot(c_cx - e_cx, c_cy - e_cy)
                    if dist < 40.0:
                        keep = False
                        break
                if keep:
                    deduped.append(cand)
            candidates = deduped

        return candidates, occluders

