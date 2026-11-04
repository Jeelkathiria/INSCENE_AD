"""
Custom Centroid & Geometric Tracker for planar candidate surfaces.
Maintains candidate-surface tracklet IDs across video frames, tracks corner motion,
and aggregates temporal visibility and occlusion histories.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from src.config import TrackerConfig
from src.detector import CandidateDetection, OccluderDetection
from src.geometry import compute_occlusion_ratio, compute_quad_angles


@dataclass
class TrackletFrameData:
    """Per-frame snapshot of a candidate surface."""
    frame_idx: int
    timestamp_sec: float
    quad: np.ndarray                 # 4x2 points
    centroid: Tuple[float, float]
    area_fraction: float
    occlusion_ratio: float
    displacement: float             # Corner displacement from previous frame


class Tracklet:
    """Represents a continuous temporal track of a candidate ad surface."""

    def __init__(self, track_id: int, initial_detection: CandidateDetection, frame_idx: int, timestamp: float, frame_area: float):
        self.track_id = track_id
        self.class_name = initial_detection.class_name
        self.confidence = initial_detection.confidence

        self.start_frame = frame_idx
        self.end_frame = frame_idx
        self.start_time = timestamp
        self.end_time = timestamp
        self.lost_frames = 0
        self.is_active = True

        cx = float(np.mean(initial_detection.quad[:, 0]))
        cy = float(np.mean(initial_detection.quad[:, 1]))
        init_area = float(cv2_polygon_area(initial_detection.quad)) / max(1.0, frame_area)

        first_snapshot = TrackletFrameData(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            quad=initial_detection.quad.copy(),
            centroid=(cx, cy),
            area_fraction=init_area,
            occlusion_ratio=0.0,
            displacement=0.0
        )
        self.history: List[TrackletFrameData] = [first_snapshot]

    @property
    def current_quad(self) -> np.ndarray:
        return self.history[-1].quad

    @property
    def current_centroid(self) -> Tuple[float, float]:
        return self.history[-1].centroid

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def total_frames(self) -> int:
        return len(self.history)

    def update(
        self,
        detection: CandidateDetection,
        frame_idx: int,
        timestamp: float,
        occlusion_ratio: float,
        frame_area: float
    ):
        """Updates tracklet state with a new frame detection."""
        self.end_frame = frame_idx
        self.end_time = timestamp
        self.lost_frames = 0
        self.is_active = True

        cx = float(np.mean(detection.quad[:, 0]))
        cy = float(np.mean(detection.quad[:, 1]))
        area_frac = float(cv2_polygon_area(detection.quad)) / max(1.0, frame_area)

        # Compute corner displacement from previous quad
        prev_quad = self.history[-1].quad
        disp = float(np.mean(np.linalg.norm(detection.quad - prev_quad, axis=1)))

        snapshot = TrackletFrameData(
            frame_idx=frame_idx,
            timestamp_sec=timestamp,
            quad=detection.quad.copy(),
            centroid=(cx, cy),
            area_fraction=area_frac,
            occlusion_ratio=occlusion_ratio,
            displacement=disp
        )
        self.history.append(snapshot)

    def mark_missed(self, frame_idx: int = 0, timestamp: float = 0.0, max_coast: int = 55):
        """
        Called when no detection matches this tracklet in the current frame.
        If within coast tolerance (<= max_coast), keeps the track alive by interpolating the previous quad
        so the ad overlay remains smoothly continuous without flickering.
        """
        self.lost_frames += 1
        if self.lost_frames <= max_coast and len(self.history) > 0:
            prev = self.history[-1]
            coasted_snapshot = TrackletFrameData(
                frame_idx=frame_idx if frame_idx > 0 else prev.frame_idx + 1,
                timestamp_sec=timestamp if timestamp > 0.0 else prev.timestamp_sec + 0.033,
                quad=prev.quad.copy(),
                centroid=prev.centroid,
                area_fraction=prev.area_fraction,
                occlusion_ratio=prev.occlusion_ratio,
                displacement=0.0
            )
            self.history.append(coasted_snapshot)
            self.end_frame = coasted_snapshot.frame_idx
            self.end_time = coasted_snapshot.timestamp_sec


def cv2_polygon_area(quad: np.ndarray) -> float:
    """Helper to compute polygon area via Shoelace formula."""
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class CentroidTracker:
    """
    Centroid & Euclidean distance tracker tailored for planar surfaces.
    Maintains tracklet lifecycle and smooth trajectories.
    """

    def __init__(self, tracker_cfg: TrackerConfig):
        self.cfg = tracker_cfg
        self.next_track_id = 1
        self.active_tracks: Dict[int, Tracklet] = {}
        self.completed_tracks: List[Tracklet] = []

    def _try_reidentify_candidate(
        self,
        cand: CandidateDetection,
        frame_idx: int,
        timestamp_sec: float,
        occluder_polys: list,
        canvas_shape: Tuple[int, int],
        frame_area: float
    ) -> Optional[Tracklet]:
        """
        Attempts to re-identify an incoming candidate with a recently completed tracklet
        that occupied the same spatial coordinates. If found, revives that tracklet,
        interpolates any missing frames across the gap, and returns it.
        """
        cx = float(np.mean(cand.quad[:, 0]))
        cy = float(np.mean(cand.quad[:, 1]))
        cand_bbox = (float(np.min(cand.quad[:, 0])), float(np.min(cand.quad[:, 1])),
                     float(np.max(cand.quad[:, 0])), float(np.max(cand.quad[:, 1])))

        max_reid_frames = getattr(self.cfg, "max_reid_frames", 120)
        max_reid_dist = getattr(self.cfg, "max_reid_distance", 80.0)

        best_match_idx = -1
        best_match_dist = float("inf")

        # Search completed tracks in reverse (most recently completed first)
        for i in range(len(self.completed_tracks) - 1, -1, -1):
            prev_track = self.completed_tracks[i]
            frame_gap = frame_idx - prev_track.end_frame
            if frame_gap > max_reid_frames:
                continue
            if frame_gap <= 0:
                continue

            tcx, tcy = prev_track.current_centroid
            dist = math.hypot(cx - tcx, cy - tcy)

            # Check bbox IoU as well
            prev_quad = prev_track.current_quad
            p_bbox = (float(np.min(prev_quad[:, 0])), float(np.min(prev_quad[:, 1])),
                      float(np.max(prev_quad[:, 0])), float(np.max(prev_quad[:, 1])))
            
            inter_x1 = max(cand_bbox[0], p_bbox[0])
            inter_y1 = max(cand_bbox[1], p_bbox[1])
            inter_x2 = min(cand_bbox[2], p_bbox[2])
            inter_y2 = min(cand_bbox[3], p_bbox[3])
            inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            b1_area = (cand_bbox[2] - cand_bbox[0]) * (cand_bbox[3] - cand_bbox[1])
            b2_area = (p_bbox[2] - p_bbox[0]) * (p_bbox[3] - p_bbox[1])
            iou = inter_area / max(1e-5, b1_area + b2_area - inter_area)

            if dist <= max_reid_dist or iou >= 0.35:
                if dist < best_match_dist:
                    best_match_dist = dist
                    best_match_idx = i
        
        if best_match_idx >= 0:
            revived = self.completed_tracks.pop(best_match_idx)
            # Interpolate missing frames across the gap
            gap_frames = frame_idx - revived.end_frame
            if gap_frames > 1 and len(revived.history) > 0:
                start_snap = revived.history[-1]
                start_q = start_snap.quad
                end_q = cand.quad
                start_t = start_snap.timestamp_sec
                end_t = timestamp_sec
                for step in range(1, gap_frames):
                    alpha = float(step) / float(gap_frames)
                    interp_q = (1.0 - alpha) * start_q + alpha * end_q
                    interp_t = (1.0 - alpha) * start_t + alpha * end_t
                    icx = float(np.mean(interp_q[:, 0]))
                    icy = float(np.mean(interp_q[:, 1]))
                    interp_snap = TrackletFrameData(
                        frame_idx=revived.end_frame + step,
                        timestamp_sec=interp_t,
                        quad=interp_q,
                        centroid=(icx, icy),
                        area_fraction=(1.0 - alpha) * start_snap.area_fraction + alpha * (cv2_polygon_area(cand.quad) / max(1.0, frame_area)),
                        occlusion_ratio=start_snap.occlusion_ratio,
                        displacement=0.0
                    )
                    revived.history.append(interp_snap)
            
            occ_ratio, _ = compute_occlusion_ratio(cand.quad, occluder_polys, canvas_shape)
            revived.update(cand, frame_idx, timestamp_sec, occ_ratio, frame_area)
            self.active_tracks[revived.track_id] = revived
            return revived
        
        return None

    def update(
        self,
        candidates: List[CandidateDetection],
        occluders: List[OccluderDetection],
        frame_idx: int,
        timestamp_sec: float,
        canvas_shape: Tuple[int, int]
    ) -> List[Tracklet]:
        """
        Associates candidates in the current frame with active tracklets.
        """
        h, w = canvas_shape
        frame_area = float(h * w)
        occluder_polys = [occ.poly for occ in occluders]

        # If no active tracks, initialize or re-identify candidates
        if len(self.active_tracks) == 0:
            for cand in candidates:
                occ_ratio, _ = compute_occlusion_ratio(cand.quad, occluder_polys, canvas_shape)
                revived = self._try_reidentify_candidate(cand, frame_idx, timestamp_sec, occluder_polys, canvas_shape, frame_area)
                if not revived:
                    track = Tracklet(self.next_track_id, cand, frame_idx, timestamp_sec, frame_area)
                    track.history[0].occlusion_ratio = occ_ratio
                    self.active_tracks[self.next_track_id] = track
                    self.next_track_id += 1
            return list(self.active_tracks.values())

        track_ids = list(self.active_tracks.keys())
        track_centroids = [self.active_tracks[tid].current_centroid for tid in track_ids]

        if len(candidates) == 0:
            # Mark all active tracks as missed
            for tid in list(self.active_tracks.keys()):
                t = self.active_tracks[tid]
                t.mark_missed(frame_idx, timestamp_sec, self.cfg.max_coast_frames)
                if t.lost_frames > self.cfg.max_lost_frames:
                    t.is_active = False
                    self.completed_tracks.append(t)
                    del self.active_tracks[tid]
            return [t for t in self.active_tracks.values() if t.is_active and t.lost_frames <= self.cfg.max_coast_frames]

        # Compute Euclidean distance matrix between track centroids and detection centroids
        cand_centroids = [
            (float(np.mean(c.quad[:, 0])), float(np.mean(c.quad[:, 1])))
            for c in candidates
        ]

        dist_matrix = np.zeros((len(track_ids), len(candidates)), dtype=np.float32)
        for i, (tx, ty) in enumerate(track_centroids):
            for j, (cx, cy) in enumerate(cand_centroids):
                dist_matrix[i, j] = math.hypot(tx - cx, ty - cy)

        # Greedy matching based on lowest distance
        matched_tracks = set()
        matched_cands = set()

        sorted_indices = np.dstack(np.unravel_index(np.argsort(dist_matrix.ravel()), dist_matrix.shape))[0]

        for r, c in sorted_indices:
            if r in matched_tracks or c in matched_cands:
                continue

            dist = dist_matrix[r, c]
            if dist > self.cfg.max_distance_threshold:
                continue

            # Match found
            tid = track_ids[r]
            cand = candidates[c]
            occ_ratio, _ = compute_occlusion_ratio(cand.quad, occluder_polys, canvas_shape)
            self.active_tracks[tid].update(cand, frame_idx, timestamp_sec, occ_ratio, frame_area)

            matched_tracks.add(r)
            matched_cands.add(c)

        # Handle unmatched active tracks
        for r, tid in enumerate(track_ids):
            if r not in matched_tracks:
                t = self.active_tracks[tid]
                t.mark_missed(frame_idx, timestamp_sec, self.cfg.max_coast_frames)
                if t.lost_frames > self.cfg.max_lost_frames:
                    t.is_active = False
                    self.completed_tracks.append(t)
                    del self.active_tracks[tid]

        # Handle unmatched new candidates (attempt re-identification first)
        for c, cand in enumerate(candidates):
            if c not in matched_cands:
                occ_ratio, _ = compute_occlusion_ratio(cand.quad, occluder_polys, canvas_shape)
                revived = self._try_reidentify_candidate(cand, frame_idx, timestamp_sec, occluder_polys, canvas_shape, frame_area)
                if not revived:
                    new_track = Tracklet(self.next_track_id, cand, frame_idx, timestamp_sec, frame_area)
                    new_track.history[0].occlusion_ratio = occ_ratio
                    self.active_tracks[self.next_track_id] = new_track
                    self.next_track_id += 1

        # Return tracks that are actively visible or coasting in the current frame
        return [t for t in self.active_tracks.values() if t.is_active and t.lost_frames <= self.cfg.max_coast_frames]

    def get_all_tracklets(self) -> List[Tracklet]:
        """Returns both active and finalized tracklets."""
        return list(self.active_tracks.values()) + self.completed_tracks

