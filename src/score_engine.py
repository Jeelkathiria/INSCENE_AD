"""
Multi-factor ScoreEngine for evaluating planar ad space suitability.
Implements the scoring formulation from the research paper:
'A Score-Based System for Detecting Optimal Planar Advertisement Spaces in Video'
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.config import ScoreWeights, TrackerConfig
from src.tracker import Tracklet, TrackletFrameData
from src.geometry import compute_homography_quality
from src.detector import CandidateDetection


def format_seconds_to_timeline(start_s: float, end_s: float) -> str:
    """Formats start and end seconds into clean M:SS - M:SS format."""
    s_int = max(0, int(round(start_s)))
    e_int = max(s_int + 1, int(round(end_s)))
    s_m, s_sec = divmod(s_int, 60)
    e_m, e_sec = divmod(e_int, 60)
    return f"{s_m}:{s_sec:02d} - {e_m}:{e_sec:02d}"


def are_cospatial_tracklets(
    t1: Tracklet,
    t2: Tracklet,
    max_centroid_dist: float = 75.0,
    min_iou: float = 0.30,
    max_gap_sec: float = 6.0
) -> bool:
    """
    Determines if two candidate tracklets correspond to the exact same physical billboard
    or planar ad space on screen based on spatial coordinates, bounding box overlap,
    area consistency, and class compatibility.
    """
    if len(t1.history) == 0 or len(t2.history) == 0:
        return False

    # Centroid distance between average spatial positions
    c1_x = float(np.mean([s.centroid[0] for s in t1.history]))
    c1_y = float(np.mean([s.centroid[1] for s in t1.history]))
    c2_x = float(np.mean([s.centroid[0] for s in t2.history]))
    c2_y = float(np.mean([s.centroid[1] for s in t2.history]))
    centroid_dist = math.hypot(c1_x - c2_x, c1_y - c2_y)

    # Average bounding box IoU
    def get_quad_bbox(quad: np.ndarray) -> Tuple[float, float, float, float]:
        xs = quad[:, 0]
        ys = quad[:, 1]
        return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))

    b1 = get_quad_bbox(t1.current_quad)
    b2 = get_quad_bbox(t2.current_quad)
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    iou = inter_area / max(1e-5, a1 + a2 - inter_area)

    # Area similarity ratio
    mean_a1 = float(np.mean([s.area_fraction for s in t1.history]))
    mean_a2 = float(np.mean([s.area_fraction for s in t2.history]))
    area_ratio = min(mean_a1, mean_a2) / max(1e-5, max(mean_a1, mean_a2))

    # Class compatibility
    planar_classes = {
        "billboard", "advertising board", "signboard", "poster", "display screen",
        "banner", "wall", "storefront", "shop sign", "blank wall", "whiteboard",
        "building facade"
    }
    c1_is_planar = t1.class_name in planar_classes
    c2_is_planar = t2.class_name in planar_classes
    classes_compatible = (c1_is_planar and c2_is_planar) or (t1.class_name == t2.class_name)

    if not classes_compatible:
        return False

    # Temporal gap
    t_earlier, t_later = (t1, t2) if t1.start_time <= t2.start_time else (t2, t1)
    time_gap = max(0.0, t_later.start_time - t_earlier.end_time)

    # Check match condition:
    # High IoU OR close centroid with consistent area
    is_spatial_match = (iou >= min_iou) or (centroid_dist <= max_centroid_dist and area_ratio >= 0.45)

    if is_spatial_match and (time_gap <= max_gap_sec or centroid_dist <= 45.0):
        return True

    return False


def consolidate_cospatial_tracklets(tracklets: List[Tracklet]) -> List[Tracklet]:
    """
    Consolidates fragmented tracklets that correspond to the exact same physical billboard
    or planar ad surface across the video.
    Merges their timelines, interpolates any missing frames across gaps, and combines their histories
    so that a single continuous ad placement is produced.
    """
    if len(tracklets) <= 1:
        return tracklets

    active_pool = list(tracklets)
    merged_any = True

    while merged_any:
        merged_any = False
        new_pool: List[Tracklet] = []
        used_indices = set()

        for i in range(len(active_pool)):
            if i in used_indices:
                continue
            base = active_pool[i]

            for j in range(i + 1, len(active_pool)):
                if j in used_indices:
                    continue
                cand = active_pool[j]

                if are_cospatial_tracklets(base, cand):
                    used_indices.add(j)
                    merged_any = True

                    # Determine chronological ordering
                    t_earlier, t_later = (base, cand) if base.start_time <= cand.start_time else (cand, base)

                    # Interpolate any intermediate frames between disjoint segments
                    gap_frames = t_later.start_frame - t_earlier.end_frame
                    interpolated_snaps = []
                    if gap_frames > 1 and len(t_earlier.history) > 0 and len(t_later.history) > 0:
                        start_snap = t_earlier.history[-1]
                        end_snap = t_later.history[0]
                        start_q = start_snap.quad
                        end_q = end_snap.quad
                        start_t = start_snap.timestamp_sec
                        end_t = end_snap.timestamp_sec
                        for step in range(1, gap_frames):
                            alpha = float(step) / float(gap_frames)
                            interp_q = (1.0 - alpha) * start_q + alpha * end_q
                            interp_t = (1.0 - alpha) * start_t + alpha * end_t
                            icx = float(np.mean(interp_q[:, 0]))
                            icy = float(np.mean(interp_q[:, 1]))
                            interp_snap = TrackletFrameData(
                                frame_idx=t_earlier.end_frame + step,
                                timestamp_sec=interp_t,
                                quad=interp_q,
                                centroid=(icx, icy),
                                area_fraction=(1.0 - alpha) * start_snap.area_fraction + alpha * end_snap.area_fraction,
                                occlusion_ratio=(1.0 - alpha) * start_snap.occlusion_ratio + alpha * end_snap.occlusion_ratio,
                                displacement=0.0
                            )
                            interpolated_snaps.append(interp_snap)

                    # Combine and deduplicate history
                    combined_hist = base.history + cand.history + interpolated_snaps
                    combined_hist.sort(key=lambda s: s.frame_idx)
                    seen_frames = set()
                    unique_hist = []
                    for snap in combined_hist:
                        if snap.frame_idx not in seen_frames:
                            seen_frames.add(snap.frame_idx)
                            unique_hist.append(snap)
                    base.history = unique_hist

                    base.start_frame = min(base.start_frame, cand.start_frame)
                    base.end_frame = max(base.end_frame, cand.end_frame)
                    base.start_time = min(base.start_time, cand.start_time)
                    base.end_time = max(base.end_time, cand.end_time)

                    # Upgrade class name if one is a more specific ad terminology
                    preferred_names = ["advertising board", "billboard", "display screen", "signboard"]
                    if cand.class_name in preferred_names and base.class_name not in preferred_names:
                        base.class_name = cand.class_name

            new_pool.append(base)

        active_pool = new_pool

    return active_pool


def split_tracklet_on_gaps(track: Tracklet, max_gap_sec: float = 2.0) -> List[Tracklet]:
    """
    Splits a tracklet into strictly continuous sub-tracklets if any internal temporal gap
    exceeds max_gap_sec (e.g. 2.0 seconds).
    This guarantees that every reported timeline slot (e.g. 0:00 - 0:30) has the object
    consistently present on screen.
    """
    if len(track.history) <= 1:
        return [track]

    segments: List[List[TrackletFrameData]] = []
    current_segment = [track.history[0]]

    for i in range(1, len(track.history)):
        prev_snap = track.history[i - 1]
        curr_snap = track.history[i]
        time_gap = curr_snap.timestamp_sec - prev_snap.timestamp_sec

        if time_gap > max_gap_sec:
            # Temporal gap detected (object disappeared and reappeared)
            segments.append(current_segment)
            current_segment = [curr_snap]
        else:
            current_segment.append(curr_snap)

    if current_segment:
        segments.append(current_segment)

    if len(segments) <= 1:
        return [track]

    split_tracks: List[Tracklet] = []
    for sub_idx, seg in enumerate(segments):
        if len(seg) == 0:
            continue
        xs = seg[0].quad[:, 0]
        ys = seg[0].quad[:, 1]
        bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
        new_t = Tracklet(
            track_id=track.track_id * 100 + sub_idx,
            initial_detection=CandidateDetection(
                bbox=bbox,
                quad=seg[0].quad.copy(),
                class_name=track.class_name,
                confidence=getattr(track, "confidence", 0.8)
            ),
            frame_idx=seg[0].frame_idx,
            timestamp_sec=seg[0].timestamp_sec,
            frame_area=1.0
        )
        new_t.history = seg
        new_t.start_frame = seg[0].frame_idx
        new_t.start_time = seg[0].timestamp_sec
        new_t.end_frame = seg[-1].frame_idx
        new_t.end_time = seg[-1].timestamp_sec
        new_t.is_active = False
        split_tracks.append(new_t)

    return split_tracks


@dataclass
class TrackletScore:
    """Detailed score report for an ad placement candidate tracklet."""
    track_id: int
    class_name: str
    start_time: float
    end_time: float
    duration: float
    total_frames: int

    # Individual factor scores (0.0 to 100.0)
    score_duration: float
    score_area: float
    score_occlusion: float
    score_stability: float
    score_geometry: float

    # Composite weighted score (0.0 to 100.0)
    final_score: float

    # Raw metrics
    mean_area_fraction: float
    mean_occlusion_ratio: float
    motion_jitter: float
    mean_geometry_score: float

    # Ranking and evaluation notes
    rank: int = 0
    is_recommended: bool = False
    audit_notes: List[str] = field(default_factory=list)

    # Sequential timeline slot allocation (non-overlapping ad windows)
    slot_start_time: float = 0.0
    slot_end_time: float = 0.0
    timeline_str: str = ""
    is_sequential_slot: bool = True

    def __post_init__(self):
        if not self.timeline_str:
            self.slot_start_time = self.start_time
            self.slot_end_time = self.end_time
            self.timeline_str = format_seconds_to_timeline(self.start_time, self.end_time)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes score data for JSON / CSV export."""
        return {
            "rank": self.rank,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "start_time_sec": round(self.slot_start_time, 2),
            "end_time_sec": round(self.slot_end_time, 2),
            "raw_start_time_sec": round(self.start_time, 2),
            "raw_end_time_sec": round(self.end_time, 2),
            "duration_sec": round(max(0.1, self.slot_end_time - self.slot_start_time), 2),
            "total_frames": self.total_frames,
            "timeline_slot_str": self.timeline_str,
            "is_sequential_slot": self.is_sequential_slot,
            "final_score": round(self.final_score, 1),
            "score_duration": round(self.score_duration, 1),
            "score_area": round(self.score_area, 1),
            "score_occlusion": round(self.score_occlusion, 1),
            "score_stability": round(self.score_stability, 1),
            "score_geometry": round(self.score_geometry, 1),
            "mean_area_pct": round(self.mean_area_fraction * 100, 2),
            "mean_occlusion_pct": round(self.mean_occlusion_ratio * 100, 2),
            "motion_jitter_px": round(self.motion_jitter, 2),
            "is_recommended": self.is_recommended,
            "audit_notes": "; ".join(self.audit_notes)
        }


class ScoreEngine:
    """
    Evaluates candidate tracklets using a mathematically grounded multi-factor score.
    Combines duration, scale, occlusion, motion stability, and planar geometry.
    """

    def __init__(self, weights: ScoreWeights, tracker_cfg: TrackerConfig):
        self.weights = weights
        self.tracker_cfg = tracker_cfg

    def evaluate_tracklet(self, track: Tracklet) -> TrackletScore:
        """Computes comprehensive score breakdown for a single candidate tracklet."""
        hist = track.history
        n_frames = len(hist)

        # 1. Screen-Time / Duration Score
        duration = track.duration_sec
        if duration < self.tracker_cfg.min_tracklet_duration_sec:
            # Harsh penalty for fleeting flashes (< 1 sec)
            s_duration = max(0.0, (duration / self.tracker_cfg.min_tracklet_duration_sec) * 30.0)
        else:
            # Linear ramp up to target optimal duration (e.g. 5 sec)
            ratio = min(1.0, duration / self.tracker_cfg.target_optimal_duration_sec)
            s_duration = 30.0 + 70.0 * ratio

        # 2. Surface Area / Scale Score
        areas = [f.area_fraction for f in hist]
        mean_area = float(np.mean(areas)) if areas else 0.0
        # Optimal candidate area is ~10% to 25% of frame
        if mean_area < 0.02:
            s_area = max(0.0, (mean_area / 0.02) * 50.0)
        elif 0.05 <= mean_area <= 0.35:
            # Ideal in-scene range
            s_area = 100.0
        elif mean_area < 0.05:
            s_area = 50.0 + 50.0 * ((mean_area - 0.02) / 0.03)
        else:
            # Overly large surfaces (>35%) begin to dominate the screen excessively
            s_area = max(40.0, 100.0 - ((mean_area - 0.35) / 0.30) * 60.0)

        # 3. Occlusion Penalty Score
        occs = [f.occlusion_ratio for f in hist]
        mean_occ = float(np.mean(occs)) if occs else 0.0
        # 0% occlusion -> 100 score; 50% occlusion -> 0 score
        s_occlusion = max(0.0, 100.0 * (1.0 - 2.0 * mean_occ))

        # 4. Motion Stability Score
        displacements = [f.displacement for f in hist[1:]] if n_frames > 1 else [0.0]
        motion_jitter = float(np.std(displacements)) if len(displacements) > 1 else 0.0
        # Exponential decay for high jitter
        s_stability = 100.0 * math.exp(-motion_jitter / 6.0)

        # 5. Geometric Quality & Orthogonality Score
        geom_scores = [compute_homography_quality(f.quad) for f in hist]
        mean_geom = float(np.mean(geom_scores)) if geom_scores else 50.0
        s_geometry = mean_geom

        # Composite weighted score
        w = self.weights
        total_weight = w.duration + w.area + w.occlusion + w.stability + w.geometry
        final_score = (
            w.duration * s_duration +
            w.area * s_area +
            w.occlusion * s_occlusion +
            w.stability * s_stability +
            w.geometry * s_geometry
        ) / max(1e-5, total_weight)

        final_score = float(np.clip(final_score, 0.0, 100.0))

        # Generate explanatory audit notes
        notes = []
        if duration >= self.tracker_cfg.target_optimal_duration_sec:
            notes.append(f"Excellent screen time ({duration:.1f}s)")
        elif duration < self.tracker_cfg.min_tracklet_duration_sec:
            notes.append(f"Short duration penalty ({duration:.1f}s)")

        if mean_occ < 0.05:
            notes.append("Virtually unoccluded")
        elif mean_occ > 0.25:
            notes.append(f"Heavily occluded ({mean_occ*100:.1f}%)")

        if motion_jitter < 2.0:
            notes.append("Smooth camera / planar stability")
        else:
            notes.append("Camera jitter / rapid motion detected")

        if mean_geom >= 80.0:
            notes.append("Clean planar quadrilateral geometry")

        is_recommended = (
            final_score >= 65.0 and
            duration >= self.tracker_cfg.min_tracklet_duration_sec and
            mean_occ <= 0.35
        )

        return TrackletScore(
            track_id=track.track_id,
            class_name=track.class_name,
            start_time=track.start_time,
            end_time=track.end_time,
            duration=duration,
            total_frames=n_frames,
            score_duration=s_duration,
            score_area=s_area,
            score_occlusion=s_occlusion,
            score_stability=s_stability,
            score_geometry=s_geometry,
            final_score=final_score,
            mean_area_fraction=mean_area,
            mean_occlusion_ratio=mean_occ,
            motion_jitter=motion_jitter,
            mean_geometry_score=mean_geom,
            is_recommended=is_recommended,
            audit_notes=notes
        )

    def evaluate_and_rank_all(self, tracklets: List[Tracklet]) -> List[TrackletScore]:
        """
        Evaluates and ranks all candidates from best to worst.
        Consolidates co-spatial tracklets representing the exact same physical billboard/space
        so that objects present throughout the clip are detected as ONE continuous ad placement.
        Allocates sequential, non-overlapping ad placement intervals along the video timeline.
        """
        # 1. Consolidate co-spatial tracklets (merges fragmented tracks of the same billboard)
        consolidated_tracklets = consolidate_cospatial_tracklets(tracklets)

        # 2. Enforce continuity: split any tracklets with prolonged internal gaps > max_gap
        max_gap = getattr(self.tracker_cfg, "max_continuous_gap_sec", 2.0)
        continuous_tracklets: List[Tracklet] = []
        for t in consolidated_tracklets:
            continuous_tracklets.extend(split_tracklet_on_gaps(t, max_gap_sec=max_gap))

        scores: List[TrackletScore] = []
        min_dur = getattr(self.tracker_cfg, "min_tracklet_duration_sec", 0.5)
        for t in continuous_tracklets:
            if t.total_frames < 3 or t.duration_sec < min_dur:
                # Filter out single-frame false detections and fleeting noise
                continue
            scores.append(self.evaluate_tracklet(t))

        # Sort descending by final score
        scores.sort(key=lambda s: s.final_score, reverse=True)

        for i, s in enumerate(scores):
            s.rank = i + 1

        # Sequential Ad Slot Partitioning (Enforce distinct, non-overlapping timelines)
        # If candidate #1 is 1:00 - 1:03, subsequent slots must be strictly beyond (> 1:03) or before (< 1:00).
        allocated_windows: List[Tuple[float, float, int]] = []  # (start, end, track_id)

        for s in scores:
            if not allocated_windows:
                # Rank 1 gets its full primary duration
                s.slot_start_time = s.start_time
                s.slot_end_time = s.end_time
                s.timeline_str = format_seconds_to_timeline(s.slot_start_time, s.slot_end_time)
                s.is_sequential_slot = True
                allocated_windows.append((s.slot_start_time, s.slot_end_time, s.track_id))
                continue

            # Check overlap against all allocated windows
            cand_start = s.start_time
            cand_end = s.end_time

            max_overlap = 0.0
            overlapping_window = None
            for w_start, w_end, _ in allocated_windows:
                overlap = max(0.0, min(cand_end, w_end) - max(cand_start, w_start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    overlapping_window = (w_start, w_end)

            if max_overlap < 0.6:
                # Completely distinct time slot (before or beyond)
                s.slot_start_time = cand_start
                s.slot_end_time = cand_end
                s.timeline_str = format_seconds_to_timeline(s.slot_start_time, s.slot_end_time)
                s.is_sequential_slot = True
                allocated_windows.append((s.slot_start_time, s.slot_end_time, s.track_id))
            else:
                # Overlaps significantly with an existing ad slot.
                # Attempt to allocate the portion strictly beyond or before that window.
                w_start, w_end = overlapping_window
                portion_beyond = (w_end, cand_end)
                portion_before = (cand_start, w_start)

                dur_beyond = portion_beyond[1] - portion_beyond[0]
                dur_before = portion_before[1] - portion_before[0]

                if dur_beyond >= 1.0:
                    # Allocate portion strictly beyond
                    s.slot_start_time = portion_beyond[0]
                    s.slot_end_time = portion_beyond[1]
                    s.timeline_str = format_seconds_to_timeline(s.slot_start_time, s.slot_end_time)
                    s.is_sequential_slot = True
                    s.audit_notes.append(f"Sequential slot scheduled beyond previous ad (from {s.timeline_str})")
                    allocated_windows.append((s.slot_start_time, s.slot_end_time, s.track_id))
                elif dur_before >= 1.0:
                    # Allocate portion strictly before
                    s.slot_start_time = portion_before[0]
                    s.slot_end_time = portion_before[1]
                    s.timeline_str = format_seconds_to_timeline(s.slot_start_time, s.slot_end_time)
                    s.is_sequential_slot = True
                    s.audit_notes.append(f"Sequential slot scheduled before next ad (from {s.timeline_str})")
                    allocated_windows.append((s.slot_start_time, s.slot_end_time, s.track_id))
                else:
                    # Fully enclosed or remaining time < 1.0s: mark as co-occurring alternative space
                    s.slot_start_time = cand_start
                    s.slot_end_time = cand_end
                    s.timeline_str = format_seconds_to_timeline(cand_start, cand_end)
                    s.is_sequential_slot = False
                    s.audit_notes.append("Co-temporal alternative surface during active ad slot")

        return scores

