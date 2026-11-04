"""
Automated unit and integration tests for AI-based In-Scene Advertising.
Validates geometry, occlusion, tracker, and ScoreEngine logic.
"""

import pytest
import numpy as np

from src.config import SystemConfig, ScoreWeights, TrackerConfig, GeometryConfig
from src.geometry import (
    order_points,
    extract_quad_from_contour,
    compute_quad_angles,
    is_valid_planar_quad,
    compute_occlusion_ratio,
    compute_homography_quality
)
from src.detector import CandidateDetection, OccluderDetection
from src.tracker import CentroidTracker
from src.score_engine import ScoreEngine


def test_order_points():
    """Verifies that 4 points are sorted correctly in clockwise order."""
    pts = np.array([[200, 200], [100, 100], [200, 100], [100, 200]], dtype=np.float32)
    ordered = order_points(pts)

    # Expected: TL=(100,100), TR=(200,100), BR=(200,200), BL=(100,200)
    assert np.allclose(ordered[0], [100, 100])
    assert np.allclose(ordered[1], [200, 100])
    assert np.allclose(ordered[2], [200, 200])
    assert np.allclose(ordered[3], [100, 200])


def test_quad_angles_and_homography():
    """Verifies internal angles of a perfect rectangle equal 90 degrees."""
    rect = np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32)
    angles = compute_quad_angles(rect)

    for a in angles:
        assert abs(a - 90.0) < 1.0

    # Orthogonality score for perfect rectangle should be 100
    quality = compute_homography_quality(rect)
    assert quality >= 99.0


def test_occlusion_ratio_calculation():
    """Tests exact occlusion intersection calculation."""
    # 100x100 square candidate
    cand_quad = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)

    # Occluder covering exactly right half of candidate: [150, 100] to [200, 200]
    occ_box = np.array([[150, 100], [200, 100], [200, 200], [150, 200]], dtype=np.float32)

    ratio, _ = compute_occlusion_ratio(cand_quad, [occ_box], (400, 400))
    # Expected ratio ~0.50 (50%)
    assert abs(ratio - 0.50) < 0.02


def test_score_engine_preferences():
    """
    Validates that an optimal, unoccluded candidate receives a substantially
    higher fitness score than an occluded, fleeting candidate.
    """
    weights = ScoreWeights()
    tracker_cfg = TrackerConfig(min_tracklet_duration_sec=1.0, target_optimal_duration_sec=4.0)
    engine = ScoreEngine(weights, tracker_cfg)

    # 1. Ideal Candidate
    det1 = CandidateDetection(
        bbox=(100, 100, 300, 200),
        quad=np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32),
        class_name="billboard",
        confidence=0.9
    )
    from src.tracker import Tracklet
    track_ideal = Tracklet(1, det1, frame_idx=0, timestamp=0.0, frame_area=640*360)
    # Simulate 4 seconds of smooth visibility with 0 occlusion
    for f in range(1, 120):
        track_ideal.update(det1, frame_idx=f, timestamp=f/30.0, occlusion_ratio=0.0, frame_area=640*360)

    score_ideal = engine.evaluate_tracklet(track_ideal)

    # 2. Fleeting, Occluded Candidate (only 0.5s visible, 40% occluded)
    det2 = CandidateDetection(
        bbox=(50, 50, 150, 100),
        quad=np.array([[50, 50], [150, 50], [150, 100], [50, 100]], dtype=np.float32),
        class_name="poster",
        confidence=0.7
    )
    track_poor = Tracklet(2, det2, frame_idx=0, timestamp=0.0, frame_area=640*360)
    for f in range(1, 15):
        track_poor.update(det2, frame_idx=f, timestamp=f/30.0, occlusion_ratio=0.40, frame_area=640*360)

    score_poor = engine.evaluate_tracklet(track_poor)

    assert score_ideal.final_score > score_poor.final_score
    assert score_ideal.is_recommended is True
    assert score_poor.is_recommended is False

