"""
Configuration settings for AI-based In-Scene Advertising.
Project: A Score-Based System for Detecting Optimal Planar Advertisement Spaces in Video
"""

import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Create data directories if they don't exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


@dataclass
class ScoreWeights:
    """Weights for the multi-factor ScoreEngine."""
    duration: float = 0.25      # Screen-time / continuous visibility
    area: float = 0.20          # Surface scale relative to frame
    occlusion: float = 0.25     # Penalty for foreground occluders
    stability: float = 0.15     # Motion smoothness / low jitter
    geometry: float = 0.15      # Planarity, internal angles, rectangularity


@dataclass
class DetectorConfig:
    """Parameters for candidate surface and occluder detection."""
    model_name: str = "yolov8s-worldv2.pt"
    fallback_model: str = "yolov8n.pt"
    conf_threshold: float = 0.18        # Sensitive enough for distant/angled ad planes
    iou_threshold: float = 0.45
    inference_imgsz: int = 480          # Fast inference resolution (speeds up CPU by 3x)
    frame_stride: int = 3               # Process every 3rd frame (speeds up video by 3x)

    # Open-vocabulary prompts for candidate planar surfaces & transit ad vehicles
    candidate_classes: List[str] = field(default_factory=lambda: [
        "billboard",
        "advertising board",
        "signboard",
        "poster",
        "display screen",
        "banner",
        "bus",
        "truck",
        "car",
        "sports car",
        "side of car",
        "car door",
        "side of truck",
        "side of bus",
        "building facade",
        "wall",
        "storefront",
        "shop sign",
        "blank wall",
        "whiteboard"
    ])

    # Open-vocabulary prompts for foreground occluders
    occluder_classes: List[str] = field(default_factory=lambda: [
        "person",
        "pedestrian",
        "pole",
        "lamp post",
        "traffic sign",
        "tree branch",
        "umbrella"
    ])

    # Fallback standard COCO mappings when using base YOLO models
    coco_candidate_classes: List[str] = field(default_factory=lambda: [
        "tv", "laptop", "refrigerator", "book"
    ])
    coco_occluder_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "chair"
    ])


@dataclass
class GeometryConfig:
    """Thresholds for 4-point quadrilateral verification."""
    min_area_fraction: float = 0.015    # Min candidate area (1.5% of total frame area)
    max_area_fraction: float = 0.75     # Max candidate area (75% of total frame area)
    min_internal_angle_deg: float = 30.0  # Reject hyper-acute perspective angles
    max_internal_angle_deg: float = 150.0
    epsilon_approx_ratio: float = 0.03  # Douglas-Peucker approximation factor


@dataclass
class TrackerConfig:
    """Tracking and tracklet management settings."""
    max_distance_threshold: float = 140.0 # Max pixel distance for centroid matching
    max_lost_frames: int = 60            # Keep track alive for up to 60 frames (~2.0s) through dropouts
    max_coast_frames: int = 55           # Smoothly interpolate across missed frames so the box never flickers or disappears
    max_continuous_gap_sec: float = 2.0  # Split tracklet only if an unobserved gap exceeds 2.0s
    min_tracklet_duration_sec: float = 0.5  # Filter out very short flashes (< 0.5 sec)
    target_optimal_duration_sec: float = 5.0 # Max score achieved around this duration
    max_reid_distance: float = 80.0      # Spatial radius (px) to re-identify and revive recently lost surfaces
    max_reid_frames: int = 120           # Memory window (~4.0s) to revive a temporarily lost tracklet


@dataclass
class SystemConfig:
    """Master configuration class."""
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    target_fps: int = 30

