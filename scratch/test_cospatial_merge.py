import numpy as np
import math
from typing import List, Tuple

class DummySnap:
    def __init__(self, frame_idx, t, quad, cx, cy, area=0.0158):
        self.frame_idx = frame_idx
        self.timestamp_sec = t
        self.quad = np.array(quad, dtype=np.float32)
        self.centroid = (cx, cy)
        self.area_fraction = area
        self.occlusion_ratio = 0.0
        self.displacement = 0.5

class DummyTrack:
    def __init__(self, tid, class_name, start_t, end_t, cx, cy, quad, area=0.0158):
        self.track_id = tid
        self.class_name = class_name
        self.start_frame = int(start_t * 30)
        self.end_frame = int(end_t * 30)
        self.start_time = start_t
        self.end_time = end_t
        self.current_centroid = (cx, cy)
        self.current_quad = np.array(quad, dtype=np.float32)
        # Create history snapshots every 3 frames
        self.history = []
        for f in range(self.start_frame, self.end_frame + 1, 3):
            self.history.append(DummySnap(f, f / 30.0, quad, cx, cy, area))

def bbox_from_quad(quad: np.ndarray) -> Tuple[float, float, float, float]:
    xs = quad[:, 0]
    ys = quad[:, 1]
    return float(np.min(xs)), float(np.min(ys)), float(np.max(xs)), float(np.max(ys))

def bbox_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / max(1e-5, union)

def are_cospatial(t1, t2, max_dist=75.0, min_iou=0.30) -> bool:
    # 1. Centroid distance
    c1 = (float(np.mean([s.centroid[0] for s in t1.history])), float(np.mean([s.centroid[1] for s in t1.history])))
    c2 = (float(np.mean([s.centroid[0] for s in t2.history])), float(np.mean([s.centroid[1] for s in t2.history])))
    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    # 2. IoU
    b1 = bbox_from_quad(t1.current_quad)
    b2 = bbox_from_quad(t2.current_quad)
    iou = bbox_iou(b1, b2)

    # 3. Area similarity
    a1 = float(np.mean([s.area_fraction for s in t1.history]))
    a2 = float(np.mean([s.area_fraction for s in t2.history]))
    area_sim = min(a1, a2) / max(1e-5, max(a1, a2))

    # Match if either high IoU or close centroid with similar area
    if (dist <= max_dist and area_sim >= 0.50) or iou >= min_iou:
        return True
    return False

# Test with our 9 CITY_VIDEO tracklets
quad_base = [[540, 200], [700, 210], [695, 340], [538, 335]]
cx_base, cy_base = 618.0, 271.0

tracks = [
    DummyTrack(1, "banner", 0.0, 8.11, cx_base, cy_base, quad_base),
    DummyTrack(3, "advertising board", 8.91, 10.21, cx_base + 2, cy_base - 1, quad_base),
    DummyTrack(5, "banner", 11.01, 12.11, cx_base + 1, cy_base + 1, quad_base),
    DummyTrack(6, "banner", 12.41, 16.22, cx_base, cy_base, quad_base),
    DummyTrack(8, "advertising board", 16.82, 20.92, cx_base + 3, cy_base, quad_base),
    DummyTrack(10, "banner", 22.62, 23.12, cx_base - 1, cy_base + 2, quad_base),
    DummyTrack(11, "banner", 23.72, 26.23, cx_base, cy_base, quad_base),
    DummyTrack(12, "banner", 26.83, 29.03, cx_base + 1, cy_base - 1, quad_base),
    DummyTrack(13, "advertising board", 29.23, 30.23, cx_base, cy_base, quad_base),
]

print("Initial count:", len(tracks))

# Test pairwise consolidation
merged = True
while merged:
    merged = False
    new_tracks = []
    used = set()
    for i in range(len(tracks)):
        if i in used:
            continue
        base = tracks[i]
        for j in range(i + 1, len(tracks)):
            if j in used:
                continue
            cand = tracks[j]
            if are_cospatial(base, cand):
                # Merge cand into base
                used.add(j)
                merged = True
                # Expand times
                base.start_time = min(base.start_time, cand.start_time)
                base.end_time = max(base.end_time, cand.end_time)
                base.start_frame = min(base.start_frame, cand.start_frame)
                base.end_frame = max(base.end_frame, cand.end_frame)
                # Combine histories
                base.history.extend(cand.history)
                base.history.sort(key=lambda s: s.frame_idx)
                # Upgrade class name if advertising board
                if cand.class_name == "advertising board":
                    base.class_name = "advertising board"
        new_tracks.append(base)
    tracks = new_tracks

print("Final consolidated count:", len(tracks))
for t in tracks:
    print(f"Track {t.track_id} ({t.class_name}): {t.start_time:.2f}s - {t.end_time:.2f}s (dur: {t.end_time - t.start_time:.2f}s), total snaps: {len(t.history)}")

