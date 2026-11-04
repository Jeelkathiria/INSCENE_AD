"""
End-to-end Video Analysis Pipeline for Ad Space Identification.
Processes video frames, maintains surface tracklets, scores candidates,
annotates frames with diagnostic overlays, and exports structured reports.
"""

import os
import json
import csv
import time
import base64
import logging
import queue
import threading
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any

from src.config import SystemConfig, OUTPUTS_DIR
from src.detector import SurfaceDetector
from src.tracker import CentroidTracker, Tracklet
from src.score_engine import ScoreEngine, TrackletScore

logger = logging.getLogger(__name__)


def ensure_browser_compatible_h264(video_path: str):
    """
    Re-encodes MP4 video to standard H.264 (yuv420p) so all web browsers
    (Chrome, Edge, Firefox, Safari) can decode and play it natively in HTML5 <video>.
    """
    if not video_path or not os.path.exists(video_path):
        return
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        temp_h264 = video_path + ".browser.mp4"
        cmd = [
            ffmpeg_exe, "-y",
            "-i", video_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "22",
            temp_h264
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(temp_h264) and os.path.getsize(temp_h264) > 1000:
            time.sleep(0.3)
            try:
                os.replace(temp_h264, video_path)
            except PermissionError:
                time.sleep(1.0)
                os.replace(temp_h264, video_path)
            logger.info("Successfully converted %s to browser-compatible H.264", video_path)
    except Exception as e:
        logger.warning("Browser H264 conversion note: %s", e)


class ThreadedFrameReader:
    """
    Decoupled Asynchronous Frame Extraction Worker (Producer Thread).
    Continuously decodes video frames into a memory buffer queue in a background thread,
    enabling frame extraction and neural detection/tracking to execute in true concurrent
    parallel streaming without blocking on disk IO.
    """
    def __init__(self, cap: cv2.VideoCapture, max_queue_size: int = 48):
        self.cap = cap
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader_worker, daemon=True)
        self.thread.start()

    def _reader_worker(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.queue.put((False, None))
                break
            self.queue.put((True, frame))

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.queue.get()

    def stop(self):
        self.stopped = True
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class VideoPlacementPipeline:
    """
    Coordinates detection, tracking, scoring, and visualization
    to identify optimal ad placement spaces in a video stream.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.detector = SurfaceDetector(self.config.detector, self.config.geometry)
        self.tracker = CentroidTracker(self.config.tracker)
        self.score_engine = ScoreEngine(self.config.score_weights, self.config.tracker)

    def draw_annotations(
        self,
        frame: np.ndarray,
        active_tracks: List[Tracklet],
        scored_tracks: Dict[int, TrackletScore],
        occluders: list,
        timestamp_sec: float
    ) -> np.ndarray:
        """
        Renders HUD overlays, quadrilateral boundaries, corner points,
        and score badges directly onto the video frame.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw occluders with semi-transparent red overlay
        if occluders:
            overlay = annotated.copy()
            for occ in occluders:
                pts = occ.poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 0, 220))
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # 2. Draw candidate surfaces
        for track in active_tracks:
            tid = track.track_id
            score_data = scored_tracks.get(tid)
            is_rank_1 = (score_data and score_data.rank == 1)

            # Color scheme: Bright Neon Green for Selected Rank 1 Ad Square, Cyan for secondary
            if is_rank_1:
                color = (0, 255, 0)        # Bright Green (Selected ad placement square)
                thickness = 3
                fill_alpha = 0.28
            elif score_data and score_data.final_score >= 60.0:
                color = (0, 230, 255)      # Cyan / Yellow (Alternative ad space)
                thickness = 2
                fill_alpha = 0.15
            else:
                color = (0, 165, 255)      # Orange
                thickness = 2
                fill_alpha = 0.12

            quad = track.current_quad.astype(np.int32)
            pts = quad.reshape((-1, 1, 2))

            # Draw polygon perimeter (Selected Square Boundary)
            cv2.polylines(annotated, [pts], True, color, thickness, cv2.LINE_AA)

            # High-visibility semi-transparent fill for candidate ad canvas
            cand_overlay = annotated.copy()
            cv2.fillPoly(cand_overlay, [pts], color)
            cv2.addWeighted(cand_overlay, fill_alpha, annotated, 1.0 - fill_alpha, 0, annotated)

            # 3D Perspective Grid Mesh (projectively interpolated cross-sections conforming to depth)
            p1, p2, p3, p4 = track.current_quad[0], track.current_quad[1], track.current_quad[2], track.current_quad[3]
            grid_overlay = annotated.copy()
            for t in [0.33, 0.67]:
                pt_top = p1 * (1 - t) + p2 * t
                pt_bot = p4 * (1 - t) + p3 * t
                cv2.line(grid_overlay, (int(pt_top[0]), int(pt_top[1])), (int(pt_bot[0]), int(pt_bot[1])), (255, 255, 255), 1, cv2.LINE_AA)
                pt_l = p1 * (1 - t) + p4 * t
                pt_r = p2 * (1 - t) + p3 * t
                cv2.line(grid_overlay, (int(pt_l[0]), int(pt_l[1])), (int(pt_r[0]), int(pt_r[1])), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(grid_overlay, 0.35, annotated, 0.65, 0, annotated)

            # Draw center label inside selected ad square
            cx = int(np.mean(quad[:, 0]))
            cy = int(np.mean(quad[:, 1]))
            center_txt = "* 3D SELECTED AD SQUARE *" if is_rank_1 else f"3D AD SPACE #{score_data.rank if score_data else tid}"
            (ctw, cth), _ = cv2.getTextSize(center_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (cx - ctw//2 - 6, cy - cth//2 - 4), (cx + ctw//2 + 6, cy + cth//2 + 4), (10, 10, 10), -1)
            cv2.rectangle(annotated, (cx - ctw//2 - 6, cy - cth//2 - 4), (cx + ctw//2 + 6, cy + cth//2 + 4), color, 1)
            cv2.putText(annotated, center_txt, (cx - ctw//2, cy + cth//2 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 3D Coordinate Triad [X, Y, Z_3D] at centroid
            u_dir = (p2 - p1).astype(float)
            u_norm = max(1e-5, np.linalg.norm(u_dir))
            u_unit = u_dir / u_norm

            v_dir = (p4 - p1).astype(float)
            v_norm = max(1e-5, np.linalg.norm(v_dir))
            v_unit = v_dir / v_norm

            x_end = (int(cx + u_unit[0] * 32), int(cy + u_unit[1] * 32))
            cv2.arrowedLine(annotated, (cx, cy), x_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)

            y_end = (int(cx + v_unit[0] * 28), int(cy + v_unit[1] * 28))
            cv2.arrowedLine(annotated, (cx, cy), y_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)

            n_dir = np.array([-v_unit[1], v_unit[0]])
            if n_dir[1] > 0:
                n_dir = -n_dir
            z_end = (int(cx + n_dir[0] * 26), int(cy + n_dir[1] * 26))
            cv2.arrowedLine(annotated, (cx, cy), z_end, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(annotated, "3D", (z_end[0] + 3, z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

            # Draw 4 corner dots [P1, P2, P3, P4]
            labels = ["P1(TL)", "P2(TR)", "P3(BR)", "P4(BL)"]
            for idx, pt in enumerate(quad):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 3, color, -1, cv2.LINE_AA)
                cv2.putText(
                    annotated, labels[idx], (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA
                )

            # Compute 3D Depth Foreshortening Ratio
            h_left = abs(p4[1] - p1[1])
            h_right = abs(p3[1] - p2[1])
            depth_ratio = float(h_right / max(1.0, h_left))
            if depth_ratio < 0.90:
                depth_str = " | 3D: Far-Right"
            elif depth_ratio > 1.10:
                depth_str = " | 3D: Far-Left"
            else:
                depth_str = " | 3D: Frontal"

            # Draw floating HUD badge above candidate
            top_pt = quad[0] if quad[0][1] < quad[1][1] else quad[1]
            bx = max(10, int(top_pt[0]) - 40)
            by = max(30, int(top_pt[1]) - 15)

            score_str = f"Score: {score_data.final_score:.1f}" if score_data else "Evaluating..."
            rank_str = f"#{score_data.rank} " if (score_data and score_data.rank > 0) else ""
            badge_text = f"[{rank_str}{track.class_name.upper()} | ID:{tid} | {score_str}{depth_str}]"

            # Badge background banner
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (bx - 4, by - th - 6), (bx + tw + 6, by + 6), (20, 20, 20), -1)
            cv2.rectangle(annotated, (bx - 4, by - th - 6), (bx + tw + 6, by + 6), color, 1)
            cv2.putText(
                annotated, badge_text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        # 3. Draw Header HUD Bar
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 15, 15), -1)
        hud_info = (
            f"In-Scene Ad Space Placement Analyzer | Time: {timestamp_sec:05.2f}s | "
            f"Active Surfaces: {len(active_tracks)}"
        )
        cv2.putText(
            annotated, hud_info, (15, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA
        )

        return annotated

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None
    ) -> Tuple[List[TrackletScore], Dict[str, str]]:
        """
        Executes analysis on the input video.
        Returns:
            (ranked_scores, file_export_paths)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path requested
        writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        scored_lookup: Dict[int, TrackletScore] = {}
        stride = getattr(self.config.detector, "frame_stride", 3)
        last_candidates = []
        last_occluders = []
        start_processing_time = time.time()

        # Decoupled Asynchronous Producer-Consumer Pipeline:
        # Worker 1 continuously extracts frames in background thread into a prefetch queue
        # Worker 2 processes neural surface detection & tracking concurrently in parallel
        reader = ThreadedFrameReader(cap, max_queue_size=48)

        try:
            while True:
                ret, frame = reader.read()
                if not ret or frame is None:
                    break

                timestamp_sec = frame_idx / fps

                # 1. Detect candidate surfaces & occluders (Keyframe stride optimization)
                if frame_idx % stride == 0 or frame_idx == 0:
                    candidates, occluders = self.detector.detect(frame)
                    last_candidates = candidates
                    last_occluders = occluders
                else:
                    candidates = last_candidates
                    occluders = last_occluders

                # 2. Update tracking
                active_tracks = self.tracker.update(
                    candidates, occluders, frame_idx, timestamp_sec, (height, width)
                )

                # 3. Live scoring update
                for track in active_tracks:
                    score = self.score_engine.evaluate_tracklet(track)
                    scored_lookup[track.track_id] = score

                # Assign dynamic ranks among active candidates
                active_scores = [scored_lookup[t.track_id] for t in active_tracks if t.track_id in scored_lookup]
                active_scores.sort(key=lambda s: s.final_score, reverse=True)
                for r_idx, s in enumerate(active_scores):
                    s.rank = r_idx + 1

                # 4. Render overlay frame
                annotated_frame = self.draw_annotations(
                    frame, active_tracks, scored_lookup, occluders, timestamp_sec
                )

                if writer is not None:
                    writer.write(annotated_frame)

                if progress_callback:
                    # Generate lightweight preview thumbnail for live UI visualizer
                    preview_b64 = None
                    if frame_idx % 2 == 0 or frame_idx == total_frames - 1:
                        try:
                            thumb = cv2.resize(annotated_frame, (320, 180))
                            _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 55])
                            preview_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")
                        except Exception:
                            pass

                    elapsed = time.time() - start_processing_time
                    fps_calc = (frame_idx + 1) / max(0.001, elapsed)

                    try:
                        progress_callback(
                            frame_idx + 1,
                            total_frames,
                            timestamp_sec,
                            len(active_tracks),
                            preview_b64,
                            round(elapsed, 1),
                            round(fps_calc, 1)
                        )
                    except TypeError:
                        try:
                            progress_callback(frame_idx + 1, total_frames, timestamp_sec, len(active_tracks))
                        except TypeError:
                            progress_callback(frame_idx + 1, total_frames, timestamp_sec)

                frame_idx += 1

        finally:
            reader.stop()
            cap.release()
            if writer is not None:
                writer.release()
                if output_video_path and os.path.exists(output_video_path):
                    ensure_browser_compatible_h264(output_video_path)

        total_processing_time = round(time.time() - start_processing_time, 2)
        avg_processing_fps = round(total_frames / max(0.001, total_processing_time), 1)
        video_duration_sec = round(total_frames / fps, 2)
        speedup_factor = round(video_duration_sec / max(0.001, total_processing_time), 2)

        # Final evaluation and ranking across the full video
        all_tracks = self.tracker.get_all_tracklets()
        final_scores = self.score_engine.evaluate_and_rank_all(all_tracks)

        # Build chronological sequential timeline summary
        sequential_slots = [s for s in final_scores if getattr(s, "is_sequential_slot", True)]
        sequential_slots.sort(key=lambda s: getattr(s, "slot_start_time", s.start_time))
        timeline_summary = " , ".join([s.timeline_str for s in sequential_slots]) if sequential_slots else "No sequential intervals"

        metadata = {
            "pipeline_architecture": "Pipelined Parallel Stream (Producer-Consumer Queue)",
            "total_processing_time_sec": total_processing_time,
            "processing_fps": avg_processing_fps,
            "total_frames_extracted": total_frames,
            "video_duration_sec": video_duration_sec,
            "speedup_factor": speedup_factor,
            "sequential_timeline_summary": timeline_summary
        }

        # Export reports
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.json")
        csv_path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.csv")

        self.export_reports(final_scores, json_path, csv_path, metadata=metadata)

        exports = {
            "json": json_path,
            "csv": csv_path,
            "video": output_video_path or "",
            "metadata": metadata
        }
        return final_scores, exports

    def process_image(
        self,
        image_path: str,
        output_image_path: Optional[str] = None
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Analyzes a single still photo/image to detect and score planar ad placement spaces.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        start_photo_time = time.time()
        frame = cv2.imread(image_path)
        if frame is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        h, w = frame.shape[:2]
        frame_area = float(h * w)

        # Detect candidate surfaces and occluders
        candidates, occluders = self.detector.detect(frame)

        # Occluder polygons for overlap
        occluder_polys = [occ.poly for occ in occluders]

        from src.geometry import compute_occlusion_ratio, compute_homography_quality
        from src.tracker import cv2_polygon_area

        scored_candidates = []
        for idx, cand in enumerate(candidates):
            quad = cand.quad
            area = float(cv2_polygon_area(quad))
            area_frac = area / max(1.0, frame_area)

            # Area score: optimal 5% - 35%
            if area_frac < 0.02:
                s_area = max(0.0, (area_frac / 0.02) * 50.0)
            elif 0.05 <= area_frac <= 0.35:
                s_area = 100.0
            elif area_frac < 0.05:
                s_area = 50.0 + 50.0 * ((area_frac - 0.02) / 0.03)
            else:
                s_area = max(40.0, 100.0 - ((area_frac - 0.35) / 0.30) * 60.0)

            # Occlusion score
            occ_ratio, _ = compute_occlusion_ratio(quad, occluder_polys, (h, w))
            s_occlusion = max(0.0, 100.0 * (1.0 - 2.0 * occ_ratio))

            # Geometry score
            s_geometry = compute_homography_quality(quad)

            # Composite photo score (Area 35%, Occlusion 35%, Geometry 30%)
            final_score = float(0.35 * s_area + 0.35 * s_occlusion + 0.30 * s_geometry)

            min_x = int(np.min(quad[:, 0]))
            max_x = int(np.max(quad[:, 0]))
            min_y = int(np.min(quad[:, 1]))
            max_y = int(np.max(quad[:, 1]))
            width_px = max_x - min_x
            height_px = max_y - min_y

            scored_candidates.append({
                "candidate_id": idx + 1,
                "class_name": cand.class_name,
                "final_score": round(final_score, 1),
                "score_area": round(s_area, 1),
                "score_occlusion": round(s_occlusion, 1),
                "score_geometry": round(s_geometry, 1),
                "mean_area_pct": round(area_frac * 100, 2),
                "mean_occlusion_pct": round(occ_ratio * 100, 2),
                "position_xy": f"({min_x}, {min_y})",
                "dimensions_wh": f"{width_px}x{height_px}px",
                "quad_points": quad.tolist(),
                "is_recommended": (final_score >= 65.0 and occ_ratio <= 0.35),
                "audit_notes": f"Canvas {width_px}x{height_px}px ({area_frac*100:.1f}% area); Occlusion: {occ_ratio*100:.1f}%; Planar score: {s_geometry:.1f}"
            })

        # Sort descending by score
        scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)
        for rank_idx, c in enumerate(scored_candidates):
            c["rank"] = rank_idx + 1

        # Render annotated image
        annotated = frame.copy()

        # Draw occluders
        if occluders:
            overlay = annotated.copy()
            for occ in occluders:
                pts = occ.poly.astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 0, 220))
                cv2.polylines(annotated, [pts], True, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        # Draw candidates
        for c in scored_candidates:
            is_rank_1 = (c["rank"] == 1)
            color = (0, 255, 0) if is_rank_1 else (0, 230, 255)
            thickness = 3 if is_rank_1 else 2
            fill_alpha = 0.28 if is_rank_1 else 0.15

            quad = np.array(c["quad_points"], dtype=np.float32)
            pts = quad.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, color, thickness, cv2.LINE_AA)

            cand_overlay = annotated.copy()
            cv2.fillPoly(cand_overlay, [pts], color)
            cv2.addWeighted(cand_overlay, fill_alpha, annotated, 1.0 - fill_alpha, 0, annotated)

            # 3D Perspective Grid Mesh (projectively interpolated cross-sections conforming to depth)
            p1, p2, p3, p4 = quad[0], quad[1], quad[2], quad[3]
            grid_overlay = annotated.copy()
            for t in [0.33, 0.67]:
                pt_top = p1 * (1 - t) + p2 * t
                pt_bot = p4 * (1 - t) + p3 * t
                cv2.line(grid_overlay, (int(pt_top[0]), int(pt_top[1])), (int(pt_bot[0]), int(pt_bot[1])), (255, 255, 255), 1, cv2.LINE_AA)
                pt_l = p1 * (1 - t) + p4 * t
                pt_r = p2 * (1 - t) + p3 * t
                cv2.line(grid_overlay, (int(pt_l[0]), int(pt_l[1])), (int(pt_r[0]), int(pt_r[1])), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.addWeighted(grid_overlay, 0.35, annotated, 0.65, 0, annotated)

            cx = int(np.mean(quad[:, 0]))
            cy = int(np.mean(quad[:, 1]))
            center_txt = "* 3D SELECTED AD SQUARE *" if is_rank_1 else f"3D AD SPACE #{c['rank']}"
            (ctw, cth), _ = cv2.getTextSize(center_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (cx - ctw//2 - 6, cy - cth//2 - 4), (cx + ctw//2 + 6, cy + cth//2 + 4), (10, 10, 10), -1)
            cv2.rectangle(annotated, (cx - ctw//2 - 6, cy - cth//2 - 4), (cx + ctw//2 + 6, cy + cth//2 + 4), color, 1)
            cv2.putText(annotated, center_txt, (cx - ctw//2, cy + cth//2 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # 3D Coordinate Triad [X, Y, Z_3D] at centroid
            u_dir = (p2 - p1).astype(float)
            u_norm = max(1e-5, np.linalg.norm(u_dir))
            u_unit = u_dir / u_norm

            v_dir = (p4 - p1).astype(float)
            v_norm = max(1e-5, np.linalg.norm(v_dir))
            v_unit = v_dir / v_norm

            x_end = (int(cx + u_unit[0] * 32), int(cy + u_unit[1] * 32))
            cv2.arrowedLine(annotated, (cx, cy), x_end, (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25)

            y_end = (int(cx + v_unit[0] * 28), int(cy + v_unit[1] * 28))
            cv2.arrowedLine(annotated, (cx, cy), y_end, (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)

            n_dir = np.array([-v_unit[1], v_unit[0]])
            if n_dir[1] > 0:
                n_dir = -n_dir
            z_end = (int(cx + n_dir[0] * 26), int(cy + n_dir[1] * 26))
            cv2.arrowedLine(annotated, (cx, cy), z_end, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(annotated, "3D", (z_end[0] + 3, z_end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

            labels = ["P1(TL)", "P2(TR)", "P3(BR)", "P4(BL)"]
            for p_idx, pt in enumerate(quad):
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (px, py), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), 3, color, -1, cv2.LINE_AA)
                cv2.putText(annotated, labels[p_idx], (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)

            # Floating HUD Badge with 3D Depth status
            h_left = abs(p4[1] - p1[1])
            h_right = abs(p3[1] - p2[1])
            depth_ratio = float(h_right / max(1.0, h_left))
            if depth_ratio < 0.90:
                depth_str = " | 3D: Far-Right"
            elif depth_ratio > 1.10:
                depth_str = " | 3D: Far-Left"
            else:
                depth_str = " | 3D: Frontal"

            top_pt = quad[0] if quad[0][1] < quad[1][1] else quad[1]
            bx = max(10, int(top_pt[0]) - 40)
            by = max(30, int(top_pt[1]) - 15)
            badge_text = f"[#{c['rank']} {c['class_name'].upper()} | Score: {c['final_score']:.1f}{depth_str}]"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (bx - 4, by - th - 6), (bx + tw + 6, by + 6), (20, 20, 20), -1)
            cv2.rectangle(annotated, (bx - 4, by - th - 6), (bx + tw + 6, by + 6), color, 1)
            cv2.putText(annotated, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Header HUD
        cv2.rectangle(annotated, (0, 0), (w, 36), (15, 15, 15), -1)
        hud_info = f"AI In-Scene Ad Space Placement Analyzer (Photo Mode) | Candidates Detected: {len(scored_candidates)}"
        cv2.putText(annotated, hud_info, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        total_processing_time = round(time.time() - start_photo_time, 2)

        if output_image_path:
            cv2.imwrite(output_image_path, annotated)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(OUTPUTS_DIR, f"{base_name}_photo_placements.json")
        csv_path = os.path.join(OUTPUTS_DIR, f"{base_name}_photo_placements.csv")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_candidates": len(scored_candidates),
                "placements": scored_candidates,
                "total_processing_time_sec": total_processing_time
            }, f, indent=2)

        if scored_candidates:
            keys = list(scored_candidates[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[k for k in keys if k != "quad_points"])
                writer.writeheader()
                for c in scored_candidates:
                    row = {k: v for k, v in c.items() if k != "quad_points"}
                    writer.writerow(row)

        exports = {
            "json": json_path,
            "csv": csv_path,
            "annotated_image": output_image_path or "",
            "total_processing_time_sec": total_processing_time
        }
        return scored_candidates, exports

    def export_reports(
        self,
        scores: List[TrackletScore],
        json_path: str,
        csv_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Saves ranked candidate results to JSON and CSV formats."""
        records = [s.to_dict() for s in scores]

        out_data = {
            "total_candidates": len(records),
            "placements": records
        }
        if metadata:
            out_data.update(metadata)

        # JSON export
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        # CSV export
        if records:
            keys = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("rank,track_id,class_name,start_time_sec,end_time_sec,duration_sec,final_score\n")

        logger.info("Reports exported successfully to %s and %s", json_path, csv_path)

