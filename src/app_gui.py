"""
Desktop Research & Analytics GUI for AI-based In-Scene Advertising.
Built with Tkinter, ttk, and Pillow (as specified in Slide 12).
Provides interactive video playback, candidate inspection, live ScoreEngine gauges,
and export features.
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, List, Dict

from src.config import SystemConfig, SAMPLES_DIR, OUTPUTS_DIR
from src.detector import SurfaceDetector
from src.tracker import CentroidTracker, Tracklet
from src.score_engine import ScoreEngine, TrackletScore
from src.pipeline import VideoPlacementPipeline


class PlacementDashboardApp:
    """Tkinter-based GUI dashboard for analyzing and scoring video ad placements."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("AI-based In-Scene Advertising | Ad Space Placement Analyzer (Group 28)")
        self.root.geometry("1280x820")
        self.root.minsize(1050, 700)

        # Style configuration
        self._setup_styles()

        # State variables
        self.video_path: Optional[str] = None
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_playing: bool = False
        self.current_frame_idx: int = 0
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.frame_width: int = 640
        self.frame_height: int = 360

        # Analysis pipeline
        self.config = SystemConfig()
        self.pipeline = VideoPlacementPipeline(self.config)
        self.active_tracks: List[Tracklet] = []
        self.scored_tracks: Dict[int, TrackletScore] = {}
        self.final_rankings: List[TrackletScore] = []

        # Toggles
        self.show_quads_var = tk.BooleanVar(value=True)
        self.show_corners_var = tk.BooleanVar(value=True)
        self.show_occluders_var = tk.BooleanVar(value=True)
        self.only_top1_var = tk.BooleanVar(value=False)

        # Build UI layout
        self._build_ui()

        # Start playback loop in main thread via Tk after
        self.root.after(30, self._playback_tick)

    def _setup_styles(self):
        """Sets a sleek modern dark/slate theme."""
        style = ttk.Style()
        style.theme_use("clam")

        # Color palette
        self.bg_dark = "#1a1c23"
        self.bg_panel = "#242731"
        self.bg_card = "#2d313f"
        self.accent_green = "#00e676"
        self.accent_blue = "#2979ff"
        self.text_main = "#f0f2f5"
        self.text_muted = "#9ba1b0"

        self.root.configure(bg=self.bg_dark)

        style.configure(".", background=self.bg_panel, foreground=self.text_main, font=("Segoe UI", 9))
        style.configure("TLabel", background=self.bg_panel, foreground=self.text_main)
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground=self.accent_green)
        style.configure("Card.TFrame", background=self.bg_card)
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.configure("Green.TButton", background=self.accent_green, foreground="#000000")
        style.configure("TCheckbutton", background=self.bg_panel, foreground=self.text_main)
        style.configure("Horizontal.TScale", background=self.bg_panel)

    def _build_ui(self):
        """Constructs the layout: Top toolbar, Center video canvas, Right rankings, Bottom score gauges."""
        # 1. Top Toolbar
        top_bar = tk.Frame(self.root, bg=self.bg_panel, height=48, padx=12, pady=6)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(top_bar, text="📁 Open Video", command=self._choose_video)
        btn_open.pack(side=tk.LEFT, padx=4)

        btn_samples = ttk.Button(top_bar, text="🎬 Sample Videos", command=self._show_sample_selector)
        btn_samples.pack(side=tk.LEFT, padx=4)

        btn_export = ttk.Button(top_bar, text="📊 Export JSON / CSV", command=self._export_reports)
        btn_export.pack(side=tk.LEFT, padx=4)

        # Status / Title label in top bar
        self.lbl_title = tk.Label(
            top_bar,
            text="Autonomous In-Scene Planar Ad Space Placement Detector",
            bg=self.bg_panel, fg=self.accent_green, font=("Segoe UI", 11, "bold")
        )
        self.lbl_title.pack(side=tk.RIGHT, padx=10)

        # 2. Main Middle Workspace (Video + Right Leaderboard)
        mid_pane = tk.Frame(self.root, bg=self.bg_dark)
        mid_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Left Column: Video player & playback slider
        video_col = tk.Frame(mid_pane, bg=self.bg_panel, bd=1, relief=tk.SOLID)
        video_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(video_col, bg="#0d0e12", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Video controls bar
        ctrl_bar = tk.Frame(video_col, bg=self.bg_panel, padx=8, pady=6)
        ctrl_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_play = ttk.Button(ctrl_bar, text="▶ Play", width=8, command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=4)

        btn_step_prev = ttk.Button(ctrl_bar, text="⏮ Step", width=7, command=self._step_backward)
        btn_step_prev.pack(side=tk.LEFT, padx=2)

        btn_step_next = ttk.Button(ctrl_bar, text="Step ⏭", width=7, command=self._step_forward)
        btn_step_next.pack(side=tk.LEFT, padx=2)

        self.lbl_time = tk.Label(ctrl_bar, text="00:00.0 / 00:00.0", bg=self.bg_panel, fg=self.text_main, width=16)
        self.lbl_time.pack(side=tk.LEFT, padx=6)

        self.slider_var = tk.DoubleVar()
        self.slider = ttk.Scale(ctrl_bar, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.slider_var, command=self._on_scrub)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Right Column: Top Ad Placements Leaderboard
        right_col = tk.Frame(mid_pane, bg=self.bg_panel, width=380, bd=1, relief=tk.SOLID)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_col.pack_propagate(False)

        lbl_rankings = ttk.Label(right_col, text="🏆 Top Recommended Placements", style="Header.TLabel")
        lbl_rankings.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(10, 4))

        lbl_sub = tk.Label(
            right_col,
            text="Ranked by Multi-Factor ScoreEngine",
            bg=self.bg_panel, fg=self.text_muted, font=("Segoe UI", 8)
        )
        lbl_sub.pack(side=tk.TOP, anchor=tk.W, padx=10, pady=(0, 8))

        # Scrollable Frame for Leaderboard cards
        container = tk.Frame(right_col, bg=self.bg_panel)
        container.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.rank_canvas = tk.Canvas(container, bg=self.bg_panel, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.rank_canvas.yview)
        self.cards_frame = tk.Frame(self.rank_canvas, bg=self.bg_panel)

        self.cards_frame.bind("<Configure>", lambda e: self.rank_canvas.configure(scrollregion=self.rank_canvas.bbox("all")))
        self.rank_canvas.create_window((0, 0), window=self.cards_frame, anchor="nw", width=350)
        self.rank_canvas.configure(yscrollcommand=scrollbar.set)

        self.rank_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # 3. Bottom Inspector: Live ScoreEngine Breakdown & Toggles
        bottom_pane = tk.Frame(self.root, bg=self.bg_card, height=140, bd=1, relief=tk.SOLID, padx=12, pady=8)
        bottom_pane.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        # Split into left gauges and right toggles
        gauges_box = tk.Frame(bottom_pane, bg=self.bg_card)
        gauges_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        lbl_inspector = tk.Label(
            gauges_box, text="Live Placement Inspector (ScoreEngine Factors)",
            bg=self.bg_card, fg=self.accent_green, font=("Segoe UI", 10, "bold")
        )
        lbl_inspector.pack(anchor=tk.W, pady=(0, 4))

        # Metrics Grid
        grid_frame = tk.Frame(gauges_box, bg=self.bg_card)
        grid_frame.pack(fill=tk.X)

        self.val_duration = self._add_metric_tile(grid_frame, "⏱ Duration Score", "0.0s", 0)
        self.val_area = self._add_metric_tile(grid_frame, "📐 Area Ratio", "0.0%", 1)
        self.val_occlusion = self._add_metric_tile(grid_frame, "🛡 Occlusion Penalty", "0.0%", 2)
        self.val_stability = self._add_metric_tile(grid_frame, "📈 Motion Stability", "100.0", 3)
        self.val_geometry = self._add_metric_tile(grid_frame, "🎯 Geometric Ortho.", "100.0", 4)
        self.val_composite = self._add_metric_tile(grid_frame, "🏆 Total Fitness Score", "0.0 / 100", 5, highlight=True)

        # Toggles box (right side of bottom pane)
        toggles_box = tk.Frame(bottom_pane, bg=self.bg_card, padx=16)
        toggles_box.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(toggles_box, text="Display Overlays:", bg=self.bg_card, fg=self.text_main, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Checkbutton(toggles_box, text="Polygon Quads", variable=self.show_quads_var).pack(anchor=tk.W)
        ttk.Checkbutton(toggles_box, text="Corner Dots (P1-P4)", variable=self.show_corners_var).pack(anchor=tk.W)
        ttk.Checkbutton(toggles_box, text="Foreground Occluders", variable=self.show_occluders_var).pack(anchor=tk.W)

    def _add_metric_tile(self, parent, title: str, init_val: str, col: int, highlight: bool = False):
        card = tk.Frame(parent, bg="#1e212b", padx=8, pady=4, bd=1, relief=tk.RIDGE)
        card.grid(row=0, column=col, padx=4, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)

        fg_color = self.accent_green if highlight else self.text_main
        tk.Label(card, text=title, bg="#1e212b", fg=self.text_muted, font=("Segoe UI", 8)).pack(anchor=tk.W)
        val_lbl = tk.Label(card, text=init_val, bg="#1e212b", fg=fg_color, font=("Segoe UI", 11, "bold"))
        val_lbl.pack(anchor=tk.W)
        return val_lbl

    def _choose_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if file_path:
            self.load_video(file_path)

    def _show_sample_selector(self):
        """Presents dropdown/modal to load one of the 5 project presentation test videos."""
        samples = [
            ("Test Case 1: Static Object", "test_1_static_object.mp4"),
            ("Test Case 2: Dynamic Motion", "test_2_dynamic_motion.mp4"),
            ("Test Case 3: Centrally Occluded", "test_3_central_occlusion.mp4"),
            ("Test Case 4: Partially Occluded", "test_4_partial_occlusion.mp4"),
            ("Test Case 5: Visual Clutter / Multi Surfaces", "test_5_visual_clutter.mp4")
        ]

        top = tk.Toplevel(self.root)
        top.title("Select Project Test Scenario")
        top.geometry("420x280")
        top.configure(bg=self.bg_panel)

        tk.Label(top, text="Select Scenario from Slide 10:", bg=self.bg_panel, fg=self.accent_green, font=("Segoe UI", 10, "bold")).pack(pady=10)

        for name, filename in samples:
            target_path = os.path.join(SAMPLES_DIR, filename)
            btn = ttk.Button(
                top, text=name,
                command=lambda p=target_path: [self.load_video(p), top.destroy()]
            )
            btn.pack(fill=tk.X, padx=20, pady=4)

    def load_video(self, path: str):
        """Loads a video file, resets tracker and analysis state."""
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File does not exist: {path}")
            return

        if self.cap is not None:
            self.cap.release()

        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.slider.configure(to=self.total_frames - 1)

        self.current_frame_idx = 0
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")

        # Reset pipeline state
        self.pipeline = VideoPlacementPipeline(self.config)
        self.scored_tracks.clear()
        self._clear_leaderboard()

        self.root.title(f"Planar Ad Space Placement Analyzer - [{os.path.basename(path)}]")
        self._read_and_display_current_frame()

    def _toggle_play(self):
        if self.cap is None:
            return
        self.is_playing = not self.is_playing
        self.btn_play.configure(text="⏸ Pause" if self.is_playing else "▶ Play")

    def _step_forward(self):
        if self.cap is None:
            return
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        if self.current_frame_idx < self.total_frames - 1:
            self.current_frame_idx += 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self._read_and_display_current_frame()

    def _step_backward(self):
        if self.cap is None:
            return
        self.is_playing = False
        self.btn_play.configure(text="▶ Play")
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self._read_and_display_current_frame()

    def _on_scrub(self, val):
        if self.cap is None:
            return
        target_frame = int(float(val))
        if target_frame != self.current_frame_idx:
            self.current_frame_idx = target_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            self._read_and_display_current_frame()

    def seek_to_time(self, timestamp_sec: float):
        """Jumps directly to a given timestamp."""
        if self.cap is None:
            return
        target_frame = int(timestamp_sec * self.fps)
        target_frame = max(0, min(self.total_frames - 1, target_frame))
        self.current_frame_idx = target_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.slider_var.set(self.current_frame_idx)
        self._read_and_display_current_frame()

    def _playback_tick(self):
        """Called periodically by Tkinter event loop."""
        if self.is_playing and self.cap is not None:
            if self.current_frame_idx < self.total_frames - 1:
                self.current_frame_idx += 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self.slider_var.set(self.current_frame_idx)
                self._read_and_display_current_frame()
            else:
                self.is_playing = False
                self.btn_play.configure(text="▶ Play")

        delay_ms = int(1000.0 / self.fps) if self.fps > 0 else 33
        self.root.after(delay_ms, self._playback_tick)

    def _read_and_display_current_frame(self):
        """Reads frame, runs detection, tracking & scoring, and renders to canvas."""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        h, w = frame.shape[:2]
        timestamp = self.current_frame_idx / self.fps

        # Run pipeline stages
        candidates, occluders = self.pipeline.detector.detect(frame)
        self.active_tracks = self.pipeline.tracker.update(
            candidates, occluders, self.current_frame_idx, timestamp, (h, w)
        )

        for t in self.active_tracks:
            score = self.pipeline.score_engine.evaluate_tracklet(t)
            self.scored_tracks[t.track_id] = score

        # Sort and assign ranks
        active_scores = [self.scored_tracks[t.track_id] for t in self.active_tracks if t.track_id in self.scored_tracks]
        active_scores.sort(key=lambda s: s.final_score, reverse=True)
        for idx, s in enumerate(active_scores):
            s.rank = idx + 1

        # Render overlays
        annotated = self.pipeline.draw_annotations(
            frame, self.active_tracks, self.scored_tracks, occluders, timestamp
        )

        # Update Live ScoreEngine Inspector with highest ranked or primary surface
        if active_scores:
            top_cand = active_scores[0]
            self.val_duration.configure(text=f"{top_cand.duration:.1f}s (Score: {top_cand.score_duration:.0f})")
            self.val_area.configure(text=f"{top_cand.mean_area_fraction*100:.1f}%")
            self.val_occlusion.configure(text=f"{top_cand.mean_occlusion_ratio*100:.1f}% (Score: {top_cand.score_occlusion:.0f})")
            self.val_stability.configure(text=f"{top_cand.score_stability:.1f} / 100")
            self.val_geometry.configure(text=f"{top_cand.score_geometry:.1f} / 100")
            self.val_composite.configure(text=f"{top_cand.final_score:.1f} / 100")
        else:
            self.val_duration.configure(text="--")
            self.val_area.configure(text="--")
            self.val_occlusion.configure(text="--")
            self.val_stability.configure(text="--")
            self.val_geometry.configure(text="--")
            self.val_composite.configure(text="0.0 / 100")

        # Update time label
        cur_str = time.strftime("%M:%S", time.gmtime(timestamp)) + f".{int((timestamp%1)*10):01d}"
        tot_sec = self.total_frames / self.fps
        tot_str = time.strftime("%M:%S", time.gmtime(tot_sec)) + f".{int((tot_sec%1)*10):01d}"
        self.lbl_time.configure(text=f"{cur_str} / {tot_str}")

        # Draw frame to Canvas
        self._render_to_canvas(annotated)

        # Update Leaderboard rankings periodically
        if self.current_frame_idx % 10 == 0:
            self._update_leaderboard()

    def _render_to_canvas(self, bgr_frame: np.ndarray):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 760, 480

        h, w = bgr_frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.tk_img)

    def _update_leaderboard(self):
        """Refreshes the right-hand ranking cards with active and completed tracks."""
        all_tracks = self.pipeline.tracker.get_all_tracklets()
        all_scores = self.pipeline.score_engine.evaluate_and_rank_all(all_tracks)
        self.final_rankings = all_scores

        self._clear_leaderboard()

        if not all_scores:
            lbl = tk.Label(self.cards_frame, text="No candidate surfaces identified yet.", bg=self.bg_panel, fg=self.text_muted)
            lbl.pack(pady=20)
            return

        for score in all_scores[:6]:  # Display top 6 placements
            card = tk.Frame(self.cards_frame, bg=self.bg_card, bd=1, relief=tk.SOLID, padx=8, pady=6)
            card.pack(fill=tk.X, expand=True, pady=4)

            top_row = tk.Frame(card, bg=self.bg_card)
            top_row.pack(fill=tk.X)

            rank_color = self.accent_green if score.rank == 1 else self.text_main
            lbl_rank = tk.Label(
                top_row, text=f"#{score.rank} {score.class_name.upper()}",
                bg=self.bg_card, fg=rank_color, font=("Segoe UI", 10, "bold")
            )
            lbl_rank.pack(side=tk.LEFT)

            lbl_score = tk.Label(
                top_row, text=f"⭐ {score.final_score:.1f}",
                bg=self.bg_card, fg=self.accent_green, font=("Segoe UI", 10, "bold")
            )
            lbl_score.pack(side=tk.RIGHT)

            # Details
            info_txt = f"Interval: {score.start_time:.1f}s – {score.end_time:.1f}s (Dur: {score.duration:.1f}s)"
            tk.Label(card, text=info_txt, bg=self.bg_card, fg=self.text_muted, font=("Segoe UI", 8)).pack(anchor=tk.W)

            metrics_txt = f"Area: {score.mean_area_fraction*100:.1f}% | Occ: {score.mean_occlusion_ratio*100:.1f}%"
            tk.Label(card, text=metrics_txt, bg=self.bg_card, fg=self.text_muted, font=("Segoe UI", 8)).pack(anchor=tk.W)

            btn_jump = ttk.Button(card, text="▶ Jump to Scene", command=lambda t=score.start_time: self.seek_to_time(t))
            btn_jump.pack(anchor=tk.E, pady=(4, 0))

    def _clear_leaderboard(self):
        for widget in self.cards_frame.winfo_children():
            widget.destroy()

    def _export_reports(self):
        """Exports JSON and CSV placement logs."""
        all_tracks = self.pipeline.tracker.get_all_tracklets()
        scores = self.pipeline.score_engine.evaluate_and_rank_all(all_tracks)
        if not scores:
            messagebox.showinfo("Export", "No placements evaluated yet. Play or scrub the video first.")
            return

        base_name = os.path.splitext(os.path.basename(self.video_path or "video"))[0]
        json_path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.json")
        csv_path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.csv")

        self.pipeline.export_reports(scores, json_path, csv_path)
        messagebox.showinfo("Success", f"Reports exported successfully!\nJSON: {json_path}\nCSV: {csv_path}")


def launch_gui():
    """Initializes and runs the Tkinter application."""
    root = tk.Tk()
    app = PlacementDashboardApp(root)
    # Check if sample video exists and load automatically
    sample_file = os.path.join(SAMPLES_DIR, "test_1_static_object.mp4")
    if os.path.exists(sample_file):
        app.load_video(sample_file)
    root.mainloop()

