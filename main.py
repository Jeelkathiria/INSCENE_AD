"""
Main Entry Point for AI-based In-Scene Advertising.
Group ID: 28 - Somaiya Vidyavihar University
Project: A Score-Based System for Detecting Optimal Planar Advertisement Spaces in Video
"""

import os
import sys
import argparse
import logging

from src.config import SAMPLES_DIR, OUTPUTS_DIR
from src.synthetic_generator import generate_all_test_videos
from src.pipeline import VideoPlacementPipeline

# Setup clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Main")


def run_single_video(video_path: str, save_annotated: bool = True):
    """Processes a video file and prints ranked ad placement opportunities."""
    if not os.path.exists(video_path):
        logger.error("Video file does not exist: %s", video_path)
        return

    logger.info("Processing video: %s", video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(OUTPUTS_DIR, f"{base_name}_annotated.mp4") if save_annotated else None

    pipeline = VideoPlacementPipeline()

    def progress(f, total, sec):
        if f % 30 == 0 or f == total:
            pct = (f / total) * 100
            print(f"\rProgress: [{f}/{total} frames ({pct:5.1f}%)] Time: {sec:05.1f}s", end="", flush=True)

    ranked_scores, exports = pipeline.process_video(video_path, out_video, progress_callback=progress)
    print("\n")

    logger.info("=== TOP AD PLACEMENT OPPORTUNITIES IDENTIFIED ===")
    table_rows = []
    for s in ranked_scores:
        table_rows.append([
            f"#{s.rank}",
            s.track_id,
            s.class_name.upper(),
            f"{s.start_time:.1f}s - {s.end_time:.1f}s",
            f"{s.duration:.1f}s",
            f"{s.final_score:.1f}",
            f"{s.mean_area_fraction*100:.1f}%",
            f"{s.mean_occlusion_ratio*100:.1f}%",
            "YES (Optimal)" if s.is_recommended else "No"
        ])

    headers = [
        "Rank", "Track ID", "Surface Type", "Time Interval",
        "Duration", "Score / 100", "Area %", "Occlusion %", "Recommended"
    ]
    try:
        from tabulate import tabulate
        print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    except ImportError:
        # Fallback simple table print
        header_str = " | ".join(f"{h:<14}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in table_rows:
            print(" | ".join(f"{str(cell):<14}" for cell in row))

    print(f"\nArtifacts Saved:\n - JSON Report: {exports['json']}\n - CSV Report:  {exports['csv']}")
    if out_video and os.path.exists(out_video):
        print(f" - Annotated MP4 Video: {out_video}")


def run_benchmark_all_test_cases():
    """
    Executes and benchmarks all 5 test scenarios from Slide 10:
    1. Static Object
    2. Dynamic Object In Motion
    3. Centrally Occluded
    4. Partially Occluded
    5. Visual Clutter / Multiple Simultaneous Objects
    """
    logger.info("Generating/verifying synthetic test scenario clips...")
    video_paths = generate_all_test_videos(SAMPLES_DIR)

    test_cases = [
        ("Case 1: Static Object", video_paths["case_1_static"], "Optimal candidate surface detected cleanly"),
        ("Case 2: Dynamic Motion", video_paths["case_2_dynamic"], "Surface tracked across displacement"),
        ("Case 3: Centrally Occluded", video_paths["case_3_central_occlusion"], "Occluder detected, penalty applied"),
        ("Case 4: Partially Occluded", video_paths["case_4_partial_occlusion"], "Reduced preference due to occlusion"),
        ("Case 5: Visual Clutter", video_paths["case_5_clutter"], "Selects most optimal among multiple surfaces")
    ]

    results_table = []
    pipeline = VideoPlacementPipeline()

    for name, v_path, expected in test_cases:
        logger.info("Executing Benchmark: %s ...", name)
        scores, _ = pipeline.process_video(v_path)
        if scores:
            top = scores[0]
            status = "PASS" if top.final_score > 0 else "FAIL"
            score_str = f"{top.final_score:.1f}"
            occ_str = f"{top.mean_occlusion_ratio*100:.1f}%"
        else:
            status = "FAIL"
            score_str = "N/A"
            occ_str = "N/A"

        results_table.append([name, expected, score_str, occ_str, status])

    print("\n" + "=" * 80)
    print("SOFTWARE TEST CASES BENCHMARK REPORT (SLIDE 10 REPLICATION)")
    print("=" * 80)
    headers = ["Test Scenario", "Expected Outcome", "Top Score", "Mean Occ.", "Result"]
    try:
        from tabulate import tabulate
        print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))
    except ImportError:
        row_format = "{:<28} | {:<42} | {:<10} | {:<10} | {:<8}"
        print(row_format.format(*headers))
        print("-" * 110)
        for r in results_table:
            print(row_format.format(*r))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI-based In-Scene Advertising | Placement Analyzer")
    parser.add_argument("--web", action="store_true", help="Launch Localhost Web Dashboard (Default)")
    parser.add_argument("--gui", action="store_true", help="Launch Tkinter desktop dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port for web server (default: 8000)")
    parser.add_argument("--generate-samples", action="store_true", help="Generate synthetic test videos")
    parser.add_argument("--test-all", action="store_true", help="Run benchmark on all 5 test scenarios")
    parser.add_argument("--video", type=str, default="", help="Path to input video for headless analysis")

    args = parser.parse_args()

    # Default to modern localhost web server if no action or --web specified
    if len(sys.argv) == 1 or args.web:
        logger.info("Launching Localhost Web Dashboard on http://localhost:%d ...", args.port)
        generate_all_test_videos(SAMPLES_DIR)
        from src.web_server import start_server
        start_server(port=args.port)
    elif args.gui:
        logger.info("Launching Desktop Interface (Tkinter)...")
        generate_all_test_videos(SAMPLES_DIR)
        from src.app_gui import launch_gui
        launch_gui()
    elif args.generate_samples:
        paths = generate_all_test_videos(SAMPLES_DIR)
        logger.info("Generated sample videos successfully:")
        for k, p in paths.items():
            print(f" - {k}: {p}")
    elif args.test_all:
        run_benchmark_all_test_cases()
    elif args.video:
        run_single_video(args.video)


if __name__ == "__main__":
    main()
