import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
import json
from src.pipeline import VideoPlacementPipeline
from src.config import SystemConfig

print("Starting test on CITY_VIDEO with enhanced tracking and consolidation...")
t0 = time.time()
pipeline = VideoPlacementPipeline()

# Process video without writing video output for fast benchmark, or write output
out_mp4 = "data/outputs/test_consolidated_city_annotated.mp4"
scores, exports = pipeline.process_video(
    video_path="data/uploads/9e44283a_CITY_VIDEO.mp4",
    output_video_path=out_mp4
)

print(f"Finished in {time.time() - t0:.1f}s")
with open(exports["json"], "r") as f:
    data = json.load(f)

print(f"Total candidates detected: {data.get('total_candidates', len(data.get('placements', [])))}")
for p in data.get("placements", []):
    print(f"Rank #{p['rank']} | ID {p['track_id']} | Class: {p['class_name']} | Timeline: {p['timeline_slot_str']} ({p['start_time_sec']}s - {p['end_time_sec']}s, dur: {p['duration_sec']}s) | Score: {p['final_score']} | Frames: {p['total_frames']} | Area: {p['mean_area_pct']}%")

print("Sequential timeline summary:", exports.get("metadata", {}).get("sequential_timeline_summary"))
