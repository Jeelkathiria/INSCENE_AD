import json
import numpy as np

with open('data/outputs/9e44283a_CITY_VIDEO_placements.json') as f:
    data = json.load(f)

print(f"Total placements: {len(data['placements'])}")
for p in data['placements']:
    print(f"Rank {p['rank']} | ID {p['track_id']} | Class: {p['class_name']} | Time: {p['timeline_slot_str']} ({p['start_time_sec']:.2f}s - {p['end_time_sec']:.2f}s, dur {p['duration_sec']:.2f}s) | Frames: {p['total_frames']} | Area: {p['mean_area_pct']}%")

