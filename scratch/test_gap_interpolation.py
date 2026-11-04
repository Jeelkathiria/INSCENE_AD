import numpy as np

def interpolate_history_gaps(history, max_allowed_frame_gap=150):
    if len(history) <= 1:
        return history
    interpolated = [history[0]]
    for i in range(1, len(history)):
        prev = interpolated[-1]
        curr = history[i]
        frame_diff = curr.frame_idx - prev.frame_idx
        if 1 < frame_diff <= max_allowed_frame_gap:
            # Interpolate missing frames
            q_start = prev.quad
            q_end = curr.quad
            t_start = prev.timestamp_sec
            t_end = curr.timestamp_sec
            for f in range(prev.frame_idx + 1, curr.frame_idx):
                alpha = (f - prev.frame_idx) / float(frame_diff)
                q_interp = (1.0 - alpha) * q_start + alpha * q_end
                t_interp = (1.0 - alpha) * t_start + alpha * t_end
                cx = float(np.mean(q_interp[:, 0]))
                cy = float(np.mean(q_interp[:, 1]))
                class MockSnap:
                    pass
                s = MockSnap()
                s.frame_idx = f
                s.timestamp_sec = t_interp
                s.quad = q_interp
                s.centroid = (cx, cy)
                s.area_fraction = (1.0 - alpha) * prev.area_fraction + alpha * curr.area_fraction
                s.occlusion_ratio = (1.0 - alpha) * prev.occlusion_ratio + alpha * curr.occlusion_ratio
                s.displacement = 0.0
                interpolated.append(s)
        interpolated.append(curr)
    return interpolated

