"""
FastAPI Web Application for AI-based In-Scene Advertising.
Supports asynchronous video ingestion and live real-time frame extraction progress.
"""

import os
import json
import logging
import uuid
import shutil
import threading
from typing import Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import DATA_DIR, BASE_DIR
from src.pipeline import VideoPlacementPipeline, ensure_browser_compatible_h264

logger = logging.getLogger("WebServer")
app = FastAPI(title="AI-based In-Scene Advertising | Placement Analyzer")


def log_terminal(msg: str):
    """Outputs real-time log messages to terminal console with flush."""
    try:
        print(msg, flush=True)
    except Exception:
        safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe_msg, flush=True)
    logger.info(msg)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Live progress tracking registry
active_jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main web dashboard interface."""
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Frontend template index.html not found.")
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


def _process_video_job(job_id: str, video_path: str, filename: str, saved_filename: str):
    """Background worker function updating live extraction percentage, preview frame, and processing time."""
    base_name = os.path.splitext(saved_filename)[0]
    out_annotated_mp4 = os.path.join(OUTPUTS_DIR, f"{base_name}_annotated.mp4")

    log_terminal("=" * 75)
    log_terminal(f"-> [JOB START] Processing Video: {filename} (Job ID: {job_id})")
    log_terminal("=" * 75)

    last_logged_frame = [-1]

    def progress_handler(
        current_frame: int,
        total_frames: int,
        timestamp_sec: float,
        active_count: int = 0,
        preview_b64: Optional[str] = None,
        elapsed_sec: float = 0.0,
        fps: float = 0.0
    ):
        pct = round((current_frame / max(1, total_frames)) * 100.0, 1)
        update_data = {
            "current_frame": current_frame,
            "total_frames": total_frames,
            "percent": pct,
            "timestamp_sec": round(timestamp_sec, 2),
            "active_surfaces": active_count,
            "elapsed_sec": elapsed_sec,
            "fps": fps,
            "status": "processing"
        }
        if preview_b64 is not None:
            update_data["preview_frame"] = preview_b64
        active_jobs[job_id].update(update_data)

        # Print real-time console log periodically (every 15 frames or at final frame)
        if current_frame - last_logged_frame[0] >= 15 or current_frame == total_frames:
            last_logged_frame[0] = current_frame
            log_terminal(
                f"[*] [FRAME EXTRACTION] Frame {current_frame:4d}/{total_frames:4d} ({pct:5.1f}%) | "
                f"Video Time: {timestamp_sec:05.2f}s | Active 3D Surfaces: {active_count} | Speed: {fps:4.1f} FPS | Elapsed: {elapsed_sec:4.1f}s"
            )

    try:
        pipeline = VideoPlacementPipeline()
        ranked_scores, exports = pipeline.process_video(
            video_path=video_path,
            output_video_path=out_annotated_mp4,
            progress_callback=progress_handler
        )

        # Ensure raw upload is also browser-playable H.264
        ensure_browser_compatible_h264(video_path)

        with open(exports["json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        data["job_id"] = job_id
        data["filename"] = filename
        data["annotated_video_url"] = f"/media/video/{os.path.basename(out_annotated_mp4)}"
        data["raw_video_url"] = f"/media/uploads/{saved_filename}"
        data["csv_export_url"] = f"/api/download/{base_name}/csv"
        data["json_export_url"] = f"/api/download/{base_name}/json"
        
        # Merge metadata (total processing time, fps, timeline summary)
        if "metadata" in exports:
            data.update(exports["metadata"])

        active_jobs[job_id].update({
            "status": "completed",
            "percent": 100.0,
            "result": data
        })
        log_terminal(f"[+] [VIDEO COMPLETED] Job {job_id} finished in {data.get('total_processing_time_sec', 'N/A')}s (avg {data.get('processing_fps', 'N/A')} FPS)")
        log_terminal(f"   Sequential Ad Timelines: {data.get('sequential_timeline_summary', 'None')}")
        log_terminal(f"   Annotated 3D Output: {out_annotated_mp4}")
        log_terminal("=" * 75)

    except Exception as e:
        log_terminal(f"[X] [JOB FAILED] {job_id}: {e}")
        logger.error("Job %s failed: %s", job_id, e, exc_info=True)
        active_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })


@app.post("/api/start_analysis")
async def start_video_analysis(file: UploadFile = File(...)):
    """
    Accepts video upload, registers a background job,
    and returns job_id immediately so client can poll live frame extraction.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No video file uploaded.")

    job_id = str(uuid.uuid4())[:8]
    saved_filename = f"{job_id}_{file.filename}"
    video_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    active_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "initializing",
        "percent": 0.0,
        "current_frame": 0,
        "total_frames": 1,
        "timestamp_sec": 0.0,
        "active_surfaces": 0,
        "error": None,
        "result": None
    }

    # Spawn processing thread
    thread = threading.Thread(
        target=_process_video_job,
        args=(job_id, video_path, file.filename, saved_filename),
        daemon=True
    )
    thread.start()

    return {"job_id": job_id, "message": "Analysis started."}


@app.get("/api/job_progress/{job_id}")
async def get_job_progress(job_id: str):
    """Returns real-time frame extraction percentage, frame counts, and candidate stats."""
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(content=job)


@app.post("/api/analyze_photo")
async def analyze_photo_upload(file: UploadFile = File(...)):
    """
    Accepts still photo/image upload (.jpg, .png, .jpeg, .webp),
    detects planar ad placement spaces, and returns ranked square locations and scores.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No photo file provided.")

    photo_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    saved_filename = f"{photo_id}_{file.filename}"
    image_path = os.path.join(UPLOADS_DIR, saved_filename)

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    base_name = os.path.splitext(saved_filename)[0]
    out_annotated_img = os.path.join(OUTPUTS_DIR, f"{base_name}_annotated{ext}")

    log_terminal("=" * 75)
    log_terminal(f"-> [PHOTO UPLOAD] Analyzing Photo: {file.filename} (Photo ID: {photo_id})")
    log_terminal("=" * 75)

    try:
        pipeline = VideoPlacementPipeline()
        ranked_scores, exports = pipeline.process_image(
            image_path=image_path,
            output_image_path=out_annotated_img
        )

        with open(exports["json"], "r", encoding="utf-8") as f:
            data = json.load(f)

        data["photo_id"] = photo_id
        data["filename"] = file.filename
        data["mode"] = "photo"
        data["annotated_media_url"] = f"/media/images/{os.path.basename(out_annotated_img)}"
        data["raw_media_url"] = f"/media/uploads/{saved_filename}"
        data["csv_export_url"] = f"/api/download/{base_name}_photo/csv"
        data["json_export_url"] = f"/api/download/{base_name}_photo/json"

        placements = data.get("placements", [])
        log_terminal(f"[+] [PHOTO COMPLETED] Detected {len(placements)} planar ad candidate(s) in {data.get('total_processing_time_sec', 'N/A')}s")
        if placements:
            top = placements[0]
            log_terminal(f"   Rank #1: {top.get('class_name')} | Score: {top.get('final_score')} | Dimensions: {top.get('dimensions_wh')} | 3D Quad: {top.get('position_xy')}")
        log_terminal(f"   Annotated 3D Output: {out_annotated_img}")
        log_terminal("=" * 75)

        return JSONResponse(content=data)

    except Exception as e:
        logger.error("Photo analysis failed for %s: %s", file.filename, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Photo analysis failed: {str(e)}")


@app.get("/media/images/{filename}")
async def stream_output_image(filename: str):
    """Streams processed annotated image."""
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image file not found.")
    ext = os.path.splitext(filename)[1].lower()
    media_type = "image/png" if ext == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media_type)


@app.get("/media/video/{filename}")
async def stream_output_video(filename: str):
    """Streams processed annotated video."""
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(path, media_type="video/mp4")


@app.get("/media/uploads/{filename}")
async def stream_raw_upload(filename: str):
    """Streams raw uploaded video or photo."""
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Raw media file not found.")
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        media_type = "image/jpeg"
    elif ext == ".png":
        media_type = "image/png"
    elif ext == ".webp":
        media_type = "image/webp"
    else:
        media_type = "video/mp4"
    return FileResponse(path, media_type=media_type)


@app.get("/api/download/{base_name}/{fmt}")
async def download_report(base_name: str, fmt: str):
    """Downloads placement report in CSV or JSON format."""
    if fmt == "csv":
        path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.csv")
        media = "text/csv"
    else:
        path = os.path.join(OUTPUTS_DIR, f"{base_name}_placements.json")
        media = "application/json"

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report file not found.")

    return FileResponse(path, media_type=media, filename=os.path.basename(path))


def start_server(port: int = 8000, host: str = "127.0.0.1"):
    """Starts the FastAPI Uvicorn web server."""
    print(f"\n=======================================================")
    print(f"🚀 AI-based In-Scene Advertising Analyzer")
    print(f"🔗 Dashboard URL: http://localhost:{port}")
    print(f"=======================================================\n")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_server()
