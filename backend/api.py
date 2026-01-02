from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.camera_manager import camera_manager
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import os
import json
import cv2
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
import sys
import threading

# Add root to path for imports if needed (already in main.py but good for linting)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_loader import get_config
from image_processing import (
    auto_white_balance,
    calculate_marker_rotation,
    correct_rotation,
    perspective_transform_from_marker,
)

router = APIRouter()
config = get_config()

# Ensure captures directory exists
CAPTURES_DIR = os.path.join(os.getcwd(), config.get_captures_dir())
os.makedirs(CAPTURES_DIR, exist_ok=True)

class SettingsUpdate(BaseModel):
    mappings: dict

class OCRRequest(BaseModel):
    image_path: Optional[str] = None
    use_last_capture: bool = True

@router.get("/stream")
async def video_stream():
    """Stream video from the camera"""
    return StreamingResponse(
        camera_manager.generate_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/status")
async def get_status():
    return {"status": "ok", "camera_connected": camera_manager.cap is not None}

@router.post("/capture")
async def capture_image(background_tasks: BackgroundTasks):
    """Capture current frame and save it"""
    frame = camera_manager.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    # Deep copy for processing
    process_frame = frame.copy()

    # Optional: Logic from main.py's take_picture / process flow could be applied here
    # For now, just save the raw or simple processed frame.
    # If we want the "AI processing" flow from the desktop app:
    # 1. ArUco detection -> 2. Rotation/Perspective -> 3. Save

    # We will use helper logic directly here or in a helper function
    # Adapt logic from main.py (automatic processing often happens for OCR, but raw capture is good too)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(CAPTURES_DIR, filename)

    cv2.imwrite(filepath, process_frame)

    # Trigger background OCR
    background_tasks.add_task(perform_ocr_background, filepath)

    return {
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "url": f"/api/captures/{filename}"
    }

def perform_ocr_background(image_path: str):
    """Background task to run OCR and save results"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image for OCR: {image_path}")
            return

        from yomitoku import OCR
        # Initialize OCR (Ideally should be shared/Global to avoid reloading)
        # Note: Initializing model inside a background task might be slow every time.
        # For a hackathon/demo, it's acceptable but maybe inefficient.
        ocr = OCR(visualize=True, device="cpu")
        results, ocr_vis = ocr(image)

        # Save visualization (optional, maybe we don't need it if we have overlay)
        # But keeping it for debug or fallback
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        vis_filename = f"{base_name}_ocr.jpg"
        vis_path = os.path.join(os.path.dirname(image_path), vis_filename)
        cv2.imwrite(vis_path, ocr_vis)

        # Save JSON
        json_filename = f"{base_name}.json"
        json_path = os.path.join(os.path.dirname(image_path), json_filename)

        # Ensure results are JSON serializable
        # Yomitoku results (OCRSchema) are Pydantic models or similar
        json_results = results
        if hasattr(results, 'model_dump'):
             json_results = results.model_dump()
        elif hasattr(results, 'dict'):
             json_results = results.dict()
        elif isinstance(results, list):
             # List of objects?
             json_results = [r.model_dump() if hasattr(r, 'model_dump') else (r.dict() if hasattr(r, 'dict') else r) for r in results]

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        print(f"OCR completed for {image_path}")

    except Exception as e:
        print(f"Background OCR Error: {e}")

@router.get("/captures/{filename}/ocr")
async def get_ocr_result(filename: str):
    """Get OCR JSON for a specific capture"""
    # filename includes extension e.g. "capture_123.jpg"
    # we need "capture_123.json"

    # Handle subdirectory paths if present in filename (from frontend)
    # filename could be "Math/capture_123.jpg"

    full_path = os.path.join(CAPTURES_DIR, filename)
    base, _ = os.path.splitext(full_path)
    json_path = f"{base}.json"

    if not os.path.exists(json_path):
        return {"results": None, "status": "not_found"}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"results": data, "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr")
async def perform_ocr(request: OCRRequest):
    """Perform OCR on an image"""
    target_path = None

    if request.use_last_capture:
        # Find latest file in captures dir
        files = glob_captures()
        if not files:
            raise HTTPException(status_code=404, detail="No captures found")
        target_path = files[0]['filepath'] # First is newest
    elif request.image_path:
        target_path = request.image_path
        if not os.path.exists(target_path):
             # Try relative to captures dir
            target_path = os.path.join(CAPTURES_DIR, os.path.basename(request.image_path))
            if not os.path.exists(target_path):
                raise HTTPException(status_code=404, detail="Image file not found")
    else:
        raise HTTPException(status_code=400, detail="No image specified")

    # Load image
    image = cv2.imread(target_path)
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to load image")

    # Pre-processing (ArUco, Dewarping, etc.) should ideally happen here
    # to maintain consistency with the desktop app's "AI processing".
    # We will implement a simplified synchronous version or reuse the worker logic.
    # Since YomiToku is heavy, we should probably run it.

    try:
        from yomitoku import OCR
        # Initialize OCR (Should be global or cached to avoid reloading model)
        # For now, minimal implementation
        ocr = OCR(visualize=True, device="cpu") # Force CPU or let it decide
        results, ocr_vis = ocr(image)

        # Save visualization
        base_name = os.path.splitext(os.path.basename(target_path))[0]
        vis_filename = f"{base_name}_ocr.jpg"
        vis_path = os.path.join(CAPTURES_DIR, vis_filename)
        cv2.imwrite(vis_path, ocr_vis)

        # Extract text (JSON serializable)
        # results structure depends on yomitoku version, typically list of blocks/lines
        # We'll just return the full result structure as JSON

        # Adapt results to be JSON serializable if needed
        # Assuming results is dict or list of dicts with primitives

        return {
            "success": True,
            "results": results,
            "vis_image_url": f"/api/captures/{vis_filename}"
        }

    except Exception as e:
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
async def get_settings():
    """Get subject mappings and other settings"""
    mapping_file = os.path.join(os.getcwd(), config.get_subject_mappings_file())
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
    else:
        mappings = {}
    return {"mappings": mappings}

@router.post("/settings")
async def update_settings(settings: SettingsUpdate):
    """Update subject mappings"""
    mapping_file = os.path.join(os.getcwd(), config.get_subject_mappings_file())
    try:
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(settings.mappings, f, ensure_ascii=False, indent=2)
        # Force config reload if needed, or just reload the specific mapping in memory if we cached it
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history():
    """Get list of captured images"""
    return glob_captures()

@router.get("/captures/{filename:path}")
async def get_capture_image(filename: str):
    """Serve capture file"""
    # Securely join path
    file_path = os.path.abspath(os.path.join(CAPTURES_DIR, filename))
    if not file_path.startswith(CAPTURES_DIR) or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

def glob_captures():
    """Helper to list captures sorted by date desc"""
    files = []
    if not os.path.exists(CAPTURES_DIR):
        return []

    # Walk through directory
    for root, dirs, files_in_dir in os.walk(CAPTURES_DIR):
        for f in files_in_dir:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('_ocr.jpg'):
                full_path = os.path.join(root, f)
                # Get relative path for URL (e.g., "Math/capture.jpg")
                rel_path = os.path.relpath(full_path, CAPTURES_DIR)
                # URL encode path segments
                url_path = "/".join([p for p in rel_path.split(os.sep)])

                stats = os.stat(full_path)
                files.append({
                    "filename": f,
                    "filepath": full_path,
                    "created_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "url": f"/api/captures/{url_path}",
                    "subject": os.path.basename(root) if root != CAPTURES_DIR else "Unclassified"
                })

    # Sort by mtime desc
    files.sort(key=lambda x: x['created_at'], reverse=True)
    return files

def manual_trigger_auto_capture(frame: np.ndarray):
    """Callback for auto-capture from CameraManager"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(CAPTURES_DIR, filename)

        # Save image (Sync IO in thread is okay here as it's the capture thread)
        cv2.imwrite(filepath, frame)
        print(f"Auto-saved: {filepath}")

        # Trigger background OCR
        # We need a new thread because we are already in the capture thread
        # and we don't want to block it with heavy OCR
        threading.Thread(target=perform_ocr_background, args=(filepath,), daemon=True).start()

    except Exception as e:
        print(f"Auto-capture callback failed: {e}")

# Register callback
camera_manager.set_capture_callback(manual_trigger_auto_capture)
