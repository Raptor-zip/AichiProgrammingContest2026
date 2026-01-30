from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from backend.camera_manager import camera_manager
from backend.llm_service import llm_service
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
import asyncio

from config_loader import get_config
from image_processing import (
    process_with_green_background,
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

class StudyRequest(BaseModel):
    text: str
    type: str  # "explain" or "problem"
    context: Optional[str] = None  # Full document context

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[dict]  # List of {role, content}
    context: Optional[str] = None  # Document context

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

@router.get("/capture_status")
async def capture_status_stream(request: Request):
    """SSE stream for capture status"""
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break

            # Get state
            progress = camera_manager.current_progress
            triggered = camera_manager.auto_capture_triggered

            data = json.dumps({"progress": progress, "triggered": triggered})
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.05) # 20fps updates

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/capture")
async def capture_image(background_tasks: BackgroundTasks):
    """Capture current frame and save it"""
    frame = camera_manager.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"capture_{timestamp}.jpg"
    filepath = os.path.join(CAPTURES_DIR, filename)

    # Save original
    original_filename = f"capture_{timestamp}_original.jpg"
    original_filepath = os.path.join(CAPTURES_DIR, original_filename)
    cv2.imwrite(original_filepath, frame)

    # Process with green background detection
    process_frame, success = process_with_green_background(frame, enhance=True)

    # Resize if too large
    max_dim = 2000
    h, w = process_frame.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        process_frame = cv2.resize(process_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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

@router.get("/captures/{filename:path}/ocr")
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

@router.post("/study_support")
async def study_support(request: StudyRequest):
    """Generate study support content using LLM"""
    if not request.text:
         raise HTTPException(status_code=400, detail="Text is required")

    result = ""
    if request.type == "explain":
        result = llm_service.explain_text(request.text, request.context)
    elif request.type == "problem":
        result = llm_service.create_problems(request.text, request.context)
    else:
        raise HTTPException(status_code=400, detail="Invalid support type")

    return {"content": result}


@router.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Continue a conversation with AI"""
    if not request.message:
        raise HTTPException(status_code=400, detail="Message is required")

    result = llm_service.chat(request.message, request.history, request.context)
    return {"content": result}


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
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('_ocr.jpg') and not f.endswith('_original.jpg'):
                full_path = os.path.join(root, f)
                # Get relative path for URL (e.g., "Math/capture.jpg")
                rel_path = os.path.relpath(full_path, CAPTURES_DIR)
                # URL encode path segments
                url_path = "/".join([p for p in rel_path.split(os.sep)])

                stats = os.stat(full_path)

                # Check for metadata
                base_name = os.path.splitext(f)[0]
                info_path = os.path.join(root, f"{base_name}_info.json")
                detected_id = None
                if os.path.exists(info_path):
                    try:
                        with open(info_path, 'r') as meta_f:
                            meta = json.load(meta_f)
                            detected_id = meta.get("detected_id")
                    except:
                        pass

                # Check for original image
                base_name = os.path.splitext(f)[0]
                original_filename = f"{base_name}_original.jpg" # Assuming jpg
                original_path = os.path.join(root, original_filename)
                url_original = None
                if os.path.exists(original_path):
                     rel_path_orig = os.path.relpath(original_path, CAPTURES_DIR)
                     url_path_orig = "/".join([p for p in rel_path_orig.split(os.sep)])
                     url_original = f"/api/captures/{url_path_orig}"

                files.append({
                    "filename": f,
                    "filepath": full_path,
                    "created_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    "url": f"/api/captures/{url_path}",
                    "url_original": url_original,
                    "subject": os.path.basename(root) if root != CAPTURES_DIR else "Unclassified",
                    "detected_id": detected_id,
                    "relative_path": url_path # For API calls needing path
                })

    # Sort by mtime desc
    files.sort(key=lambda x: x['created_at'], reverse=True)
    return files

def manual_trigger_auto_capture(frame: np.ndarray, detected_ids: List[int] = [], detected_corners: List[Any] = []):
    """Callback for auto-capture from CameraManager"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load mappings
        mapping_file = os.path.join(os.getcwd(), config.get_subject_mappings_file())
        subject_mappings = {}
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                subject_mappings = json.load(f)

        # Determine subject
        target_dir = CAPTURES_DIR # Default to root/Unclassified effectively
        subject_name = "Unclassified"
        detected_id = None

        # Check detected IDs
        # Priority: First mapped ID found
        for mid in detected_ids:
            str_id = str(mid)
            if str_id in subject_mappings:
                subject_name = subject_mappings[str_id]
                target_dir = os.path.join(CAPTURES_DIR, subject_name)
                break
            else:
                # If unmapped, we note the first one to register
                if detected_id is None:
                    detected_id = mid

        if subject_name == "Unclassified":
            target_dir = os.path.join(CAPTURES_DIR, "Unclassified")

        os.makedirs(target_dir, exist_ok=True)
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(target_dir, filename)

        # Save Original Image
        original_filename = f"capture_{timestamp}_original.jpg"
        original_filepath = os.path.join(target_dir, original_filename)
        cv2.imwrite(original_filepath, frame)

        # --- Image Processing with Green Background Detection ---
        processing_frame, success = process_with_green_background(frame, enhance=True)

        if success:
            print(f"[GreenDetect] Paper detected and transformed successfully")
        else:
            print(f"[GreenDetect] Paper detection failed, using enhanced original")

        # Resize if too large
        max_dim = 2000
        h, w = processing_frame.shape[:2]
        if h > max_dim or w > max_dim:
            scale = max_dim / max(h, w)
            processing_frame = cv2.resize(processing_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Save image
        cv2.imwrite(filepath, processing_frame)
        print(f"Auto-saved to: {filepath} (Subject: {subject_name})")

        # Save Metadata if Unclassified and has ID
        if subject_name == "Unclassified" and detected_id is not None:
             meta_filename = f"capture_{timestamp}_info.json"
             meta_path = os.path.join(target_dir, meta_filename)
             with open(meta_path, 'w', encoding='utf-8') as f:
                 json.dump({"detected_id": int(detected_id)}, f)

        # Trigger background OCR
        threading.Thread(target=perform_ocr_background, args=(filepath,), daemon=True).start()

    except Exception as e:
        print(f"Auto-capture callback failed: {e}")

# Register callback
camera_manager.set_capture_callback(manual_trigger_auto_capture)
