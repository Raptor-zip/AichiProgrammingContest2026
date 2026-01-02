import cv2
import threading
import time
from typing import Optional, Generator
import numpy as np
import os
import sys

# Ensure we can import from root
if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_loader import get_config
from image_processing import (
    auto_white_balance,
)
import cv2.aruco as aruco

class CameraManager:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        self.config = get_config()
        self.current_frame: Optional[np.ndarray] = None
        self.camera_paused = False
        self.white_balance_enabled = self.config.get_white_balance_enabled_by_default()

        # ArUco setup
        dict_type_name = self.config.get_aruco_dict_type()
        dict_type = getattr(aruco, dict_type_name, aruco.DICT_4X4_50)
        self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

        # Auto-capture state
        self.last_marker_time = 0.0
        self.auto_capture_triggered = False
        self.auto_capture_triggered = False
        self.on_capture_callback = None
        self.capture_flash_time = 0.0 # Time when capture happened for visual feedback
        self.current_progress = 0.0 # For progress bar visualization

        # Threading support
        self.running = False
        self.thread = None

    def initialize(self):
        """Initialize the camera based on config"""
        # Similar logic to main.py try_open_capture
        def try_open_capture(source, tries=3):
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                return None
            for _ in range(tries):
                ret, _ = cap.read()
                if ret:
                    return cap
            cap.release()
            return None

        print(f"Attempting to open network camera: {self.config.get_network_video_url()}")
        self.cap = try_open_capture(
            self.config.get_network_video_url(),
            tries=self.config.get_network_retry_count(),
        )

        if self.cap is None:
            print(f"Attempting to open local camera: {self.config.get_local_device_index()}")
            self.cap = try_open_capture(
                self.config.get_local_device_index(),
                tries=self.config.get_network_retry_count(),
            )

        if self.cap is None:
            raise RuntimeError("Failed to open any camera source")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.get_buffer_size())
        print("Camera initialized successfully")

        # Start background thread
        self.start_capture_thread()

    def start_capture_thread(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop_capture_thread(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _capture_loop(self):
        """Continuous capture and processing loop"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                # Try to reconnect or just wait
                time.sleep(0.1)
                continue

            # Update current frame safely
            with self.lock:
                self.current_frame = frame.copy()

            # Run auto-capture logic immediately
            self.check_auto_capture(frame)

            # Rate limit slightly to avoid 100% CPU if camera is very fast
            # But usually cap.read() blocks until frame is ready.

    def check_auto_capture(self, frame):
        """Check markers and trigger capture if stable"""
        try:
            # We need to detect markers here for logic,
            # even if we also do it for visualization in stream.
            # Optimization: Cache the detection result?
            # For now, let's just detect. It might be redundant with stream processing
            # but safer for decoupled logic.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                cur_time = time.time()
                if self.last_marker_time == 0:
                    self.last_marker_time = cur_time

                # Check duration
                elapsed = (cur_time - self.last_marker_time) * 1000
                if elapsed >= self.config.get_auto_capture_delay_ms():
                    self.current_progress = 1.0
                    if not self.auto_capture_triggered:
                        print(f"Auto-capture triggered! (stable for {elapsed:.0f}ms)")
                        self.auto_capture_triggered = True
                        self.capture_flash_time = time.time() # Trigger flash
                        if self.on_capture_callback:
                            # Invoke callback with copy of frame and detected IDs
                            # ids is usually [[id]], so we might want to flatten or pass as is
                            # Let's pass the raw ids array (numpy) or list
                            detected_ids = ids.flatten().tolist() if ids is not None else []
                            # Pass corners as list of lists (or similar structure compatible with JSON/simple passing)
                            # corners is tuple of arrays. image_processing expects list of arrays or tuple of arrays.
                            # We'll pass it as is (list of numpy arrays) inside the thread.
                            detected_corners = [c.tolist() for c in corners] if corners else []
                            self.on_capture_callback(frame.copy(), detected_ids, detected_corners)
                else:
                    # Update progress
                    self.current_progress = elapsed / self.config.get_auto_capture_delay_ms()
            else:
                self.last_marker_time = 0
                self.current_progress = 0.0
                self.auto_capture_triggered = False


        except Exception as e:
            print(f"Error in auto-capture logic: {e}")


    def release(self):
        self.stop_capture_thread()
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.current_frame is None:
                return None
            return self.current_frame.copy()

    def process_frame_for_stream(self, frame: np.ndarray) -> bytes:
        """Process frame (ArUco, WB, etc) and return JPEG bytes"""
        # Copy to avoid modifying the original for other uses if needed
        display_frame = frame.copy()

        # ArUco detection (simplified adaptation from main.py)
        gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(display_frame, corners, ids)

        # JPEG encoding
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            return b""
        return buffer.tobytes()

    def generate_stream(self) -> Generator[bytes, None, None]:
        """Generator for MJPEG stream"""
        while True:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue

            jpeg_bytes = self.process_frame_for_stream(frame)
            if not jpeg_bytes:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')

            # Control framerate roughly
            time.sleep(self.config.get_frame_interval_ms() / 1000.0)


    def set_capture_callback(self, callback):
        self.on_capture_callback = callback

camera_manager = CameraManager()
