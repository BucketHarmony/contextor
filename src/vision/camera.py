"""Camera capture module for video input."""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """A captured video frame."""
    image: np.ndarray
    timestamp: datetime
    frame_number: int


class Camera:
    """Camera capture interface supporting USB and CSI cameras."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.camera_id = config.camera_id
        self.width = config.camera_width
        self.height = config.camera_height
        self.fps = config.camera_fps

        self._capture: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_count = 0
        self._last_frame: Optional[Frame] = None
        self._frame_lock = threading.Lock()

        # Callbacks for frames
        self._callbacks: list[Callable[[Frame], None]] = []

    def _get_capture_source(self) -> str | int:
        """Get the appropriate capture source for the camera."""
        # Check if it's a CSI camera path
        if isinstance(self.camera_id, str):
            if self.camera_id.startswith("/dev/video"):
                return self.camera_id
            # GStreamer pipeline for Jetson CSI camera
            if self.camera_id == "csi" or self.camera_id.startswith("nvargus"):
                return self._get_gstreamer_pipeline()
            return self.camera_id

        return self.camera_id

    def _get_gstreamer_pipeline(self) -> str:
        """Get GStreamer pipeline for Jetson CSI camera."""
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )

    def _capture_loop(self) -> None:
        """Main capture loop."""
        target_interval = 1.0 / self.fps
        last_capture_time = 0.0

        while self._running:
            current_time = time.time()
            elapsed = current_time - last_capture_time

            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)
                continue

            if self._capture is None or not self._capture.isOpened():
                logger.warning("Camera not available, retrying...")
                time.sleep(1.0)
                self._open_camera()
                continue

            ret, frame = self._capture.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue

            last_capture_time = time.time()
            self._frame_count += 1

            frame_obj = Frame(
                image=frame,
                timestamp=datetime.now(),
                frame_number=self._frame_count,
            )

            with self._frame_lock:
                self._last_frame = frame_obj

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    callback(frame_obj)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")

    def _open_camera(self) -> bool:
        """Open the camera capture."""
        source = self._get_capture_source()
        logger.info(f"Opening camera: {source}")

        try:
            if isinstance(source, str) and "nvargus" in source:
                self._capture = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
            else:
                self._capture = cv2.VideoCapture(source)

            if not self._capture.isOpened():
                logger.error("Failed to open camera")
                return False

            # Set properties for USB camera
            if not isinstance(source, str) or "nvargus" not in source:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self._capture.set(cv2.CAP_PROP_FPS, self.fps)

            actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._capture.get(cv2.CAP_PROP_FPS)

            logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True

        except Exception as e:
            logger.error(f"Failed to open camera: {e}")
            return False

    def add_callback(self, callback: Callable[[Frame], None]) -> None:
        """Register a callback for frames."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Frame], None]) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> bool:
        """Start camera capture."""
        if self._running:
            logger.warning("Camera already running")
            return True

        if not self._open_camera():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        logger.info("Camera capture started")
        return True

    def stop(self) -> None:
        """Stop camera capture."""
        if not self._running:
            return

        logger.info("Stopping camera capture")
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        logger.info("Camera capture stopped")

    def get_frame(self) -> Optional[Frame]:
        """Get the most recent frame."""
        with self._frame_lock:
            return self._last_frame

    def capture_image(self) -> Optional[np.ndarray]:
        """Capture a single high-quality image."""
        frame = self.get_frame()
        if frame is not None:
            return frame.image.copy()
        return None

    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running

    @property
    def frame_count(self) -> int:
        """Get total captured frame count."""
        return self._frame_count

    @staticmethod
    def list_cameras() -> list[dict]:
        """List available cameras."""
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    "id": i,
                    "width": width,
                    "height": height,
                })
                cap.release()
        return cameras
