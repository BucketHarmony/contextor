"""Motion detection using frame differencing."""

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import VisionConfig
from .camera import Frame

logger = logging.getLogger(__name__)


@dataclass
class MotionEvent:
    """A detected motion event."""
    timestamp: datetime
    frame: np.ndarray
    contour_area: int
    bounding_box: tuple[int, int, int, int]  # x, y, w, h


class MotionDetector:
    """Motion detection using frame differencing and contour analysis."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.threshold = config.motion_threshold
        self.min_area = config.motion_min_area
        self.blur_size = config.motion_blur_size
        self.detection_fps = config.motion_detection_fps

        self._previous_frame: Optional[np.ndarray] = None
        self._last_detection_time = 0.0
        self._frame_interval = 1.0 / self.detection_fps
        self._motion_events: list[MotionEvent] = []
        self._events_lock = threading.Lock()

        # Callbacks
        self._on_motion: list[Callable[[MotionEvent], None]] = []

        # Statistics
        self._total_motion_events = 0

    def on_motion(self, callback: Callable[[MotionEvent], None]) -> None:
        """Register callback for motion events."""
        self._on_motion.append(callback)

    def process_frame(self, frame: Frame) -> Optional[MotionEvent]:
        """
        Process a frame for motion detection.

        Args:
            frame: Input frame to analyze

        Returns:
            MotionEvent if motion detected, None otherwise
        """
        current_time = time.time()

        # Rate limit detection
        if current_time - self._last_detection_time < self._frame_interval:
            return None

        self._last_detection_time = current_time

        # Convert to grayscale
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

        # First frame - initialize
        if self._previous_frame is None:
            self._previous_frame = gray
            return None

        # Compute absolute difference
        frame_delta = cv2.absdiff(self._previous_frame, gray)

        # Apply threshold
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Dilate to fill gaps
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Update previous frame
        self._previous_frame = gray

        # Find largest contour above threshold
        motion_event = None
        max_area = 0
        max_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.min_area and area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)

            motion_event = MotionEvent(
                timestamp=frame.timestamp,
                frame=frame.image.copy(),
                contour_area=int(max_area),
                bounding_box=(x, y, w, h),
            )

            self._total_motion_events += 1

            # Store event
            with self._events_lock:
                self._motion_events.append(motion_event)

            logger.debug(f"Motion detected: area={max_area}, bbox=({x},{y},{w},{h})")

            # Trigger callbacks
            for callback in self._on_motion:
                try:
                    callback(motion_event)
                except Exception as e:
                    logger.error(f"Motion callback error: {e}")

        return motion_event

    def get_recent_events(self, clear: bool = False) -> list[MotionEvent]:
        """Get recent motion events, optionally clearing the buffer."""
        with self._events_lock:
            events = list(self._motion_events)
            if clear:
                self._motion_events.clear()
        return events

    def clear_events(self) -> None:
        """Clear the motion events buffer."""
        with self._events_lock:
            self._motion_events.clear()

    def reset(self) -> None:
        """Reset the motion detector state."""
        self._previous_frame = None
        self.clear_events()

    @property
    def total_motion_events(self) -> int:
        """Get total motion events count."""
        return self._total_motion_events

    @property
    def event_count(self) -> int:
        """Get current buffered event count."""
        with self._events_lock:
            return len(self._motion_events)
