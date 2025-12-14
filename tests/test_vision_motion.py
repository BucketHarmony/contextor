"""Tests for the motion detection module."""

import time
from datetime import datetime
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.vision.camera import Frame
from src.vision.motion import MotionDetector, MotionEvent
from src.config import VisionConfig


class TestMotionEvent:
    """Tests for MotionEvent dataclass."""

    def test_motion_event_creation(self, sample_frame):
        """Test creating a motion event."""
        event = MotionEvent(
            timestamp=datetime.now(),
            frame=sample_frame,
            contour_area=1500,
            bounding_box=(100, 100, 200, 150),
        )

        assert event.contour_area == 1500
        assert event.bounding_box == (100, 100, 200, 150)


class TestMotionDetector:
    """Tests for MotionDetector class."""

    @pytest.fixture
    def motion_config(self):
        """Create test motion config."""
        return VisionConfig(
            motion_detection_fps=30,  # High FPS for testing
            motion_threshold=25,
            motion_min_area=500,
            motion_blur_size=21,
        )

    @pytest.fixture
    def detector(self, motion_config):
        """Create MotionDetector instance."""
        return MotionDetector(motion_config)

    def test_init(self, detector, motion_config):
        """Test MotionDetector initialization."""
        assert detector.threshold == motion_config.motion_threshold
        assert detector.min_area == motion_config.motion_min_area
        assert detector.total_motion_events == 0

    def test_on_motion_callback(self, detector):
        """Test registering motion callback."""
        callback = MagicMock()
        detector.on_motion(callback)
        assert callback in detector._on_motion

    def test_process_frame_first_frame(self, detector, sample_frame):
        """Test processing first frame (initialization)."""
        frame = Frame(
            image=sample_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )

        result = detector.process_frame(frame)

        assert result is None  # First frame doesn't detect motion
        assert detector._previous_frame is not None

    def test_process_frame_no_motion(self, detector, sample_frame):
        """Test processing frames with no motion."""
        frame1 = Frame(
            image=sample_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )
        frame2 = Frame(
            image=sample_frame.copy(),  # Same frame
            timestamp=datetime.now(),
            frame_number=2,
        )

        detector.process_frame(frame1)
        time.sleep(0.1)  # Wait for rate limit
        result = detector.process_frame(frame2)

        assert result is None

    def test_process_frame_with_motion(self, detector, black_frame, white_frame):
        """Test processing frames with significant motion."""
        frame1 = Frame(
            image=black_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )
        frame2 = Frame(
            image=white_frame,
            timestamp=datetime.now(),
            frame_number=2,
        )

        detector.process_frame(frame1)
        time.sleep(0.1)
        result = detector.process_frame(frame2)

        assert result is not None
        assert isinstance(result, MotionEvent)
        assert detector.total_motion_events == 1

    def test_process_frame_rate_limiting(self, detector, sample_frame):
        """Test frame rate limiting."""
        detector._frame_interval = 1.0  # 1 second interval

        frame1 = Frame(
            image=sample_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )
        frame2 = Frame(
            image=np.zeros_like(sample_frame),
            timestamp=datetime.now(),
            frame_number=2,
        )

        detector.process_frame(frame1)
        result = detector.process_frame(frame2)  # Should be rate limited

        assert result is None

    def test_motion_callback_triggered(self, detector, black_frame, white_frame):
        """Test motion callback is triggered."""
        callback = MagicMock()
        detector.on_motion(callback)

        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        callback.assert_called()

    def test_callback_error_handling(self, detector, black_frame, white_frame):
        """Test callback error handling."""
        def error_callback(event):
            raise ValueError("Callback error")

        detector.on_motion(error_callback)

        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        # Should not raise
        detector.process_frame(frame2)

    def test_get_recent_events(self, detector, black_frame, white_frame):
        """Test getting recent events."""
        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        events = detector.get_recent_events()
        assert len(events) >= 1

    def test_get_recent_events_with_clear(self, detector, black_frame, white_frame):
        """Test getting recent events with clear."""
        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        events = detector.get_recent_events(clear=True)
        assert len(events) >= 1

        events_after = detector.get_recent_events()
        assert len(events_after) == 0

    def test_clear_events(self, detector, black_frame, white_frame):
        """Test clearing events."""
        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        detector.clear_events()

        assert detector.event_count == 0

    def test_reset(self, detector, sample_frame):
        """Test resetting detector."""
        frame = Frame(image=sample_frame, timestamp=datetime.now(), frame_number=1)
        detector.process_frame(frame)

        detector.reset()

        assert detector._previous_frame is None
        assert detector.event_count == 0

    def test_event_count_property(self, detector, black_frame, white_frame):
        """Test event_count property."""
        assert detector.event_count == 0

        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        assert detector.event_count >= 1

    def test_total_motion_events_property(self, detector, black_frame, white_frame):
        """Test total_motion_events property."""
        frame1 = Frame(image=black_frame, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=white_frame, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        detector.process_frame(frame2)

        # Clear events but total should remain
        detector.clear_events()

        assert detector.total_motion_events >= 1

    def test_small_motion_ignored(self, detector):
        """Test that small motion is ignored."""
        # Create frames with tiny difference
        frame1_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2_img = frame1_img.copy()
        # Add tiny change (below min_area)
        frame2_img[0:5, 0:5] = 255

        frame1 = Frame(image=frame1_img, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=frame2_img, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        result = detector.process_frame(frame2)

        # Small motion should be ignored
        assert result is None

    def test_large_motion_detected(self, detector):
        """Test that large motion is detected."""
        # Create frames with large difference
        frame1_img = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2_img = frame1_img.copy()
        # Add large change (above min_area)
        frame2_img[100:300, 100:300] = 255

        frame1 = Frame(image=frame1_img, timestamp=datetime.now(), frame_number=1)
        frame2 = Frame(image=frame2_img, timestamp=datetime.now(), frame_number=2)

        detector.process_frame(frame1)
        time.sleep(0.1)
        result = detector.process_frame(frame2)

        assert result is not None
        assert result.contour_area > detector.min_area
