"""Tests for the camera module."""

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest

from src.vision.camera import Camera, Frame
from src.config import VisionConfig


class TestFrame:
    """Tests for Frame dataclass."""

    def test_frame_creation(self, sample_frame):
        """Test creating a frame."""
        frame = Frame(
            image=sample_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )

        assert frame.frame_number == 1
        assert frame.image.shape == sample_frame.shape


class TestCamera:
    """Tests for Camera class."""

    @pytest.fixture
    def camera_config(self):
        """Create test camera config."""
        return VisionConfig(
            camera_id=0,
            camera_width=640,
            camera_height=480,
            camera_fps=30,
        )

    @pytest.fixture
    def camera(self, camera_config):
        """Create Camera instance."""
        return Camera(camera_config)

    def test_init(self, camera, camera_config):
        """Test Camera initialization."""
        assert camera.camera_id == camera_config.camera_id
        assert camera.width == camera_config.camera_width
        assert camera.height == camera_config.camera_height
        assert not camera.is_running()

    def test_add_callback(self, camera):
        """Test adding callbacks."""
        callback = MagicMock()
        camera.add_callback(callback)
        assert callback in camera._callbacks

    def test_remove_callback(self, camera):
        """Test removing callbacks."""
        callback = MagicMock()
        camera.add_callback(callback)
        camera.remove_callback(callback)
        assert callback not in camera._callbacks

    def test_remove_nonexistent_callback(self, camera):
        """Test removing non-existent callback."""
        callback = MagicMock()
        camera.remove_callback(callback)  # Should not raise

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_open_camera_success(self, mock_capture_class, camera):
        """Test successful camera opening."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture_class.return_value = mock_capture

        result = camera._open_camera()

        assert result is True
        mock_capture_class.assert_called_with(0)

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_open_camera_failure(self, mock_capture_class, camera):
        """Test camera opening failure."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_capture_class.return_value = mock_capture

        result = camera._open_camera()

        assert result is False

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_open_camera_exception(self, mock_capture_class, camera):
        """Test camera opening with exception."""
        mock_capture_class.side_effect = Exception("Camera error")

        result = camera._open_camera()

        assert result is False

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_start(self, mock_capture_class, camera):
        """Test starting camera."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_capture

        result = camera.start()

        assert result is True
        assert camera.is_running()

        camera.stop()

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_start_failure(self, mock_capture_class, camera):
        """Test starting camera failure."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = False
        mock_capture_class.return_value = mock_capture

        result = camera.start()

        assert result is False
        assert not camera.is_running()

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_start_already_running(self, mock_capture_class, camera):
        """Test starting when already running."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_capture

        camera.start()
        result = camera.start()  # Should return True without restarting

        assert result is True
        camera.stop()

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_stop(self, mock_capture_class, camera):
        """Test stopping camera."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_capture

        camera.start()
        camera.stop()

        assert not camera.is_running()
        mock_capture.release.assert_called_once()

    def test_stop_not_running(self, camera):
        """Test stopping when not running."""
        camera.stop()  # Should not raise

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_get_frame(self, mock_capture_class, camera, sample_frame):
        """Test getting current frame."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, sample_frame)
        mock_capture_class.return_value = mock_capture

        camera.start()
        time.sleep(0.2)  # Wait for capture

        frame = camera.get_frame()
        camera.stop()

        assert frame is not None
        assert isinstance(frame, Frame)

    def test_get_frame_not_running(self, camera):
        """Test getting frame when not running."""
        frame = camera.get_frame()
        assert frame is None

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_capture_image(self, mock_capture_class, camera, sample_frame):
        """Test capturing single image."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, sample_frame)
        mock_capture_class.return_value = mock_capture

        camera.start()
        time.sleep(0.2)

        image = camera.capture_image()
        camera.stop()

        assert image is not None
        assert image.shape == sample_frame.shape

    def test_capture_image_not_running(self, camera):
        """Test capturing image when not running."""
        image = camera.capture_image()
        assert image is None

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_frame_count(self, mock_capture_class, camera, sample_frame):
        """Test frame count property."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, sample_frame)
        mock_capture_class.return_value = mock_capture

        assert camera.frame_count == 0

        camera.start()
        time.sleep(0.2)
        camera.stop()

        assert camera.frame_count > 0

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_callback_triggered(self, mock_capture_class, camera, sample_frame):
        """Test that callbacks are triggered."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, sample_frame)
        mock_capture_class.return_value = mock_capture

        callback = MagicMock()
        camera.add_callback(callback)

        camera.start()
        time.sleep(0.2)
        camera.stop()

        assert callback.called

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_callback_error_handling(self, mock_capture_class, camera, sample_frame):
        """Test callback error handling."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, sample_frame)
        mock_capture_class.return_value = mock_capture

        def error_callback(frame):
            raise ValueError("Callback error")

        camera.add_callback(error_callback)

        camera.start()
        time.sleep(0.2)
        camera.stop()

    def test_get_capture_source_integer(self, camera):
        """Test getting capture source for integer ID."""
        source = camera._get_capture_source()
        assert source == 0

    def test_get_capture_source_device_path(self):
        """Test getting capture source for device path."""
        config = VisionConfig(camera_id="/dev/video1")
        camera = Camera(config)
        # camera_id is int, so this tests string handling
        camera.camera_id = "/dev/video1"
        source = camera._get_capture_source()
        assert source == "/dev/video1"

    def test_get_capture_source_csi(self):
        """Test getting capture source for CSI camera."""
        config = VisionConfig(camera_id=0)
        camera = Camera(config)
        camera.camera_id = "csi"
        source = camera._get_capture_source()
        assert "nvargus" in source

    def test_get_gstreamer_pipeline(self, camera):
        """Test GStreamer pipeline generation."""
        pipeline = camera._get_gstreamer_pipeline()
        assert "nvarguscamerasrc" in pipeline
        assert str(camera.width) in pipeline
        assert str(camera.height) in pipeline

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_list_cameras(self, mock_capture_class):
        """Test listing cameras."""
        # Mock camera 0 as available, camera 1 as unavailable
        def mock_capture(idx):
            cap = MagicMock()
            if idx == 0:
                cap.isOpened.return_value = True
                cap.get.return_value = 640
            else:
                cap.isOpened.return_value = False
            return cap

        mock_capture_class.side_effect = mock_capture

        cameras = Camera.list_cameras()

        assert len(cameras) >= 1
        assert cameras[0]["id"] == 0

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_capture_loop_read_failure(self, mock_capture_class, camera):
        """Test capture loop handling read failure."""
        mock_capture = MagicMock()
        mock_capture.isOpened.return_value = True
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (False, None)  # Read fails
        mock_capture_class.return_value = mock_capture

        camera.start()
        time.sleep(0.2)
        camera.stop()

        # Should handle gracefully

    @patch("src.vision.camera.cv2.VideoCapture")
    def test_capture_loop_camera_disconnect(self, mock_capture_class, camera):
        """Test capture loop handling camera disconnect."""
        mock_capture = MagicMock()
        # First call succeeds, then fails
        mock_capture.isOpened.side_effect = [True, True, False, True, True]
        mock_capture.get.return_value = 640
        mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_capture_class.return_value = mock_capture

        camera.start()
        time.sleep(0.3)
        camera.stop()
