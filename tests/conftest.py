"""Pytest configuration and shared fixtures."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ==================== Path Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file."""
    config_path = temp_dir / "settings.yaml"
    config_content = """
audio:
  device: "default"
  sample_rate: 16000
  channels: 1
  chunk_duration_ms: 512
  vad_threshold: 0.5
  whisper_model: "tiny"
  whisper_device: "cpu"

vision:
  camera_id: 0
  camera_width: 640
  camera_height: 480
  motion_threshold: 25
  yolo_model: "yolov8n.pt"
  yolo_confidence: 0.5

context:
  output_dir: "{output_dir}"
  images_dir: "{images_dir}"
  interval: 300
  keep_images: true
  max_storage_gb: 1.0

logging:
  level: "DEBUG"
  file: null
""".format(output_dir=str(temp_dir / "context"), images_dir=str(temp_dir / "images"))

    config_path.write_text(config_content)
    return config_path


# ==================== Audio Fixtures ====================

@pytest.fixture
def sample_audio_chunk():
    """Generate a sample audio chunk."""
    # 512ms of audio at 16kHz = 8192 samples
    duration_samples = int(16000 * 0.512)
    # Generate some noise with speech-like characteristics
    audio = np.random.randn(duration_samples).astype(np.float32) * 0.1
    return audio


@pytest.fixture
def silence_audio_chunk():
    """Generate a silent audio chunk."""
    duration_samples = int(16000 * 0.512)
    return np.zeros(duration_samples, dtype=np.float32)


@pytest.fixture
def mock_audio_config():
    """Create a mock audio config."""
    from src.config import AudioConfig
    return AudioConfig(
        device="default",
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=512,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_min_silence_ms=500,
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_compute_type="float32",
    )


# ==================== Vision Fixtures ====================

@pytest.fixture
def sample_frame():
    """Generate a sample video frame."""
    # 640x480 BGR image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def black_frame():
    """Generate a black frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    """Generate a white frame."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


@pytest.fixture
def mock_vision_config():
    """Create a mock vision config."""
    from src.config import VisionConfig
    return VisionConfig(
        camera_id=0,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        motion_detection_fps=5,
        motion_threshold=25,
        motion_min_area=500,
        motion_blur_size=21,
        capture_interval=300,
        yolo_model="yolov8n.pt",
        yolo_confidence=0.5,
        yolo_device="cpu",
    )


# ==================== Transcript Fixtures ====================

@pytest.fixture
def sample_transcript_segment():
    """Create a sample transcript segment."""
    from src.audio.transcriber import TranscriptSegment
    return TranscriptSegment(
        text="Hello, this is a test transcript.",
        timestamp=datetime.now() - timedelta(seconds=30),
        end_timestamp=datetime.now() - timedelta(seconds=25),
        confidence=0.95,
        duration_ms=5000,
    )


@pytest.fixture
def sample_transcript_segments():
    """Create multiple sample transcript segments."""
    from src.audio.transcriber import TranscriptSegment
    now = datetime.now()
    return [
        TranscriptSegment(
            text="First segment.",
            timestamp=now - timedelta(minutes=5),
            end_timestamp=now - timedelta(minutes=4, seconds=55),
            confidence=0.9,
            duration_ms=5000,
        ),
        TranscriptSegment(
            text="Second segment.",
            timestamp=now - timedelta(minutes=3),
            end_timestamp=now - timedelta(minutes=2, seconds=55),
            confidence=0.85,
            duration_ms=5000,
        ),
        TranscriptSegment(
            text="Third segment.",
            timestamp=now - timedelta(minutes=1),
            end_timestamp=now - timedelta(seconds=55),
            confidence=0.92,
            duration_ms=5000,
        ),
    ]


# ==================== Detection Fixtures ====================

@pytest.fixture
def sample_detection_result():
    """Create a sample detection result."""
    from src.vision.detector import DetectedObject, DetectionResult
    return DetectionResult(
        timestamp=datetime.now(),
        image_path="/tmp/test_image.jpg",
        objects=[
            DetectedObject(
                label="person",
                confidence=0.95,
                bbox=(100, 100, 200, 300),
                center=(150, 200),
            ),
            DetectedObject(
                label="laptop",
                confidence=0.87,
                bbox=(250, 150, 400, 250),
                center=(325, 200),
            ),
        ],
        inference_time_ms=45.5,
        trigger="motion",
    )


@pytest.fixture
def sample_motion_event(sample_frame):
    """Create a sample motion event."""
    from src.vision.motion import MotionEvent
    return MotionEvent(
        timestamp=datetime.now(),
        frame=sample_frame,
        contour_area=1500,
        bounding_box=(100, 100, 200, 150),
    )


# ==================== Context Fixtures ====================

@pytest.fixture
def mock_context_config(temp_dir):
    """Create a mock context config."""
    from src.config import ContextConfig
    return ContextConfig(
        output_dir=str(temp_dir / "context"),
        images_dir=str(temp_dir / "images"),
        interval=300,
        keep_images=True,
        max_storage_gb=1.0,
        max_context_files=10,
    )


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_whisper_model():
    """Create a mock Whisper model."""
    mock_model = MagicMock()
    mock_segment = MagicMock()
    mock_segment.text = "Test transcription"
    mock_segment.avg_logprob = -0.5
    mock_model.transcribe.return_value = ([mock_segment], MagicMock())
    return mock_model


@pytest.fixture
def mock_yolo_model():
    """Create a mock YOLO model."""
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = MagicMock()
    mock_result.boxes.__len__ = MagicMock(return_value=1)
    mock_result.boxes.__iter__ = MagicMock(return_value=iter([]))
    mock_result.names = {0: "person"}
    mock_result.speed = {"inference": 50.0}
    mock_model.predict.return_value = [mock_result]
    return mock_model


@pytest.fixture
def mock_vad_model():
    """Create a mock Silero VAD model."""
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
    mock_model.reset_states = MagicMock()
    return mock_model


@pytest.fixture
def mock_camera():
    """Create a mock OpenCV VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.return_value = 640
    return mock_cap
