"""Tests for the main orchestrator module."""

import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.config import Config, AudioConfig, VisionConfig, ContextConfig
from src.main import Contextor, main


class TestContextor:
    """Tests for Contextor class."""

    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create mock configuration."""
        return Config(
            audio=AudioConfig(
                device="default",
                whisper_model="tiny",
                whisper_device="cpu",
            ),
            vision=VisionConfig(
                camera_id=0,
                capture_interval=1,  # Short for testing
            ),
            context=ContextConfig(
                output_dir=str(temp_dir / "context"),
                images_dir=str(temp_dir / "images"),
                interval=1,  # Short for testing
            ),
        )

    @pytest.fixture
    @patch("src.main.ObjectDetector")
    @patch("src.main.MotionDetector")
    @patch("src.main.Camera")
    @patch("src.main.Transcriber")
    @patch("src.main.VoiceActivityDetector")
    @patch("src.main.AudioCapture")
    def contextor(
        self,
        mock_audio,
        mock_vad,
        mock_transcriber,
        mock_camera,
        mock_motion,
        mock_detector,
        mock_config,
    ):
        """Create Contextor with mocked components."""
        return Contextor(mock_config)

    def test_init(self, contextor):
        """Test Contextor initialization."""
        assert not contextor._running
        assert contextor.audio_capture is not None
        assert contextor.camera is not None

    @patch("src.main.ObjectDetector")
    @patch("src.main.MotionDetector")
    @patch("src.main.Camera")
    @patch("src.main.Transcriber")
    @patch("src.main.VoiceActivityDetector")
    @patch("src.main.AudioCapture")
    def test_start(
        self,
        mock_audio,
        mock_vad,
        mock_transcriber,
        mock_camera,
        mock_motion,
        mock_detector,
        mock_config,
    ):
        """Test starting Contextor."""
        contextor = Contextor(mock_config)

        with patch.object(contextor, "_start_web_server"):
            contextor.start(enable_web=False)

        assert contextor._running
        contextor.stop()

    @patch("src.main.ObjectDetector")
    @patch("src.main.MotionDetector")
    @patch("src.main.Camera")
    @patch("src.main.Transcriber")
    @patch("src.main.VoiceActivityDetector")
    @patch("src.main.AudioCapture")
    def test_start_already_running(
        self,
        mock_audio,
        mock_vad,
        mock_transcriber,
        mock_camera,
        mock_motion,
        mock_detector,
        mock_config,
    ):
        """Test starting when already running."""
        contextor = Contextor(mock_config)

        with patch.object(contextor, "_start_web_server"):
            contextor.start(enable_web=False)
            contextor.start(enable_web=False)  # Should not fail

        contextor.stop()

    @patch("src.main.ObjectDetector")
    @patch("src.main.MotionDetector")
    @patch("src.main.Camera")
    @patch("src.main.Transcriber")
    @patch("src.main.VoiceActivityDetector")
    @patch("src.main.AudioCapture")
    def test_stop(
        self,
        mock_audio,
        mock_vad,
        mock_transcriber,
        mock_camera,
        mock_motion,
        mock_detector,
        mock_config,
    ):
        """Test stopping Contextor."""
        contextor = Contextor(mock_config)

        with patch.object(contextor, "_start_web_server"):
            contextor.start(enable_web=False)
        contextor.stop()

        assert not contextor._running

    @patch("src.main.ObjectDetector")
    @patch("src.main.MotionDetector")
    @patch("src.main.Camera")
    @patch("src.main.Transcriber")
    @patch("src.main.VoiceActivityDetector")
    @patch("src.main.AudioCapture")
    def test_stop_not_running(
        self,
        mock_audio,
        mock_vad,
        mock_transcriber,
        mock_camera,
        mock_motion,
        mock_detector,
        mock_config,
    ):
        """Test stopping when not running."""
        contextor = Contextor(mock_config)
        contextor.stop()  # Should not fail

    def test_on_transcription(self, contextor):
        """Test transcription callback."""
        from src.audio.transcriber import TranscriptSegment

        transcript = TranscriptSegment(
            text="Test",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.9,
            duration_ms=1000,
        )

        contextor._on_transcription(transcript)

        assert contextor.aggregator.transcript_count == 1

    def test_on_frame(self, contextor, sample_frame):
        """Test frame callback."""
        from src.vision.camera import Frame

        frame = Frame(
            image=sample_frame,
            timestamp=datetime.now(),
            frame_number=1,
        )

        # Motion detector should be called
        contextor._on_frame(frame)

    def test_on_motion(self, contextor, sample_motion_event):
        """Test motion callback."""
        with patch.object(contextor.storage, "save_image", return_value="/tmp/test.jpg"):
            with patch.object(contextor.object_detector, "detect") as mock_detect:
                from src.vision.detector import DetectionResult
                mock_detect.return_value = DetectionResult(
                    timestamp=datetime.now(),
                    image_path="/tmp/test.jpg",
                )

                contextor._on_motion(sample_motion_event)

                mock_detect.assert_called_once()

    def test_scheduled_capture_not_time(self, contextor):
        """Test scheduled capture before interval."""
        contextor._last_scheduled_capture = time.time()

        with patch.object(contextor.camera, "capture_image") as mock_capture:
            contextor._scheduled_capture()
            mock_capture.assert_not_called()

    def test_scheduled_capture_time_elapsed(self, contextor):
        """Test scheduled capture after interval."""
        contextor._last_scheduled_capture = time.time() - 1000

        with patch.object(contextor.camera, "capture_image", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            with patch.object(contextor.storage, "save_image", return_value="/tmp/test.jpg"):
                with patch.object(contextor.object_detector, "detect"):
                    contextor._scheduled_capture()

    def test_scheduled_capture_no_image(self, contextor):
        """Test scheduled capture when no image returned."""
        contextor._last_scheduled_capture = time.time() - 1000

        with patch.object(contextor.camera, "capture_image", return_value=None):
            contextor._scheduled_capture()

    def test_generate_context_not_time(self, contextor):
        """Test context generation before interval."""
        contextor._last_context_generation = time.time()

        result = contextor._generate_context()

        assert result is None

    def test_generate_context_time_elapsed(self, contextor):
        """Test context generation after interval."""
        contextor._last_context_generation = time.time() - 1000

        with patch.object(contextor.generator, "generate", return_value="/tmp/context.json"):
            with patch.object(contextor.storage, "cleanup_old_files", return_value=0):
                result = contextor._generate_context()

                assert result == "/tmp/context.json"

    def test_generate_context_forced(self, contextor):
        """Test forced context generation."""
        contextor._last_context_generation = time.time()

        with patch.object(contextor.generator, "generate", return_value="/tmp/context.json"):
            with patch.object(contextor.storage, "cleanup_old_files", return_value=0):
                result = contextor._generate_context(force=True)

                assert result == "/tmp/context.json"

    def test_check_trigger_no_file(self, contextor):
        """Test trigger check when no file."""
        contextor._trigger_file = Path("/tmp/nonexistent_trigger")

        result = contextor._check_trigger()

        assert result is False

    def test_check_trigger_with_file(self, contextor, temp_dir):
        """Test trigger check with file."""
        trigger_file = temp_dir / "trigger"
        trigger_file.touch()
        contextor._trigger_file = trigger_file

        result = contextor._check_trigger()

        assert result is True
        assert not trigger_file.exists()

    def test_trigger_context_now(self, contextor):
        """Test manual trigger."""
        with patch.object(contextor, "_generate_context", return_value="/tmp/ctx.json") as mock_gen:
            result = contextor.trigger_context_now()

            mock_gen.assert_called_with(force=True)
            assert result == "/tmp/ctx.json"

    def test_get_status(self, contextor):
        """Test getting status."""
        status = contextor.get_status()

        assert "running" in status
        assert "audio" in status
        assert "vision" in status
        assert "storage" in status

    @patch("src.main.uvicorn")
    def test_start_web_server(self, mock_uvicorn, contextor):
        """Test starting web server."""
        mock_server = MagicMock()
        mock_uvicorn.Server.return_value = mock_server

        contextor._start_web_server(port=8080)

        assert contextor._web_thread is not None

    def test_stop_web_server(self, contextor):
        """Test stopping web server."""
        mock_server = MagicMock()
        contextor._web_server = mock_server
        contextor._web_thread = MagicMock()

        contextor._stop_web_server()

        assert mock_server.should_exit is True

    def test_stop_web_server_not_running(self, contextor):
        """Test stopping web server when not running."""
        contextor._stop_web_server()  # Should not fail

    def test_main_loop_trigger(self, contextor):
        """Test main loop processes trigger."""
        with patch.object(contextor, "_check_trigger", side_effect=[True, False]):
            with patch.object(contextor, "_generate_context"):
                with patch.object(contextor, "_scheduled_capture"):
                    contextor._running = True

                    def stop_loop():
                        time.sleep(0.2)
                        contextor._shutdown_event.set()

                    stopper = threading.Thread(target=stop_loop)
                    stopper.start()

                    contextor._main_loop()

                    stopper.join()

    def test_main_loop_error_handling(self, contextor):
        """Test main loop handles errors."""
        with patch.object(contextor, "_check_trigger", side_effect=Exception("Error")):
            contextor._running = True

            def stop_loop():
                time.sleep(0.3)
                contextor._shutdown_event.set()

            stopper = threading.Thread(target=stop_loop)
            stopper.start()

            contextor._main_loop()  # Should not raise

            stopper.join()


class TestMain:
    """Tests for main function."""

    @patch("src.main.AudioCapture")
    def test_list_audio(self, mock_audio, capsys):
        """Test listing audio devices."""
        mock_audio.list_devices.return_value = [
            {"id": 0, "name": "Test Device", "channels": 2}
        ]

        with patch("sys.argv", ["contextor", "--list-audio"]):
            main()

        captured = capsys.readouterr()
        assert "Test Device" in captured.out

    @patch("src.main.Camera")
    def test_list_cameras(self, mock_camera, capsys):
        """Test listing cameras."""
        mock_camera.list_cameras.return_value = [
            {"id": 0, "width": 640, "height": 480}
        ]

        with patch("sys.argv", ["contextor", "--list-cameras"]):
            main()

        captured = capsys.readouterr()
        assert "640" in captured.out

    @patch("src.main.Contextor")
    @patch("src.main.load_config")
    def test_main_with_config(self, mock_load, mock_contextor_class, temp_dir):
        """Test main with config file."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_contextor = MagicMock()
        mock_contextor_class.return_value = mock_contextor

        # Simulate KeyboardInterrupt after start
        def raise_keyboard():
            raise KeyboardInterrupt()

        with patch("time.sleep", side_effect=raise_keyboard):
            with patch("sys.argv", ["contextor", "-c", str(temp_dir / "config.yaml")]):
                main()

        mock_contextor.start.assert_called_once()
        mock_contextor.stop.assert_called_once()

    @patch("src.main.Contextor")
    @patch("src.main.load_config")
    def test_main_no_web(self, mock_load, mock_contextor_class):
        """Test main with --no-web flag."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_contextor = MagicMock()
        mock_contextor_class.return_value = mock_contextor

        def raise_keyboard():
            raise KeyboardInterrupt()

        with patch("time.sleep", side_effect=raise_keyboard):
            with patch("sys.argv", ["contextor", "--no-web"]):
                main()

        mock_contextor.start.assert_called_with(enable_web=True, web_port=8080)

    @patch("src.main.Contextor")
    @patch("src.main.load_config")
    def test_main_custom_port(self, mock_load, mock_contextor_class):
        """Test main with custom port."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_contextor = MagicMock()
        mock_contextor_class.return_value = mock_contextor

        def raise_keyboard():
            raise KeyboardInterrupt()

        with patch("time.sleep", side_effect=raise_keyboard):
            with patch("sys.argv", ["contextor", "-p", "9000"]):
                main()

        mock_contextor.start.assert_called_with(enable_web=True, web_port=9000)

    @patch("src.main.Contextor")
    @patch("src.main.load_config")
    @patch("src.main.signal.signal")
    def test_signal_handlers(self, mock_signal, mock_load, mock_contextor_class):
        """Test signal handlers are set up."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config

        mock_contextor = MagicMock()
        mock_contextor_class.return_value = mock_contextor

        def raise_keyboard():
            raise KeyboardInterrupt()

        with patch("time.sleep", side_effect=raise_keyboard):
            with patch("sys.argv", ["contextor"]):
                main()

        # Signal handlers should be registered
        assert mock_signal.call_count >= 2  # SIGINT, SIGTERM
