"""Tests for the audio capture module."""

import queue
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.capture import AudioCapture
from src.config import AudioConfig


class TestAudioCapture:
    """Tests for AudioCapture class."""

    @pytest.fixture
    def audio_config(self):
        """Create test audio config."""
        return AudioConfig(
            device="default",
            sample_rate=16000,
            channels=1,
            chunk_duration_ms=512,
        )

    @pytest.fixture
    def audio_capture(self, audio_config):
        """Create AudioCapture instance."""
        return AudioCapture(audio_config)

    def test_init(self, audio_capture, audio_config):
        """Test AudioCapture initialization."""
        assert audio_capture.sample_rate == audio_config.sample_rate
        assert audio_capture.channels == audio_config.channels
        assert audio_capture.chunk_samples == int(16000 * 0.512)
        assert not audio_capture.is_running()

    def test_add_callback(self, audio_capture):
        """Test adding callbacks."""
        callback = MagicMock()
        audio_capture.add_callback(callback)
        assert callback in audio_capture._callbacks

    def test_remove_callback(self, audio_capture):
        """Test removing callbacks."""
        callback = MagicMock()
        audio_capture.add_callback(callback)
        audio_capture.remove_callback(callback)
        assert callback not in audio_capture._callbacks

    def test_remove_nonexistent_callback(self, audio_capture):
        """Test removing non-existent callback doesn't raise."""
        callback = MagicMock()
        audio_capture.remove_callback(callback)  # Should not raise

    @patch("src.audio.capture.sd.InputStream")
    def test_start(self, mock_input_stream, audio_capture):
        """Test starting audio capture."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()

        assert audio_capture.is_running()
        mock_input_stream.assert_called_once()
        mock_stream.start.assert_called_once()

        # Cleanup
        audio_capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_start_already_running(self, mock_input_stream, audio_capture):
        """Test starting when already running."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()
        audio_capture.start()  # Should log warning but not fail

        assert mock_input_stream.call_count == 1

        audio_capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_stop(self, mock_input_stream, audio_capture):
        """Test stopping audio capture."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()
        audio_capture.stop()

        assert not audio_capture.is_running()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    def test_stop_not_running(self, audio_capture):
        """Test stopping when not running doesn't raise."""
        audio_capture.stop()  # Should not raise

    @patch("src.audio.capture.sd.InputStream")
    def test_audio_callback_triggers_callbacks(self, mock_input_stream, audio_capture):
        """Test that audio callback triggers registered callbacks."""
        callback = MagicMock()
        audio_capture.add_callback(callback)

        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()

        # Simulate audio callback
        test_audio = np.random.randn(8192, 1).astype(np.float32)
        audio_capture._audio_callback(test_audio, 8192, {}, None)

        # Wait for processing
        time.sleep(0.2)

        callback.assert_called()
        audio_capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_audio_callback_with_status(self, mock_input_stream, audio_capture):
        """Test audio callback handles status flags."""
        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()

        # Simulate callback with status
        test_audio = np.random.randn(8192, 1).astype(np.float32)
        status = MagicMock()
        audio_capture._audio_callback(test_audio, 8192, {}, status)

        audio_capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_callback_error_handling(self, mock_input_stream, audio_capture):
        """Test that callback errors are caught."""
        def error_callback(audio):
            raise ValueError("Test error")

        audio_capture.add_callback(error_callback)

        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        audio_capture.start()

        # Simulate audio callback - should not raise
        test_audio = np.random.randn(8192, 1).astype(np.float32)
        audio_capture._audio_callback(test_audio, 8192, {}, None)

        time.sleep(0.2)
        audio_capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_start_with_numeric_device(self, mock_input_stream):
        """Test starting with numeric device ID."""
        config = AudioConfig(device="1")
        capture = AudioCapture(config)

        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture.start()

        call_kwargs = mock_input_stream.call_args[1]
        assert call_kwargs["device"] == 1

        capture.stop()

    @patch("src.audio.capture.sd.InputStream")
    def test_start_with_string_device(self, mock_input_stream):
        """Test starting with string device name."""
        config = AudioConfig(device="hw:1,0")
        capture = AudioCapture(config)

        mock_stream = MagicMock()
        mock_input_stream.return_value = mock_stream

        capture.start()

        call_kwargs = mock_input_stream.call_args[1]
        assert call_kwargs["device"] == "hw:1,0"

        capture.stop()

    @patch("src.audio.capture.sd.query_devices")
    def test_list_devices(self, mock_query):
        """Test listing audio devices."""
        mock_query.return_value = [
            {"name": "Device 1", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "Device 2", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "Device 3", "max_input_channels": 1, "default_samplerate": 16000},
        ]

        devices = AudioCapture.list_devices()

        assert len(devices) == 2  # Only input devices
        assert devices[0]["name"] == "Device 1"
        assert devices[0]["channels"] == 2
        assert devices[1]["name"] == "Device 3"

    def test_process_loop_timeout(self, audio_capture):
        """Test process loop handles empty queue timeout."""
        audio_capture._running = True

        # Run process loop briefly
        def stop_soon():
            time.sleep(0.2)
            audio_capture._running = False

        stopper = threading.Thread(target=stop_soon)
        stopper.start()

        audio_capture._process_loop()  # Should exit cleanly

        stopper.join()
