"""Tests for the voice activity detection module."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.audio.vad import SpeechSegment, VoiceActivityDetector
from src.config import AudioConfig


class TestSpeechSegment:
    """Tests for SpeechSegment dataclass."""

    def test_speech_segment_creation(self, sample_audio_chunk):
        """Test creating a speech segment."""
        segment = SpeechSegment(
            audio=sample_audio_chunk,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=500,
        )

        assert segment.duration_ms == 500
        assert len(segment.audio) == len(sample_audio_chunk)


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    @pytest.fixture
    def mock_vad_config(self):
        """Create test VAD config."""
        return AudioConfig(
            sample_rate=16000,
            vad_threshold=0.5,
            vad_min_speech_ms=250,
            vad_min_silence_ms=500,
        )

    @pytest.fixture
    @patch("src.audio.vad.torch.hub.load")
    def vad(self, mock_load, mock_vad_config, mock_vad_model):
        """Create VAD instance with mocked model."""
        mock_load.return_value = (mock_vad_model, None)
        return VoiceActivityDetector(mock_vad_config)

    @patch("src.audio.vad.torch.hub.load")
    def test_init(self, mock_load, mock_vad_config, mock_vad_model):
        """Test VAD initialization."""
        mock_load.return_value = (mock_vad_model, None)
        vad = VoiceActivityDetector(mock_vad_config)

        assert vad.sample_rate == 16000
        assert vad.threshold == 0.5
        assert not vad.is_speaking

    @patch("src.audio.vad.torch.hub.load")
    def test_load_model_failure(self, mock_load, mock_vad_config):
        """Test model loading failure."""
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            VoiceActivityDetector(mock_vad_config)

    def test_reset_state(self, vad):
        """Test resetting VAD state."""
        vad._is_speaking = True
        vad._speech_buffer = [np.zeros(100)]
        vad._silence_samples = 1000

        vad.reset_state()

        assert not vad._is_speaking
        assert len(vad._speech_buffer) == 0
        assert vad._silence_samples == 0

    def test_on_speech_end_callback(self, vad):
        """Test registering speech end callback."""
        callback = MagicMock()
        vad.on_speech_end(callback)

        assert callback in vad._on_speech_end

    def test_process_audio_no_model(self, mock_vad_config):
        """Test process_audio when model not loaded."""
        with patch("src.audio.vad.torch.hub.load") as mock_load:
            mock_load.return_value = (MagicMock(), None)
            vad = VoiceActivityDetector(mock_vad_config)
            vad._model = None

            result = vad.process_audio(np.zeros(8192, dtype=np.float32))
            assert result is None

    def test_process_audio_speech_detected(self, vad, sample_audio_chunk):
        """Test processing audio with speech."""
        # Configure model to return high probability
        vad._model.return_value = MagicMock(item=MagicMock(return_value=0.9))

        prob = vad.process_audio(sample_audio_chunk)

        assert prob == 0.9
        assert vad._is_speaking

    def test_process_audio_silence_detected(self, vad, silence_audio_chunk):
        """Test processing audio with silence."""
        # Configure model to return low probability
        vad._model.return_value = MagicMock(item=MagicMock(return_value=0.1))

        prob = vad.process_audio(silence_audio_chunk)

        assert prob == 0.1
        assert not vad._is_speaking

    def test_speech_segment_creation_on_silence(self, vad, sample_audio_chunk):
        """Test speech segment is created after silence."""
        callback = MagicMock()
        vad.on_speech_end(callback)

        # Simulate speech
        vad._model.return_value = MagicMock(item=MagicMock(return_value=0.9))
        for _ in range(5):
            vad.process_audio(sample_audio_chunk)

        # Simulate silence to trigger segment end
        vad._model.return_value = MagicMock(item=MagicMock(return_value=0.1))
        silence_chunk = np.zeros(vad.min_silence_samples + 1000, dtype=np.float32)
        vad.process_audio(silence_chunk)

        # Should have triggered callback
        assert callback.called

    def test_short_speech_discarded(self, vad):
        """Test that short speech segments are discarded."""
        callback = MagicMock()
        vad.on_speech_end(callback)

        # Very short speech
        vad._is_speaking = True
        vad._speech_buffer = [np.zeros(100, dtype=np.float32)]  # Very short
        vad._speech_start_time = datetime.now()
        vad._silence_samples = vad.min_silence_samples + 100

        vad._end_speech_segment(datetime.now())

        callback.assert_not_called()

    def test_flush_with_pending_speech(self, vad, sample_audio_chunk):
        """Test flushing with pending speech."""
        callback = MagicMock()
        vad.on_speech_end(callback)

        # Set up speech state
        vad._is_speaking = True
        vad._speech_buffer = [sample_audio_chunk for _ in range(10)]
        vad._speech_start_time = datetime.now()
        vad._silence_samples = 0

        vad.flush()

        # Should have triggered callback with speech segment
        callback.assert_called()

    def test_flush_without_pending_speech(self, vad):
        """Test flushing without pending speech."""
        vad._is_speaking = False
        result = vad.flush()
        assert result is None

    def test_is_speaking_property(self, vad):
        """Test is_speaking property."""
        assert not vad.is_speaking

        vad._is_speaking = True
        assert vad.is_speaking

    def test_callback_error_handling(self, vad, sample_audio_chunk):
        """Test callback errors are caught."""
        def error_callback(segment):
            raise ValueError("Callback error")

        vad.on_speech_end(error_callback)

        # Set up for segment creation
        vad._is_speaking = True
        vad._speech_buffer = [sample_audio_chunk for _ in range(10)]
        vad._speech_start_time = datetime.now()
        vad._silence_samples = 0

        # Should not raise
        vad._end_speech_segment(datetime.now())

    def test_update_state_speech_start(self, vad, sample_audio_chunk):
        """Test state update when speech starts."""
        vad._is_speaking = False
        vad._update_state(sample_audio_chunk, 0.9)

        assert vad._is_speaking
        assert vad._speech_start_time is not None
        assert len(vad._speech_buffer) == 1

    def test_update_state_continued_speech(self, vad, sample_audio_chunk):
        """Test state update during continued speech."""
        vad._is_speaking = True
        vad._speech_start_time = datetime.now()
        vad._speech_buffer = [sample_audio_chunk]

        vad._update_state(sample_audio_chunk, 0.9)

        assert len(vad._speech_buffer) == 2
        assert vad._silence_samples == 0

    def test_update_state_silence_during_speech(self, vad, silence_audio_chunk):
        """Test state update when silence occurs during speech."""
        vad._is_speaking = True
        vad._speech_start_time = datetime.now()
        vad._speech_buffer = [np.zeros(8192, dtype=np.float32)]
        vad._silence_samples = 0

        vad._update_state(silence_audio_chunk, 0.1)

        assert vad._is_speaking  # Still speaking
        assert vad._silence_samples == len(silence_audio_chunk)
