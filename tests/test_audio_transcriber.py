"""Tests for the transcriber module."""

import queue
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.transcriber import Transcriber, TranscriptSegment
from src.audio.vad import SpeechSegment
from src.config import AudioConfig


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_transcript_segment_creation(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            text="Hello world",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.95,
            duration_ms=1000,
        )

        assert segment.text == "Hello world"
        assert segment.confidence == 0.95
        assert segment.duration_ms == 1000


class TestTranscriber:
    """Tests for Transcriber class."""

    @pytest.fixture
    def transcriber_config(self):
        """Create test config."""
        return AudioConfig(
            sample_rate=16000,
            whisper_model="tiny",
            whisper_device="cpu",
            whisper_compute_type="float32",
        )

    @pytest.fixture
    def speech_segment(self, sample_audio_chunk):
        """Create test speech segment."""
        return SpeechSegment(
            audio=sample_audio_chunk,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=512,
        )

    @pytest.fixture
    @patch("src.audio.transcriber.WhisperModel")
    def transcriber(self, mock_whisper, transcriber_config):
        """Create Transcriber with mocked model."""
        transcriber = Transcriber(transcriber_config)
        return transcriber

    def test_init(self, transcriber_config):
        """Test Transcriber initialization."""
        transcriber = Transcriber(transcriber_config)

        assert transcriber.model_name == "tiny"
        assert transcriber.device == "cpu"
        assert not transcriber.is_running()

    @patch("src.audio.transcriber.WhisperModel")
    def test_load_model(self, mock_whisper_class, transcriber):
        """Test model loading."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        transcriber._load_model()

        mock_whisper_class.assert_called_with(
            "tiny",
            device="cpu",
            compute_type="float32",
        )
        assert transcriber._model == mock_model

    @patch("src.audio.transcriber.WhisperModel")
    def test_load_model_failure(self, mock_whisper_class, transcriber):
        """Test model loading failure."""
        mock_whisper_class.side_effect = Exception("Load failed")

        with pytest.raises(Exception):
            transcriber._load_model()

    @patch("src.audio.transcriber.WhisperModel")
    def test_start(self, mock_whisper_class, transcriber):
        """Test starting transcriber."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        transcriber.start()

        assert transcriber.is_running()
        assert transcriber._thread is not None

        transcriber.stop()

    @patch("src.audio.transcriber.WhisperModel")
    def test_start_already_running(self, mock_whisper_class, transcriber):
        """Test starting when already running."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        transcriber.start()
        transcriber.start()  # Should not fail

        transcriber.stop()

    def test_stop_not_running(self, transcriber):
        """Test stopping when not running."""
        transcriber.stop()  # Should not raise

    @patch("src.audio.transcriber.WhisperModel")
    def test_stop(self, mock_whisper_class, transcriber):
        """Test stopping transcriber."""
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model

        transcriber.start()
        transcriber.stop()

        assert not transcriber.is_running()

    def test_on_transcription_callback(self, transcriber):
        """Test registering transcription callback."""
        callback = MagicMock()
        transcriber.on_transcription(callback)

        assert callback in transcriber._on_transcription

    def test_queue_segment(self, transcriber, speech_segment):
        """Test queuing a speech segment."""
        transcriber.queue_segment(speech_segment)

        assert not transcriber._transcription_queue.empty()

    @patch("src.audio.transcriber.WhisperModel")
    def test_transcribe_segment(self, mock_whisper_class, transcriber, speech_segment):
        """Test transcribing a segment."""
        # Setup mock model
        mock_segment = MagicMock()
        mock_segment.text = "Test transcription"
        mock_segment.avg_logprob = -0.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        transcriber._model = mock_model

        callback = MagicMock()
        transcriber.on_transcription(callback)

        transcriber._transcribe_segment(speech_segment)

        callback.assert_called_once()
        transcript = callback.call_args[0][0]
        assert "Test transcription" in transcript.text

    def test_transcribe_segment_no_model(self, transcriber, speech_segment):
        """Test transcribing without model."""
        transcriber._model = None
        transcriber._transcribe_segment(speech_segment)  # Should not raise

    @patch("src.audio.transcriber.WhisperModel")
    def test_transcribe_segment_empty_result(self, mock_whisper_class, transcriber, speech_segment):
        """Test transcribing with empty result."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([], MagicMock())
        transcriber._model = mock_model

        callback = MagicMock()
        transcriber.on_transcription(callback)

        transcriber._transcribe_segment(speech_segment)

        callback.assert_not_called()

    @patch("src.audio.transcriber.WhisperModel")
    def test_transcribe_segment_error(self, mock_whisper_class, transcriber, speech_segment):
        """Test transcribing with error."""
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Transcription error")
        transcriber._model = mock_model

        # Should not raise
        transcriber._transcribe_segment(speech_segment)

    def test_get_recent_transcripts(self, transcriber):
        """Test getting recent transcripts."""
        segment = TranscriptSegment(
            text="Test",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.9,
            duration_ms=1000,
        )
        transcriber._recent_transcripts.append(segment)

        transcripts = transcriber.get_recent_transcripts()

        assert len(transcripts) == 1
        assert transcripts[0].text == "Test"

    def test_get_recent_transcripts_with_clear(self, transcriber):
        """Test getting recent transcripts with clear."""
        segment = TranscriptSegment(
            text="Test",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.9,
            duration_ms=1000,
        )
        transcriber._recent_transcripts.append(segment)

        transcripts = transcriber.get_recent_transcripts(clear=True)

        assert len(transcripts) == 1
        assert len(transcriber._recent_transcripts) == 0

    def test_clear_transcripts(self, transcriber):
        """Test clearing transcripts."""
        segment = TranscriptSegment(
            text="Test",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.9,
            duration_ms=1000,
        )
        transcriber._recent_transcripts.append(segment)

        transcriber.clear_transcripts()

        assert len(transcriber._recent_transcripts) == 0

    @patch("src.audio.transcriber.WhisperModel")
    def test_callback_error_handling(self, mock_whisper_class, transcriber, speech_segment):
        """Test callback error handling."""
        mock_segment = MagicMock()
        mock_segment.text = "Test"
        mock_segment.avg_logprob = -0.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        transcriber._model = mock_model

        def error_callback(transcript):
            raise ValueError("Callback error")

        transcriber.on_transcription(error_callback)

        # Should not raise
        transcriber._transcribe_segment(speech_segment)

    @patch("src.audio.transcriber.WhisperModel")
    def test_transcription_loop(self, mock_whisper_class, transcriber, speech_segment):
        """Test transcription processing loop."""
        mock_segment = MagicMock()
        mock_segment.text = "Loop test"
        mock_segment.avg_logprob = -0.5

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], MagicMock())
        mock_whisper_class.return_value = mock_model

        transcriber.start()
        transcriber.queue_segment(speech_segment)

        # Wait for processing
        time.sleep(0.5)

        transcripts = transcriber.get_recent_transcripts()
        transcriber.stop()

        assert len(transcripts) >= 1
