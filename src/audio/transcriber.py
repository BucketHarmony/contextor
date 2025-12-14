"""Speech-to-text transcription using faster-whisper."""

import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from faster_whisper import WhisperModel

from ..config import AudioConfig
from .vad import SpeechSegment

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A transcribed speech segment."""
    text: str
    timestamp: datetime
    end_timestamp: datetime
    confidence: float
    duration_ms: int


class Transcriber:
    """Speech-to-text transcription using faster-whisper."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.model_name = config.whisper_model
        self.device = config.whisper_device
        self.compute_type = config.whisper_compute_type
        self.sample_rate = config.sample_rate

        self._model: Optional[WhisperModel] = None
        self._transcription_queue: queue.Queue[SpeechSegment] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Callbacks for transcription results
        self._on_transcription: list[Callable[[TranscriptSegment], None]] = []

        # Recent transcripts buffer
        self._recent_transcripts: list[TranscriptSegment] = []
        self._transcripts_lock = threading.Lock()

    def _load_model(self) -> None:
        """Load the Whisper model."""
        logger.info(f"Loading Whisper model: {self.model_name} on {self.device}")
        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info("Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _transcription_loop(self) -> None:
        """Process speech segments from queue."""
        while self._running:
            try:
                segment = self._transcription_queue.get(timeout=1.0)
                self._transcribe_segment(segment)
            except queue.Empty:
                continue

    def _transcribe_segment(self, segment: SpeechSegment) -> None:
        """Transcribe a speech segment."""
        if self._model is None:
            logger.warning("Whisper model not loaded")
            return

        try:
            # Transcribe
            segments, info = self._model.transcribe(
                segment.audio,
                beam_size=5,
                language="en",
                vad_filter=False,  # We already did VAD
            )

            # Collect transcription
            texts = []
            total_confidence = 0.0
            segment_count = 0

            for seg in segments:
                texts.append(seg.text.strip())
                total_confidence += seg.avg_logprob
                segment_count += 1

            if not texts:
                return

            full_text = " ".join(texts)
            avg_confidence = (
                np.exp(total_confidence / segment_count) if segment_count > 0 else 0.0
            )

            # Create transcript segment
            transcript = TranscriptSegment(
                text=full_text,
                timestamp=segment.start_time,
                end_timestamp=segment.end_time,
                confidence=float(avg_confidence),
                duration_ms=segment.duration_ms,
            )

            logger.info(f"Transcribed: '{full_text[:50]}...' (conf: {avg_confidence:.2f})")

            # Store in recent transcripts
            with self._transcripts_lock:
                self._recent_transcripts.append(transcript)

            # Trigger callbacks
            for callback in self._on_transcription:
                try:
                    callback(transcript)
                except Exception as e:
                    logger.error(f"Transcription callback error: {e}")

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def on_transcription(self, callback: Callable[[TranscriptSegment], None]) -> None:
        """Register callback for transcription results."""
        self._on_transcription.append(callback)

    def queue_segment(self, segment: SpeechSegment) -> None:
        """Add a speech segment to the transcription queue."""
        self._transcription_queue.put(segment)

    def start(self) -> None:
        """Start the transcriber."""
        if self._running:
            logger.warning("Transcriber already running")
            return

        self._load_model()
        self._running = True

        self._thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._thread.start()

        logger.info("Transcriber started")

    def stop(self) -> None:
        """Stop the transcriber."""
        if not self._running:
            return

        logger.info("Stopping transcriber")
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Clear queue
        while not self._transcription_queue.empty():
            try:
                self._transcription_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Transcriber stopped")

    def get_recent_transcripts(self, clear: bool = False) -> list[TranscriptSegment]:
        """Get recent transcripts, optionally clearing the buffer."""
        with self._transcripts_lock:
            transcripts = list(self._recent_transcripts)
            if clear:
                self._recent_transcripts.clear()
        return transcripts

    def clear_transcripts(self) -> None:
        """Clear the transcript buffer."""
        with self._transcripts_lock:
            self._recent_transcripts.clear()

    def is_running(self) -> bool:
        """Check if transcriber is running."""
        return self._running
