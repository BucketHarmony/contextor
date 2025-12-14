"""Voice Activity Detection using Silero VAD."""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import torch

from ..config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment."""
    audio: np.ndarray
    start_time: datetime
    end_time: datetime
    duration_ms: int


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD model."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.threshold = config.vad_threshold
        self.min_speech_ms = config.vad_min_speech_ms
        self.min_silence_ms = config.vad_min_silence_ms

        # Calculate samples for timing thresholds
        self.min_speech_samples = int(self.min_speech_ms * self.sample_rate / 1000)
        self.min_silence_samples = int(self.min_silence_ms * self.sample_rate / 1000)

        # State tracking
        self._is_speaking = False
        self._speech_buffer: list[np.ndarray] = []
        self._silence_samples = 0
        self._speech_start_time: Optional[datetime] = None

        # Callbacks
        self._on_speech_end: list[Callable[[SpeechSegment], None]] = []

        # Load Silero VAD model
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Silero VAD model."""
        logger.info("Loading Silero VAD model...")
        try:
            self._model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._model.eval()
            logger.info("Silero VAD model loaded")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def reset_state(self) -> None:
        """Reset VAD state."""
        self._is_speaking = False
        self._speech_buffer.clear()
        self._silence_samples = 0
        self._speech_start_time = None
        if self._model is not None:
            self._model.reset_states()

    def on_speech_end(self, callback: Callable[[SpeechSegment], None]) -> None:
        """Register callback for when speech segment ends."""
        self._on_speech_end.append(callback)

    def process_audio(self, audio_chunk: np.ndarray) -> Optional[float]:
        """
        Process an audio chunk through VAD.

        Args:
            audio_chunk: Audio samples as float32 numpy array

        Returns:
            Speech probability (0.0-1.0) or None if model not loaded
        """
        if self._model is None:
            return None

        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_chunk).float()

        # Get speech probability
        with torch.no_grad():
            speech_prob = self._model(audio_tensor, self.sample_rate).item()

        # Update state machine
        self._update_state(audio_chunk, speech_prob)

        return speech_prob

    def _update_state(self, audio_chunk: np.ndarray, speech_prob: float) -> None:
        """Update speech detection state machine."""
        is_speech = speech_prob >= self.threshold
        current_time = datetime.now()

        if is_speech:
            if not self._is_speaking:
                # Speech started
                self._is_speaking = True
                self._speech_start_time = current_time
                self._speech_buffer = []
                logger.debug("Speech started")

            self._speech_buffer.append(audio_chunk)
            self._silence_samples = 0

        else:  # Silence
            if self._is_speaking:
                # Still in speech segment, buffer the silence
                self._speech_buffer.append(audio_chunk)
                self._silence_samples += len(audio_chunk)

                # Check if silence is long enough to end speech
                if self._silence_samples >= self.min_silence_samples:
                    self._end_speech_segment(current_time)

    def _end_speech_segment(self, end_time: datetime) -> None:
        """End the current speech segment and trigger callbacks."""
        if not self._speech_buffer or self._speech_start_time is None:
            self.reset_state()
            return

        # Concatenate all buffered audio
        full_audio = np.concatenate(self._speech_buffer)

        # Check minimum speech duration
        speech_samples = len(full_audio) - self._silence_samples
        if speech_samples < self.min_speech_samples:
            logger.debug("Speech segment too short, discarding")
            self.reset_state()
            return

        # Create speech segment
        duration_ms = int(len(full_audio) * 1000 / self.sample_rate)
        segment = SpeechSegment(
            audio=full_audio,
            start_time=self._speech_start_time,
            end_time=end_time,
            duration_ms=duration_ms,
        )

        logger.debug(f"Speech segment: {duration_ms}ms")

        # Trigger callbacks
        for callback in self._on_speech_end:
            try:
                callback(segment)
            except Exception as e:
                logger.error(f"Speech callback error: {e}")

        self.reset_state()

    def flush(self) -> Optional[SpeechSegment]:
        """Flush any pending speech buffer."""
        if self._is_speaking and self._speech_buffer:
            self._end_speech_segment(datetime.now())
        return None

    @property
    def is_speaking(self) -> bool:
        """Check if currently detecting speech."""
        return self._is_speaking
