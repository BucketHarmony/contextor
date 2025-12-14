"""Context aggregator for collecting data from audio and vision pipelines."""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from ..audio.transcriber import TranscriptSegment
from ..vision.detector import DetectionResult
from ..vision.motion import MotionEvent

logger = logging.getLogger(__name__)

# Default history duration: 1 hour
TRANSCRIPT_HISTORY_DURATION = timedelta(hours=1)


@dataclass
class ContextData:
    """Aggregated context data for a time period."""
    period_start: datetime
    period_end: datetime

    # Audio data
    transcript_segments: list[TranscriptSegment] = field(default_factory=list)

    # Vision data
    motion_events: list[MotionEvent] = field(default_factory=list)
    detection_results: list[DetectionResult] = field(default_factory=list)

    @property
    def full_transcript(self) -> str:
        """Get concatenated transcript text."""
        return " ".join(seg.text for seg in self.transcript_segments)

    @property
    def motion_event_count(self) -> int:
        """Get number of motion events."""
        return len(self.motion_events)

    @property
    def images_captured(self) -> int:
        """Get number of images captured."""
        return len(self.detection_results)

    def get_object_summary(self) -> dict[str, dict]:
        """Get summary of all detected objects."""
        object_data: dict[str, list[float]] = {}

        for result in self.detection_results:
            for obj in result.objects:
                if obj.label not in object_data:
                    object_data[obj.label] = []
                object_data[obj.label].append(obj.confidence)

        summary = {}
        for label, confidences in object_data.items():
            summary[label] = {
                "count": len(confidences),
                "avg_confidence": round(sum(confidences) / len(confidences), 3),
            }

        return summary

    def get_unique_objects(self) -> list[str]:
        """Get list of unique detected objects."""
        objects = set()
        for result in self.detection_results:
            for obj in result.objects:
                objects.add(obj.label)
        return sorted(list(objects))


class ContextAggregator:
    """Aggregates context data from multiple sources."""

    def __init__(self, history_duration: timedelta = TRANSCRIPT_HISTORY_DURATION):
        self._lock = threading.Lock()
        self._period_start: datetime = datetime.now()
        self._history_duration = history_duration

        # Buffers for current period
        self._transcripts: list[TranscriptSegment] = []
        self._motion_events: list[MotionEvent] = []
        self._detections: list[DetectionResult] = []

        # Long-term transcript history (last hour)
        self._transcript_history: deque[TranscriptSegment] = deque()

    def _prune_transcript_history(self) -> None:
        """Remove transcripts older than history duration."""
        cutoff = datetime.now() - self._history_duration
        while self._transcript_history and self._transcript_history[0].timestamp < cutoff:
            self._transcript_history.popleft()

    def add_transcript(self, transcript: TranscriptSegment) -> None:
        """Add a transcript segment."""
        with self._lock:
            self._transcripts.append(transcript)
            # Also add to history
            self._transcript_history.append(transcript)
            self._prune_transcript_history()
            logger.debug(f"Added transcript: {transcript.text[:30]}...")

    def add_motion_event(self, event: MotionEvent) -> None:
        """Add a motion event."""
        with self._lock:
            self._motion_events.append(event)
            logger.debug(f"Added motion event at {event.timestamp}")

    def add_detection(self, detection: DetectionResult) -> None:
        """Add a detection result."""
        with self._lock:
            self._detections.append(detection)
            logger.debug(f"Added detection with {len(detection.objects)} objects")

    def collect(self, clear: bool = True) -> ContextData:
        """
        Collect all aggregated data.

        Args:
            clear: Whether to clear buffers after collection

        Returns:
            ContextData containing all aggregated information
        """
        with self._lock:
            now = datetime.now()

            context = ContextData(
                period_start=self._period_start,
                period_end=now,
                transcript_segments=list(self._transcripts),
                motion_events=list(self._motion_events),
                detection_results=list(self._detections),
            )

            if clear:
                self._transcripts.clear()
                self._motion_events.clear()
                self._detections.clear()
                self._period_start = now

            return context

    def get_transcript_history(self) -> list[TranscriptSegment]:
        """
        Get the last hour of transcripts.

        Returns:
            List of transcript segments from the last hour
        """
        with self._lock:
            self._prune_transcript_history()
            return list(self._transcript_history)

    def get_transcript_history_text(self) -> str:
        """
        Get the last hour of transcripts as concatenated text.

        Returns:
            Full transcript text from the last hour
        """
        history = self.get_transcript_history()
        return " ".join(seg.text for seg in history)

    def clear(self) -> None:
        """Clear all buffers and reset period start."""
        with self._lock:
            self._transcripts.clear()
            self._motion_events.clear()
            self._detections.clear()
            self._period_start = datetime.now()

    def clear_history(self) -> None:
        """Clear transcript history."""
        with self._lock:
            self._transcript_history.clear()

    @property
    def transcript_count(self) -> int:
        """Get current transcript count."""
        with self._lock:
            return len(self._transcripts)

    @property
    def history_count(self) -> int:
        """Get transcript history count."""
        with self._lock:
            return len(self._transcript_history)

    @property
    def motion_count(self) -> int:
        """Get current motion event count."""
        with self._lock:
            return len(self._motion_events)

    @property
    def detection_count(self) -> int:
        """Get current detection count."""
        with self._lock:
            return len(self._detections)

    @property
    def period_start(self) -> datetime:
        """Get current period start time."""
        return self._period_start
