"""Tests for the context aggregator module."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.audio.transcriber import TranscriptSegment
from src.context.aggregator import ContextAggregator, ContextData
from src.vision.detector import DetectedObject, DetectionResult
from src.vision.motion import MotionEvent


class TestContextData:
    """Tests for ContextData dataclass."""

    def test_context_data_creation(self):
        """Test creating context data."""
        now = datetime.now()
        context = ContextData(
            period_start=now - timedelta(minutes=5),
            period_end=now,
        )

        assert context.transcript_segments == []
        assert context.motion_events == []
        assert context.detection_results == []

    def test_full_transcript_empty(self):
        """Test full_transcript with no segments."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
        )

        assert context.full_transcript == ""

    def test_full_transcript_with_segments(self, sample_transcript_segments):
        """Test full_transcript with segments."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
            transcript_segments=sample_transcript_segments,
        )

        transcript = context.full_transcript
        assert "First segment" in transcript
        assert "Second segment" in transcript
        assert "Third segment" in transcript

    def test_motion_event_count(self, sample_motion_event):
        """Test motion_event_count property."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
            motion_events=[sample_motion_event, sample_motion_event],
        )

        assert context.motion_event_count == 2

    def test_images_captured(self, sample_detection_result):
        """Test images_captured property."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
            detection_results=[sample_detection_result],
        )

        assert context.images_captured == 1

    def test_get_object_summary(self, sample_detection_result):
        """Test get_object_summary method."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
            detection_results=[sample_detection_result],
        )

        summary = context.get_object_summary()

        assert "person" in summary
        assert summary["person"]["count"] == 1
        assert "laptop" in summary

    def test_get_object_summary_empty(self):
        """Test get_object_summary with no detections."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
        )

        summary = context.get_object_summary()
        assert summary == {}

    def test_get_unique_objects(self, sample_detection_result):
        """Test get_unique_objects method."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
            detection_results=[sample_detection_result],
        )

        objects = context.get_unique_objects()

        assert "laptop" in objects
        assert "person" in objects
        assert objects == sorted(objects)  # Should be sorted


class TestContextAggregator:
    """Tests for ContextAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create ContextAggregator instance."""
        return ContextAggregator()

    def test_init(self, aggregator):
        """Test ContextAggregator initialization."""
        assert aggregator.transcript_count == 0
        assert aggregator.motion_count == 0
        assert aggregator.detection_count == 0

    def test_add_transcript(self, aggregator, sample_transcript_segment):
        """Test adding transcript."""
        aggregator.add_transcript(sample_transcript_segment)

        assert aggregator.transcript_count == 1

    def test_add_motion_event(self, aggregator, sample_motion_event):
        """Test adding motion event."""
        aggregator.add_motion_event(sample_motion_event)

        assert aggregator.motion_count == 1

    def test_add_detection(self, aggregator, sample_detection_result):
        """Test adding detection result."""
        aggregator.add_detection(sample_detection_result)

        assert aggregator.detection_count == 1

    def test_collect(self, aggregator, sample_transcript_segment, sample_motion_event, sample_detection_result):
        """Test collecting context data."""
        aggregator.add_transcript(sample_transcript_segment)
        aggregator.add_motion_event(sample_motion_event)
        aggregator.add_detection(sample_detection_result)

        context = aggregator.collect()

        assert len(context.transcript_segments) == 1
        assert len(context.motion_events) == 1
        assert len(context.detection_results) == 1

    def test_collect_with_clear(self, aggregator, sample_transcript_segment):
        """Test collect clears buffers."""
        aggregator.add_transcript(sample_transcript_segment)

        context = aggregator.collect(clear=True)

        assert len(context.transcript_segments) == 1
        assert aggregator.transcript_count == 0

    def test_collect_without_clear(self, aggregator, sample_transcript_segment):
        """Test collect without clearing."""
        aggregator.add_transcript(sample_transcript_segment)

        context = aggregator.collect(clear=False)

        assert len(context.transcript_segments) == 1
        assert aggregator.transcript_count == 1

    def test_clear(self, aggregator, sample_transcript_segment):
        """Test clearing aggregator."""
        aggregator.add_transcript(sample_transcript_segment)
        aggregator.clear()

        assert aggregator.transcript_count == 0
        assert aggregator.motion_count == 0
        assert aggregator.detection_count == 0

    def test_transcript_history(self, aggregator):
        """Test transcript history is maintained."""
        # Add transcripts
        for i in range(5):
            segment = TranscriptSegment(
                text=f"Segment {i}",
                timestamp=datetime.now(),
                end_timestamp=datetime.now(),
                confidence=0.9,
                duration_ms=1000,
            )
            aggregator.add_transcript(segment)

        history = aggregator.get_transcript_history()
        assert len(history) == 5

    def test_transcript_history_pruning(self, aggregator):
        """Test old transcripts are pruned from history."""
        # Set short history duration for testing
        aggregator._history_duration = timedelta(seconds=1)

        # Add old transcript
        old_segment = TranscriptSegment(
            text="Old segment",
            timestamp=datetime.now() - timedelta(seconds=5),
            end_timestamp=datetime.now() - timedelta(seconds=4),
            confidence=0.9,
            duration_ms=1000,
        )
        aggregator._transcript_history.append(old_segment)

        # Add new transcript (triggers pruning)
        new_segment = TranscriptSegment(
            text="New segment",
            timestamp=datetime.now(),
            end_timestamp=datetime.now(),
            confidence=0.9,
            duration_ms=1000,
        )
        aggregator.add_transcript(new_segment)

        history = aggregator.get_transcript_history()
        assert len(history) == 1
        assert history[0].text == "New segment"

    def test_get_transcript_history_text(self, aggregator):
        """Test getting transcript history as text."""
        for i in range(3):
            segment = TranscriptSegment(
                text=f"Word{i}",
                timestamp=datetime.now(),
                end_timestamp=datetime.now(),
                confidence=0.9,
                duration_ms=1000,
            )
            aggregator.add_transcript(segment)

        text = aggregator.get_transcript_history_text()
        assert "Word0" in text
        assert "Word1" in text
        assert "Word2" in text

    def test_clear_history(self, aggregator, sample_transcript_segment):
        """Test clearing transcript history."""
        aggregator.add_transcript(sample_transcript_segment)

        aggregator.clear_history()

        assert aggregator.history_count == 0

    def test_history_count_property(self, aggregator, sample_transcript_segment):
        """Test history_count property."""
        assert aggregator.history_count == 0

        aggregator.add_transcript(sample_transcript_segment)

        assert aggregator.history_count == 1

    def test_period_start_property(self, aggregator):
        """Test period_start property."""
        start = aggregator.period_start
        assert isinstance(start, datetime)

    def test_period_start_updated_on_collect(self, aggregator, sample_transcript_segment):
        """Test period_start is updated on collect."""
        initial_start = aggregator.period_start

        aggregator.add_transcript(sample_transcript_segment)
        import time
        time.sleep(0.01)  # Small delay

        aggregator.collect(clear=True)

        new_start = aggregator.period_start
        assert new_start > initial_start

    def test_thread_safety(self, aggregator):
        """Test thread safety of aggregator."""
        import threading

        def add_transcripts():
            for i in range(100):
                segment = TranscriptSegment(
                    text=f"Thread segment {i}",
                    timestamp=datetime.now(),
                    end_timestamp=datetime.now(),
                    confidence=0.9,
                    duration_ms=1000,
                )
                aggregator.add_transcript(segment)

        threads = [threading.Thread(target=add_transcripts) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have all transcripts
        assert aggregator.transcript_count == 500
