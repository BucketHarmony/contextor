"""Tests for the context generator module."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.audio.transcriber import TranscriptSegment
from src.context.aggregator import ContextData
from src.context.generator import ContextGenerator
from src.vision.detector import DetectedObject, DetectionResult


class TestContextGenerator:
    """Tests for ContextGenerator class."""

    @pytest.fixture
    def output_dir(self, temp_dir):
        """Create output directory."""
        out = temp_dir / "context"
        out.mkdir()
        return out

    @pytest.fixture
    def generator(self, output_dir):
        """Create ContextGenerator instance."""
        return ContextGenerator(str(output_dir))

    @pytest.fixture
    def context_data(self, sample_transcript_segments, sample_detection_result):
        """Create test context data."""
        return ContextData(
            period_start=datetime.now() - timedelta(minutes=5),
            period_end=datetime.now(),
            transcript_segments=sample_transcript_segments,
            detection_results=[sample_detection_result],
        )

    def test_init(self, generator, output_dir):
        """Test ContextGenerator initialization."""
        assert generator.output_dir == output_dir
        assert output_dir.exists()

    def test_init_creates_directory(self, temp_dir):
        """Test initialization creates directory."""
        new_dir = temp_dir / "new_context"
        generator = ContextGenerator(str(new_dir))

        assert new_dir.exists()

    def test_generate(self, generator, context_data, output_dir):
        """Test generating context file."""
        filepath = generator.generate(context_data)

        assert Path(filepath).exists()
        assert filepath.endswith(".json")

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "period" in data
        assert "audio" in data
        assert "vision" in data
        assert "summary" in data

    def test_generate_with_filename(self, generator, context_data, output_dir):
        """Test generating with custom filename."""
        filepath = generator.generate(context_data, filename="custom.json")

        assert Path(filepath).name == "custom.json"

    def test_generate_creates_latest(self, generator, context_data, output_dir):
        """Test generate creates context.json."""
        generator.generate(context_data)

        latest = output_dir / "context.json"
        assert latest.exists()

    def test_generate_with_transcript_history(self, generator, context_data):
        """Test generate with transcript history."""
        history = [
            TranscriptSegment(
                text="History segment",
                timestamp=datetime.now() - timedelta(minutes=30),
                end_timestamp=datetime.now() - timedelta(minutes=29),
                confidence=0.9,
                duration_ms=1000,
            )
        ]

        filepath = generator.generate(context_data, transcript_history=history)

        with open(filepath, "r") as f:
            data = json.load(f)

        assert "last_hour" in data["audio"]
        assert data["audio"]["last_hour"]["segment_count"] == 1
        assert "History segment" in data["audio"]["last_hour"]["full_transcript"]

    def test_build_json_structure(self, generator, context_data):
        """Test JSON structure."""
        data = generator._build_json(context_data)

        assert "generated_at" in data
        assert "period" in data
        assert "start" in data["period"]
        assert "end" in data["period"]
        assert "duration_seconds" in data["period"]

    def test_build_audio_section(self, generator, context_data):
        """Test audio section building."""
        data = generator._build_audio_section(context_data, None)

        assert "current_period" in data
        assert "segment_count" in data["current_period"]
        assert "transcript_segments" in data["current_period"]
        assert "full_transcript" in data["current_period"]

    def test_build_audio_section_with_history(self, generator, context_data):
        """Test audio section with history."""
        history = [
            TranscriptSegment(
                text="History",
                timestamp=datetime.now(),
                end_timestamp=datetime.now(),
                confidence=0.9,
                duration_ms=1000,
            )
        ]

        data = generator._build_audio_section(context_data, history)

        assert "last_hour" in data
        assert data["last_hour"]["segment_count"] == 1

    def test_build_vision_section(self, generator, context_data):
        """Test vision section building."""
        data = generator._build_vision_section(context_data)

        assert "motion_events" in data
        assert "images_captured" in data
        assert "objects_detected" in data
        assert "unique_objects" in data
        assert "images" in data

    def test_generate_summary_with_transcripts(self, generator, context_data):
        """Test summary generation with transcripts."""
        summary = generator._generate_summary(context_data)

        assert "speech segments" in summary

    def test_generate_summary_with_objects(self, generator, context_data):
        """Test summary generation with objects."""
        summary = generator._generate_summary(context_data)

        assert "Objects detected" in summary

    def test_generate_summary_empty(self, generator):
        """Test summary generation with no activity."""
        context = ContextData(
            period_start=datetime.now(),
            period_end=datetime.now(),
        )

        summary = generator._generate_summary(context)

        assert summary == "No activity detected"

    def test_generate_summary_with_history(self, generator, context_data):
        """Test summary prefers history count."""
        history = [
            TranscriptSegment(
                text="H",
                timestamp=datetime.now(),
                end_timestamp=datetime.now(),
                confidence=0.9,
                duration_ms=100,
            )
            for _ in range(10)
        ]

        summary = generator._generate_summary(context_data, history)

        assert "10 speech segments in the last hour" in summary

    def test_calculate_history_duration_empty(self, generator):
        """Test history duration with no history."""
        duration = generator._calculate_history_duration(None)
        assert duration == "0 minutes"

        duration = generator._calculate_history_duration([])
        assert duration == "0 minutes"

    def test_calculate_history_duration_minutes(self, generator):
        """Test history duration in minutes."""
        now = datetime.now()
        history = [
            TranscriptSegment(
                text="A",
                timestamp=now - timedelta(minutes=30),
                end_timestamp=now - timedelta(minutes=29),
                confidence=0.9,
                duration_ms=1000,
            ),
            TranscriptSegment(
                text="B",
                timestamp=now - timedelta(minutes=1),
                end_timestamp=now,
                confidence=0.9,
                duration_ms=1000,
            ),
        ]

        duration = generator._calculate_history_duration(history)
        assert "minutes" in duration

    def test_calculate_history_duration_hours(self, generator):
        """Test history duration in hours."""
        now = datetime.now()
        history = [
            TranscriptSegment(
                text="A",
                timestamp=now - timedelta(hours=2),
                end_timestamp=now - timedelta(hours=1, minutes=59),
                confidence=0.9,
                duration_ms=1000,
            ),
            TranscriptSegment(
                text="B",
                timestamp=now - timedelta(minutes=1),
                end_timestamp=now,
                confidence=0.9,
                duration_ms=1000,
            ),
        ]

        duration = generator._calculate_history_duration(history)
        assert "hour" in duration

    def test_get_latest_context(self, generator, context_data, output_dir):
        """Test getting latest context."""
        generator.generate(context_data)

        latest = generator.get_latest_context()

        assert latest is not None
        assert "generated_at" in latest

    def test_get_latest_context_not_exists(self, generator):
        """Test getting latest when not exists."""
        latest = generator.get_latest_context()
        assert latest is None

    def test_legacy_fields_present(self, generator, context_data):
        """Test legacy fields are present for backward compatibility."""
        filepath = generator.generate(context_data)

        with open(filepath, "r") as f:
            data = json.load(f)

        audio = data["audio"]
        assert "segment_count" in audio
        assert "transcript_segments" in audio
        assert "full_transcript" in audio
