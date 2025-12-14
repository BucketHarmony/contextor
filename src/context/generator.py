"""Context JSON file generator."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ..audio.transcriber import TranscriptSegment
from .aggregator import ContextData

logger = logging.getLogger(__name__)


class ContextGenerator:
    """Generates context JSON files from aggregated data."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        context: ContextData,
        filename: Optional[str] = None,
        transcript_history: Optional[list[TranscriptSegment]] = None,
    ) -> str:
        """
        Generate a context JSON file.

        Args:
            context: Aggregated context data for current period
            filename: Optional filename (auto-generated if not provided)
            transcript_history: Optional list of transcripts from the last hour

        Returns:
            Path to the generated file
        """
        if filename is None:
            timestamp = context.period_end.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"context_{timestamp}.json"

        output_path = self.output_dir / filename

        # Build JSON structure
        data = self._build_json(context, transcript_history)

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Generated context file: {output_path}")

        # Also write to latest.json for easy access
        latest_path = self.output_dir / "context.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        return str(output_path)

    def _build_json(
        self,
        context: ContextData,
        transcript_history: Optional[list[TranscriptSegment]] = None,
    ) -> dict[str, Any]:
        """Build the JSON structure from context data."""
        data = {
            "generated_at": context.period_end.isoformat(),
            "period": {
                "start": context.period_start.isoformat(),
                "end": context.period_end.isoformat(),
                "duration_seconds": (
                    context.period_end - context.period_start
                ).total_seconds(),
            },
            "audio": self._build_audio_section(context, transcript_history),
            "vision": self._build_vision_section(context),
            "summary": self._generate_summary(context, transcript_history),
        }

        return data

    def _build_audio_section(
        self,
        context: ContextData,
        transcript_history: Optional[list[TranscriptSegment]] = None,
    ) -> dict[str, Any]:
        """Build the audio section of the JSON."""
        # Current period segments
        current_segments = []
        for seg in context.transcript_segments:
            current_segments.append({
                "timestamp": seg.timestamp.isoformat(),
                "end_timestamp": seg.end_timestamp.isoformat(),
                "text": seg.text,
                "confidence": round(seg.confidence, 3),
                "duration_ms": seg.duration_ms,
            })

        # Last hour history segments
        history_segments = []
        if transcript_history:
            for seg in transcript_history:
                history_segments.append({
                    "timestamp": seg.timestamp.isoformat(),
                    "end_timestamp": seg.end_timestamp.isoformat(),
                    "text": seg.text,
                    "confidence": round(seg.confidence, 3),
                    "duration_ms": seg.duration_ms,
                })

        # Full transcript from history
        history_full_transcript = ""
        if transcript_history:
            history_full_transcript = " ".join(seg.text for seg in transcript_history)

        return {
            "current_period": {
                "segment_count": len(current_segments),
                "transcript_segments": current_segments,
                "full_transcript": context.full_transcript,
            },
            "last_hour": {
                "segment_count": len(history_segments),
                "transcript_segments": history_segments,
                "full_transcript": history_full_transcript,
                "duration_covered": self._calculate_history_duration(transcript_history),
            },
            # Legacy fields for backward compatibility
            "segment_count": len(current_segments),
            "transcript_segments": current_segments,
            "full_transcript": context.full_transcript,
        }

    def _calculate_history_duration(
        self,
        transcript_history: Optional[list[TranscriptSegment]],
    ) -> str:
        """Calculate the time span covered by transcript history."""
        if not transcript_history:
            return "0 minutes"

        earliest = min(seg.timestamp for seg in transcript_history)
        latest = max(seg.end_timestamp for seg in transcript_history)
        duration = latest - earliest

        minutes = int(duration.total_seconds() / 60)
        if minutes < 60:
            return f"{minutes} minutes"
        else:
            hours = minutes // 60
            mins = minutes % 60
            if mins > 0:
                return f"{hours} hours {mins} minutes"
            return f"{hours} hours"

    def _build_vision_section(self, context: ContextData) -> dict[str, Any]:
        """Build the vision section of the JSON."""
        # Build images list
        images = []
        for result in context.detection_results:
            objects = []
            for obj in result.objects:
                objects.append({
                    "label": obj.label,
                    "confidence": round(obj.confidence, 3),
                    "bbox": list(obj.bbox),
                    "center": list(obj.center),
                })

            images.append({
                "filename": result.image_path,
                "timestamp": result.timestamp.isoformat(),
                "trigger": result.trigger,
                "inference_time_ms": round(result.inference_time_ms, 1),
                "objects": objects,
            })

        return {
            "motion_events": context.motion_event_count,
            "images_captured": context.images_captured,
            "objects_detected": context.get_object_summary(),
            "unique_objects": context.get_unique_objects(),
            "images": images,
        }

    def _generate_summary(
        self,
        context: ContextData,
        transcript_history: Optional[list[TranscriptSegment]] = None,
    ) -> str:
        """Generate a human-readable summary."""
        parts = []

        # Audio summary - mention both current and history
        if transcript_history:
            parts.append(f"{len(transcript_history)} speech segments in the last hour")
        elif context.transcript_segments:
            parts.append(f"{len(context.transcript_segments)} speech segments recorded")

        # Vision summary
        unique_objects = context.get_unique_objects()
        if unique_objects:
            # Get top objects by count
            summary = context.get_object_summary()
            top_objects = sorted(
                summary.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )[:5]
            object_strs = [f"{obj[0]} ({obj[1]['count']}x)" for obj in top_objects]
            parts.append(f"Objects detected: {', '.join(object_strs)}")

        if context.motion_event_count:
            parts.append(f"Motion detected {context.motion_event_count} times")

        if context.images_captured:
            parts.append(f"{context.images_captured} images captured")

        return ". ".join(parts) if parts else "No activity detected"

    def get_latest_context(self) -> Optional[dict]:
        """Read the latest context file."""
        latest_path = self.output_dir / "context.json"
        if not latest_path.exists():
            return None

        with open(latest_path, "r", encoding="utf-8") as f:
            return json.load(f)
