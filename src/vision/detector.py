"""Object detection using YOLOv8."""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from ..config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """A detected object in an image."""
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    center: tuple[int, int]


@dataclass
class DetectionResult:
    """Result of object detection on an image."""
    timestamp: datetime
    image_path: Optional[str]
    objects: list[DetectedObject] = field(default_factory=list)
    inference_time_ms: float = 0.0
    trigger: str = "unknown"  # "motion", "scheduled", "manual"


class ObjectDetector:
    """Object detection using YOLOv8."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.model_path = config.yolo_model
        self.confidence_threshold = config.yolo_confidence
        self.device = config.yolo_device

        self._model: Optional[YOLO] = None
        self._results: list[DetectionResult] = []
        self._results_lock = threading.Lock()

        # Object statistics
        self._object_counts: dict[str, int] = {}
        self._object_confidences: dict[str, list[float]] = {}

    def load_model(self) -> None:
        """Load the YOLO model."""
        logger.info(f"Loading YOLO model: {self.model_path}")
        try:
            self._model = YOLO(self.model_path)
            # Warm up
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            logger.info("YOLO model loaded")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        trigger: str = "unknown",
        image_path: Optional[str] = None,
    ) -> DetectionResult:
        """
        Run object detection on an image.

        Args:
            image: BGR image as numpy array
            trigger: What triggered this detection
            image_path: Optional path where image is saved

        Returns:
            DetectionResult with detected objects
        """
        if self._model is None:
            self.load_model()

        timestamp = datetime.now()
        detected_objects = []

        try:
            # Run inference
            results = self._model.predict(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
            )

            inference_time = results[0].speed.get("inference", 0)

            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for i in range(len(boxes)):
                    box = boxes[i]
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    label = result.names[cls_id]

                    # Get bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    detected_objects.append(DetectedObject(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                    ))

                    # Update statistics
                    self._object_counts[label] = self._object_counts.get(label, 0) + 1
                    if label not in self._object_confidences:
                        self._object_confidences[label] = []
                    self._object_confidences[label].append(confidence)

            detection_result = DetectionResult(
                timestamp=timestamp,
                image_path=image_path,
                objects=detected_objects,
                inference_time_ms=inference_time,
                trigger=trigger,
            )

            # Store result
            with self._results_lock:
                self._results.append(detection_result)

            logger.debug(
                f"Detected {len(detected_objects)} objects in {inference_time:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Detection error: {e}")
            detection_result = DetectionResult(
                timestamp=timestamp,
                image_path=image_path,
                trigger=trigger,
            )

        return detection_result

    def get_recent_results(self, clear: bool = False) -> list[DetectionResult]:
        """Get recent detection results, optionally clearing the buffer."""
        with self._results_lock:
            results = list(self._results)
            if clear:
                self._results.clear()
        return results

    def clear_results(self) -> None:
        """Clear the results buffer."""
        with self._results_lock:
            self._results.clear()

    def get_object_summary(self, clear: bool = False) -> dict:
        """
        Get summary of detected objects.

        Returns:
            Dictionary with object counts and average confidences
        """
        summary = {}
        for label, count in self._object_counts.items():
            confidences = self._object_confidences.get(label, [])
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            summary[label] = {
                "count": count,
                "avg_confidence": round(avg_confidence, 3),
            }

        if clear:
            self._object_counts.clear()
            self._object_confidences.clear()

        return summary

    def get_unique_objects(self) -> list[str]:
        """Get list of unique object labels detected."""
        return list(self._object_counts.keys())

    def reset_statistics(self) -> None:
        """Reset object statistics."""
        self._object_counts.clear()
        self._object_confidences.clear()

    def save_annotated_image(
        self,
        image: np.ndarray,
        detections: list[DetectedObject],
        output_path: str,
    ) -> None:
        """Save image with detection annotations."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label_text = f"{det.label} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.imwrite(output_path, annotated)
        logger.debug(f"Saved annotated image: {output_path}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None
