"""Configuration management for Contextor."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    device: str = "default"
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 512
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 500
    whisper_model: str = "small.en"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"


@dataclass
class VisionConfig:
    """Vision pipeline configuration."""
    camera_id: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    motion_detection_fps: int = 5
    motion_threshold: int = 25
    motion_min_area: int = 500
    motion_blur_size: int = 21
    capture_interval: int = 300  # seconds
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.5
    yolo_device: str = "cuda"


@dataclass
class ContextConfig:
    """Context generation configuration."""
    output_dir: str = "./data/context"
    images_dir: str = "./data/images"
    interval: int = 300  # seconds
    keep_images: bool = True
    max_storage_gb: float = 10.0
    max_context_files: int = 288  # 24 hours of 5-min intervals


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: Optional[str] = "./logs/contextor.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration container."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            audio=AudioConfig(**data.get("audio", {})),
            vision=VisionConfig(**data.get("vision", {})),
            context=ContextConfig(**data.get("context", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "audio": {
                "device": self.audio.device,
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "chunk_duration_ms": self.audio.chunk_duration_ms,
                "vad_threshold": self.audio.vad_threshold,
                "vad_min_speech_ms": self.audio.vad_min_speech_ms,
                "vad_min_silence_ms": self.audio.vad_min_silence_ms,
                "whisper_model": self.audio.whisper_model,
                "whisper_device": self.audio.whisper_device,
                "whisper_compute_type": self.audio.whisper_compute_type,
            },
            "vision": {
                "camera_id": self.vision.camera_id,
                "camera_width": self.vision.camera_width,
                "camera_height": self.vision.camera_height,
                "camera_fps": self.vision.camera_fps,
                "motion_detection_fps": self.vision.motion_detection_fps,
                "motion_threshold": self.vision.motion_threshold,
                "motion_min_area": self.vision.motion_min_area,
                "motion_blur_size": self.vision.motion_blur_size,
                "capture_interval": self.vision.capture_interval,
                "yolo_model": self.vision.yolo_model,
                "yolo_confidence": self.vision.yolo_confidence,
                "yolo_device": self.vision.yolo_device,
            },
            "context": {
                "output_dir": self.context.output_dir,
                "images_dir": self.context.images_dir,
                "interval": self.context.interval,
                "keep_images": self.context.keep_images,
                "max_storage_gb": self.context.max_storage_gb,
                "max_context_files": self.context.max_context_files,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "format": self.logging.format,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def setup_logging(self) -> None:
        """Configure logging based on settings."""
        log_level = getattr(logging, self.logging.level.upper(), logging.INFO)

        handlers = [logging.StreamHandler()]

        if self.logging.file:
            log_path = Path(self.logging.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(log_path))

        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            handlers=handlers,
        )

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        Path(self.context.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.context.images_dir).mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from file or environment."""
    if path is None:
        path = os.environ.get("CONTEXTOR_CONFIG", "config/settings.yaml")
    return Config.from_yaml(path)
