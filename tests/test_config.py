"""Tests for the config module."""

import logging
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.config import (
    AudioConfig,
    Config,
    ContextConfig,
    LoggingConfig,
    VisionConfig,
    load_config,
)


class TestAudioConfig:
    """Tests for AudioConfig dataclass."""

    def test_default_values(self):
        """Test default AudioConfig values."""
        config = AudioConfig()
        assert config.device == "default"
        assert config.sample_rate == 16000
        assert config.channels == 1
        assert config.chunk_duration_ms == 512
        assert config.vad_threshold == 0.5
        assert config.whisper_model == "small.en"
        assert config.whisper_device == "cuda"

    def test_custom_values(self):
        """Test AudioConfig with custom values."""
        config = AudioConfig(
            device="hw:1,0",
            sample_rate=44100,
            channels=2,
            whisper_model="tiny",
        )
        assert config.device == "hw:1,0"
        assert config.sample_rate == 44100
        assert config.channels == 2
        assert config.whisper_model == "tiny"


class TestVisionConfig:
    """Tests for VisionConfig dataclass."""

    def test_default_values(self):
        """Test default VisionConfig values."""
        config = VisionConfig()
        assert config.camera_id == 0
        assert config.camera_width == 1280
        assert config.camera_height == 720
        assert config.camera_fps == 30
        assert config.motion_threshold == 25
        assert config.yolo_model == "yolov8n.pt"

    def test_custom_values(self):
        """Test VisionConfig with custom values."""
        config = VisionConfig(
            camera_id=1,
            camera_width=1920,
            camera_height=1080,
            motion_threshold=50,
        )
        assert config.camera_id == 1
        assert config.camera_width == 1920
        assert config.camera_height == 1080
        assert config.motion_threshold == 50


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_values(self):
        """Test default ContextConfig values."""
        config = ContextConfig()
        assert config.output_dir == "./data/context"
        assert config.images_dir == "./data/images"
        assert config.interval == 300
        assert config.keep_images is True
        assert config.max_storage_gb == 10.0

    def test_custom_values(self):
        """Test ContextConfig with custom values."""
        config = ContextConfig(
            output_dir="/custom/path",
            interval=600,
            max_storage_gb=5.0,
        )
        assert config.output_dir == "/custom/path"
        assert config.interval == 600
        assert config.max_storage_gb == 5.0


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default LoggingConfig values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.file == "./logs/contextor.log"

    def test_custom_values(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(level="DEBUG", file=None)
        assert config.level == "DEBUG"
        assert config.file is None


class TestConfig:
    """Tests for main Config class."""

    def test_default_config(self):
        """Test default Config values."""
        config = Config()
        assert isinstance(config.audio, AudioConfig)
        assert isinstance(config.vision, VisionConfig)
        assert isinstance(config.context, ContextConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_from_yaml_missing_file(self, temp_dir):
        """Test loading from non-existent file returns defaults."""
        config = Config.from_yaml(temp_dir / "nonexistent.yaml")
        assert config.audio.device == "default"
        assert config.vision.camera_id == 0

    def test_from_yaml_valid_file(self, temp_dir):
        """Test loading from valid YAML file."""
        yaml_content = {
            "audio": {"device": "test_device", "sample_rate": 44100},
            "vision": {"camera_id": 2, "motion_threshold": 30},
            "context": {"interval": 600},
            "logging": {"level": "DEBUG"},
        }
        yaml_path = temp_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = Config.from_yaml(yaml_path)
        assert config.audio.device == "test_device"
        assert config.audio.sample_rate == 44100
        assert config.vision.camera_id == 2
        assert config.vision.motion_threshold == 30
        assert config.context.interval == 600
        assert config.logging.level == "DEBUG"

    def test_from_yaml_empty_file(self, temp_dir):
        """Test loading from empty YAML file."""
        yaml_path = temp_dir / "empty.yaml"
        yaml_path.write_text("")

        config = Config.from_yaml(yaml_path)
        assert config.audio.device == "default"

    def test_from_yaml_partial_config(self, temp_dir):
        """Test loading from YAML with partial config."""
        yaml_content = {"audio": {"device": "partial_device"}}
        yaml_path = temp_dir / "partial.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = Config.from_yaml(yaml_path)
        assert config.audio.device == "partial_device"
        assert config.vision.camera_id == 0  # Default

    def test_to_yaml(self, temp_dir):
        """Test saving config to YAML file."""
        config = Config(
            audio=AudioConfig(device="saved_device"),
            vision=VisionConfig(camera_id=5),
        )
        yaml_path = temp_dir / "output.yaml"
        config.to_yaml(yaml_path)

        assert yaml_path.exists()

        with open(yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded["audio"]["device"] == "saved_device"
        assert loaded["vision"]["camera_id"] == 5

    def test_to_yaml_creates_parent_dirs(self, temp_dir):
        """Test to_yaml creates parent directories."""
        config = Config()
        yaml_path = temp_dir / "nested" / "dir" / "config.yaml"
        config.to_yaml(yaml_path)

        assert yaml_path.exists()

    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        # Reset logging to ensure clean state
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.setLevel(logging.WARNING)  # Reset to default

        config = Config(
            logging=LoggingConfig(level="DEBUG", file=None)
        )
        config.setup_logging()

        logger = logging.getLogger()
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, temp_dir):
        """Test logging setup with file."""
        log_path = temp_dir / "test.log"
        config = Config(
            logging=LoggingConfig(level="INFO", file=str(log_path))
        )
        config.setup_logging()

        assert log_path.parent.exists()

    def test_ensure_directories(self, temp_dir):
        """Test directory creation."""
        config = Config(
            context=ContextConfig(
                output_dir=str(temp_dir / "new_context"),
                images_dir=str(temp_dir / "new_images"),
            )
        )
        config.ensure_directories()

        assert (temp_dir / "new_context").exists()
        assert (temp_dir / "new_images").exists()
        assert (Path("models")).exists()


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_default_path(self, temp_dir, monkeypatch):
        """Test load_config with default path."""
        # Create config at default location
        config_dir = temp_dir / "config"
        config_dir.mkdir()
        config_path = config_dir / "settings.yaml"
        config_path.write_text("audio:\n  device: 'env_device'\n")

        monkeypatch.chdir(temp_dir)
        config = load_config(str(config_path))

        assert config.audio.device == "env_device"

    def test_load_config_from_env(self, temp_dir, monkeypatch):
        """Test load_config from environment variable."""
        config_path = temp_dir / "env_config.yaml"
        config_path.write_text("audio:\n  device: 'from_env'\n")

        monkeypatch.setenv("CONTEXTOR_CONFIG", str(config_path))
        config = load_config()

        assert config.audio.device == "from_env"

    def test_load_config_explicit_path(self, temp_dir):
        """Test load_config with explicit path."""
        config_path = temp_dir / "explicit.yaml"
        config_path.write_text("vision:\n  camera_id: 99\n")

        config = load_config(str(config_path))

        assert config.vision.camera_id == 99
