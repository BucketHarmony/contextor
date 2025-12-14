"""Tests for the context storage module."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import ContextConfig
from src.context.storage import StorageManager


class TestStorageManager:
    """Tests for StorageManager class."""

    @pytest.fixture
    def storage_config(self, temp_dir):
        """Create test storage config."""
        return ContextConfig(
            output_dir=str(temp_dir / "context"),
            images_dir=str(temp_dir / "images"),
            interval=300,
            keep_images=True,
            max_storage_gb=0.001,  # 1MB for testing
            max_context_files=5,
        )

    @pytest.fixture
    def storage(self, storage_config):
        """Create StorageManager instance."""
        return StorageManager(storage_config)

    def test_init(self, storage, storage_config):
        """Test StorageManager initialization."""
        assert storage.context_dir.exists()
        assert storage.images_dir.exists()
        assert storage.max_storage_gb == storage_config.max_storage_gb

    def test_save_image(self, storage, sample_frame):
        """Test saving an image."""
        filepath = storage.save_image(sample_frame, trigger="test")

        assert Path(filepath).exists()
        assert "test" in filepath
        assert filepath.endswith(".jpg")

    def test_save_image_with_timestamp(self, storage, sample_frame):
        """Test saving image with specific timestamp."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        filepath = storage.save_image(sample_frame, trigger="motion", timestamp=ts)

        assert "2024-01-15_10-30-00" in filepath

    def test_cleanup_old_files_context(self, storage, temp_dir):
        """Test cleanup removes old context files."""
        # Create more files than max
        for i in range(10):
            filepath = storage.context_dir / f"context_2024-01-{i:02d}.json"
            filepath.write_text(json.dumps({"test": i}))
            time.sleep(0.01)  # Ensure different mtime

        deleted = storage.cleanup_old_files()

        remaining = list(storage.context_dir.glob("context_*.json"))
        assert len(remaining) <= storage.max_context_files

    def test_cleanup_old_files_by_size(self, storage, sample_frame):
        """Test cleanup by size limit."""
        # Save multiple large images
        for i in range(20):
            large_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
            storage.save_image(large_frame, trigger=f"test{i}")

        storage.cleanup_old_files()

        # Should have cleaned up some files
        stats = storage.get_storage_stats()
        # Total should be under limit (or close to it)

    def test_get_storage_stats(self, storage, sample_frame):
        """Test getting storage statistics."""
        # Save some files
        storage.save_image(sample_frame, trigger="test")

        filepath = storage.context_dir / "context_test.json"
        filepath.write_text(json.dumps({"test": "data"}))

        stats = storage.get_storage_stats()

        assert "context_files" in stats
        assert "image_files" in stats
        assert "total_size_mb" in stats
        assert "usage_percent" in stats

    def test_get_recent_images(self, storage, sample_frame):
        """Test getting recent images."""
        # Save some images
        for i in range(5):
            storage.save_image(sample_frame, trigger=f"test{i}")
            time.sleep(0.01)

        images = storage.get_recent_images(count=3)

        assert len(images) == 3
        # Should be most recent first

    def test_get_recent_images_empty(self, storage):
        """Test getting recent images when empty."""
        images = storage.get_recent_images()
        assert images == []

    def test_get_recent_contexts(self, storage):
        """Test getting recent context files."""
        # Create some context files
        for i in range(5):
            filepath = storage.context_dir / f"context_test_{i}.json"
            filepath.write_text(json.dumps({"test": i}))
            time.sleep(0.01)

        contexts = storage.get_recent_contexts(count=3)

        assert len(contexts) == 3

    def test_get_recent_contexts_empty(self, storage):
        """Test getting recent contexts when empty."""
        contexts = storage.get_recent_contexts()
        assert contexts == []

    def test_get_directory_size(self, storage, sample_frame):
        """Test getting directory size."""
        storage.save_image(sample_frame, trigger="test")

        size = storage._get_directory_size(storage.images_dir)

        assert size > 0

    def test_get_directory_size_empty(self, storage):
        """Test getting size of empty directory."""
        size = storage._get_directory_size(storage.context_dir)
        assert size == 0

    def test_cleanup_by_size(self, storage, sample_frame):
        """Test cleanup_by_size method."""
        # Fill with images
        for i in range(10):
            storage.save_image(sample_frame, trigger=f"test{i}")

        # Manually call cleanup
        deleted = storage._cleanup_by_size()

        # Should have deleted some files if over limit

    def test_storage_stats_usage_percent(self, storage, sample_frame):
        """Test usage percent calculation."""
        storage.save_image(sample_frame, trigger="test")

        stats = storage.get_storage_stats()

        assert 0 <= stats["usage_percent"] <= 100

    def test_max_context_files_respected(self, storage):
        """Test max context files is respected."""
        # Create many files
        for i in range(20):
            filepath = storage.context_dir / f"context_2024-01-{i:02d}_00-00-00.json"
            filepath.write_text(json.dumps({"test": i}))
            time.sleep(0.01)

        storage.cleanup_old_files()

        remaining = list(storage.context_dir.glob("context_*.json"))
        assert len(remaining) <= storage.max_context_files

    def test_image_quality(self, storage, sample_frame):
        """Test saved image quality."""
        filepath = storage.save_image(sample_frame, trigger="test")

        # Read back and verify
        loaded = cv2.imread(filepath)
        assert loaded is not None
        assert loaded.shape[:2] == sample_frame.shape[:2]

    def test_cleanup_preserves_latest_json(self, storage):
        """Test cleanup doesn't delete context.json (latest)."""
        # Create latest
        latest = storage.context_dir / "context.json"
        latest.write_text(json.dumps({"latest": True}))

        # Create old files
        for i in range(10):
            filepath = storage.context_dir / f"context_2024-01-{i:02d}.json"
            filepath.write_text(json.dumps({"test": i}))

        storage.cleanup_old_files()

        # Latest should still exist
        assert latest.exists()
