"""Storage management for context files and images."""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import ContextConfig

logger = logging.getLogger(__name__)


class StorageManager:
    """Manages storage of context files and images."""

    def __init__(self, config: ContextConfig):
        self.config = config
        self.context_dir = Path(config.output_dir)
        self.images_dir = Path(config.images_dir)
        self.max_storage_gb = config.max_storage_gb
        self.max_context_files = config.max_context_files
        self.keep_images = config.keep_images

        # Create directories
        self.context_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        image: np.ndarray,
        trigger: str = "unknown",
        timestamp: Optional[datetime] = None,
    ) -> str:
        """
        Save an image to storage.

        Args:
            image: BGR image as numpy array
            trigger: What triggered the capture (motion, scheduled, manual)
            timestamp: Optional timestamp (uses now if not provided)

        Returns:
            Path to saved image
        """
        if timestamp is None:
            timestamp = datetime.now()

        filename = f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_{trigger}.jpg"
        filepath = self.images_dir / filename

        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.debug(f"Saved image: {filepath}")

        return str(filepath)

    def cleanup_old_files(self) -> int:
        """
        Clean up old files to stay within storage limits.

        Returns:
            Number of files deleted
        """
        deleted_count = 0

        # Clean up old context files
        context_files = sorted(
            self.context_dir.glob("context_*.json"),
            key=lambda p: p.stat().st_mtime,
        )

        while len(context_files) > self.max_context_files:
            oldest = context_files.pop(0)
            oldest.unlink()
            deleted_count += 1
            logger.debug(f"Deleted old context file: {oldest}")

        # Check storage usage
        total_size = self._get_directory_size(self.context_dir)
        total_size += self._get_directory_size(self.images_dir)
        total_size_gb = total_size / (1024 ** 3)

        if total_size_gb > self.max_storage_gb:
            deleted_count += self._cleanup_by_size()

        return deleted_count

    def _cleanup_by_size(self) -> int:
        """Clean up files to meet storage limit."""
        deleted_count = 0
        target_size = self.max_storage_gb * 0.8 * (1024 ** 3)  # 80% of max

        # Get all files sorted by age
        all_files = []

        for f in self.context_dir.glob("context_*.json"):
            all_files.append(f)

        for f in self.images_dir.glob("*.jpg"):
            all_files.append(f)

        all_files.sort(key=lambda p: p.stat().st_mtime)

        current_size = sum(f.stat().st_size for f in all_files)

        while current_size > target_size and all_files:
            oldest = all_files.pop(0)
            file_size = oldest.stat().st_size
            oldest.unlink()
            current_size -= file_size
            deleted_count += 1
            logger.debug(f"Deleted to free space: {oldest}")

        return deleted_count

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def get_storage_stats(self) -> dict:
        """Get storage statistics."""
        context_size = self._get_directory_size(self.context_dir)
        images_size = self._get_directory_size(self.images_dir)
        total_size = context_size + images_size

        context_files = list(self.context_dir.glob("context_*.json"))
        image_files = list(self.images_dir.glob("*.jpg"))

        return {
            "context_files": len(context_files),
            "image_files": len(image_files),
            "context_size_mb": round(context_size / (1024 ** 2), 2),
            "images_size_mb": round(images_size / (1024 ** 2), 2),
            "total_size_mb": round(total_size / (1024 ** 2), 2),
            "max_storage_gb": self.max_storage_gb,
            "usage_percent": round(total_size / (self.max_storage_gb * 1024 ** 3) * 100, 1),
        }

    def get_recent_images(self, count: int = 10) -> list[str]:
        """Get paths to most recent images."""
        images = sorted(
            self.images_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [str(img) for img in images[:count]]

    def get_recent_contexts(self, count: int = 10) -> list[str]:
        """Get paths to most recent context files."""
        contexts = sorted(
            self.context_dir.glob("context_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [str(ctx) for ctx in contexts[:count]]
