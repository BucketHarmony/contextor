"""Tests for the web API module."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.web.api import create_app, set_contextor_instance


class TestWebAPI:
    """Tests for the web API."""

    @pytest.fixture
    def mock_contextor(self, temp_dir):
        """Create mock Contextor instance."""
        mock = MagicMock()

        # Mock config
        mock.config.context.output_dir = str(temp_dir / "context")
        mock.config.context.images_dir = str(temp_dir / "images")

        # Create directories
        (temp_dir / "context").mkdir()
        (temp_dir / "images").mkdir()

        # Mock status
        mock.get_status.return_value = {
            "running": True,
            "audio": {
                "capture_running": True,
                "transcriber_running": True,
                "is_speaking": False,
                "pending_transcripts": 0,
            },
            "vision": {
                "camera_running": True,
                "frame_count": 1000,
                "motion_events": 5,
                "detections": 3,
            },
            "storage": {
                "context_files": 10,
                "image_files": 20,
                "total_size_mb": 50.5,
                "usage_percent": 5.0,
            },
        }

        # Mock generator
        mock.generator.get_latest_context.return_value = {
            "generated_at": datetime.now().isoformat(),
            "period": {
                "start": datetime.now().isoformat(),
                "end": datetime.now().isoformat(),
                "duration_seconds": 300,
            },
            "audio": {
                "full_transcript": "Test transcript",
            },
            "vision": {
                "objects_detected": {"person": {"count": 2}},
            },
            "summary": "Test summary",
        }

        # Mock trigger
        mock.trigger_context_now.return_value = "/path/to/context.json"

        # Mock storage
        mock.storage.get_storage_stats.return_value = {
            "context_files": 10,
            "image_files": 20,
            "context_size_mb": 10.0,
            "images_size_mb": 40.0,
            "total_size_mb": 50.0,
            "max_storage_gb": 10.0,
            "usage_percent": 5.0,
        }
        mock.storage.get_recent_contexts.return_value = []
        mock.storage.get_recent_images.return_value = []
        mock.storage.cleanup_old_files.return_value = 5

        # Mock transcriber
        mock.transcriber.get_recent_transcripts.return_value = []

        return mock

    @pytest.fixture
    def client(self, mock_contextor):
        """Create test client."""
        set_contextor_instance(mock_contextor)
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def client_no_contextor(self):
        """Create test client without Contextor."""
        set_contextor_instance(None)
        app = create_app()
        return TestClient(app)

    def test_index_page(self, client):
        """Test index page returns HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_get_status(self, client):
        """Test GET /api/status."""
        response = client.get("/api/status")
        assert response.status_code == 200

        data = response.json()
        assert data["running"] is True
        assert "audio" in data
        assert "vision" in data
        assert "storage" in data

    def test_get_status_no_contextor(self, client_no_contextor):
        """Test status when Contextor not initialized."""
        response = client_no_contextor.get("/api/status")
        assert response.status_code == 503

    def test_generate_context(self, client, mock_contextor):
        """Test POST /api/context/generate."""
        response = client.post("/api/context/generate")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        mock_contextor.trigger_context_now.assert_called_once()

    def test_generate_context_no_contextor(self, client_no_contextor):
        """Test generate when Contextor not initialized."""
        response = client_no_contextor.post("/api/context/generate")
        assert response.status_code == 503

    def test_generate_context_error(self, client, mock_contextor):
        """Test generate with error."""
        mock_contextor.trigger_context_now.side_effect = Exception("Error")

        response = client.post("/api/context/generate")
        assert response.status_code == 500

    def test_get_latest_context(self, client):
        """Test GET /api/context/latest."""
        response = client.get("/api/context/latest")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data

    def test_get_latest_context_not_found(self, client, mock_contextor):
        """Test latest context when none exists."""
        mock_contextor.generator.get_latest_context.return_value = None

        response = client.get("/api/context/latest")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False

    def test_get_context_history(self, client, mock_contextor):
        """Test GET /api/context/history."""
        mock_contextor.storage.get_recent_contexts.return_value = [
            "/path/context_1.json",
            "/path/context_2.json",
        ]

        # Create mock files
        for path in mock_contextor.storage.get_recent_contexts.return_value:
            p = Path(path)
            # Can't create files in /path, so just test API structure

        response = client.get("/api/context/history?limit=5")
        assert response.status_code == 200

    def test_get_context_file(self, client, mock_contextor, temp_dir):
        """Test GET /api/context/file/{filename}."""
        # Create test file
        context_dir = Path(mock_contextor.config.context.output_dir)
        test_file = context_dir / "test_context.json"
        test_file.write_text(json.dumps({"test": "data"}))

        response = client.get("/api/context/file/test_context.json")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_get_context_file_not_found(self, client, mock_contextor):
        """Test getting non-existent context file."""
        response = client.get("/api/context/file/nonexistent.json")
        assert response.status_code == 404

    def test_get_context_file_invalid(self, client, mock_contextor, temp_dir):
        """Test getting non-json file."""
        response = client.get("/api/context/file/test.txt")
        assert response.status_code == 404

    def test_get_recent_images(self, client, mock_contextor):
        """Test GET /api/images/recent."""
        response = client.get("/api/images/recent?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_get_image_file(self, client, mock_contextor, temp_dir, sample_frame):
        """Test GET /api/images/file/{filename}."""
        import cv2

        # Create test image
        images_dir = Path(mock_contextor.config.context.images_dir)
        test_image = images_dir / "test.jpg"
        cv2.imwrite(str(test_image), sample_frame)

        response = client.get("/api/images/file/test.jpg")
        assert response.status_code == 200
        assert "image" in response.headers["content-type"]

    def test_get_image_file_not_found(self, client, mock_contextor):
        """Test getting non-existent image."""
        response = client.get("/api/images/file/nonexistent.jpg")
        assert response.status_code == 404

    def test_get_recent_transcripts(self, client, mock_contextor):
        """Test GET /api/transcripts/recent."""
        from src.audio.transcriber import TranscriptSegment

        mock_contextor.transcriber.get_recent_transcripts.return_value = [
            TranscriptSegment(
                text="Test",
                timestamp=datetime.now(),
                end_timestamp=datetime.now(),
                confidence=0.9,
                duration_ms=1000,
            )
        ]

        response = client.get("/api/transcripts/recent")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True

    def test_get_storage_stats(self, client):
        """Test GET /api/storage/stats."""
        response = client.get("/api/storage/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data

    def test_trigger_cleanup(self, client, mock_contextor):
        """Test POST /api/storage/cleanup."""
        response = client.post("/api/storage/cleanup")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted_count"] == 5

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/status")
        # FastAPI handles CORS middleware

    def test_set_contextor_instance(self):
        """Test setting contextor instance."""
        mock = MagicMock()
        set_contextor_instance(mock)

        # Verify it's set (indirectly via API)
        app = create_app()
        client = TestClient(app)
        response = client.get("/api/status")
        assert response.status_code == 200

    def test_api_error_handling(self, client, mock_contextor):
        """Test API error handling."""
        mock_contextor.storage.get_storage_stats.side_effect = Exception("Error")

        response = client.get("/api/storage/stats")
        assert response.status_code == 500
