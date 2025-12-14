"""FastAPI REST API for Contextor."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Will be set by main.py
_contextor_instance = None


class ContextResponse(BaseModel):
    """Response model for context data."""
    success: bool
    message: str
    data: Optional[dict] = None


class StatusResponse(BaseModel):
    """Response model for system status."""
    running: bool
    uptime_seconds: float
    audio: dict
    vision: dict
    storage: dict


def set_contextor_instance(instance) -> None:
    """Set the Contextor instance for API access."""
    global _contextor_instance
    _contextor_instance = instance


def create_app(static_dir: Optional[Path] = None) -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Contextor API",
        description="Jetson Orin Nano Context Recorder API",
        version="0.1.0",
    )

    # CORS for local network access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files and templates
    if static_dir is None:
        static_dir = Path(__file__).parent / "static"

    templates_dir = Path(__file__).parent / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    # Store start time for uptime calculation
    app.state.start_time = datetime.now()

    # ==================== Frontend Routes ====================

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Serve the main dashboard."""
        return templates.TemplateResponse("index.html", {"request": request})

    # ==================== API Routes ====================

    @app.get("/api/status", response_model=StatusResponse)
    async def get_status():
        """Get current system status."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        status = _contextor_instance.get_status()
        uptime = (datetime.now() - app.state.start_time).total_seconds()

        return StatusResponse(
            running=status["running"],
            uptime_seconds=uptime,
            audio=status["audio"],
            vision=status["vision"],
            storage=status["storage"],
        )

    @app.post("/api/context/generate", response_model=ContextResponse)
    async def generate_context():
        """Trigger immediate context generation."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            filepath = _contextor_instance.trigger_context_now()
            return ContextResponse(
                success=True,
                message=f"Context generated: {filepath}",
                data={"filepath": filepath},
            )
        except Exception as e:
            logger.error(f"Failed to generate context: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/context/latest", response_model=ContextResponse)
    async def get_latest_context():
        """Get the most recent context data."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            context = _contextor_instance.generator.get_latest_context()
            if context is None:
                return ContextResponse(
                    success=False,
                    message="No context available yet",
                )

            return ContextResponse(
                success=True,
                message="Latest context retrieved",
                data=context,
            )
        except Exception as e:
            logger.error(f"Failed to get context: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/context/history")
    async def get_context_history(limit: int = 10):
        """Get list of recent context files."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            files = _contextor_instance.storage.get_recent_contexts(limit)
            history = []

            for filepath in files:
                path = Path(filepath)
                history.append({
                    "filename": path.name,
                    "filepath": filepath,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "size_bytes": path.stat().st_size,
                })

            return {"success": True, "data": history}
        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/context/file/{filename}")
    async def get_context_file(filename: str):
        """Get a specific context file by filename."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        filepath = Path(_contextor_instance.config.context.output_dir) / filename

        if not filepath.exists() or not filepath.suffix == ".json":
            raise HTTPException(status_code=404, detail="Context file not found")

        with open(filepath, "r") as f:
            data = json.load(f)

        return {"success": True, "data": data}

    @app.get("/api/images/recent")
    async def get_recent_images(limit: int = 20):
        """Get list of recent captured images."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            images = _contextor_instance.storage.get_recent_images(limit)
            result = []

            for filepath in images:
                path = Path(filepath)
                result.append({
                    "filename": path.name,
                    "url": f"/api/images/file/{path.name}",
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                    "size_bytes": path.stat().st_size,
                })

            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Failed to get images: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/images/file/{filename}")
    async def get_image_file(filename: str):
        """Serve a captured image file."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        filepath = Path(_contextor_instance.config.context.images_dir) / filename

        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(filepath, media_type="image/jpeg")

    @app.get("/api/transcripts/recent")
    async def get_recent_transcripts():
        """Get recent transcript segments from current period."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            transcripts = _contextor_instance.transcriber.get_recent_transcripts()
            result = []

            for t in transcripts:
                result.append({
                    "text": t.text,
                    "timestamp": t.timestamp.isoformat(),
                    "confidence": t.confidence,
                    "duration_ms": t.duration_ms,
                })

            return {"success": True, "data": result}
        except Exception as e:
            logger.error(f"Failed to get transcripts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/storage/stats")
    async def get_storage_stats():
        """Get storage usage statistics."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            stats = _contextor_instance.storage.get_storage_stats()
            return {"success": True, "data": stats}
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/storage/cleanup")
    async def trigger_cleanup():
        """Manually trigger storage cleanup."""
        if _contextor_instance is None:
            raise HTTPException(status_code=503, detail="Contextor not initialized")

        try:
            deleted = _contextor_instance.storage.cleanup_old_files()
            return {
                "success": True,
                "message": f"Cleaned up {deleted} files",
                "data": {"deleted_count": deleted},
            }
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app
