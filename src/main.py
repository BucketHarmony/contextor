"""Main orchestrator for Contextor - Jetson Orin Nano Context Recorder."""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn

from .audio.capture import AudioCapture
from .audio.transcriber import Transcriber
from .audio.vad import VoiceActivityDetector
from .config import Config, load_config
from .context.aggregator import ContextAggregator
from .context.generator import ContextGenerator
from .context.storage import StorageManager
from .vision.camera import Camera
from .vision.detector import ObjectDetector
from .vision.motion import MotionDetector
from .web.api import create_app, set_contextor_instance

logger = logging.getLogger(__name__)


class Contextor:
    """Main application orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self._running = False
        self._shutdown_event = threading.Event()

        # Initialize components
        self._init_audio()
        self._init_vision()
        self._init_context()

        # Timers
        self._last_scheduled_capture = 0.0
        self._last_context_generation = 0.0

        # On-demand trigger
        self._trigger_file = Path("/tmp/contextor_trigger")

        # Web server
        self._web_thread: Optional[threading.Thread] = None
        self._web_server: Optional[uvicorn.Server] = None

    def _init_audio(self) -> None:
        """Initialize audio pipeline components."""
        logger.info("Initializing audio pipeline...")
        self.audio_capture = AudioCapture(self.config.audio)
        self.vad = VoiceActivityDetector(self.config.audio)
        self.transcriber = Transcriber(self.config.audio)

        # Wire up callbacks
        self.audio_capture.add_callback(self.vad.process_audio)
        self.vad.on_speech_end(self.transcriber.queue_segment)
        self.transcriber.on_transcription(self._on_transcription)

    def _init_vision(self) -> None:
        """Initialize vision pipeline components."""
        logger.info("Initializing vision pipeline...")
        self.camera = Camera(self.config.vision)
        self.motion_detector = MotionDetector(self.config.vision)
        self.object_detector = ObjectDetector(self.config.vision)

        # Wire up callbacks
        self.camera.add_callback(self._on_frame)
        self.motion_detector.on_motion(self._on_motion)

    def _init_context(self) -> None:
        """Initialize context management components."""
        logger.info("Initializing context manager...")
        self.aggregator = ContextAggregator()
        self.generator = ContextGenerator(self.config.context.output_dir)
        self.storage = StorageManager(self.config.context)

    def _on_transcription(self, transcript) -> None:
        """Handle transcription results."""
        self.aggregator.add_transcript(transcript)

    def _on_frame(self, frame) -> None:
        """Handle camera frames."""
        # Process for motion detection
        motion_event = self.motion_detector.process_frame(frame)

        if motion_event is not None:
            self._on_motion(motion_event)

    def _on_motion(self, event) -> None:
        """Handle motion detection events."""
        logger.info("Motion detected, capturing image...")

        # Save image
        image_path = self.storage.save_image(
            event.frame,
            trigger="motion",
            timestamp=event.timestamp,
        )

        # Run object detection
        result = self.object_detector.detect(
            event.frame,
            trigger="motion",
            image_path=image_path,
        )

        self.aggregator.add_motion_event(event)
        self.aggregator.add_detection(result)

        logger.info(f"Motion capture: {len(result.objects)} objects detected")

    def _scheduled_capture(self) -> None:
        """Perform scheduled photo capture."""
        current_time = time.time()
        interval = self.config.vision.capture_interval

        if current_time - self._last_scheduled_capture < interval:
            return

        self._last_scheduled_capture = current_time

        logger.info("Scheduled capture...")

        image = self.camera.capture_image()
        if image is None:
            logger.warning("Failed to capture scheduled image")
            return

        # Save image
        image_path = self.storage.save_image(image, trigger="scheduled")

        # Run object detection
        result = self.object_detector.detect(
            image,
            trigger="scheduled",
            image_path=image_path,
        )

        self.aggregator.add_detection(result)

        logger.info(f"Scheduled capture: {len(result.objects)} objects detected")

    def _generate_context(self, force: bool = False) -> Optional[str]:
        """Generate context file if interval elapsed or forced."""
        current_time = time.time()
        interval = self.config.context.interval

        if not force and current_time - self._last_context_generation < interval:
            return None

        self._last_context_generation = current_time

        logger.info("Generating context file...")

        # Collect current period data and pass transcript history
        context_data = self.aggregator.collect(clear=True)

        # Add the last hour of transcript history to context
        transcript_history = self.aggregator.get_transcript_history()
        filepath = self.generator.generate(context_data, transcript_history=transcript_history)

        # Cleanup old files
        deleted = self.storage.cleanup_old_files()
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old files")

        return filepath

    def _check_trigger(self) -> bool:
        """Check for on-demand trigger."""
        if self._trigger_file.exists():
            self._trigger_file.unlink()
            return True
        return False

    def _main_loop(self) -> None:
        """Main processing loop."""
        logger.info("Starting main loop...")

        while not self._shutdown_event.is_set():
            try:
                # Check for on-demand trigger
                if self._check_trigger():
                    logger.info("On-demand context generation triggered")
                    self._generate_context(force=True)

                # Scheduled capture
                self._scheduled_capture()

                # Context generation
                self._generate_context()

                # Small sleep to prevent busy loop
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1.0)

    def _start_web_server(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the web server in a background thread."""
        logger.info(f"Starting web server on {host}:{port}...")

        # Set the contextor instance for API access
        set_contextor_instance(self)

        # Create FastAPI app
        app = create_app()

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )
        self._web_server = uvicorn.Server(config)

        # Run in thread
        def run_server():
            self._web_server.run()

        self._web_thread = threading.Thread(target=run_server, daemon=True)
        self._web_thread.start()

        logger.info(f"Web server started at http://{host}:{port}")

    def _stop_web_server(self) -> None:
        """Stop the web server."""
        if self._web_server is not None:
            logger.info("Stopping web server...")
            self._web_server.should_exit = True
            if self._web_thread is not None:
                self._web_thread.join(timeout=5.0)

    def start(self, enable_web: bool = True, web_port: int = 8080) -> None:
        """Start all components."""
        if self._running:
            logger.warning("Contextor already running")
            return

        logger.info("Starting Contextor...")

        self._running = True
        self._shutdown_event.clear()

        # Ensure directories exist
        self.config.ensure_directories()

        # Start components
        self.transcriber.start()
        self.audio_capture.start()
        self.camera.start()

        # Load YOLO model
        self.object_detector.load_model()

        # Initialize timers
        self._last_scheduled_capture = time.time()
        self._last_context_generation = time.time()

        # Start main loop in thread
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()

        # Start web server
        if enable_web:
            self._start_web_server(port=web_port)

        logger.info("Contextor started successfully")

    def stop(self) -> None:
        """Stop all components gracefully."""
        if not self._running:
            return

        logger.info("Stopping Contextor...")

        self._running = False
        self._shutdown_event.set()

        # Stop web server first
        self._stop_web_server()

        # Generate final context
        logger.info("Generating final context...")
        self.vad.flush()
        time.sleep(0.5)  # Allow pending transcriptions
        self._generate_context(force=True)

        # Stop components
        self.audio_capture.stop()
        self.transcriber.stop()
        self.camera.stop()

        if hasattr(self, "_main_thread"):
            self._main_thread.join(timeout=5.0)

        logger.info("Contextor stopped")

    def trigger_context_now(self) -> str:
        """Manually trigger context generation."""
        return self._generate_context(force=True)

    def get_status(self) -> dict:
        """Get current status of all components."""
        return {
            "running": self._running,
            "audio": {
                "capture_running": self.audio_capture.is_running(),
                "transcriber_running": self.transcriber.is_running(),
                "is_speaking": self.vad.is_speaking,
                "pending_transcripts": self.aggregator.transcript_count,
            },
            "vision": {
                "camera_running": self.camera.is_running(),
                "frame_count": self.camera.frame_count,
                "motion_events": self.aggregator.motion_count,
                "detections": self.aggregator.detection_count,
            },
            "storage": self.storage.get_storage_stats(),
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Contextor - Context Recorder")
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8080,
        help="Web server port (default: 8080)",
    )
    parser.add_argument(
        "--no-web",
        action="store_true",
        help="Disable web server",
    )
    parser.add_argument(
        "--list-audio",
        action="store_true",
        help="List available audio devices",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available cameras",
    )
    args = parser.parse_args()

    # List devices if requested
    if args.list_audio:
        print("Available audio devices:")
        for dev in AudioCapture.list_devices():
            print(f"  [{dev['id']}] {dev['name']} ({dev['channels']}ch)")
        return

    if args.list_cameras:
        print("Available cameras:")
        for cam in Camera.list_cameras():
            print(f"  [{cam['id']}] {cam['width']}x{cam['height']}")
        return

    # Load configuration
    config = load_config(args.config)
    config.setup_logging()

    logger.info("=" * 50)
    logger.info("Contextor - Jetson Orin Nano Context Recorder")
    logger.info("=" * 50)

    # Create application
    app = Contextor(config)

    # Signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        app.stop()
        sys.exit(0)

    def usr1_handler(signum, frame):
        logger.info("USR1 received - generating context")
        app.trigger_context_now()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # USR1 for on-demand context (Unix only)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, usr1_handler)

    # Start application
    app.start(enable_web=not args.no_web, web_port=args.port)

    if not args.no_web:
        logger.info(f"Dashboard available at http://localhost:{args.port}")

    # Run until interrupted
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
