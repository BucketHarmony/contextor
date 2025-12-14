"""Audio capture module for continuous microphone input."""

import logging
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from ..config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCapture:
    """Continuous audio capture from microphone."""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.sample_rate = config.sample_rate
        self.channels = config.channels
        self.chunk_samples = int(config.sample_rate * config.chunk_duration_ms / 1000)

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: list[Callable[[np.ndarray], None]] = []

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for sounddevice stream."""
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Convert to mono float32 and normalize
        audio_data = indata.copy().flatten().astype(np.float32)

        # Put in queue for processing
        self._audio_queue.put(audio_data)

    def _process_loop(self) -> None:
        """Process audio chunks from queue."""
        while self._running:
            try:
                audio_chunk = self._audio_queue.get(timeout=1.0)
                for callback in self._callbacks:
                    try:
                        callback(audio_chunk)
                    except Exception as e:
                        logger.error(f"Audio callback error: {e}")
            except queue.Empty:
                continue

    def add_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback for audio chunks."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Unregister a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            logger.warning("Audio capture already running")
            return

        logger.info(f"Starting audio capture: {self.sample_rate}Hz, {self.channels}ch")

        # Resolve device
        device = None
        if self.config.device != "default":
            try:
                device = int(self.config.device)
            except ValueError:
                device = self.config.device

        self._running = True

        # Start processing thread
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

        # Start audio stream
        self._stream = sd.InputStream(
            device=device,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

        logger.info("Audio capture started")

    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return

        logger.info("Stopping audio capture")
        self._running = False

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("Audio capture stopped")

    def is_running(self) -> bool:
        """Check if capture is running."""
        return self._running

    @staticmethod
    def list_devices() -> list[dict]:
        """List available audio input devices."""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] > 0:
                devices.append({
                    "id": i,
                    "name": device["name"],
                    "channels": device["max_input_channels"],
                    "sample_rate": device["default_samplerate"],
                })
        return devices
