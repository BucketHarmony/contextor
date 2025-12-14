# Contextor - Jetson Orin Nano Context Recorder

## Overview

A Python-based context recording system that continuously captures audio transcriptions and visual observations, generating periodic context snapshots.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTEXTOR                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │   Audio Pipeline │         │  Vision Pipeline │              │
│  ├──────────────────┤         ├──────────────────┤              │
│  │ Microphone Input │         │   Camera Input   │              │
│  │        ↓         │         │        ↓         │              │
│  │  VAD (Silero)    │         │ Motion Detector  │──→ On Motion │
│  │        ↓         │         │        ↓         │    Capture   │
│  │ Whisper (small)  │         │  YOLO Detector   │              │
│  │        ↓         │         │        ↓         │              │
│  │ Transcript Queue │         │  Objects Queue   │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           │                            │                         │
│           └──────────┬─────────────────┘                         │
│                      ↓                                           │
│           ┌──────────────────┐                                   │
│           │ Context Manager  │                                   │
│           ├──────────────────┤                                   │
│           │ - 5-min timer    │                                   │
│           │ - On-demand trigger                                  │
│           │ - JSON generator │                                   │
│           └────────┬─────────┘                                   │
│                    ↓                                             │
│           ┌──────────────────┐                                   │
│           │  context.json    │                                   │
│           │  images/         │                                   │
│           │  transcripts/    │                                   │
│           └──────────────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Audio Pipeline (`src/audio/`)

**Components:**
- `capture.py` - Continuous audio capture using PyAudio/sounddevice
- `vad.py` - Voice Activity Detection using Silero VAD (efficient, runs on CPU)
- `transcriber.py` - Speech-to-text using Whisper (small model, GPU accelerated)

**Flow:**
1. Continuously capture audio in chunks (e.g., 512ms)
2. VAD filters out silence (saves compute, only transcribe speech)
3. Buffer speech segments until pause detected
4. Send complete utterances to Whisper for transcription
5. Store transcripts with timestamps in thread-safe queue

**Models:**
- Silero VAD: ~1MB, CPU-only, very fast
- Whisper small.en: ~500MB, optimized for English, good accuracy/speed tradeoff

### 2. Vision Pipeline (`src/vision/`)

**Components:**
- `camera.py` - Camera interface (CSI or USB camera)
- `motion.py` - Motion detection using frame differencing + contours
- `detector.py` - Object detection using YOLOv8n (nano)
- `scheduler.py` - 5-minute photo capture timer

**Flow:**
1. Capture frames at low FPS (2-5 FPS) for motion detection
2. Compare consecutive frames, detect significant changes
3. On motion: capture full-res image, run YOLO, store results
4. Every 5 minutes: capture image regardless of motion
5. Store detected objects with confidence scores and bounding boxes

**Models:**
- YOLOv8n: ~6MB, optimized for edge devices, 80 COCO classes
- Can be swapped for custom-trained model later

### 3. Context Manager (`src/context/`)

**Components:**
- `aggregator.py` - Collects data from both pipelines
- `generator.py` - Creates JSON context files
- `storage.py` - File management, rotation, cleanup

**Output Format (`context.json`):**
```json
{
  "generated_at": "2024-01-15T10:30:00Z",
  "period": {
    "start": "2024-01-15T10:25:00Z",
    "end": "2024-01-15T10:30:00Z"
  },
  "audio": {
    "transcript_segments": [
      {
        "timestamp": "2024-01-15T10:26:15Z",
        "text": "Hey, can you pass me the coffee?",
        "confidence": 0.92
      }
    ],
    "full_transcript": "Hey, can you pass me the coffee? Sure, here you go. Thanks!"
  },
  "vision": {
    "motion_events": 3,
    "images_captured": 4,
    "objects_detected": {
      "person": {"count": 12, "avg_confidence": 0.87},
      "cup": {"count": 5, "avg_confidence": 0.76},
      "laptop": {"count": 8, "avg_confidence": 0.91}
    },
    "unique_objects": ["person", "cup", "laptop", "chair", "desk"],
    "images": [
      {
        "filename": "2024-01-15_10-26-30_motion.jpg",
        "trigger": "motion",
        "objects": [
          {"label": "person", "confidence": 0.89, "bbox": [100, 50, 300, 400]}
        ]
      }
    ]
  },
  "summary": "2 people detected discussing over coffee. Motion detected 3 times."
}
```

### 4. Main Controller (`src/main.py`)

- Initializes all pipelines
- Manages threading/async coordination
- Handles graceful shutdown (SIGINT/SIGTERM)
- Provides on-demand context generation (via signal or file trigger)

## Project Structure

```
contextor/
├── src/
│   ├── __init__.py
│   ├── main.py              # Entry point, orchestration
│   ├── config.py            # Configuration management
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── capture.py       # Audio input handling
│   │   ├── vad.py           # Voice activity detection
│   │   └── transcriber.py   # Whisper integration
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py        # Camera interface
│   │   ├── motion.py        # Motion detection
│   │   └── detector.py      # YOLO object detection
│   └── context/
│       ├── __init__.py
│       ├── aggregator.py    # Data collection
│       ├── generator.py     # JSON output
│       └── storage.py       # File management
├── data/
│   ├── context/             # Generated context files
│   ├── images/              # Captured images
│   └── audio/               # Optional: raw audio segments
├── models/                  # Downloaded model files
├── config/
│   └── settings.yaml        # User configuration
├── scripts/
│   ├── install.sh           # Jetson setup script
│   └── trigger_context.sh   # On-demand context generation
├── tests/
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Dependencies

```
# Core
python>=3.10

# Audio
sounddevice>=0.4.6          # Audio capture
numpy>=1.24.0               # Array operations
torch>=2.0.0                # ML framework (Jetson optimized)
torchaudio>=2.0.0           # Audio processing
faster-whisper>=0.10.0      # Optimized Whisper (CTranslate2)
silero-vad                  # Voice activity detection

# Vision
opencv-python>=4.8.0        # Image capture/processing
ultralytics>=8.0.0          # YOLOv8

# Utils
pyyaml>=6.0                 # Configuration
schedule>=1.2.0             # Periodic tasks
python-json-logger>=2.0.0   # Structured logging
```

## Implementation Steps

### Phase 1: Project Setup
1. Create project structure and configuration system
2. Set up logging and error handling
3. Create Jetson-specific installation script

### Phase 2: Audio Pipeline
4. Implement audio capture with sounddevice
5. Integrate Silero VAD for speech detection
6. Set up faster-whisper for transcription
7. Create transcript buffering and timestamping

### Phase 3: Vision Pipeline
8. Implement camera capture (CSI/USB support)
9. Build motion detection using frame differencing
10. Integrate YOLOv8 for object detection
11. Create 5-minute photo scheduler

### Phase 4: Context Generation
12. Build data aggregator to collect from both pipelines
13. Implement JSON context generator
14. Add file rotation and storage management
15. Create on-demand trigger mechanism

### Phase 5: Integration & Polish
16. Create main orchestrator with proper threading
17. Add graceful shutdown handling
18. Write systemd service for auto-start
19. Create setup documentation

## Hardware Considerations

**Jetson Orin Nano Specs:**
- 6 TOPS AI performance
- 8GB RAM (shared CPU/GPU)
- NVIDIA Ampere GPU

**Resource Allocation:**
- Whisper (small): ~1.5GB VRAM
- YOLOv8n: ~0.5GB VRAM
- System overhead: ~2GB
- Available for buffers: ~4GB

**Optimization Strategies:**
- Use `faster-whisper` (CTranslate2) for 4x faster inference
- Run VAD on CPU, Whisper/YOLO on GPU
- Motion detection on CPU (OpenCV)
- Batch object detection when possible
- Use INT8 quantization if memory constrained

## Configuration Options

```yaml
# config/settings.yaml
audio:
  device: "default"           # Audio input device
  sample_rate: 16000
  vad_threshold: 0.5          # Voice activity sensitivity
  whisper_model: "small.en"   # tiny, base, small, medium

vision:
  camera_id: 0                # Camera index or CSI path
  motion_threshold: 25        # Pixel difference threshold
  motion_min_area: 500        # Minimum contour area
  capture_interval: 300       # Seconds between scheduled captures
  yolo_model: "yolov8n.pt"
  yolo_confidence: 0.5

context:
  output_dir: "./data/context"
  interval: 300               # Context generation interval (seconds)
  keep_images: true           # Store captured images
  max_storage_gb: 10          # Auto-cleanup threshold

logging:
  level: "INFO"
  file: "./logs/contextor.log"
```

## On-Demand Triggers

Three methods to trigger immediate context generation:

1. **Signal:** `kill -USR1 <pid>`
2. **File:** Create `/tmp/contextor_trigger`
3. **GPIO:** Button press on Jetson GPIO pin (optional)

## Future Enhancements

- Face recognition for person identification
- LLM-based summary generation (local Llama or API)
- Audio event detection (doorbell, alarm, etc.)
- Multi-room support with multiple devices
- Mobile app for remote viewing
- Home Assistant integration
