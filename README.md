# Contextor

**Jetson Orin Nano Context Recorder** - Always listening, always watching.

A Python-based context recording system that continuously captures audio transcriptions and visual observations, generating periodic context snapshots.

## Features

- **Continuous Speech Transcription**: Always-on audio capture with voice activity detection and on-device transcription using Whisper
- **Motion Detection**: Frame differencing-based motion detection triggers image capture
- **Object Detection**: YOLOv8 identifies objects in captured images
- **Scheduled Capture**: Automatic photos every 5 minutes (configurable)
- **Context Generation**: Periodic JSON files summarizing what was heard and seen
- **On-Demand Triggers**: Generate context immediately via file or signal

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTEXTOR                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │   Audio Pipeline │         │  Vision Pipeline │              │
│  ├──────────────────┤         ├──────────────────┤              │
│  │ Microphone → VAD │         │ Camera → Motion  │              │
│  │        ↓         │         │        ↓         │              │
│  │    Whisper       │         │      YOLO        │              │
│  │        ↓         │         │        ↓         │              │
│  │  Transcripts     │         │    Objects       │              │
│  └────────┬─────────┘         └────────┬─────────┘              │
│           └──────────┬─────────────────┘                         │
│                      ↓                                           │
│              Context Manager → context.json                      │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- Jetson Orin Nano (8GB recommended)
- JetPack 6.0 or later
- USB microphone
- USB or CSI camera
- Python 3.10+

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/contextor.git
cd contextor

# Run the installation script
chmod +x scripts/install.sh
./scripts/install.sh
```

## Configuration

Edit `config/settings.yaml` to configure:

```yaml
audio:
  device: "default"           # Audio device
  whisper_model: "small.en"   # tiny, base, small, medium

vision:
  camera_id: 0                # Camera index
  capture_interval: 300       # Seconds between captures
  yolo_model: "yolov8n.pt"    # YOLO model

context:
  interval: 300               # Context generation interval
  output_dir: "./data/context"
```

## Usage

### Manual Run

```bash
source venv/bin/activate
python -m src.main -c config/settings.yaml
```

### As a Service

```bash
# Enable and start
sudo systemctl enable contextor
sudo systemctl start contextor

# View logs
journalctl -u contextor -f
```

### List Devices

```bash
python -m src.main --list-audio
python -m src.main --list-cameras
```

### Trigger Context Generation

```bash
# Method 1: File trigger
touch /tmp/contextor_trigger

# Method 2: Signal
kill -USR1 $(pgrep -f contextor)
```

## Output Format

The `context.json` file contains:

```json
{
  "generated_at": "2024-01-15T10:30:00",
  "period": {
    "start": "2024-01-15T10:25:00",
    "end": "2024-01-15T10:30:00"
  },
  "audio": {
    "transcript_segments": [...],
    "full_transcript": "Conversation text here..."
  },
  "vision": {
    "motion_events": 3,
    "images_captured": 4,
    "objects_detected": {
      "person": {"count": 12, "avg_confidence": 0.87},
      "cup": {"count": 5, "avg_confidence": 0.76}
    },
    "unique_objects": ["person", "cup", "laptop"],
    "images": [...]
  },
  "summary": "2 speech segments recorded. Objects: person (12x), cup (5x)..."
}
```

## Resource Usage

On Jetson Orin Nano (8GB):
- Whisper (small): ~1.5GB VRAM
- YOLOv8n: ~0.5GB VRAM
- System: ~2GB
- Available: ~4GB for buffers

## Models Used

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| VAD | Silero VAD | ~1MB | Voice activity detection |
| STT | Whisper small.en | ~500MB | Speech transcription |
| Detection | YOLOv8n | ~6MB | Object detection |

## Project Structure

```
contextor/
├── src/
│   ├── main.py           # Entry point
│   ├── config.py         # Configuration
│   ├── audio/            # Audio pipeline
│   ├── vision/           # Vision pipeline
│   └── context/          # Context generation
├── data/
│   ├── context/          # Generated JSON files
│   └── images/           # Captured images
├── config/
│   └── settings.yaml     # Configuration file
└── scripts/
    └── install.sh        # Setup script
```

## License

MIT License
