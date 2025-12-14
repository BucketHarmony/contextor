# Contextor

**Jetson Orin Nano Context Recorder** - Always listening, always watching.

A Python-based context recording system that continuously captures audio transcriptions and visual observations, generating periodic context snapshots.

## Features

- **Continuous Speech Transcription**: Always-on audio capture with voice activity detection and on-device transcription using Whisper
- **Motion Detection**: Frame differencing-based motion detection triggers image capture
- **Object Detection**: YOLOv8 identifies objects in captured images
- **Scheduled Capture**: Automatic photos every 5 minutes (configurable)
- **Context Generation**: Periodic JSON files summarizing what was heard and seen
- **On-Demand Triggers**: Generate context immediately via file, signal, or web API
- **Web Dashboard**: Real-time monitoring and control from any device on the network

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
│                      ↓                                           │
│               Web Dashboard (port 8080)                          │
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
git clone https://github.com/BucketHarmony/contextor.git
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

### Command Line Options

```bash
python -m src.main --help

Options:
  -c, --config PATH    Path to configuration file
  -p, --port PORT      Web server port (default: 8080)
  --no-web             Disable web interface
  --list-audio         List available audio devices
  --list-cameras       List available cameras
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

# Method 3: Web API (see below)
curl -X POST http://localhost:8080/api/context/generate
```

## Web Dashboard

Contextor includes a web-based dashboard for monitoring and control, accessible from any device on your network.

### Starting the Web Interface

The web server starts automatically with Contextor on port 8080:

```bash
# Start with web interface (default)
python -m src.main

# Start on a custom port
python -m src.main -p 9000

# Start without web interface
python -m src.main --no-web
```

### Accessing from Other Systems

1. **Find your Jetson's IP address:**
   ```bash
   hostname -I
   # Example output: 192.168.1.100
   ```

2. **Access the dashboard from any browser on the same network:**
   ```
   http://192.168.1.100:8080
   ```

3. **For remote access outside your network**, consider using:
   - SSH port forwarding: `ssh -L 8080:localhost:8080 user@jetson-ip`
   - A reverse proxy (nginx, caddy)
   - VPN connection to your network

### Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web dashboard |
| `/api/status` | GET | System status |
| `/api/context/generate` | POST | Trigger context generation |
| `/api/context/latest` | GET | Get latest context data |
| `/api/context/history` | GET | List recent context files |
| `/api/context/file/{name}` | GET | Get specific context file |
| `/api/images/recent` | GET | List recent images |
| `/api/images/file/{name}` | GET | Get specific image |
| `/api/transcripts/recent` | GET | Get recent transcripts |
| `/api/storage/stats` | GET | Storage statistics |
| `/api/storage/cleanup` | POST | Trigger storage cleanup |

### API Examples

```bash
# Get system status
curl http://192.168.1.100:8080/api/status

# Generate context immediately
curl -X POST http://192.168.1.100:8080/api/context/generate

# Get latest context
curl http://192.168.1.100:8080/api/context/latest

# Get recent transcripts
curl http://192.168.1.100:8080/api/transcripts/recent?limit=10
```

## Running as a System Service

### Automatic Setup (via install.sh)

The installation script automatically creates a systemd service. To enable it:

```bash
# Enable service to start on boot
sudo systemctl enable contextor

# Start the service now
sudo systemctl start contextor
```

### Manual Service Setup

If you need to create the service manually:

1. **Create the service file:**
   ```bash
   sudo nano /etc/systemd/system/contextor.service
   ```

2. **Add the following content** (adjust paths as needed):
   ```ini
   [Unit]
   Description=Contextor - Context Recorder
   After=network.target

   [Service]
   Type=simple
   User=your-username
   WorkingDirectory=/home/your-username/contextor
   ExecStart=/home/your-username/contextor/venv/bin/python -m src.main -c config/settings.yaml
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal

   # Resource limits (optional, recommended for Jetson)
   MemoryMax=6G
   CPUQuota=80%

   [Install]
   WantedBy=multi-user.target
   ```

3. **Reload systemd and enable the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable contextor
   sudo systemctl start contextor
   ```

### Service Management Commands

```bash
# Start the service
sudo systemctl start contextor

# Stop the service
sudo systemctl stop contextor

# Restart the service
sudo systemctl restart contextor

# Check service status
sudo systemctl status contextor

# View live logs
journalctl -u contextor -f

# View recent logs
journalctl -u contextor -n 100

# Disable auto-start on boot
sudo systemctl disable contextor
```

### Troubleshooting Service Issues

```bash
# Check if service is running
systemctl is-active contextor

# Check for errors
journalctl -u contextor -p err -b

# Check resource usage
systemctl status contextor --no-pager -l

# Reset failed state
sudo systemctl reset-failed contextor
```

## Updating Contextor

### Using the Update Script

The easiest way to update is using the provided script:

```bash
chmod +x scripts/update.sh
./scripts/update.sh
```

The script will:
1. Check for uncommitted changes
2. Fetch and display available updates
3. Stop the service if running
4. Pull the latest code
5. Update dependencies if needed
6. Restart the service

### Manual Update

```bash
# Stop the service
sudo systemctl stop contextor

# Pull latest changes
cd ~/contextor
git pull origin master

# Update dependencies (if needed)
source venv/bin/activate
pip install -r requirements.txt

# Restart the service
sudo systemctl start contextor
```

### Rollback to Previous Version

If an update causes issues:

```bash
# Find previous commit
git log --oneline -5

# Reset to previous version
git reset --hard <commit-hash>

# Restart service
sudo systemctl restart contextor
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
    "full_transcript": "Conversation text here...",
    "last_hour": {
      "segments": [...],
      "full_transcript": "Last hour of conversation..."
    }
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
│   ├── context/          # Context generation
│   └── web/              # Web dashboard & API
├── data/
│   ├── context/          # Generated JSON files
│   └── images/           # Captured images
├── config/
│   └── settings.yaml     # Configuration file
├── scripts/
│   ├── install.sh        # Installation script
│   ├── update.sh         # Update script
│   └── trigger_context.sh
└── tests/                # Unit tests
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

MIT License
