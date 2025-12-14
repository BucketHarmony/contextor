#!/bin/bash
# Contextor Installation Script for Jetson Orin Nano
# Run this script on a fresh JetPack 6.0+ installation

set -e

echo "=========================================="
echo "Contextor - Jetson Orin Nano Setup"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This doesn't appear to be a Jetson device"
    echo "Some optimizations may not apply"
fi

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install system dependencies
echo "[2/8] Installing system dependencies..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly

# Create virtual environment
echo "[3/8] Creating Python virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch from NVIDIA (Jetson optimized)
echo "[4/8] Installing PyTorch (Jetson optimized)..."
# For JetPack 6.0 (L4T R36.x)
pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://developer.download.nvidia.com/compute/redist/jp/v60

# Install CUDA-optimized OpenCV
echo "[5/8] Installing OpenCV with CUDA support..."
# Use system OpenCV if available (pre-installed on JetPack)
if python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())" 2>/dev/null; then
    echo "Using system OpenCV with CUDA"
else
    pip install opencv-python
fi

# Install Python dependencies
echo "[6/8] Installing Python dependencies..."
pip install -r requirements.txt

# Download models
echo "[7/8] Downloading models..."

# Download YOLOv8n
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/ 2>/dev/null || true

# Pre-download Whisper model (will cache in ~/.cache/huggingface)
python3 -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu')"

# Pre-download Silero VAD
python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad')"

# Create systemd service
echo "[8/8] Creating systemd service..."
sudo tee /etc/systemd/system/contextor.service > /dev/null << EOF
[Unit]
Description=Contextor - Context Recorder
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/python -m src.main -c config/settings.yaml
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=6G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Edit config/settings.yaml to configure your devices"
echo "  2. Test with: source venv/bin/activate && python -m src.main"
echo "  3. Enable service: sudo systemctl enable contextor"
echo "  4. Start service: sudo systemctl start contextor"
echo ""
echo "Useful commands:"
echo "  View logs: journalctl -u contextor -f"
echo "  Trigger context: touch /tmp/contextor_trigger"
echo "  Or send signal: kill -USR1 \$(pgrep -f contextor)"
echo ""
