#!/bin/bash
# Contextor Update Script
# Updates the codebase to the latest version from the repository

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Contextor - Update Script"
echo "=========================================="

cd "$PROJECT_DIR"

# Check if this is a git repository
if [ ! -d ".git" ]; then
    echo "Error: This directory is not a git repository"
    echo "Please clone the repository first:"
    echo "  git clone https://github.com/BucketHarmony/contextor.git"
    exit 1
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "Warning: You have uncommitted changes"
    echo "Please commit or stash them before updating"
    git status --short
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get current version
CURRENT_COMMIT=$(git rev-parse --short HEAD)
echo "Current version: $CURRENT_COMMIT"

# Fetch latest changes
echo ""
echo "[1/4] Fetching latest changes..."
git fetch origin

# Check if update is available
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/master)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "Already up to date!"
    exit 0
fi

# Show changes
echo ""
echo "[2/4] Changes to be applied:"
git log --oneline HEAD..origin/master
echo ""

# Stop service if running
echo "[3/4] Stopping contextor service (if running)..."
if systemctl is-active --quiet contextor 2>/dev/null; then
    sudo systemctl stop contextor
    SERVICE_WAS_RUNNING=true
    echo "Service stopped"
else
    SERVICE_WAS_RUNNING=false
    echo "Service was not running"
fi

# Pull changes
echo ""
echo "[4/4] Pulling latest changes..."
git pull origin master

# Update dependencies if requirements changed
if git diff --name-only "$CURRENT_COMMIT" HEAD | grep -q "requirements"; then
    echo ""
    echo "Requirements changed, updating dependencies..."
    if [ -d "venv" ]; then
        source venv/bin/activate
        pip install -r requirements.txt
        deactivate
    else
        echo "Warning: Virtual environment not found at ./venv"
        echo "Please run: pip install -r requirements.txt"
    fi
fi

# Restart service if it was running
if [ "$SERVICE_WAS_RUNNING" = true ]; then
    echo ""
    echo "Restarting contextor service..."
    sudo systemctl start contextor
    echo "Service restarted"
fi

NEW_COMMIT=$(git rev-parse --short HEAD)
echo ""
echo "=========================================="
echo "Update complete!"
echo "=========================================="
echo "Updated from $CURRENT_COMMIT to $NEW_COMMIT"
echo ""
echo "If you encounter issues, you can rollback with:"
echo "  git reset --hard $CURRENT_COMMIT"
echo ""
