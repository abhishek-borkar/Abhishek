#!/usr/bin/env bash
# =============================================================================
#  setup_pi4b.sh — One-time setup for HOD Face Detector on Raspberry Pi 4B
#  Works on: Raspberry Pi OS Bookworm (64-bit) & Bullseye (32/64-bit)
#
#  Usage:
#      bash setup_pi4b.sh
#      source venv/bin/activate
#      python3 hod_detection_pi4b.py
# =============================================================================

set -euo pipefail

VENV_DIR="venv"
PI_MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "unknown")

echo ""
echo "=============================================="
echo "  HOD Face Detector — Pi 4B Setup"
echo "  Detected board: $PI_MODEL"
echo "=============================================="
echo ""

# ── Step 1: System packages ───────────────────────────────────────────────────
echo "[1/5] Installing system dependencies (apt)..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavformat-dev \
    libswscale-dev \
    libgtk-3-dev \
    python3-numpy \
    --no-install-recommends
echo "    [OK] System packages installed."

# ── Step 2: Expand swap to 2 GB (dlib compile needs ~1.5 GB RAM+swap) ────────
echo ""
echo "[2/5] Configuring swap (needed for dlib compilation)..."
SWAP_FILE="/etc/dphys-swapfile"
if [ -f "$SWAP_FILE" ]; then
    current_swap=$(grep "^CONF_SWAPSIZE" "$SWAP_FILE" | cut -d= -f2 | tr -d ' ')
    if [ "${current_swap:-0}" -lt 2048 ]; then
        sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=2048/' "$SWAP_FILE"
        sudo systemctl restart dphys-swapfile
        echo "    [OK] Swap expanded to 2048 MB."
    else
        echo "    [OK] Swap already ${current_swap} MB — no change needed."
    fi
else
    echo "    [WARN] dphys-swapfile not found — skipping swap config."
fi

# ── Step 3: Create Python virtual environment ─────────────────────────────────
echo ""
echo "[3/5] Creating Python virtual environment in './$VENV_DIR'..."
# Bookworm requires venv; --system-site-packages gives us numpy from apt
python3 -m venv --system-site-packages "$VENV_DIR"
echo "    [OK] Virtual environment created."

# ── Step 4: Install Python packages inside venv ───────────────────────────────
echo ""
echo "[4/5] Installing Python packages (dlib compiles from source ~15-20 min)..."
"$VENV_DIR/bin/pip" install --upgrade pip wheel setuptools

# Install dlib first (longest step — compiles C++ on ARM)
"$VENV_DIR/bin/pip" install dlib

# Install headless OpenCV (no GUI dependencies — works on headless Pi)
"$VENV_DIR/bin/pip" install opencv-python-headless

# Install face_recognition
"$VENV_DIR/bin/pip" install face_recognition

echo "    [OK] All Python packages installed."

# ── Step 5: Restore swap to default to save SD card writes ────────────────────
echo ""
echo "[5/5] Restoring swap to 200 MB (saves SD card wear)..."
if [ -f "$SWAP_FILE" ]; then
    sudo sed -i 's/^CONF_SWAPSIZE=.*/CONF_SWAPSIZE=200/' "$SWAP_FILE"
    sudo systemctl restart dphys-swapfile
    echo "    [OK] Swap restored to 200 MB."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "  Next steps:"
echo "  1. Place your reference image:  hod.jpg"
echo "  2. Add test images to folder:   test_images/"
echo "  3. Run:"
echo "       source $VENV_DIR/bin/activate"
echo "       python3 hod_detection_pi4b.py"
echo ""
echo "  Results will be saved in:  detection_results/"
echo "=============================================="
