#!/usr/bin/env bash
# ============================================================
#  run.sh  —  Quick launcher for the Garbage Segregation System
# ============================================================
set -e

echo ""
echo "  ♻️  AI Garbage Segregation System"
echo "  ─────────────────────────────────"
echo ""

# 1. Create virtual env if missing
if [ ! -d "venv" ]; then
  echo "[setup] Creating virtual environment …"
  python3 -m venv venv
fi

# 2. Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# 3. Install deps
echo "[setup] Installing dependencies …"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "  Choose an action:"
echo "  1) Train model"
echo "  2) Launch web app (Streamlit)"
echo "  3) Live webcam detection"
echo "  4) Predict single image"
echo ""
read -p "  Enter choice [1-4]: " CHOICE

case $CHOICE in
  1)
    echo "[train] Starting training …"
    python train.py
    ;;
  2)
    echo "[app] Launching Streamlit …"
    streamlit run app.py
    ;;
  3)
    echo "[webcam] Starting webcam detection …"
    python webcam_detection.py
    ;;
  4)
    read -p "  Image path: " IMG_PATH
    python predict.py "$IMG_PATH"
    ;;
  *)
    echo "Invalid choice."
    ;;
esac
