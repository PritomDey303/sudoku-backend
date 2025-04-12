#!/usr/bin/env bash
# build.sh

# Install system dependencies
apt-get update
apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    python3-opencv  # This ensures OpenCV system dependencies are installed

# Install Python dependencies
pip install --upgrade pip  # Ensure pip is latest
pip install -r requirements.txt