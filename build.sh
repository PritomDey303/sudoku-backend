#!/usr/bin/env bash
# build.sh

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr libtesseract-dev

# Install Python dependencies with specific versions
pip install numpy==1.21.6
pip install -r requirements.txt