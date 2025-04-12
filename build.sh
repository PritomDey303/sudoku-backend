#!/usr/bin/env bash
# build.sh

# Install system dependencies
apt-get update
apt-get install -y tesseract-ocr
apt-get install -y libtesseract-dev

# Install Python dependencies
pip install -r requirements.txt