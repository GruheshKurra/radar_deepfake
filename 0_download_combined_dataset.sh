#!/bin/bash

set -e

echo "======================================"
echo "RADAR Combined Dataset Downloader"
echo "======================================"
echo ""

OUTPUT_DIR="${1:-./data}"
DOWNLOAD_FILE="radar-deepfake-frames.zip"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/gruheshkurra/radar-deepfake-frames"

echo "Download directory: $OUTPUT_DIR"
echo ""

if [[ "$OSTYPE" == "linux-gnu"* ]] || [ -f /etc/debian_version ]; then
    if ! command -v 7z &> /dev/null; then
        echo "Installing 7zip for fast extraction..."
        apt-get update -qq
        apt-get install -y -qq p7zip-full
    fi
    if ! command -v wget &> /dev/null; then
        echo "Installing wget..."
        apt-get install -y -qq wget
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v 7z &> /dev/null; then
        echo "Installing 7zip via Homebrew..."
        brew install p7zip
    fi
    if ! command -v wget &> /dev/null; then
        echo "Installing wget via Homebrew..."
        brew install wget
    fi
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Downloading dataset (public, no auth required)..."
wget --progress=bar:force:noscroll \
     --show-progress \
     --no-check-certificate \
     -O "$DOWNLOAD_FILE" \
     "$DATASET_URL"

if [ ! -f "$DOWNLOAD_FILE" ]; then
    echo "ERROR: Download failed!"
    exit 1
fi

echo ""
echo "Download complete: $(du -h $DOWNLOAD_FILE | cut -f1)"
echo ""

echo "Extracting with 7zip (fast multi-threaded)..."
7z x "$DOWNLOAD_FILE" -o. -bsp1

if [ -d "combined" ]; then
    rm "$DOWNLOAD_FILE"
    echo ""
    echo "✓ Dataset ready at: $PWD/combined"
else
    echo "ERROR: Extraction failed or unexpected structure"
    exit 1
fi
