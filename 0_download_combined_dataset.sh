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
    if ! command -v wget &> /dev/null; then
        echo "Installing wget..."
        apt-get update -qq
        apt-get install -y -qq wget
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v wget &> /dev/null; then
        echo "Installing wget..."
        brew install wget
    fi
fi

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

if [ -f "$DOWNLOAD_FILE" ]; then
    echo "✓ Dataset already downloaded: $DOWNLOAD_FILE ($(du -h $DOWNLOAD_FILE | cut -f1))"
    echo "  Skipping download. Delete the file to re-download."
else
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
fi
echo ""
echo "✓ Dataset ready: $PWD/$DOWNLOAD_FILE"
echo ""
echo "Choose your mode:"
echo ""
echo "1. INSTANT MODE (Recommended): Train directly from zip"
echo "   python 2_train_model.py --data_dir $PWD/$DOWNLOAD_FILE"
echo ""
echo "2. FAST EXTRACT: Parallel extraction (5-10 minutes on 96 cores)"
echo "   python 1_fast_extract.py --zip_path $PWD/$DOWNLOAD_FILE --num_workers 96"
echo "   python 2_train_model.py --data_dir $PWD"
