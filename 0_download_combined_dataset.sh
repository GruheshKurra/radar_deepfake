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
    MISSING_PACKAGES=""
    [ ! -x "$(command -v wget)" ] && MISSING_PACKAGES="$MISSING_PACKAGES wget"
    [ ! -x "$(command -v bsdtar)" ] && MISSING_PACKAGES="$MISSING_PACKAGES libarchive-tools"
    [ ! -x "$(command -v pv)" ] && MISSING_PACKAGES="$MISSING_PACKAGES pv"

    if [ -n "$MISSING_PACKAGES" ]; then
        echo "Installing required packages:$MISSING_PACKAGES"
        apt-get update -qq
        apt-get install -y -qq $MISSING_PACKAGES
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    [ ! -x "$(command -v wget)" ] && brew install wget
    [ ! -x "$(command -v bsdtar)" ] && brew install libarchive
    [ ! -x "$(command -v pv)" ] && brew install pv
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

echo "Extracting (bsdtar with progress)..."
FILE_SIZE=$(stat -f%z "$DOWNLOAD_FILE" 2>/dev/null || stat -c%s "$DOWNLOAD_FILE" 2>/dev/null)
pv -s $FILE_SIZE "$DOWNLOAD_FILE" | bsdtar -xf - -C .

if [ -d "combined" ]; then
    rm "$DOWNLOAD_FILE"
    echo ""
    echo "✓ Dataset ready at: $PWD/combined"
else
    echo "ERROR: Extraction failed or unexpected structure"
    exit 1
fi
