#!/bin/bash
# Download Silero VAD ONNX model from the official repository.
#
# Usage:
#   bash tool/download_model.sh
#
# The model will be saved to assets/models/silero_vad.onnx

set -euo pipefail

MODEL_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
OUTPUT_DIR="assets/models"
OUTPUT_FILE="$OUTPUT_DIR/silero_vad.onnx"

echo "Downloading Silero VAD ONNX model..."
mkdir -p "$OUTPUT_DIR"

if command -v curl &> /dev/null; then
    curl -L -o "$OUTPUT_FILE" "$MODEL_URL"
elif command -v wget &> /dev/null; then
    wget -O "$OUTPUT_FILE" "$MODEL_URL"
else
    echo "Error: curl or wget is required to download the model." >&2
    exit 1
fi

echo "Model downloaded successfully to: $OUTPUT_FILE"
echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
