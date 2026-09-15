#!/usr/bin/env bash
# Startup script for HALAL-BERT Classifier

set -e

echo "=================================================="
echo " Starting HALAL-BERT Classifier Web Application   "
echo " Fine-Tuned PEFT LoRA by Umair Naveed             "
echo "=================================================="

PORT=${PORT:-7860}
echo "Running on http://localhost:$PORT"

python3 app.py
