#!/usr/bin/env bash
set -e

ROOT_DIR="/home/op/cozyvoice_fastapi"
MODEL_DIR="$ROOT_DIR/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B-2512"
LOG_DIR="$ROOT_DIR"
SERVER_LOG="$LOG_DIR/server.log"
DL_LOG="$LOG_DIR/hf_download.log"

cd "$ROOT_DIR"

# Download model if missing
if [ ! -d "$MODEL_DIR" ]; then
  echo "[INFO] Downloading CosyVoice model..." >> "$DL_LOG"
  huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False >> "$DL_LOG" 2>&1
fi

# Wait until model looks complete
while [ ! -f "$MODEL_DIR/config.json" ]; do
  echo "[INFO] Waiting for model files..." >> "$SERVER_LOG"
  sleep 10
done

# Watchdog loop
while true; do
  echo "[INFO] Starting CosyVoice server..." >> "$SERVER_LOG"
  python3 openai_tts_cosyvoice_server.py >> "$SERVER_LOG" 2>&1 || true
  echo "[WARN] Server crashed. Restarting in 5s..." >> "$SERVER_LOG"
  sleep 5
done
