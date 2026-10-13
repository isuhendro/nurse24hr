#!/usr/bin/env bash
# Run the nurse24hr monitor. Any arguments are passed through to the CLI.
#
#   ./run.sh                      webcam + nurse web page (default)
#   ./run.sh --source clip.mp4    a recording instead of the webcam
#   ./run.sh --headless --fast    no window, flat out
#
# First run: downloads the MediaPipe models if they are missing.

set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. See https://docs.astral.sh/uv/ (or: curl -LsSf https://astral.sh/uv/install.sh | sh)" >&2
    exit 1
fi

mkdir -p models
if [ ! -f models/hand_landmarker.task ]; then
    echo "downloading hand landmarker model..."
    curl -sSL -o models/hand_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi
if [ ! -f models/pose_landmarker_lite.task ]; then
    echo "downloading pose landmarker model..."
    curl -sSL -o models/pose_landmarker_lite.task \
        https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
fi
if [ ! -f models/face_landmarker.task ]; then
    echo "downloading face landmarker model..."
    curl -sSL -o models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
fi

uv sync --quiet

# Default: webcam with the nurse page. Any explicit flags replace the defaults.
if [ $# -eq 0 ]; then
    set -- --source webcam --web
fi

# Hide MediaPipe's startup chatter (absl logging) while keeping our own output.
exec uv run nurse24hr "$@" 2> >(grep -v -E '^(WARNING: All log messages|I0000|W0000|INFO: Created TensorFlow)' >&2)
