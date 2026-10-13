# nurse24hr

Camera-based monitor that alarms when a patient attempts to pull out a feeding tube.
Design: [stories/TS-001-initial-technical-spec.md](stories/TS-001-initial-technical-spec.md).

> ## Warning: not a medical device. Use at your own risk.
>
> This is an experimental prototype. It has **not** been tested, validated,
> certified or approved as a medical device by any regulatory body, and it has
> not been evaluated in a clinical study. It is not a substitute for clinical
> observation, nursing care, or any approved patient-monitoring equipment.
>
> It will miss events and it will raise false alarms. On the first bedside
> sessions it alarmed on a visitor leaning over the bed and on hands hidden
> under a blanket, and its hand tracking sees the hand in roughly one frame in
> ten at ward distances. Treat every output as a hint for a human to check,
> never as a reason to look away from the patient.
>
> Use it only with the informed consent of the patient or their legal
> representative, in line with local law on video recording and patient
> privacy. The authors accept no liability for any harm, injury, loss or
> damage arising from its use. You use it entirely at your own risk.

## Status

All eight build-order steps of the Phase 1 prototype are implemented. What is
**not** done, because it needs people and a camera: recording staged sessions
(§7.3), tuning thresholds on them, and training the tube detector (§7.4–7.5).
The tooling for each of those exists and is tested.

| Module | Purpose |
|--------|---------|
| `features.py` | Per-frame `Features` record: distance, pinch, oscillation, radial velocity, tube displacement, confidence |
| `config.py` / `config.yaml` | `Thresholds` and every other setting, nothing hard-coded |
| `events.py` | Pure state machine IDLE → WATCH → WARN → ALARM / FAULT: latching, hysteresis, suppression, low-confidence hold |
| `simulate.py` | Synthetic scenario generator (idle, scratch, rub, grasp, pull, yank, caregiver, occlusion, dropout) |
| `capture.py` | Capture thread for webcam, file or RTSP into a size-one latest-frame slot; file clock for deterministic replay |
| `mp_perceiver.py` | MediaPipe hands + face landmarks → `Features` (v1 nostril proxy for the tube) |
| `tube_detector.py` | Optional YOLO tube detector wrapper that replaces the proxy and computes tube displacement |
| `alarm.py` | Alarm manager: levels, latching, sound, escalation, audit log |
| `watchdog.py` | Camera loss, frozen frames, inference stall, heartbeat |
| `recorder.py` | 10 s pre / 10 s post clips on WARN, ALARM, FAULT |
| `notify.py` / `mqtt.py` / `web.py` | Notifier protocol, MQTT publisher with ack subscription, nurse web page with live state and ACK |
| `features_log.py` / `replay.py` | Parquet logging; offline replay, metrics and threshold sweeps |
| `session.py` | Staged recording protocol runner that writes video, features and script-derived labels |
| `data/` | Frame sampling, seed labels, augmentation, YOLO training scaffold |

## Setup

```bash
uv sync
uv run pytest
```

### Models

MediaPipe model files are not committed. Download once:

```bash
mkdir -p models
curl -sSL -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -sSL -o models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
curl -sSL -o models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Note for Intel Macs: MediaPipe is pinned to 0.10.21, the last release with an
x86_64 macOS wheel. Apple Silicon and Linux can use newer versions.

## Run the monitor

```bash
# live webcam with the debug window (macOS asks for camera permission)
uv run nurse24hr --source webcam

# with the nurse page at http://127.0.0.1:8765
uv run nurse24hr --source webcam --web

# a recorded session, paced to real time, logging features for later tuning
uv run nurse24hr --source data/sessions/s01/video.mp4 --log out/s01.parquet

# flat out, no window, events on stdout, no clips
uv run nurse24hr --source session.mp4 --headless --fast --no-record
```

Keys in the window: `q` quit, `a` acknowledge a latched ALARM or FAULT, space pause.
With no perception (`--perceiver none`) the system raises FAULT after five
seconds: no confident view of the patient, fail loud by design.

## Tune without a camera

```bash
# every synthetic scenario across 100 seeds
uv run python -m nurse24hr.simulate --n 100

# one scenario's event timeline
uv run python -m nurse24hr.simulate --scenario nose_scratch_short --seed 3

# replay a logged session against its labels, then sweep thresholds
uv run nurse24hr-replay out/s01.parquet --labels data/sessions/s01/labels.csv
uv run nurse24hr-replay out/s01.parquet --labels data/sessions/s01/labels.csv \
    --sweep t_grasp=1.0,1.5,2.0 --sweep osc_max=0.04,0.05,0.06
```

## Record staged sessions (needs volunteers)

```bash
uv run nurse24hr-session --dry-run                       # print the protocol schedule
uv run nurse24hr-session --out data/sessions/s01 --subject alice --side left
uv run nurse24hr-session --out data/sessions/s02 --blocks 2,6,7   # subset of blocks
```

Each session writes `video.mp4`, `features.parquet`, `labels.csv` (from the
script, no manual annotation) and `session.json`.

## Train the tube detector (needs sessions)

```bash
uv run python -m nurse24hr.data.sample_frames data/sessions/s01/video.mp4 data/sessions/s01/labels.csv datasets/tube/raw/s01
# seed + review labels (see data/seed_labels.py docstring), split train/ and val/
uv run python -m nurse24hr.data.augment datasets/tube/train datasets/tube/train_aug --n 4
uv add ultralytics
uv run python -m nurse24hr.data.train_tube --data datasets/tube --epochs 50 --model yolo11n.pt --device mps
```

Then set `perception.tube_weights` in `config.yaml` to the trained weights.

## Sounds

`sounds/alarm.wav` (two-tone siren) and `sounds/fault.wav` (low triple beep)
are generated tones and are committed, as is `sounds/warn.wav`, a soft chime
played at the bedside on WARN. Set `alarm.warn_sound: null` in `config.yaml`
for a station-only WARN. Replace any of them with your own WAV files.
