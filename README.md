# nurse24hr

Camera-based monitor that alarms when a patient attempts to pull out a feeding tube.
Design: [stories/TS-001-initial-technical-spec.md](stories/TS-001-initial-technical-spec.md).

## Status

Steps 1 and 2 of the build order are done: the pure state machine with its
simulator and tests, and the capture pipeline with a live debug window.
Perception (hand and face landmarks) is step 3, so the running app currently
raises FAULT after five seconds: no perception means no confident view of the
patient, and the system fails loud by design.

| Module | Purpose |
|--------|---------|
| `src/nurse24hr/features.py` | Per-frame `Features` record (distance, pinch, oscillation, radial velocity, tube displacement, confidence) |
| `src/nurse24hr/config.py` | `Thresholds` dataclass, loaded from `config.yaml` |
| `src/nurse24hr/events.py` | `EventMachine`: IDLE → WATCH → WARN → ALARM / FAULT, latching, suppression, low-confidence hold |
| `src/nurse24hr/simulate.py` | Scenario generator (idle, scratch, rub, grasp, pull, yank, caregiver, occlusion, dropout) and CLI |
| `src/nurse24hr/capture.py` | Capture thread for webcam, file or RTSP into a size-one latest-frame slot; file clock for deterministic replay |
| `src/nurse24hr/perception.py` | `Perceiver` interface plus the zero-confidence placeholder |
| `src/nurse24hr/overlay.py` | Debug HUD: state banner, ROI, metrics, timers |
| `src/nurse24hr/app.py` | Run loop and `nurse24hr` CLI |
| `tests/` | Unit tests for the machine plus every scenario across many seeds |

## Setup

```bash
uv sync
uv run pytest
```

## Try the simulator

```bash
# outcome histogram for every scenario across 100 random seeds
uv run python -m nurse24hr.simulate --n 100

# event timeline for one scenario
uv run python -m nurse24hr.simulate --scenario grasp_and_pull --seed 3
uv run python -m nurse24hr.simulate --scenario nose_scratch_short --seed 3

# with alternative thresholds
uv run python -m nurse24hr.simulate --config config.yaml
```

## Run the pipeline

```bash
# live webcam with the debug window (macOS will ask for camera permission)
uv run nurse24hr --source webcam

# a recorded session, paced to real time, with a bed ROI
uv run nurse24hr --source session.mp4 --roi 100,80,540,400

# as fast as possible, no window, events on stdout
uv run nurse24hr --source session.mp4 --headless --fast
```

Keys in the window: `q` quit, `a` acknowledge a latched ALARM or FAULT, space pause.

## Next

Step 3: perception producing `Features` from MediaPipe hands and face landmarks,
with Parquet logging of every frame.
