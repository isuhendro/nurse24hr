# nurse24hr

Camera-based monitor that alarms when a patient attempts to pull out a feeding tube.
Design: [stories/TS-001-initial-technical-spec.md](stories/TS-001-initial-technical-spec.md).

## Status

Step 1 of the build order is done: the pure state machine, a synthetic scenario
simulator, and the test suite. No camera code yet.

| Module | Purpose |
|--------|---------|
| `src/nurse24hr/features.py` | Per-frame `Features` record (distance, pinch, oscillation, radial velocity, tube displacement, confidence) |
| `src/nurse24hr/config.py` | `Thresholds` dataclass, loaded from `config.yaml` |
| `src/nurse24hr/events.py` | `EventMachine`: IDLE → WATCH → WARN → ALARM / FAULT, latching, suppression, low-confidence hold |
| `src/nurse24hr/simulate.py` | Scenario generator (idle, scratch, rub, grasp, pull, yank, caregiver, occlusion, dropout) and CLI |
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

## Next

Step 2: capture thread and debug overlay (webcam and file sources), then
perception producing `Features` from MediaPipe hands and face landmarks.
