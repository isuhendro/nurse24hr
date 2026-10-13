# TS-001 — Feeding-Tube Pull-Out Monitor: Initial Technical Specification

| Field        | Value                                   |
|--------------|-----------------------------------------|
| Status       | Draft                                   |
| Version      | 0.1                                     |
| Date         | 2026-10-11                              |
| Owner        | isuhendro                               |
| Scope        | Mac prototype (Phase 1) + target architecture (Phase 2/3) |

---

## 1. Problem statement

Patients with nasogastric (NG) or gastrostomy (PEG) feeding tubes sometimes pull them out, most often at night when nursing observation is thinnest. Re-insertion is painful, carries aspiration risk, and consumes staff time.

We want a **camera-based monitor** that watches the patient and **sounds an alarm when the patient attempts to remove the tube**, early enough for a nurse to intervene (target: alarm within 3 s of a grasp, before the tube is displaced).

This is a **supplementary alert**, not a replacement for nursing observation.

## 2. Goals and non-goals

### Goals

- G1. Detect a hand approaching, grasping, and pulling the feeding tube or its insertion point.
- G2. Alarm locally (audible) and remotely (nurse station / phone) within 3 s of a confirmed grasp.
- G3. Very high recall. Missing a real pull-out is the worst outcome.
- G4. Tolerable false-alarm rate so staff do not learn to ignore it (alarm fatigue).
- G5. Fail loudly. Camera loss, frozen frames, inference stall, or occlusion must raise a fault, never silence.
- G6. Keep video on-device. Only events, metadata, and short clips leave the box, encrypted.
- G7. Every alarm is reviewable: a clip plus a structured log entry.

### Non-goals (for now)

- Replacing nursing observation or restraints policy.
- Multi-bed monitoring from one camera.
- Clinical certification. Regulatory path (e.g. HSA Singapore) is tracked as a risk, not a deliverable of this spec.
- Diagnosing anything else about the patient (falls, seizures, vitals).

## 3. Key design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Classical CV in the real-time loop**, not an LLM / VLM | A grab-and-pull takes 1–3 s. VLM inference is seconds per clip, non-deterministic, costly at 10–15 fps 24/7, and cloud-dependent. Pose + nano detector runs on a ~$300 edge board. |
| D2 | **VLM only as a second stage** | Review flagged WARN clips to suppress or confirm escalation, pre-label recorded clips for training, generate incident summaries. It must never be able to delay or block the primary alarm. |
| D3 | **Temporal state machine with hysteresis** | Single-frame detections are noisy. Dwell time, pull motion, and hysteresis are what keep false alarms down. |
| D4 | **Alarms latch until a human acknowledges** | A self-clearing alarm can be missed. |
| D5 | **Low confidence holds state, sustained low confidence raises a fault** | Blanket over the face must not silently reset timers or stop monitoring. |
| D6 | **Edge compute + thin nurse client** (target) | Camera and alarm logic independent of any phone's state. One edge box can serve several beds. |
| D7 | **Prototype in Python on macOS first** | Fastest iteration on thresholds. The state machine ports to Swift or stays in Python on the edge box unchanged. |
| D8 | **Pure, I/O-free state machine module** | Unit-testable by replaying logged features. Threshold sweeps run offline in seconds. |
| D9 | **No training data is assumed at the start.** v1 uses only pretrained models; all task-specific data is self-staged or synthetic (see §7) | We have zero recorded patient footage. Pretrained hand and face models get the state machine working on day one; the tube detector is trained later from staged and synthetic data. |
| D10 | **Face scratching is a first-class hard negative**, not a tube event | Patients scratch their nose, cheek and mouth constantly. Proximity alone cannot be the trigger; grasp, dwell, pull motion, and tube displacement are what separate a pull from a scratch (see §5.3). |
| D11 | **Python for the pipeline, on Mac and on the edge box.** Swift or TypeScript only for the nurse client | The ML ecosystem is Python; the hot path runs in compiled libraries anyway; Jetson ships Python + TensorRT. See §10 for where this stops being true. |

## 4. System architecture

### 4.1 Target deployment (Phase 2/3)

```
IR camera (RTSP) ──► Edge box (Jetson / Mac mini)                  ──► Nurse client (iPhone / web)
                     ├─ Capture → Inference → Events → Alarm            ├─ push notification
                     ├─ Local buzzer / speaker (works offline)          ├─ live view
                     ├─ Clip recorder + audit log (on-device)           └─ acknowledge button
                     └─ Watchdog + heartbeat        ──MQTT / webhook──►  Nurse station
```

Options considered:

| Option | Setup | Verdict |
|--------|-------|---------|
| A. iPhone as camera + compute | One phone on a stand, Vision + CoreML | Quickest prototype. Not for deployment (foreground-only camera, thermal, battery, iOS updates). |
| **B. IP/IR camera → edge box → nurse app** | Pipeline on edge box, phone is alert/ack client | **Chosen target.** |
| C. iPhone decodes RTSP itself | Camera → phone | Only if an edge box is impossible. Needs third-party RTSP decoder, foreground-only. |

### 4.2 Phase 1 prototype (macOS, single process)

```
CaptureThread ──► latest-frame slot (size 1, drops stale frames)
                        │
InferenceThread: pose → tube detector → Features
                        │  Features via queue
EventThread: state machine → Events
                        │
AlarmThread: sound + MQTT publish + latch / ack
        └► Recorder: 10 s ring buffer → clip on WARN / ALARM
Watchdog: heartbeats from every thread → fault alarm
```

One process, four worker threads plus the main thread (macOS requires `cv2.imshow` on the main thread). Queues between stages. The latest-frame slot is size 1 and always overwritten so latency never accumulates.

## 5. Component specification

### 5.1 Capture (`capture.py`)

- Source selectable by flag: `--source webcam | <file.mp4> | rtsp://…`. Same code path for live and recorded input.
- Target 10–15 fps. Each frame stamped with monotonic time.
- Writes into a single-slot buffer. Stale frames are dropped, never queued.
- Bed ROI defined once at setup (config). Pixels outside the ROI are ignored downstream.
- Emits heartbeat to watchdog on every frame.

### 5.2 Perception (`perception.py`)

Inputs: frame. Output: one `Features` record per frame.

Models:

- **Hand / wrist pose**: MediaPipe Hands (and Pose for body context). CPU on Apple Silicon is sufficient.
- **Face landmarks**: MediaPipe Face. In v1 the nasal tube insertion point is approximated from face landmarks (nostril region). Face width is the normalization scale.
- **Tube detector (v2)**: Ultralytics YOLO-nano fine-tuned on tube + insertion point. Runs on `mps`. Replaces the face-landmark proxy when available. Exportable to CoreML for the iPhone path.

`Features` schema:

| Field | Type | Notes |
|-------|------|-------|
| `t` | float | monotonic timestamp (s) |
| `wrist_l`, `wrist_r` | (x, y) or null | pixel coords |
| `tube_pt` | (x, y) or null | insertion point or tube anchor |
| `face_w` | float or null | pixel width, normalization scale |
| `d` | float or null | `min(dist(wrist_l, tube_pt), dist(wrist_r, tube_pt)) / face_w`, smoothed |
| `v_wrist` | float or null | velocity of the closer wrist, normalized by `face_w` per second |
| `v_radial` | float or null | component of wrist velocity **away from** `tube_pt` (positive = moving away). Pull signature. |
| `pinch` | float or null | thumb-tip to index-tip distance / `face_w` for the closer hand. Small value = pinch grip. Scratching uses extended fingers; pulling uses a pinch. |
| `osc` | float or null | the hand's own positional scatter over the last 1 s (std of the grip point in face-widths), not a distance to one anchor, so a cheek scratch looks as busy as a nose scratch. High = scratching / rubbing. Low = static grasp. |
| `tube_disp` | float or null | displacement of `tube_pt` from its rolling baseline, normalized. Scratching does not move the tube; pulling does. |
| `d_tape` | float or null | grip-point distance to the nearest **tape anchor** on the cheek (face landmarks 205 / 425), normalised. Picking at the securing tape is the usual precursor to a pull. |
| `n_persons` | int | people detected inside bed ROI |
| `conf` | float | 0–1 overall confidence of this frame |

Smoothing: 5-frame median or EMA on `d` to suppress landmark jitter.

**Pose first, face mesh second.** The face mesh alone is not a reliable "can I see the patient" signal: on the first real bedside clip (2026-10-11, ward lighting, patient seen from the side with tube and tape on the face) it failed for 10 s and produced a FAULT while the patient was plainly in view. The pose landmarker (full-range body model) found the patient in 114 of 118 frames of the same clip. Perception now runs pose on the bed ROI first: it answers patient-present, seeds the zoom crop with the head position, and when the face mesh fails it supplies coarse anchors (nose as tube point, nose–ear midpoints as tape points, ear-to-ear distance as face width) at confidence 0.7, plus wrist / index / thumb as coarse hands. FAULT therefore means nobody is visible at all. Cost: ~50 ms per frame extra on the Intel Mac (`pose_every_n` can halve it). Replaying that clip after the change: 117/118 confident frames, no FAULT.

A second bedside clip (patient wearing a nebuliser mask, head ~6 % of frame width, handheld camera) still faulted: pose on the full frame found the patient in 1 of 6 sampled gap frames. Larger pose models (full, heavy) did not help. What did: running pose on a wide crop (~10 head-widths, min 512 px) around the last known head with the detection threshold lowered to 0.25, which found 5 of 6. Pose now has its own tracking crop with a full-ROI retry. Replay after the change: unseen frames fell from ~110 to 61 of 153, longest unseen stretch 1.9 s (below `T_FAULT`), no FAULT; the remaining unseen frames include the camera physically swinging away. The lite pose model is kept.

**Range: tracking crop.** MediaPipe's face detector is short-range; a face that is a small patch of a wide frame is not found (observed live 2026-10-11 when the subject stepped back). Perception therefore runs on a crop: the static bed ROI when configured, and on top of that a tracking window of ~3 face-widths around the last seen face (digital zoom). If the face is lost inside the window the same frame is retried on the full ROI; after 1 s of misses the window is dropped. All landmarks are mapped back to full-frame pixels so the overlay, ROI and clips agree. For a ceiling camera over a whole bed the static ROI remains the primary tool; the tracking crop is what makes a 1280×720 frame usable at 2–3 m.

**"Hand not seen" is not "hand away".** On the bedside clips the hand model drops out constantly at distance, and reporting a vanished hand as far away cut every reach short before the dwell could complete (14 WATCH episodes in 5 min, none reaching WARN, during movement the owner judged dangerous, 2026-10-11). A hand that vanishes now holds its last distance for 2 s at confidence 0.6 before being treated as away. Coarse hands and anchors from the pose model report `pinch` and `osc` as unknown (None), so the missing-evidence rule lets dwell alone raise WARN. Dwell timers hold rather than reset while the hand hovers between the grasp and watch bands.

**Coarse hands alarm on nothing.** Bedside session 2026-10-11 14:00 (visitor leaning over the bed, patient's hands under the blanket or clasped at the chin) produced four ALARMs, all "grasp dwell" with no pinch / osc data. The hand model fired on 3 % of frames; every other frame used body-model wrists, which sat on the patient's face while the real hands were hidden (visibility just over the 0.3 cutoff) or 40 px off when they were clasped under the chin. The per-frame person count flipped between 1 and 2 some 230 times in three minutes, and suppression only gated the frame it was checked on while the dwell timers kept running, so the first single-person frame fired (audit: "ALARM suppressed, 2 persons" at 777088.89, ALARM at 777089.02). Three changes: the WARN-dwell ALARM now needs real grasp evidence seen at least once during the WARN (`missing_evidence_alarms`, default false; dwell alone still raises WARN); after a second person was seen the count must read one for `t_single_person` (1.5 s) before escalation and the WARN clock does not advance while suppressed; the WARN clock resets once the hand has been outside the watch band for `t_warn_reset` (1 s). Body-model hands need visibility 0.5 (`pose_hand_min_vis`). Replaying all nine logged runs: the 14:00 run goes from 4 ALARMs to 0 and 9 WARNs to 3; the morning runs keep their pinch-backed ALARMs (4 of 5) and lose 27 of 37 WARNs. Open: the hand-model hit rate itself (3 to 20 % at this distance) is the real limiter and has not been addressed.

**Whose face is it.** Live session 2026-10-12 00:29: a visitor's back and head filled the frame, the face / pose models tracked the visitor as the patient, body-model wrists dwelt at the visitor's nose and the machine alarmed ("grasp dwell 3.7s") while the patient was not in the picture at all. Nothing distinguished faces. Perception now keeps a size baseline of the accepted face-mesh width and rejects a face more than `face_scale_max` (1.6) times larger or smaller: a visitor leaning in is two to four times the patient's width (replaying the clip: 230-300 px against a 105 px patient). Among accepted faces the one nearest the tracked head wins. The body model's ear-to-ear width is not usable as a cue (2 to 77 px on one person in a profile view), so a pose is rejected by position instead: its nose inside a mesh face rejected within the last second (`reject_memory_s`). A masked visitor the mesh never sees is therefore still not caught. A rejected face still counts as a person, and while one is in view the deriver does not measure hands against the remembered anchor (those are the visitor's hands). With no accepted face for `face_relearn_s` (30 s) the baseline is dropped so a moved camera recovers. A visitor hiding the patient therefore ends in FAULT, which is an alarm, not a silence.

**Hand model at distance.** Experiment on the 2026-10-11 clips (patient face ~100 px wide in 1280x720, ward light): the MediaPipe hand landmarker found a hand in 9-11 % of frames of the clasped-hands clip and 3-9 % of the blanket clip, whichever way it was run: VIDEO or IMAGE mode, face crop at 3x or 5x, full frame, detection confidence 0.4 or 0.2, or a tight 1.5-3x crop centred on each body-model wrist (the usual pose-to-hand trick). Cropping is not the limiter; the model is. Until a better hand detector is in (RTMPose / YOLO-pose whole-body, or a hand detector fine-tuned on bedside views), coarse body-model hands carry the proximity logic and WARN is the ceiling without pinch / osc evidence.

**Repeated reaching.** A patient who keeps bringing a hand to the face is agitated and at risk even if no single reach lasts 1.5 s. Three entries into WATCH within 60 s raise WARN ("repeated reaching"); `reach_count_warn` / `reach_window_s`, 0 disables.

**Hand-over-face occlusion.** A hand at the nose hides the landmarks the face model needs, so the grasp we most care about can blind the tube anchor. When the face drops out but a hand is still visible within 10 s of the last good face, perception keeps measuring against the remembered anchor at confidence 0.6 (above `CONF_MIN`), so dwell keeps accumulating. With no hand visible either, confidence is 0 and FAULT follows as before. Observed on the first live webcam run (2026-10-11).

**v1 insertion-point proxy.** Without a tube detector, `tube_pt` is the nostril region from face landmarks for an NG tube. Note that a nose scratch then lands right on `tube_pt`, so §5.3 must not alarm on proximity and dwell alone.

### 5.3 Event logic (`events.py`) — pure, no I/O

States: `IDLE`, `WATCH`, `WARN`, `ALARM`, `FAULT`.

```
IDLE  → WATCH : d < D_WATCH
WATCH → WARN  : d < D_GRASP for > T_GRASP            (grasp)
                AND grasp_evidence
                    = osc < OSC_MAX  (hand is still, not rubbing)
                      OR pinch < PINCH_MAX  (pinch grip)
WARN  → ALARM : dwell in WARN > T_ALARM
                OR pull motion (v_radial > V_PULL while pinch held
                                OR tube_disp rising at the same time)
                OR tube_disp > DISP_MAX
any   → IDLE  : d > D_RELEASE for > T_RELEASE         (hysteresis)
ALARM          : latched; only an explicit ack returns to IDLE
```

**Tape on the cheek.** The tube is secured with tape on the cheek, and patients pick at it before pulling. A still or pinched hand dwelling at `d_tape < D_GRASP` for `T_GRASP` raises WARN ("hand at tape"). It never reaches ALARM on its own: WARN dwell and pull motion are tied to the tube anchor, and pull motion is only armed within `PULL_ARM_S` of the hand being at the tube, so a pinched hand leaving the cheek is not a pull. Hysteresis back to IDLE requires the hand to be away from both anchors. Added 2026-10-11 after the first live test.

**Scratch vs pull.** A face scratch and a tube grasp both put the hand at `tube_pt`. They differ in three measurable ways, and the machine uses all three:

| Signal | Scratch / rub | Grasp and pull |
|--------|---------------|----------------|
| Dwell at `d < D_GRASP` | short, 0.3–1.5 s, then leaves | sustained > 1.5 s |
| Hand motion | high-frequency oscillation (`osc` high), fingers extended (`pinch` large) | still hand, pinch grip (`pinch` small), then one radial move away (`v_radial` > 0) |
| Tube | `tube_disp` ≈ 0 | `tube_disp` grows with the hand |

Expected behaviour: a scratch reaches WATCH, occasionally WARN if the patient rubs for a long time, and never ALARM unless the tube actually moves. WARN plays a **soft, single bedside chime** (`sounds/warn.wav`, not looped) plus the station notification; set `alarm.warn_sound: null` for station-only if the false-WARN rate turns out to disturb patients. (Changed 2026-10-11 from station-only at the owner's request during live testing.)

Default thresholds (config, to be tuned on recorded sessions):

| Param | Default | Meaning |
|-------|---------|---------|
| `D_WATCH` | 0.6 | face-widths |
| `D_GRASP` | 0.3 | face-widths |
| `D_RELEASE` | 0.8 | face-widths |
| `T_GRASP` | 1.5 s | dwell before WARN |
| `T_ALARM` | 3.0 s | dwell before ALARM |
| `T_RELEASE` | 2.0 s | hand away before reset |
| `V_PULL` | TBD | normalized radial wrist speed away from tube |
| `DISP_MAX` | TBD | tube displacement from baseline |
| `OSC_MAX` | TBD | above this the hand is rubbing / scratching, not grasping |
| `PINCH_MAX` | TBD | below this the hand is in a pinch grip |
| `CONF_MIN` | 0.5 | below this a frame is "low confidence" |
| `T_FAULT` | 5.0 s | sustained low confidence → FAULT |

Rules:

- **Low-confidence frames hold state.** Timers are neither advanced nor reset. Sustained low confidence for `T_FAULT` enters `FAULT` ("can't see patient"). FAULT is an alarm, not a silence.
- **Suppression**: if `n_persons > 1` inside the bed ROI, transitions into WARN/ALARM are suppressed (a caregiver adjusting the tube is not an event). The suppression itself is logged.
- **WARN dwell counts time on the tube only** (`d < D_GRASP`). A hand hovering between the grasp and release bands neither advances nor resets it. Otherwise a grasp-and-release tips into ALARM during the release hysteresis window.
- **Tube displacement alarms from any non-latched state**, including IDLE, since the tube moving away from baseline is strong evidence regardless of whether a hand was tracked.
- Per-frame `dt` is capped at `MAX_DT` (0.5 s) so a stalled pipeline never jumps a timer.
- The module exposes `step(features) -> events` and holds all timers internally. It must be deterministic given the same feature sequence.
- **Ack resets the tube baseline.** After a nurse acknowledges, the detector baseline restarts (the tube has been re-seated); otherwise a still-displaced tube re-alarms immediately. Replay emulates this by rebasing `tube_disp` at ack time.
- Implemented in `src/nurse24hr/events.py`; scenarios in `src/nurse24hr/simulate.py`; tests in `tests/`. Known v1 gap, recorded as an expected failure in the suite: a sub-1.5 s yank without a tube detector is indistinguishable from a brief touch.

### 5.4 Alarm manager (`alarm.py`)

| Level | Local | Remote | Escalation |
|-------|-------|--------|------------|
| WARN  | soft single chime (config: `warn_sound`, null = silent) | nurse-station notification | none |
| ALARM | loud continuous alarm | push to nurse call / phone | re-notify if no ack in 30 s, then escalate to secondary |
| FAULT | distinct fault tone | notification with fault reason | same as ALARM |

- Local sound driven directly by the device (`afplay` or `pygame`), so it works with the network down.
- Remote via MQTT (`paho-mqtt`) or webhook.
- ALARM and FAULT **latch** until acknowledged by keypress (prototype), physical button, or ack message. Never self-clear.
- All state changes and acks are appended to the audit log.

### 5.5 Recorder (`recorder.py`)

- Keeps a 10 s rolling buffer of frames.
- On WARN or ALARM, writes pre-event buffer plus 10 s post-event to an MP4 clip, named by event ID.
- Clips stay on-device. They are the review and training set.

### 5.6 Watchdog (`watchdog.py`)

Raises FAULT if any of:

- No frame from capture for > 2 s (camera loss).
- Frame content unchanged for > 5 s (frozen stream).
- No `Features` output for > 2 s (inference stall).
- No face detected for > `T_FAULT` (occlusion / patient out of view).

Sends a heartbeat to the nurse station every 10 s. The station alarms if heartbeats stop (covers process or box death).

### 5.7 Config (`config.yaml`)

Camera source, bed ROI, all thresholds above, MQTT broker, sound file paths, clip directory. No thresholds hard-coded in Python.

## 6. Edge-to-client protocol (sketch, Phase 2)

MQTT topics, per bed:

| Topic | Direction | Payload |
|-------|-----------|---------|
| `nurse24hr/<bed>/state` | edge → client | `{state, t, d, conf}` at 1 Hz |
| `nurse24hr/<bed>/event` | edge → client | `{event_id, level, t, clip_url, summary}` |
| `nurse24hr/<bed>/heartbeat` | edge → station | `{t, fps, model_ok}` every 10 s |
| `nurse24hr/<bed>/ack` | client → edge | `{event_id, user, t}` |

QoS 1 for events and acks. Clips served over local HTTPS from the edge box, not pushed through MQTT.

## 7. Data strategy (no real data assumed)

We start with **zero recorded patient footage** and no labels. The plan is built so that nothing blocks on real data, and so that real data, once it exists, slots in without redesign.

### 7.1 What needs data and what does not

| Component | Needs task-specific data? | Source |
|-----------|---------------------------|--------|
| Hand / wrist pose (MediaPipe) | No, pretrained | — |
| Face landmarks (MediaPipe) | No, pretrained | — |
| State machine thresholds | Tuning only, no training | synthetic feature sequences (7.2) + staged sessions (7.3) |
| Evaluation set (recall, false alarms) | Labels only | staged sessions (7.3), labels come free from the recording script |
| Tube detector (YOLO-nano, v2) | Yes, bounding boxes | staged frames (7.3) + auto-labeling (7.4) + augmentation (7.5) |
| Action classifier (optional v3) | Yes, clip labels | staged sessions (7.3) + pilot clips (7.6) |

v1 therefore ships with **no trained model of our own**.

### 7.2 Synthetic feature sequences (day one, no camera)

A small simulator `simulate.py` generates `Features` time series directly, bypassing video. Each scenario is a parametrised template with random jitter:

| Scenario | `d(t)` | `osc` | `pinch` | `v_radial` | `tube_disp` | Expected final state |
|----------|--------|-------|---------|------------|-------------|----------------------|
| idle sleep | 1.5–3.0 | low | large | 0 | 0 | IDLE |
| nose scratch (short) | dips to 0.1–0.3 for 0.3–1.2 s | high | large | small ± | 0 | WATCH |
| long face rub | 0.2–0.4 for 2–6 s | high | large | small ± | 0 | WATCH, at most WARN |
| wipe mouth / adjust glasses | 0.2–0.5 for 0.5–2 s | medium | medium | ± | 0 | WATCH |
| grasp, hold, release | 0.1–0.2 for 2–4 s | low | small | 0 then + | 0 | WARN |
| grasp and pull | 0.1–0.2 for 1.5–3 s, then rising | low | small | + | rising | ALARM |
| fast yank | 0.1 for 0.5 s, then rising fast | low | small | large + | rising | ALARM |
| caregiver adjust (`n_persons = 2`) | 0.1–0.2 for 5–20 s | low | small | ± | small | WATCH, suppressed |
| blanket over face | `conf` drops to 0.1 for > 5 s | — | — | — | — | FAULT |
| tape pick (`d_tape` 0.1, `d` 0.7) | 0.7 | low | small | ± | 0 | WARN, never ALARM |
| cheek scratch at tape | 0.7, `d_tape` oscillating | high | large | ± | 0 | WATCH |
| tape peel then tube pull | tape first, then 0.15, then rising | low | small | + | rising | WARN then ALARM |
| landmark dropout | `conf` flickers 0.2–0.9 at 2 Hz during a grasp | low | small | 0 | 0 | WARN (state must hold) |

Thousands of sequences run through `events.py` in seconds. This is the unit-test suite for the state machine and the first threshold sweep. It cannot tell us real-world recall, but it guarantees the logic is correct and that scratches do not alarm by construction.

### 7.3 Staged sessions (week one, webcam or iPhone)

Volunteers wear a **stand-in tube**: 3–4 mm silicone tubing taped from nostril to cheek and ear with surgical tape, as an NG tube is. Record with the Mac webcam and iPhone (Continuity Camera) at 15 fps.

Recording protocol. Each session follows a timed script read aloud or shown on screen, so the **action label and its onset time come from the script**, not from manual annotation:

| Block | Action | Reps | Label |
|-------|--------|------|-------|
| 1 | lie still, eyes closed, small head movements | 2 min | `idle` |
| 2 | scratch nose, cheek, chin, eyebrow | 10 each | `scratch` |
| 3 | rub face with palm, wipe mouth | 10 | `rub` |
| 4 | adjust pillow, pull blanket up, roll over | 10 | `other` |
| 5 | touch the tube without gripping | 10 | `touch` |
| 10 | pick at / peel the tape on the cheek | 10 | `tape` |
| 6 | pinch and hold the tube 2–4 s, release | 10 | `grasp` |
| 7 | pinch and pull the tube out (tape re-applied each time) | 10 | `pull` |
| 8 | second person adjusts the tube | 5 | `caregiver` |
| 9 | blanket pulled over face for 10 s | 3 | `occlusion` |

Variation across sessions: at least 5 volunteers, both hands, lying on back and on side, overhead and bedside camera angle, bright / dim / near-dark lighting, with and without glasses or beard, tube taped on left and right. Target: 20–30 sessions, roughly 3–5 hours of video, a few hundred `pull` events and a few thousand `scratch` and `rub` events.

Output of each session: raw video, `Features` Parquet from `perception.py`, and a CSV of script-derived labels `(t_start, t_end, label)`. This is both the **evaluation set** for §9 metrics and the raw material for the detector.

### 7.4 Auto-labeling the tube for the detector

Manual boxing of thousands of frames is the step we want to avoid. Bootstrap:

1. **Sample** 1 frame per second from staged sessions, more densely inside `grasp`, `pull`, and `caregiver` blocks (tube is partially occluded or displaced there, which is where the detector must work).
2. **Seed** the insertion point from face landmarks (nostril) automatically.
3. **Segment** the tube with a promptable segmenter (Segment Anything or similar) using the nostril seed plus a point along the cheek. Convert the mask to an oriented box or polyline and a tight box for the insertion point.
4. **Optionally** have a VLM confirm each sampled frame ("is the tube visible and correctly boxed?") to flag bad masks.
5. **Human review** in a labeling tool (CVAT, Label Studio, or Roboflow): accept / fix / reject. Reviewing is roughly 10× faster than drawing.

Target: 2,000–5,000 reviewed frames before the first YOLO-nano training run.

### 7.5 Synthetic augmentation for the detector

To widen coverage beyond our volunteers and lighting:

- **Copy-paste**: cut tube segments from labeled frames and paste them onto unlabeled face images (public face datasets or more volunteers without a tube), anchored at the nostril landmark, with random curvature, thickness, colour, and tape patches. Labels come free.
- **IR domain simulation**: convert to grayscale, boost contrast, add sensor noise and slight blur, and darken backgrounds, to approximate what the night IR camera will see before we have one.
- **Standard augmentation** (Albumentations): random crop, rotation, motion blur, brightness, occlusion patches (simulated blanket, hand).
- **Generative inpainting** (stretch): use a diffusion inpainting model to add a tube to face photos. Useful for diversity, but domain gap is real, so keep its share under 20 % and validate on staged frames only.

The validation set is always real staged frames, never synthetic.

### 7.6 Data flywheel after the pilot

Once a prototype runs on a real bed (with consent and approvals), every WARN, ALARM, FAULT, and suppressed event produces a clip. Each clip gets a VLM pre-label plus a nurse's one-tap verdict on acknowledgement (`real pull` / `scratch` / `caregiver` / `other`). These are the first real labels and feed back into thresholds, the detector, and the optional action classifier.

### 7.7 What synthetic data will not tell us

Real patients differ from volunteers: tremor, restless sleep, confusion, restraints, oxygen tubing nearby, real IR optics and overhead mounting. Treat all pre-pilot metrics as logic validation, not clinical performance.

## 8. Dev and test tooling

- **Feature logging**: every `Features` record written to Parquet per session.
- **Offline replay**: `replay.py <features.parquet> --config x.yaml` runs `events.py` without video and prints the event timeline. Threshold sweeps are a loop over configs.
- **Debug overlay**: landmarks, `d`, state, timers, and confidence drawn on the live frame.
- **Synthetic sequence suite**: `simulate.py` scenarios from §7.2 run as pytest cases against `events.py`. Every scratch / rub scenario asserts `state != ALARM`.
- **Labeled test set**: staged sessions from §7.3 with script-derived labels. Scratch and rub are the primary hard negatives and are over-represented on purpose.
- **Metrics**: recall on `pull`, false alarms per hour overall **and per negative class** (`scratch`, `rub`, `other`, `caregiver`), time-to-alarm (grasp onset → ALARM), fault detection latency.
- **Acceptance gate for v1** (on staged data): recall on `pull` ≥ 0.95, zero ALARMs on `scratch` / `rub` blocks, time-to-alarm ≤ 3 s median.

## 9. Build order (Phase 1)

1. ~~`events.py` + `simulate.py` + pytest suite (§7.2). No camera needed. Scratch-vs-pull logic proven here first.~~ **Done 2026-10-11.**
2. ~~Capture + debug overlay, webcam and file sources.~~ **Done 2026-10-11.** `capture.py`, `overlay.py`, `app.py`, `Perceiver` interface with a zero-confidence placeholder so the pipeline fails loud (FAULT) until step 3 lands.
3. ~~Perception → `Features` (incl. `pinch`, `osc`, `v_radial`), Parquet logging.~~ **Done 2026-10-11.** `mp_perceiver.py` (MediaPipe Tasks hands + face, VIDEO mode), `features_log.py`. Deviation: `d` uses the grip point (midpoint of thumb and index tips) rather than the wrist, so `d` and `pinch` refer to the same location; wrists are still logged.
4. Staged recording sessions (§7.3); replay their features through `events.py`; first real threshold tuning. **Tooling done 2026-10-11** (`session.py` protocol runner, `replay.py` metrics and sweeps). **Recording and tuning still require volunteers and a camera.**
5. ~~Alarm manager + watchdog.~~ **Done 2026-10-11.** `alarm.py`, `watchdog.py`. Watchdog runs in its own thread and is edge-triggered; the separate-process variant is still open for Phase 3.
6. Auto-label pipeline (§7.4) + augmentation (§7.5) → train YOLO-nano tube detector → replaces face-landmark proxy, enables `tube_disp`. **Scaffolding done 2026-10-11** (`data/sample_frames.py`, `data/seed_labels.py` with a SAM hook, `data/augment.py`, `data/train_tube.py`, `tube_detector.py` wrapper). **Training requires sampled frames from step 4.**
7. ~~Clip recorder.~~ **Done 2026-10-11.** `recorder.py`, one clip per incident with a JSON sidecar.
8. ~~MQTT publish + minimal nurse client (web page or iPhone).~~ **Done 2026-10-11.** `mqtt.py` (topics per §6, ack subscription), `web.py` (stdlib HTTP + SSE page with ACK and beep).

## 10. Platform notes

**Implementation language**

Python 3.12 for everything in this spec through Phase 2.

- Why: MediaPipe, Ultralytics, PyTorch, Albumentations, Segment Anything, Parquet tooling and the labeling tools are all Python-first. Frame decode and inference run inside compiled C++ / CUDA libraries, so Python only glues stages at 10–15 fps. Iteration speed on untuned thresholds matters more than raw speed right now. Jetson ships Python, TensorRT and OpenCV, and a Python service under `systemd` is the normal deployment there. One codebase runs on Mac and edge by swapping camera source and inference backend.
- How: `uv` for dependencies, `src/` layout, type hints throughout, `dataclasses` for `Features` and thresholds, `pytest` for the simulator suite. Real threads (not asyncio) for capture and inference; OpenCV and the inference libraries release the GIL. The watchdog runs as a **separate process** (or the `systemd` watchdog) so it survives the pipeline process dying.
- Where Python is the wrong answer:
  - Nurse client: Swift (Vision / CoreML if the phone ever infers) or a small TypeScript web app for alert, ack and live view. The web app avoids App Store review.
  - Hard real-time or microcontroller-class hardware: port the pipeline to C++ or Rust. The pure state machine is small enough to port in a day and the Python version with its test suite becomes the reference implementation.

**macOS prototype**

- `cv2.imshow` must run on the main thread. Run headless when not debugging.
- Grant camera permission to Terminal / IDE.
- `mps` backend for YOLO; MediaPipe on CPU.
- Intel Macs: MediaPipe pinned to 0.10.21 (last x86_64 wheel), which forces NumPy 1.x and pyarrow < 19. Measured ~60 ms per frame for hands + face on a blank 640×480 frame, right at the 15 fps budget; the capture slot drops frames if inference falls behind.
- iPhone via Continuity Camera is a drop-in better webcam.

**iPhone (if Option A is used for a demo)**

- Swift + AVFoundation → Vision (`VNDetectHumanHandPoseRequest`) → CoreML tube detector → ported state machine → `AVAudioPlayer`.
- Camera only runs in the foreground. Dedicated plugged-in phone, `isIdleTimerDisabled = true`, Guided Access. Critical-alert entitlement needs Apple approval.
- Not a deployment platform.

**Edge box (Phase 3)**

- Jetson or Mac mini, `systemd` / `launchd` service with auto-restart.
- Benchmark latency on the real board. Mac numbers do not transfer.
- Validate low light / IR, blanket occlusion, overhead camera angle, tremor and restless sleep on the real setup.

## 11. Risks

| Risk | Mitigation |
|------|------------|
| Alarm fatigue from false positives | Hysteresis, dwell timers, multi-person suppression, VLM second-stage filter, per-night FA tracking. |
| Face scratching / rubbing triggers alarms | Grasp evidence (`osc`, `pinch`) required for WARN; ALARM needs pull motion or tube displacement; WARN is silent at bedside in v1; scratch is the dominant class in the staged test set with a zero-ALARM gate. |
| No real training data | v1 uses pretrained models only; synthetic feature sequences, staged sessions with script-derived labels, auto-labeling, and augmentation (§7). Pilot clips become the first real labels. |
| Volunteer data does not transfer to real patients (tremor, confusion, IR optics) | Treat pre-pilot metrics as logic validation; re-tune thresholds on pilot clips before trusting any FA numbers. |
| Silent failure (camera dead, model stalled) | Watchdog + station-side heartbeat monitoring. FAULT is an alarm. |
| Occlusion (blanket, turned away) | FAULT after `T_FAULT`; IR camera placement; later depth sensor. |
| Regulatory classification as a medical device | Scope as supplementary alert; consult HSA pathway before clinical use. |
| Privacy | Video never leaves the box; clips encrypted at rest and in transit; retention policy TBD. |
| Mac results not representative of edge board | Benchmark on target early (Phase 2). |

## 12. Open questions

- NG vs PEG: does v1 support both insertion points, or NG only?
- Which IR camera model and mounting height for the pilot?
- Who acknowledges alarms and from where (station PC, phone, bedside button)?
- Clip retention period and who may view them.
- Is a VLM second stage in scope for Phase 2, and which model (on-device vs API)?
- Can we recruit 5+ volunteers for staged sessions, and do we need consent paperwork for storing their video?
- Is a nose scratch that lasts > 3 s acceptable as a WARN (station notification), or must it stay at WATCH?
- Do we need a public face dataset for copy-paste augmentation, and which licence is acceptable?
