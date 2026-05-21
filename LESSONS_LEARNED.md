# Lessons learned — DRL final project

Captured 2026-05-21 (Day 1) during the first multi-ws training run, while everything is fresh. Sections map to paper sections so writing the report later is straightforward.

---

## For §1 Introduction / motivation

- **Bouncy Basketball is a real, commercial Android game** (Dreamon Studios, ~2015) with non-trivial physics: gravity, bounce, charge-and-release shooting, two-player teams. It's not a toy environment.
- The game runs as a Unity application — closed-source — so engine-state access is impossible without reverse engineering. All observation and reward must come from pixels.
- This is **representative of a broader class of real-world RL problems** where the environment isn't designed for RL: ads interrupt gameplay, controls are constrained, no exposed score API, no engine hooks.

---

## For §3 Methodology — concrete design choices

### Action space: Discrete(2) {NO_PRESS, PRESS}, not the original 6-action plan

Our initial proposal had `{NOOP, LEFT, RIGHT, JUMP, CHARGE, RELEASE}` modeled on platformer-style fighting games. After two converging pieces of evidence — a web search confirming **"one-button control"** and direct in-game probing — we found:

- Player auto-walks toward the ball (low-level control is built-in)
- A single touch-and-hold anywhere on screen makes the player jump
- Longer hold = higher jump; releasing in the air = shoot
- A second CHI player on the agent's team responds to the SAME touch (2v2 game, one button controls both teammates)

**Implication for the paper:** the policy has only one bit of decision per step. The "right" representation is one that distinguishes "moments when pressing now will produce a goal" from all other moments. This makes object-position prediction (OCA) particularly well-aligned with the task: knowing the ball's trajectory tells you exactly when to press.

### OCA target source: OpenCV HSV thresholding, not YOLO

Original plan: YOLOv8n trained on ~200 hand-labeled emulator frames. Switched to label-free OpenCV because:

- **Shoes were the original discriminator** (per proposal). On the real APK, both teams wear dark/black shoes — they don't separate by hue.
- **Jersey color is the better discriminator** (CHI red vs HOU white). HSV thresholding plus connected-component analysis recovers per-team centroids in <1 ms with zero labeling cost.
- The CV pipeline (`vision.py`) cleanly populates a 10-dim OCA target: `[xb, yb, xp, yp, sin_p, cos_p, xo, yo, sin_o, cos_o]` — ball location + each team's centroid + each team's body orientation (the angle of the vector between the two players on that team).
- **Tradeoff to disclose:** the OCA target is the output of a CV pipeline, not engine state. We measure pipeline accuracy on a hand-labeled set (§5.5) and report it as a confound. Detector noise is small — most CV failures are missed-detections that the per-element mask cleanly handles.

### Reward signal: pixel-diff on CHI score ROI, not digit OCR

The proposal sketched digit-template OCR. We replaced it with pixel-diff on a fixed ROI containing the CHI score box, with a cooldown to suppress animation double-counting (`reward.py`, threshold=25, cooldown=8 frames). This:
- Avoids needing per-digit templates (which would require capturing examples of 0–9 from the scoreboard)
- Handles arbitrary 1- and 2-digit scores
- Tolerates JPEG-like compression artifacts in screencap
- Validated on multiple manual gameplay clips (>98% agreement with manual counting)

### Action duration ≈ one agent step

`frame_skip = 4` means a PRESS holds the on-screen touch for `4 × 33 ms ≈ 132 ms` of game time. Each agent step is therefore one "atomic decision" that affects game state for 132 ms. This roughly matches the natural duration of a basketball jump.

### OCA prediction horizon: K=4, not K=1

We initially had OCA predict pose at t+1 (the very next agent step). **Empirical result: aux loss collapses to ~0 within 6 PPO updates** because the encoder learns the trivial mapping "detect objects in current frame, output as next-frame coords." Once aux loss is 0, the OCA gradient is dead and the encoder is shaped only by PPO — same as baseline. **Fix: K=4 (predict pose ~528 ms ahead).** At that horizon, objects move 30–50 pixels (significant fraction of the 84×84 frame), trajectories bend with gravity, and prediction requires real forward-dynamics learning.

This is itself a **paper-worthy methodological lesson**: when designing an auxiliary task for representation learning, the task must be *non-trivial throughout training*, not just at initialization.

---

## For §4 Environment

### Operational gotchas (mostly belong in an appendix)

| Gotcha | Mitigation | Why this matters |
|---|---|---|
| **AdMob "Test Ad" overlay** covers most of the screen mid-gameplay, breaking the observation | `adb shell settings put global airplane_mode_on 1` (baked into the AVD snapshot) | Without this, training is impossible; the agent sees a blue/black ad screen for half the time |
| **Android "Viewing full screen" banner** appears on first launch and blocks UI | `settings put secure immersive_mode_confirmations confirmed` before snapshot save | One-time but easy to miss; would force re-initial-launch on every snapshot reset otherwise |
| **2v2 mode in Quick Game** — a single touch controls both of the agent's team players simultaneously | Documented in proposal; OCA target uses per-team centroid (not per-player) | Important for interpreting actions |
| **Default-difficulty CPU opponent is weak** — HOU scored 0 across all our trials | Reported as a limitation; argue relative ordering of {baseline, OCA, DPR} is what matters | Reviewers will ask if the agent only learned to beat a trivial opponent |
| **Per-user watchdog on shared workstation** kills sessions with too many threads | Capped OMP/BLAS/cv2/torch num_threads to 1; N=3 emulators per ws (not 4); 8s launch stagger between emulators | Project might have been infeasible on ws10 alone; distributed across ws1-ws10 to bypass per-machine cap |
| **Shared adb daemon collides** when multiple students use it | Use per-user `ANDROID_ADB_SERVER_PORT` (derived from UID) | Otherwise our `adb kill-server` would kill another student's emulator |

### Snapshot strategy

- Single AVD (`pixel5_api31`), multiple **named snapshots** — one per opposing team
- All N emulators on a ws boot from the same snapshot via `-read-only` (copy-on-write overlay) — saves disk and boot time
- Each `env.reset()` randomly picks a snapshot from `EmulatorEndpoint.snapshot_names` (currently `["clean_boot" (HOU), "clean_boot_lac" (LAC)]`) and reloads it — agent sees diverse opponents
- Snapshot reload takes ~3-4 s (vs ~5 min cold boot) → tractable episode resets

### Controlled experiment validated control

Before training, we ran a 45-second controlled comparison from the same snapshot:
- All NO_PRESS: CHI 3 / HOU 0
- All PRESS (sustained 132 ms holds): CHI 1 / HOU 0

This proved (1) presses affect outcomes, (2) the team we control is CHI (not HOU), and (3) the "do nothing" policy already scores ~3 per quarter — RL must beat that baseline to demonstrate value.

---

## For §6 Evaluation — what to measure and report

### Throughput numbers (verified on ws10 / ws1-ws10)

- **Single emulator, plain adb:** ~1 SPS (steps per second). Bottleneck breakdown:
  - 132 ms — touch hold (deliberate; this is game time)
  - 100 ms — `adb exec-out screencap` (PNG fetch)
  - 100 ms — adb subprocess startup overhead
  - 500 ms — pickle of 7.5 MB `info["full_rgb"]` through `AsyncVectorEnv` pipe (BIG surprise; fixed by removing field)
- **N=3 emulators per ws:** ~3 SPS aggregate per machine
- **9 machines × N=3 emulators:** ~27 SPS cluster aggregate
- **With IPC fix (full_rgb removed):** expected ~3× per-env → ~80 SPS cluster aggregate
- **Theoretical further win** with minicap+minitouch: another 5–10× per-env

### Distributed training across CSIE workstation cluster

- CSIE has ws1–ws10 reachable as `wsN.csie.ntu.edu.tw`
- NFS-mounted home dir `/home/student/12/$USER/` is shared (writes from any ws visible from all)
- `/tmp2/$USER/` is per-machine local (each ws has its own)
- 9-run ablation matrix dispatched as: one (mode, seed) per ws
- Cron at `*/5 * * * *` mirrors a status snapshot to `~/drl_status_latest.txt` for at-a-glance monitoring; alert file `~/drl_ALERT.txt` appears iff any run is STALLED or ERROR
- Recovery: when a ws is killed by the watchdog, just rerun `scripts/launch_on_ws.sh <mode> <seed>` on that ws; logs go back to the same NFS path

### Logging / observability

- Per-update PPO line: `upd N/M step S sps X pg L_pg v L_v ent H aux L_aux ret R`
  - Tracks all standard PPO quantities + aux loss
  - `ret` is mean episode return over recent finished episodes (nan until first episode terminates)
- For 50k-step runs at N=3 (3 SPS), there are 50000 / (3 × 128) ≈ 130 PPO updates per run

---

## For §7 Limitations (the things to disclose honestly)

1. **OCA targets are CV-pipeline outputs, not engine state.** Pipeline error contaminates the auxiliary signal. We report pipeline accuracy on a hand-labeled set as a confound. The cleaner alternative — engine-state ground truth via APK instrumentation — was infeasible on a closed-source Unity game in the project timeline.

2. **Opponent is weak.** Default-difficulty HOU CPU rarely scores. The absolute scores are not generalizable to "Bouncy Basketball is solved by RL"; they only establish the relative ordering of {baseline, OCA, DPR} under matched conditions. Future work: opponent-strength sweep using the in-game CUSTOMIZE skill-point allocator.

3. **Per-step latency is high (~1000 ms reduced to ~300 ms after the IPC fix).** The agent's effective decision rate (~3 Hz) is below the game's natural decision-relevant timescale for some events (e.g., shot release timing at the peak of a jump). PPO can compensate by learning anticipatory behavior, but the lag is a real handicap to absolute performance.

4. **Limited training budget.** 50,000 environment steps × 9 runs × ~1.5 hours = within deadline budget, but well below the ~10M steps RL papers typically use for Atari-style results. Comparative results between configurations remain valid; absolute final scores should not be benchmarked against Atari-scale literature.

5. **One-game scope.** All claims are conditioned on Bouncy Basketball. Generalizing to other physics-based mobile games would require re-running the matrix.

6. **Reward signal is coarse.** Pixel-diff on the CHI score ROI detects "score changed" but cannot distinguish +1 vs +2 vs +3 point scores (3-pointers exist in basketball). We assign +1 per detected score event. This introduces ~10–20% error in the absolute reward magnitude (a constant scale factor that doesn't affect relative comparison).

7. **2 of 3 opponent snapshots used.** Time pressure → captured HOU and LAC opponents only. A richer pool (BRO, DAL, IND, …) would strengthen the §4.0 opponent-distribution claim.

---

## Methodological insights worth highlighting in §3 or §8

### 1. Auxiliary tasks must remain non-trivial throughout training

A well-designed auxiliary task must keep providing learning signal across the entire training run. Predicting pose at t+1 was trivially solvable (encoder converges to ≈0 loss within 6 updates) because the temporal gap between input frames and target was too small. Increasing the prediction horizon to t+4 (~528 ms ahead) restored a meaningful learning task. **The right metric to track is not "did the aux head converge?" but "does the aux gradient stay nonzero throughout?"**

### 2. Per-team centroid vs. per-player coordinates

For 2v2 games with shared control, modeling the two teammates as a single "team object" (centroid + body-vector orientation) gives a 10-dim OCA target instead of 18-dim per-player coords. This matches the action space (one touch → one team behavior) and reduces label dimensionality without losing information about what the policy can actually control.

### 3. Distributed training without a real distributed framework

We did not use Ray, Horovod, or any distributed RL framework. 9 completely independent PPO trainings, one per machine, with NFS for visibility. The seeds are independent → the runs are embarrassingly parallel. **Simpler than it sounds and easier to debug than any real distributed system would have been.**

### 4. The pickle tax of AsyncVectorEnv

Every `info` dict returned from `env.step()` is pickled and sent through a multiprocessing pipe. We accidentally included a 7.5 MB RGB frame in `info` for diagnostic purposes; it cost ~500 ms per step (most of our latency budget) despite the main process never using it. **Lesson: be ruthless about what crosses worker-process boundaries.**

### 5. The watchdog as soft compute constraint

Shared workstations like ws10 have monitoring scripts that kill sessions exceeding thread/process budgets. This shaped our design more than memory or CPU did. **Threading-default libraries (cv2, OpenMP, BLAS) on a 144-core machine create thousands of threads per Python worker if not capped.** All threading defaults must be explicitly set to 1.

### 6. The "do nothing" baseline can be surprisingly strong

In default-difficulty Quick Game mode, the NO_PRESS-forever policy already scored 3 in 45 s of Q2. Any learned policy must beat this. We saw firsthand that a *bad* learned policy (e.g., sustained PRESS) is provably *worse* than no policy. This is reassuring evidence the environment is sensitive to control quality.

---

## Per-day diary (terse, for paper writing)

### Day 1 (2026-05-21, ~13:00–22:00)

- **Morning:** team setup, env_setup.md walkthrough. SDK + AVD + APK install. Hit ads, fixed via airplane mode. Hit immersive banner, fixed via secure setting.
- **Afternoon:** scaffolded `env.py`, `vision.py`, `reward.py`, `adb_backend.py`, `train.py`, `orchestrate.py`. Verified Discrete(2) action space via web search + controlled experiment.
- **Evening:** distributed deploy across ws1–ws8 + ws10. First 9-run matrix launched at 19:38. Cron-based monitoring set up at 19:57. Multiple watchdog kills required N=3 fallback and ws-specific relaunches.
- **Late evening:** analyzed first training results — discovered OCA-too-easy and full_rgb pickle tax. Fixed both. Restarted matrix at 21:06 with 4 fixes applied.

### Day 2–4 (planned)

- Continue training; harvest checkpoints; run §5.3 robustness eval on the Python clone; run §5.5 detector validation; draft figures; write paper.

### Day 5 (planned)

- Final edits; submit by 23:59.
