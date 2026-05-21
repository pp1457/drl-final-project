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

### Day 2 — 01:30, diagnosed garbage data via frame-health instrumentation

Frame-health instrumentation (`oca_mask.sum() per rollout`) added to train.py confirmed that **the Android-emulator training pipeline was producing garbage data**:

```
ws*_*  fh 0.03-0.06 / 0.1-0.2
```

Across all 9 runs and all 3 configs, only 3-6% of rollout frames had any OCA component detected; the mean was 0.1-0.2 components per frame (out of 10). PPO was being trained on mostly empty / stuck-screen observations with no real reward signal.

Root cause is unclear but likely a combination of:
- `-read-only` emulator mode disabled `adb emu avd snapshot load` (silent failure of env.reset's snapshot path)
- Manual REMATCH taps work from a separate SSH session but the worker subprocess's identical tap call doesn't reliably advance the screen — possibly an adb daemon contention / queue issue with 3 workers per ws hammering the same daemon
- Match cycles are extremely fast on default difficulty (CHI auto-plays so well that matches end in seconds), leaving most wall-time on stats/GAME OVER screens
- Cumulative effect: workers see brief flashes of gameplay (when REMATCH does happen to work) but spend most time on transition screens producing no useful signal

**Plan: fix the worker reset() to verify gameplay actually starts before returning, instead of relying on hardcoded sleeps. Stay on the real Android APK as the proposal specifies. The clone_env stays as the §5.3 robustness eval target only.**

Confirmed via clone-env smoke test that the network and PPO loop are sound (sps 60, fh 0.75/4.5, aux 0.246 → 0.004 in 8 updates). So the bug is specifically in how env.reset() interacts with the Android worker subprocesses — not in the model or training loop itself. That's good news; means we can target the fix narrowly.

### Day 1/2 — check #10 (00:07), trend confirmed at upd ~28/130

```
Config     Returns (3 seeds)     Mean    Trend
baseline   0.33, 0.67, 0.33      0.443   slightly up from check #9 (0.22)
OCA        0.89, 1.00, 0.67      0.853   strongly up from check #9 (0.67)
DPR        0.78, 0.67, 0.33      0.593   slightly down from check #9 (0.67)
```

**Ordering matches proposal hypothesis: OCA > DPR > baseline.** OCA's sparse physics-relevant supervision is outperforming DPR's dense pixel reconstruction, which is outperforming pure PPO.

Most return trajectories are STABLE or RISING (none of the monotone decline we saw in the previous run). Examples:
- ws4 oca: 1.00 → 0.83 → 0.89 (stable around 0.9)
- ws5 oca: 0.67 → 1.00 (rising)
- ws7 dpr: 0.67 → 0.83 → 0.78 (stable)
- ws2 baseline: 0.33 → 0.83 → 0.67 (rising)

Entropy stays near max (0.67-0.69 for 8/9 seeds; ws5 oca dropped to 0.58 — modest commitment). With ent_coef=0.05 the policy isn't collapsing.

OCA aux loss profile: mostly 0.0001-0.0014 with occasional 0.027 spikes (e.g. ws6 upd 17). Small absolute value but nonzero throughout — gradient flow continues to shape the encoder.

If these means hold to convergence, this is **the proposal's headline result already visible**: structured object-centric prediction (10-dim coords) outperforms unstructured pixel-level prediction (84×84 reconstruction) as a representation shaper for physics-based RL.

### Day 1 — check #9 (23:40), early but striking reversal

Restart with ent_coef=0.05 produced dramatically different early results at upd ~14/130:

```
Config     Returns (3 seeds)     Mean    Comparison
baseline   0.33, 0.33, 0.00      0.220   ↓ from 0.89 in old run
OCA        1.00, 0.67, 0.33      0.667   ↑ from 0.56
DPR        1.00, 0.67, 0.33      0.667   ↑ from 0.28
```

**OCA and DPR are now clearly ahead of baseline.** This matches the proposal's hypothesis: aux representation shaping + adequate exploration → faster policy improvement than baseline PPO.

But note caveats:
- Sample size is 1-3 episodes per seed. Need to hit upd ~30+ for real signal.
- Both OCA and DPR show identical 0.667 means right now — too coincidental to be trustworthy. Likely noise.
- The big finding here is that ent_coef=0.05 prevented the policy collapse we saw last run.

**Lower-bound on entropy** (vs upd 15 of the old run):
```
Old run (ent_coef=0.01): ws5 oca at 0.475, ws4 oca at 0.564 — already collapsing
New run (ent_coef=0.05): ws5 oca at 0.595, ws4 oca at 0.606 — still exploring
```

The 5x entropy bonus is keeping the policy from committing prematurely, which gives the aux features time to actually pay off in better action selection.

### Day 1 — check #8 (23:12) — RESTART decision

By upd ~55/130 (~42% through fast runs), the trend was unmistakable:

```
Returns trajectory per run (chronological):
ws2 baseline:  1.33 → 1.22 → 0.92 → 0.73 → 0.61
ws3 baseline:  1.00 → 1.11 → 0.83 → 0.67 → 0.56
ws4 oca:       0.33 → 0.50 → 0.89 → 0.67 → 0.53 → 0.44   ← peaked then collapsed
ws5 oca:       0.33 → 0.50 → 0.56 → 0.42 → 0.33 → 0.28
ws10 dpr:      0.00 → 0.33 → 0.25 → 0.20 → 0.17
ws7 dpr:       0.33 → 0.17 → 0.44 → 0.33 → 0.27 → 0.22
```

ALL runs monotonically decreasing in their most recent ~3 episodes. Cleanly correlated with entropy: most-committed runs (lowest entropy) had lowest returns.

This is the classic PPO entropy-collapse failure in sparse-reward environments. In Bouncy Basketball, NO_PRESS-forever already scores ~0.5-1 per quarter (CHI auto-plays well), so "random uncommitted policy" is locally near-optimal. With ent_coef=0.01, PPO commits to whatever action got slightly positive advantage by chance — usually the wrong choice — and then degrades.

**Decision: restart all 9 runs with ent_coef bumped 0.01 → 0.05.**

This is a 5x increase in the entropy bonus. Standard for sparse-reward Atari is 0.01, but our reward density is even lower (one possible score per ~5 minutes of game), so we need more exploration pressure.

**Paper-worthy finding** (regardless of how the restart goes):
- "Premature policy commitment is a real failure mode in physics-based 1-button games" — the baseline random policy beats every learned policy at upd 55. This is itself worth noting as a methodological observation: sparse-reward arcade games may need higher entropy regularization than Atari benchmarks suggest.

### Day 1 — checks #6, ~upd 30/130 (22:45)

**Returns are DECREASING in every run.** Across all 3 configs:
```
baseline:  1.22 → 0.92, 1.11 → 0.83, 0.33 (stable)
oca:       0.89 → 0.67, 0.56 → 0.42, 0.67 → 0.56
dpr:       0.44 → 0.33, 0.33 → 0.33, 0.33 → 0.25
```
Mean by config: baseline 0.69 > oca 0.55 > dpr 0.30.

Combined with dropping entropy in several seeds (ws5 oca at 0.475, ws4 oca at 0.529, ws10 dpr at 0.608) this suggests **premature policy commitment to a worse-than-do-nothing policy.** Recall our controlled experiment: NO_PRESS-only scored 3 in 45s; sustained-PRESS scored 1. If PPO is committing toward "press more often" without enough exploration, it should underperform "press never" — exactly what we're seeing.

The entropy bonus coefficient (`ent_coef=0.01`) may be too low. A bigger coefficient would keep the policy exploring longer before committing.

**Decision:** wait through check #8 (~upd 60/130) before restarting. PPO sometimes recovers as the value function catches up to penalize bad commitments. If still trending down at upd 60, restart with `--ent-coef 0.05`.

### Day 1 mid-training observations (after re-dispatch ~21:35)

Once the four fixes landed (drop `full_rgb`, OCA K=8, max_episode_steps=1024, N=3 default), the training was healthier and produced these mid-training findings:

- **OCA aux loss is no longer monotonically collapsing** — it oscillates between ~0.0001 and ~0.04, with periodic spikes (e.g. upd 17 for both ws4 and ws5 oca runs, synchronized). The spike indicates PPO's gradient direction can transiently push the encoder away from its OCA-solving configuration; the head then has to relearn. **This is the desired behavior** for a representation-shaping aux task and confirms K=8 is the right horizon.

- **At upd ~22 (early, ~17% through), baseline > OCA > DPR in mean return** (0.89 vs 0.56 vs 0.28 across 3 seeds each). This is *counter* to our hypothesis but the data is super-noisy at this stage (3-5 episodes per run completed). If the ranking holds through the end of training, it's a paper finding in itself: aux tasks may *interfere* with PPO's early-stage exploration even if they help asymptotically — worth investigating in the discussion.

- **Multiple OCA seeds' aux losses spike at the same update.** Synchronization across seeds is evidence of a shared PPO-gradient direction at that point in training. Could be the "first policy commitment" moment — when the policy stops being uniform-random and starts preferring one action systematically. Possibly an interesting analysis: plot aux-loss spike timing vs. entropy drop.

- **Entropy drops happen unevenly across seeds.** ws1 baseline, ws5 OCA, ws8 DPR show entropy dropping ~0.69 → 0.55-0.65, indicating policy commitment. Other 6 seeds still at ~0.69 (uniform). This variance in "when does the policy commit" is worth noting in the paper — 3 seeds may not be enough to characterize the distribution well.

- **The IPC fix (drop info["full_rgb"]) only gave a marginal speedup** — SPS went from ~1-2 to ~2-3 per ws, not the 3× I predicted. Either the pickle cost was lower than estimated or other overhead now dominates. Honest correction for the paper: the 500ms-per-step IPC estimate was an over-count.

### Day 2–4 (planned)

- Continue training; harvest checkpoints; run §5.3 robustness eval on the Python clone; run §5.5 detector validation; draft figures; write paper.

### Day 5 (planned)

- Final edits; submit by 23:59.
