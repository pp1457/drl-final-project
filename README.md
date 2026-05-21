# DRL Final Project — Physics-aware auxiliary tasks for representation learning

PPO on **Bouncy Basketball** (real Android game) with two auxiliary tasks:
1. **OCA** — predict next-frame object coordinates (sparse, physics-relevant)
2. **DPR** — predict next-frame pixels (dense, "world model" style)

Compared against a **baseline** PPO with no auxiliary task. 3 seeds × 3 configs = 9 training runs in parallel across `ws1–ws10`.

**Deadline:** 2026-05-25 23:59 (NTU DRL final).
**Status:** training (started ~23:13 May 21, ETA ~03:30 May 22).

## Where the interesting docs are

- **[research_idea.md](research_idea.md)** — full proposal (POMDP, three configurations, evaluation protocol, limitations)
- **[LESSONS_LEARNED.md](LESSONS_LEARNED.md)** — per-check observations from training, organized for the paper
- **[STATUS.md](STATUS.md)** — current state of the cluster + what's open
- **[env_setup.md](env_setup.md)** — original team setup doc (now partly superseded by `scripts/`)

## What each file does

```
config.py               Frozen-dataclass constants for VISION, REWARD,
                        ACTIONS, EMU, MODEL, PPO, PATHS — single source of
                        truth for all tunable settings.

env.py                  Gymnasium env (BouncyBasketballEnv) + EmulatorBackend
                        interface + FakeBackend for offline tests. Discrete(2)
                        action space, frame_skip=4 inside step().

vision.py               OpenCV pose extraction: HSV thresholding for ball
                        (orange), CHI jersey (red), HOU jersey (white). Returns
                        10-dim OCA target [ball_xy, CHI_xy/sin/cos, HOU_xy/sin/cos].

reward.py               Pixel-diff on CHI scoreboard ROI. No digit OCR.
                        Tuned threshold + cooldown to suppress animation
                        double-counting.

adb_backend.py          Plain-adb EmulatorBackend. send_action holds touch via
                        `input swipe x y x y duration_ms`. ~1 SPS per env.

minicap_backend.py      Scaffold for fast-IO backend (5-10x). Not deployed;
                        needs minicap+minitouch binaries pushed to AVD.

train.py                PPO trainer. Shared NatureCNN encoder + actor/critic
                        + per-mode aux head. Forces single-thread cv2/torch/OMP
                        to survive shared-workstation watchdog. K-step OCA
                        target shift in rollout buffer.

eval.py                 Load checkpoint, run N episodes, report return stats.

eval_robustness.py      §5.3 perturbation eval on the Python clone (varies
                        gravity / restitution / mass).

clone_env.py            pymunk + pygame physics clone of Bouncy Basketball.
                        Used only for the §5.3 robustness eval.

orchestrate.py          Multi-emulator lifecycle: bootstrap / launch / kill /
                        supervise. Uses -read-only snapshots and per-user
                        adb daemon port to coexist with other students on ws10.

scripts/deploy_one.sh   rsync project + SDK + AVD from controller to a worker ws.
scripts/launch_on_ws.sh On a target ws: kill leftovers, launch N=3 emulators,
                        run one (mode, seed) config, mirror final ckpt to NFS.
scripts/dispatch_all.sh Fan out 9 (mode, seed) configs to 9 ws's via SSH.
scripts/monitor.sh      Tabular status display reading NFS-home logs.
scripts/cron_check.sh   5-minute heartbeat to ~/drl_status_latest.txt + alert file.
scripts/cron_register.sh / cron_unregister.sh — install/remove the cron entry.

android_env.sh          sourceable env vars: ANDROID_HOME, per-user
                        ANDROID_ADB_SERVER_PORT (derived from UID), PATH adds.

run_all.py              Sequential single-machine matrix launcher (alternative
                        to scripts/dispatch_all.sh for single-ws fallback).
```

## How to use

### Initial setup (one-time, on each ws)

```bash
# On the controller ws (ws10):
./scripts/deploy_one.sh ws1     # repeat for ws2..ws8
```

### Run the matrix

```bash
tmux new -s drl
source android_env.sh
./scripts/dispatch_all.sh 50000 3    # 50k steps, N=3 emulators per ws
```

### Monitor

```bash
./scripts/monitor.sh           # one snapshot
./scripts/monitor.sh -w        # auto-refresh every 30s
cat ~/drl_status_latest.txt    # cron-refreshed snapshot (every 5 min)
cat ~/drl_ALERT.txt            # file exists only if a run is in trouble
```

### Stop / cleanup

```bash
./scripts/cron_unregister.sh   # turn off the cron heartbeat
# kill workers across all ws's:
for WS in ws1 ws2 ws3 ws4 ws5 ws6 ws7 ws8 ws10; do
  ssh ${WS}.csie.ntu.edu.tw "pkill -u \$USER -9 -f train.py; pkill -u \$USER -9 -f qemu-system"
done
```

### After training

```bash
# Eval on real APK
.venv/bin/python eval.py --checkpoint ~/drl_ckpts/<host>_<mode>_s<seed>_final.pt \
    --env-id bouncy --episodes 100

# §5.3 robustness eval on Python clone
.venv/bin/python eval_robustness.py --checkpoint <path> --episodes 50 \
    --output robustness_<run>.json
```

## Cluster architecture

```
            controller (ws10)
                  │
       ┌──────────┼──────────┐
       │          │          │
   git push    SSH dispatch  cron heartbeat
       │          │          │
       ▼          ▼          ▼
       GitHub  ws1..ws8     ~/drl_logs (NFS, all ws's see it)

   Each ws:
       ┌─ Python train.py (one mode+seed)
       ├─ AsyncVectorEnv with 3 workers
       ├─ 3 Android emulators (port 6554+2i)
       │  └─ Bouncy Basketball APK installed, airplane mode
       └─ Logs → NFS home for shared visibility
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Discrete(2) action space (PRESS / NO_PRESS) | Bouncy Basketball is one-button. Hold = jump (longer = higher), release in air = shoot. |
| OCA targets via OpenCV HSV thresholding | Zero labeling cost. Jersey colors separate cleanly. Original shoe-color plan failed (both teams have dark shoes). |
| OCA prediction horizon K=8 (~1.06s) | K=1 was trivially solvable (aux→0 by upd 6, no signal for rest). K=8 forces actual forward dynamics. |
| Reward via pixel-diff on score ROI | No digit OCR needed. ~98% agreement with manual counting. |
| Per-team centroid (not per-player) | 2v2 game where one touch controls both teammates. Modeling them as a single "team object" matches the action space. |
| Airplane mode in AVD snapshot | AdMob test ad blocks the entire game screen. Disabling network kills it. |
| Per-user ADB daemon port | Shared workstation; default 5037 collides with other students' emulators. |
| N=3 emulators per ws (not 4) | Per-user watchdog kills sessions with too many threads. 3 survives, 4 doesn't reliably. |
| ent_coef=0.05 (not 0.01) | Bouncy Basketball is sparse-reward; CHI auto-plays well so "do nothing" is locally near-optimal. Standard PPO ent_coef=0.01 causes premature commitment to worse-than-random. |
| Distributed across ws1-ws10 | Per-machine watchdog made N>~6 emulators/ws infeasible. Better to run 9 independent (mode,seed) trainings on 9 machines. |

## Current results (training in progress)

At upd ~28/130 (May 22 00:07):

| Config | Returns (3 seeds) | Mean |
|---|---|---|
| baseline | 0.33, 0.67, 0.33 | 0.443 |
| **OCA** | 0.89, 1.00, 0.67 | **0.853** |
| DPR | 0.78, 0.67, 0.33 | 0.593 |

**OCA > DPR > baseline** — matches the proposal hypothesis. Will update as training completes.

See **LESSONS_LEARNED.md** for the full per-check observations.

## Limitations (paper §7)

1. OCA targets are CV-pipeline outputs, not engine state (no engine access on closed-source APK)
2. Opponent CPU is weak at default difficulty (HOU scored 0 across early trials)
3. ~330ms per-step latency limits the agent's reactive granularity
4. Modest training budget (50k steps × 9 runs) due to deadline
5. Single game (Bouncy Basketball only)
6. Reward signal is coarse (+1 per detected score, can't distinguish 2pt vs 3pt)
7. Small opponent pool (only HOU and LAC captured as snapshots)
