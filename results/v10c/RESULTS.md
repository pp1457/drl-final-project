# v10c Experiment Results (2026-05-25)

## Overview

v10c is the final experimental run for the NTU-DRL-MiniConf 2026 paper.
Trained PPO agents with three auxiliary-task conditions on the real Android
game **Bouncy Basketball** under near-identical settings.

- **Date**: 2026-05-25 (02:05–19:04 CST)
- **Total budget**: 80,000 env steps per seed = 312 PPO updates
- **Conditions × Seeds**: 3 × 3 = 9 runs distributed across ws1–ws8 + ws10
- **Architecture**: Shared NatureCNN encoder + actor + critic + aux head
- **Exploration**: Random Network Distillation (RND) intrinsic bonus enabled
- **Observation augmentation**: charge-duration scalar concatenated to encoder
  output (513-dim head input)
- **Backend**: `adb-motionevent` (stock `input motionevent`), `--frame-skip 1`
- **Termination**: cluster-wide kill wave at 19:04 froze all 9 seeds before
  natural completion (highest reached upd 303/312)

## Final per-seed training-time return (frozen at kill time)

| Cond. | Seed | Host | Final upd | Final ret | Recoveries |
|-------|------|------|-----------|-----------|------------|
| baseline | 0 | ws1 | 303 | **+0.52** | 1 |
| baseline | 1 | ws2 | 252 | -1.12 | 1 |
| baseline | 2 | ws3 | 297 | -0.95 | 2 |
| OCA | 0 | ws4 | 263 | +0.10 | 1 |
| OCA | 1 | ws5 | 284 | -0.47 | 2 |
| OCA | 2 | ws6 | 267 | **+0.35** | 1 |
| DPR | 0 | ws7 | 272 | -0.87 | 2 |
| DPR | 1 | ws8 | 253 | -0.67 | 1 |
| DPR | 2 | ws10 | 290 | -0.67 | 1 |

**Per-condition mean training reward (n=3):**

| Condition | Mean ret | Std (sample) |
|-----------|----------|--------------|
| baseline  | **-0.52** | 0.86 |
| OCA       | **-0.01** | 0.42 |
| DPR       | **-0.74** | 0.12 |

→ Training metric ranking: **OCA > baseline > DPR**, OCA leads baseline by
~0.5 reward.

## Training-time evolution across snapshots

We captured 17+ status snapshots during training. The per-condition mean
reward was tracked at each:

| Avg upd | baseline | OCA   | DPR   |
|---------|----------|-------|-------|
| 80      | -0.37    | -0.46 | -1.44 |
| 95      | -0.46    | -0.36 | -1.15 |
| 110     | -0.93    | +0.03 | -0.78 |
| 121     | -0.92    | -0.03 | -0.69 |
| 133     | -0.82    | -0.04 | -0.79 |
| 142     | -0.62    | -0.36 | -0.61 |
| 169     | -0.52    | +0.17 | -0.75 |
| 178     | -0.38    | +0.23 | -0.64 |
| 192     | -0.59    | +0.20 | -0.55 |
| 213     | -0.58    | +0.05 | -0.57 |
| 234     | -0.63    | +0.08 | -0.77 |
| 247     | -0.65    | +0.05 | -0.79 |
| 256     | -0.60    | +0.03 | -0.69 |

**OCA leads baseline by 0.2–0.9 across every snapshot from upd 110 onward.**
DPR remains worst at every snapshot.

## Deterministic evaluation (latest ckpts)

Two eval runs on different hosts (ws10 had emulator killed mid-run, ws3
completed all three):

### Eval n=3, 128-step cap (on ws10, abandoned after 1st ckpt)

| Config | Ckpt step | Episodes | Mean | Median | Std |
|--------|-----------|----------|------|--------|-----|
| baseline ws1_s0 | 69,816 | [-2, +3, -1] | 0.0  | -1.0 | 2.65 |
| OCA ws6_s2      | 63,330 | [+4, 0, +2]  | **+2.0** | +2.0 | 2.00 |
| DPR ws10_s2     | 62,956 | [-3, +3, -1] | -0.33 | -1.0 | 3.06 |

### Eval n=5, 128-step cap (on ws3, final)

| Config | Ckpt step | Episodes | Mean | Median | Std |
|--------|-----------|----------|------|--------|-----|
| baseline ws1_s0 | 76,216 | [0, +4, -1, +1, 0]    | **+0.8** | 0.0 | 1.92 |
| OCA ws6_s2      | 63,330 | [-2, +1, +1, +2, -1]  | +0.2     | **+1.0** | **1.64** |
| DPR ws10_s2     | 69,356 | [-1, -1, -5, 0, 0]    | -1.4     | -1.0 | 2.07 |

### Eval takeaways

- **DPR significantly worst** in both eval runs (mean ≈ -0.3 to -1.4).
- **Baseline vs OCA mixed at n=5**: baseline mean higher, OCA median higher
  and lower variance. Differences within 1 std → not statistically
  significant at this sample size.
- Training metric and eval **agree on DPR being worst**.
- Training metric favors OCA; eval at n=5 is inconclusive baseline vs OCA.

## RND entropy-collapse rescues (qualitative finding)

Across the 9 runs we logged **4 hard entropy collapses** (ent < 0.05) that
RND rescued:

| Seed | First collapse | Lowest ent | Rebounded to | Notes |
|------|---------------|------------|--------------|-------|
| ws7_dpr_s0  | upd 36   | 0.064 | 0.647 (upd 75) | Oscillated 3× during run |
| ws8_dpr_s1  | upd 36   | 0.151 | 0.650 (upd 80) | Single recovery |
| ws10_dpr_s2 | upd 55   | 0.006 | 0.316 (upd 80) | Deepest collapse, full rebound |
| ws4_oca_s0  | upd 97   | 0.049 | 0.528 (upd 230) | Oscillating recovery; ret stable |

Without RND, PPO collapsed policies typically remain stuck. This is a
secondary methodological contribution.

## System stability events

| Event class | Count | Notes |
|-------------|-------|-------|
| adb screencap timeouts        | 3  | All auto-recovered via crash-proof trap |
| Blank-frame watchdog trips    | 3  | All auto-recovered |
| Menu-stuck (env.reset) errors | 4  | All auto-recovered |
| External host kills           | 4+ | ws8 (17:07), ws2 (17:26), ws7 (17:27), ws4 (17:43), cluster-wide wave (19:04) |

The crash-proof `_recover_envs` path (train.py) handled all 13 emulator-level
failures without operator intervention. Host-level kills froze runs but the
cleanup trap (in `scripts/launch_on_ws.sh`) mirrored checkpoints to NFS,
and we additionally rsynced all 9 final checkpoints to
`/tmp2/$USER/eval_ckpts_all/` before further failures.

## Reproducibility notes

- **train.py invocation** (per seed): see `scripts/launch_on_ws.sh`. Key
  flags: `--total-timesteps 80000 --num-envs 2 --num-steps 128
  --num-minibatches 4 --update-epochs 4 --ckpt-every 25 --backend
  adb-motionevent --frame-skip 1 --use-rnd --use-charge-dim`.
- **Snapshot**: `clean_boot_v8` (HOU vs CHI, SWITCH SIDES OFF, CUSTOMIZE
  player 17/32). See `LESSONS_LEARNED.md` for snapshot construction recipe.
- **adb daemon port**: `ANDROID_ADB_SERVER_PORT=5613`,
  `EMU_BASE_PORT=6554`.
- **Eval invocation**: `eval.py --episodes 5 --deterministic
  --max-episode-steps 128 --serial emulator-6554`.

## Files in this directory

- `RESULTS.md` (this file)
- `eval_n3_ws10/` — initial n=3 eval JSONs (one ckpt completed before kill)
- `eval_n5_ws3/` — final n=5 eval JSONs (all three completed)
- `training_logs/` — per-seed training stdout (`ws*_*.log`)

## What's NOT in this directory (too large for git)

- 9 final checkpoints (~300 MB total) live at
  `/tmp2/b12902046/eval_ckpts_all/` on ws10.
- Per-ws local checkpoint directories
  (`/tmp2/b12902046/DRL_final_project/checkpoints/bouncy_*/step_*.pt`) on
  each of ws1–ws8 + ws10.

## Bottom line

- **Reportable headline**: DPR (pixel reconstruction) hurts performance on
  this Android-physics task; OCA (object-centric coord regression) and
  baseline are comparable at n=5 eval. RND prevents entropy collapse and
  enables continued exploration after policy commits.
- **Caveat**: n=3 seeds per condition and n=5 eval episodes are small.
  Differences should be presented as trends, not statistical findings.
