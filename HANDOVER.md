# Handover — DRL Final Project (2026-05-23 evening)

You're picking up a project that's **mid-matrix** with **a known but unverified
root-cause fix**. Read this top-to-bottom before doing anything destructive.

Deadline: **2026-05-25 23:59** (Sunday late). It's currently Saturday ~19:30.
You have ~28 hours of real time.

---

## Current state (live)

- **v5 matrix is running** across 9 workstations (ws1–ws8, ws10).
  - Backend: `adb-motionevent`, `frame_skip=1`, `n_envs=2`, `--total-timesteps=50000`.
  - Progress at 19:13: ws3 at upd 141/195, others 127–140. ~72% through.
  - ETA finish: ~22:00–23:00 tonight.
- **Persistent error monitor** task `b15f0kn2f` watching for crashes/recoveries.
- **Hourly auto-archive** cron `ada3deca` writing to `~/drl_logs_archive/`.
- **15-min status cron** `d15e383c` was firing reports through Claude; replace
  with whatever monitoring fits your style.

To check live status:
```bash
for f in ~/drl_logs/ws*_*.log; do tail -1 "$f"; done
```

To kill the matrix (do this only if you're about to relaunch):
```bash
# kills training + qemu on each ws via launch_on_ws.sh's pkill
bash scripts/dispatch_all.sh 50000 2 adb-motionevent 1   # also re-launches
# or selectively
ssh wsN.csie.ntu.edu.tw "pkill -9 -f train.py; pkill -9 -f qemu-system"
```

---

## What's broken (and why the v5 results won't be publishable as-is)

Tonight a `diag_actions.py` test (always-press vs never-press, 200 steps each)
revealed:

| trial         | CHI events | HOU events |
|---|---|---|
| always-press  | 2          | 4          |
| **never-press** | **6**    | 4          |
| alternating   | 4          | 4          |

**Our PRESS action does not control CHI — it INTERFERES with the game's
autopilot.** Bouncy Basketball's default mode is "control one player on your
team, autopilot the other(s)". Our agent's PRESS at (1170, 793) is wasting
the active player's turn while the autopilot scores for the other CHI players.

This means the v5 matrix is training on a noisy reward signal where the agent
has essentially no control. ret values converged to ~0 across all 3 configs
(baseline, OCA, DPR) — the optimal policy is "do nothing", which PPO can't
easily find from this signal.

---

## The plan (validated, not yet executed)

**Source**: in-game **Options → "Control All"** toggle disables autopilot, so
a single PRESS drives every CHI teammate simultaneously. This restores the
Discrete(2) action space the proposal assumed.

### Phase 1 (≈1h): configure the game via Options menu
1. Launch one emulator interactively (`emulator -avd pixel5_api31 -port 6554`).
2. Navigate to **Options** from the main menu (use `adb exec-out screencap -p`
   to inspect each screen; tap with `adb shell input motionevent DOWN/UP x y`).
3. Enable **Control All** (the load-bearing setting).
4. Set **Difficulty = Hard** (makes HOU autopilot stronger → defense decisions matter).
5. Set **Quarter length = 30 s**, **Quarters = 2** (1-minute episodes → 5–10×
   more episodes per training step → much denser reward gradient).
6. Optionally **Team size = 1** (cleaner 1v1, fewer OCA centroids to predict).
7. Save snapshot: `adb -s emulator-6554 emu avd snapshot save clean_boot_v6`.
8. Update `orchestrate.py`'s `DEFAULT_BOOT_SNAPSHOT` to `clean_boot_v6` and
   `EmulatorEndpoint.snapshot_names = ("clean_boot_v6",)`.

**LOAD-BEARING RISK:** the App Store description listing "Control All" is for
the current iOS version. Our APK is Android v3.2.1 from 2018. It may not have
this toggle. If you boot and the Options menu doesn't expose Control All,
fall back to Phase 1b (below) or pivot to training on `clone_env.py` (the
Python physics clone in `clone_env.py` — that has full programmatic control).

### Phase 1b (fallback): clone_env training
If Control All doesn't exist:
- `clone_env.py` is a Python physics clone of Bouncy Basketball, already in
  the repo, used for §5.3 robustness perturbation eval.
- Train PPO on the clone (it has full programmatic action space + scoring).
- Engineering story about the Android pipeline remains valid; main results
  use the clone.

### Phase 2 (≈30 min): re-verify with `diag_actions.py`
```bash
source ./android_env.sh
# Launch emu on port 6558 (or any free port)
/tmp/launch_eval_emu.sh > /tmp/eval_emu.log 2>&1 &
# Wait for boot
until adb -s emulator-6558 shell getprop sys.boot_completed | grep -q 1; do sleep 5; done
# Edit diag_actions.py to use clean_boot_v6 if you saved it under a new name
.venv/bin/python -u diag_actions.py
```

**Expected if Control All is enabled:**
- always-press: **CHI events HIGH** (we're shooting).
- never-press: **CHI events LOW** (no shots).
- alternating: in between.

If you see this monotone separation, the action space is real and PPO has a
useful gradient. Proceed to Phase 3. **If never-press still wins, Control All
didn't actually take effect — re-check the settings snapshot.**

### Phase 3 (≈3h): launch the corrected matrix
- Verify `chi_score_roi` and `hou_score_roi` are still at correct positions.
  Game may have re-laid-out at 30 s quarters / 1-player team. Run
  `diag_roi.py` to visually confirm and `diag_diff.py` to check the diff
  threshold is still ~10.
- Launch via `scripts/dispatch_all.sh 50000 2 adb-motionevent 1`.
  - With 1-minute episodes (vs current ~30 sec), you get 5–10× more episodes
    per agent step → 50k transitions might give ~500–800 episodes per run
    (vs ~30 episodes on the broken v5 setup). May be able to drop to 30k
    steps and still get tight error bars.

### Phase 4: paper writeup
- §3.5 "Diagnostic methodology": `diag_actions.py` revealed default autopilot;
  we corrected via Options/Control All. Closed-source game configuration as a
  novel sim-to-real challenge.
- §5 main result: OCA vs DPR vs baseline on real action space.
- §6 engineering contributions: motionevent backend, crash-proof recovery,
  sustained-overlay gate (quarter-end / OOB), vision watchdog, ROI diagnostic.

---

## Latency optimization (worth doing if you have ~1h spare)

Current bottleneck: `adb exec-out screencap -p` takes **~650 ms** per call
(on ws10 with 3 emulators contending; ~300 ms on idle hosts). PNG encoding
on-device is the slow part.

**Quick win — raw screencap (~30% faster, saves ~200 ms/step):**
- `adb exec-out screencap` (no `-p`) outputs raw RGBA: 10.1 MB for 2340×1080.
- Verified: PNG = 1.3s wall, RAW = 0.92s wall (30% faster).
- Format: 16-byte header (uint32 width, uint32 height, uint32 format, uint32
  colorspace), then `width*height*4` bytes of RGBA.
- Patch `AdbBackend.grab_frame` to slice and reshape into numpy directly,
  no `cv2.imdecode` needed:
  ```python
  raw = _adb(serial, "exec-out", "screencap")  # bytes, ~10 MB
  w = int.from_bytes(raw[0:4],  "little")
  h = int.from_bytes(raw[4:8],  "little")
  # raw[8:12]  = format (1 = RGBA_8888 typically)
  # raw[12:16] = colorspace
  arr = np.frombuffer(raw[16:16+w*h*4], dtype=np.uint8).reshape(h, w, 4)
  return arr[:, :, :3]  # drop alpha, returns HxWx3 RGB
  ```
- Verify with a quick `diag_latency.py` re-run before deploying.

**Bigger wins (more work):**
- **minicap** — abandoned earlier because Android 12+ blocks SurfaceFlinger
  access. STFService.apk is committed as the workaround attempt but never
  fully wired up.
- **Lower emulator resolution** — `-screen 720p` or AVD config edit. Cuts
  transfer bytes ~4× but requires re-measuring all pixel coords (PRESS_COORD,
  ROIs).

---

## Files modified tonight (all committed + pushed)

Already on `origin/main`:

| commit | file | what |
|---|---|---|
| 89cf74e | env.py | vision watchdog + multi-frame reset + sustained-overlay gate + quarter auto-advance |
| b7d8d96 | config.py | CHI/HOU ROI swap fix + threshold 25→10 |
| 8ab351e | eval.py | backend pass-through + `--serial` flag |
| af5d5ff | scripts/ | archive_logs.sh, extract_metrics.py, STFService.apk |
| 1852047 | diag_*.py | 7 diagnostic scripts (ROI, actions, latency, episode-end, …) |
| 2d1af5b | STATUS.md | overnight findings documented |

Run `git log --since='2026-05-22' --oneline` for the full list.

---

## Diagnostic scripts available

All in repo root, all target `emulator-6558` (the side eval emulator) so
they don't disturb a running matrix. Load `clean_boot` fresh each time.

| script | what it tells you |
|---|---|
| `diag_actions.py` | Does PRESS actually control CHI? (always/never/alternating × 200 steps) |
| `diag_roi.py` | Visual + RGB-mean inspection of the chi/hou score ROIs |
| `diag_roi2.py` | Wider scan across x-positions to find the actual score panels |
| `diag_diff.py` | 60-frame pixel-diff trace at the new ROI positions; calibrates threshold |
| `diag_verify.py` | End-to-end smoke test (reward + pose + game-over over 100 steps) |
| `diag_episode_end.py` | Random-policy step loop; saves last 8 frames on termination |
| `diag_why_short.py` | overlay_streak per step + frames around termination |
| `diag_game_over.py` | Red-pixel count in y=20-130 band (verify is_game_over False on gameplay) |
| `diag_latency.py` | Per-step latency breakdown (send / grab / pose / reward) |

To run any: `source ./android_env.sh && .venv/bin/python -u diag_X.py`.

---

## Workstation layout

9 workstations, each runs 2 emulators (ports 6554, 6556) + 1 training process:

| ws | config | seed | port pair |
|---|---|---|---|
| ws1 | baseline | 0 | 6554/6556 |
| ws2 | baseline | 1 | 6554/6556 |
| ws3 | baseline | 2 | 6554/6556 |
| ws4 | oca | 0 | 6554/6556 |
| ws5 | oca | 1 | 6554/6556 |
| ws6 | oca | 2 | 6554/6556 |
| ws7 | dpr | 0 | 6554/6556 |
| ws8 | dpr | 1 | 6554/6556 |
| ws10 | dpr | 2 | 6554/6556 |
| (ws9 was down at deploy time; ws10 took its run.)

ws10 doubles as the controller (where you run dispatch + eval). It also
launches the side eval emulator on port 6558 via `/tmp/launch_eval_emu.sh`.

Logs land in `~/drl_logs/<host>_<config>_s<seed>.log` (NFS-mounted home, so
visible from any ws). launch_on_ws.sh **truncates** this file at start, so
archive before re-launching.

---

## Known issues / gotchas

1. **launch_on_ws.sh pkill is broad.** `pkill -u $USER -9 -f qemu-system-x86`
   kills ALL qemu including the side eval emulator on 6558. Relaunch via
   `/tmp/launch_eval_emu.sh` if you re-dispatch.
2. **adb screencap timeout (10s) sometimes fires** on overloaded hosts; the
   trainer's `_recover_envs` catches it. Most v5 recoveries are this or
   "180 consecutive blank frames" (vision watchdog).
3. **ssh from ws10 to other ws's stays open**. dispatch_all forks 9 parallel
   ssh's that don't all exit cleanly — they appear as sshd-session zombies.
   Harmless but `ps` looks weird.
4. **`am start -n com.DreamonStudios.BouncyBasketball/...`** is in env.reset()
   to re-foreground the game if the Pixel launcher takes over. Verify the
   activity name `com.unity3d.player.UnityPlayerActivity` still works after
   any APK update.

---

## My honest read

The Control All hypothesis is the **most plausible** explanation for the
diag_actions result, and the App Store listing is suggestive evidence. But
**I haven't verified** that our 2018-vintage Android APK exposes the toggle.
Phase 1 step 3 is the load-bearing check, and you'll know in ~10 minutes.

If Control All works:
- Restart matrix with new snapshot. ~3-4 h for a fresh matrix.
- Real RL results.

If Control All doesn't exist on Android v3.2.1:
- Pivot to clone_env. Python-side training is fast (no adb overhead).
- Lose the "real Android" framing but the engineering work still publishes.
- 1 day is enough for clone-only training + paper.

Either way, v5 results should NOT be the headline of the paper — they reflect
a corrupted action interface. The engineering work (motionevent, recovery,
overlay gate, ROI fix) is the publishable contribution.

Good luck.
