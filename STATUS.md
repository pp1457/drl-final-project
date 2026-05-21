# DRL Final Project — Status

**Last updated:** 2026-05-21 ~17:55 (Day 1 of 5 — deadline 2026-05-25 23:59)

## Recent (since last status update)

- `git init` done; 4 commits: initial → frame skip → config.py refactor → thread caps
- Frame skip = 4 wired into `BouncyBasketballEnv.step()` (one adb call per agent step, held for 132 ms)
- `config.py` centralizes VISION / REWARD / ACTIONS / EMU / MODEL / PPO / PATHS constants
- Multi-snapshot opponent pool: `env.reset()` picks uniformly from `EmulatorEndpoint.snapshot_names`
- ws10 has a per-user watchdog that killed our 12-emulator launch and our first 6-worker training run (pthread exhaustion). Mitigated by:
  - 5 s launch stagger in `orchestrate.py` (was 2 s)
  - Force `OMP_NUM_THREADS=1` etc. + `cv2.setNumThreads(1)` + `torch.set_num_threads(1)` in `train.py`/`vision.py`
- `minicap_backend.py` scaffolded as a future drop-in (5–10× throughput once binaries are pushed)


## What works end-to-end

| Layer | Component | Status |
|---|---|---|
| Workstation | KVM enabled, 144 CPU, 755 GB RAM, Java 21 | ✓ ws10 verified |
| Android SDK | cmdline-tools, platform-tools, android-31 system image, emulator | ✓ at `/tmp2/$USER/DRL_final/android-sdk/` |
| AVD | `pixel5_api31` (Pixel 5, API 31, x86_64 google_apis) | ✓ `avdmanager list avd` confirms |
| Emulator | Running headless on console port 6554 | ✓ `adb -s emulator-6554 get-state` = device |
| App | Bouncy Basketball 3.2.1 installed (`com.DreamonStudios.BouncyBasketball`) | ✓ |
| Snapshot | `clean_boot` saved (266 MB), reload tested OK | ✓ but lands at end-of-Q1 — see Limitations |
| Ad blocking | airplane mode enabled, persists in snapshot | ✓ ads no longer appear |
| Per-user adb | `ANDROID_ADB_SERVER_PORT=5613` (computed from UID) | ✓ no collision with other students on ws10 |
| Python env | venv at `.venv/` with torch 2.11, cv2 4.13, gymnasium 1.3, pygame, pymunk, wandb, tensorboard | ✓ |
| Env wrapper (`env.py`) | Gymnasium env, EmulatorBackend interface, FakeBackend for offline tests | ✓ smoke test passes |
| Vision (`vision.py`) | OpenCV pose pipeline; HSV ranges are placeholders | ✓ runs on synthetic frames; NEEDS real-frame calibration |
| Reward (`reward.py`) | TemplateScoreExtractor scaffold (digit templates not yet captured) + `zero_extractor` placeholder | ✓ falls back to zero reward |
| PPO trainer (`train.py`) | Shared NatureCNN + actor/critic/OCA/DPR heads, `--aux-mode {baseline,oca,dpr}` | ✓ all 3 modes train end-to-end on FakeBackend |
| ADB backend (`adb_backend.py`) | Plain-adb screencap + input | ✓ produces real frames; touch coords placeholder |
| Orchestration (`orchestrate.py`) | bootstrap/launch/kill/supervise-once subcommands | ✓ written, scaling not yet stress-tested |
| Python clone (`clone_env.py`) | pymunk + pygame, configurable gravity/restitution/mass | ✓ smoke test passes |
| Eval (`eval.py`) | Load checkpoint, run N episodes, report stats | ✓ smoke test on tiny checkpoint passed |
| Robustness eval (`eval_robustness.py`) | 9 perturbation conditions on the Python clone | ✓ wired |
| Checkpointing | Save every N updates to `checkpoints/<run>/step_<n>.pt` | ✓ |

## What's blocked

| Blocker | Why it matters | Path forward |
|---|---|---|
| In-game touch controls unmapped | Without LEFT/RIGHT/JUMP/CHARGE/RELEASE tap coords, the agent cannot ACT on the real game. PPO would train a no-op policy. | Probe more carefully during ACTIVE Q1 play (not pre-game / end-of-quarter), OR play the game on a phone to identify control regions, OR inspect the APK assets for the UI atlas. The "hand cursor" at bottom-right of gameplay (~1900, 860) is likely the SHOOT button — it's a tutorial pointer that disappears after first use. |
| Snapshot lands at end-of-Q1 stats screen, not active play | Each `env.reset()` should put the agent in a playable state. Currently it restores to a "tap NEXT QUARTER to continue" state. | Re-save snapshot during ACTIVE Q1 play (a few seconds into the match), OR have `BouncyBasketballEnv.reset()` issue a `NEXT QUARTER` tap as part of reset. NEXT QUARTER is a green button — coords findable via OpenCV color thresholding. |
| HSV thresholds in `vision.py` not tuned | OCA target signal would be all-zeros on real frames. | We now have 50+ real frames at `/tmp2/$USER/DRL_final/frames/gameplay/`. Sample shoe/ball/jersey pixels in a notebook → set HSV ranges → validate against §5.5 thresholds. Note: shoes are dark/black on both teams. Jersey color (red CHI vs white HOU) is the better discriminator. |
| Digit templates for scoreboard not captured | No reward signal. PPO would train on a flat zero reward (no learning). | Crop digits 0-9 from the captured frames' scoreboard region. Save to `digits/<n>.png`. Update `reward.PLAYER_SCORE_ROI`. |
| Quick Game mode may be CPU-vs-CPU demo | If the player can't control either team in Quick Game, no RL is possible there. | Try other game modes (PLAY OFF / Tournament). Or read the game tutorial. The "Q1 ended 11-0 to CHI with zero input from me" observation is the smoking gun — either CHI auto-plays (CPU vs CPU) or the controls are non-obvious. |

## Configuration committed

| Setting | Value | Reasoning |
|---|---|---|
| AVD | pixel5_api31, Android 12 (API 31), x86_64 + Google APIs | Per env_setup.md |
| Display | 1080x2340 native portrait, current rotation = 90 (landscape) | Game forces landscape |
| Network | Airplane mode ON | Kills the "Test Ad" overlay that otherwise blocks the game screen |
| Immersive banner | `secure immersive_mode_confirmations=confirmed` | Suppresses Android's first-launch "Got it" overlay |
| Teams (Quick Game) | CHI (ROAD, 1-dot rating) vs HOU (HOME, 3-dot rating) | Default, no edits to CUSTOMIZE. 2v2 game where the agent's single PRESS controls both CHI players simultaneously. Strength is held constant across all 9 experimental configs. |
| Match length | Default (likely 4-min quarter or NBA-style) | Not adjusted yet; check if shorter quarters available |
| Reward | `zero_extractor` placeholder until digit templates captured | Trains with zero reward — useful only for smoke tests |
| Pose extractor | `vision.detect_pose` with placeholder HSV ranges | Will return non-zero on synthetic FakeBackend but mostly zero on real frames |

## Code layout

```
/tmp2/b12902046/DRL_final_project/
├── env_setup.md              # Original team setup doc (no longer accurate for parallel/network/ad-block)
├── research_idea.md          # Concrete research proposal (OCA via shoe/jersey CV, DPR, baseline)
├── STATUS.md                 # this file
├── android_env.sh            # source-able env vars; sets ANDROID_ADB_SERVER_PORT per user
├── env.py                    # gym env + EmulatorBackend interface + FakeBackend
├── adb_backend.py            # AdbBackend implementing EmulatorBackend via plain adb
├── orchestrate.py            # bootstrap/launch/kill/supervise multi-emulator farm
├── vision.py                 # OpenCV pose extraction (HSV placeholders)
├── reward.py                 # TemplateScoreExtractor + zero_extractor
├── train.py                  # PPO with 3-config encoder split
├── eval.py                   # checkpoint loader + episode rollouts
├── eval_robustness.py        # §5.3 perturbation eval on Python clone
├── clone_env.py              # pymunk + pygame physics clone (eval-only per design)
├── .venv/                    # Python venv with deps
└── checkpoints/              # PPO checkpoints saved here
```

External (not under project dir):
```
/tmp2/b12902046/DRL_final/
├── android-sdk/              # ~3 GB
├── .android/avd/pixel5_api31.avd/   # AVD + clean_boot snapshot
├── bouncy-basketball-3-2-1.apk      # the game (scp'd by user)
├── emu_logs/                 # emulator stdout/stderr
├── endpoints.json            # set by orchestrate.py launch
└── frames/                   # captured screenshots
    ├── launch_*.png          # title screen post-launch
    ├── menu_*.png            # main menu
    ├── team_select_*.png     # CHI vs HOU select screen
    ├── customize*.png        # CUSTOMIZE screen
    ├── gameplay/g_001..050.png  # 50 frames during a Q1 match
    └── probe/*.png           # control-mapping experiments
```

## Useful one-liners

```bash
# Start a fresh emulator from snapshot (post-bootstrap)
source /tmp2/$USER/DRL_final_project/android_env.sh
nohup emulator -avd pixel5_api31 -port 6554 -no-window -no-audio -no-boot-anim \
  -gpu swiftshader_indirect -no-metrics -snapshot clean_boot \
  > /tmp2/$USER/DRL_final/emu_logs/run.log 2>&1 &

# Wait for boot
until adb -s emulator-6554 shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do sleep 3; done

# Capture a frame
adb -s emulator-6554 exec-out screencap -p > frame.png

# Force-stop and relaunch the game (use when ad sneaks in or state is broken)
adb -s emulator-6554 shell am force-stop com.DreamonStudios.BouncyBasketball
adb -s emulator-6554 shell monkey -p com.DreamonStudios.BouncyBasketball -c android.intent.category.LAUNCHER 1

# Smoke-test training against FakeBackend (no emulator needed)
.venv/bin/python train.py --env-id fake --aux-mode oca --total-timesteps 4096 --num-envs 2

# Smoke-test training against the Python clone
.venv/bin/python train.py --env-id clone --aux-mode dpr --total-timesteps 4096 --num-envs 4
```

## Decisions made today (2026-05-21)

1. **Use OpenCV shoe/jersey color thresholding for OCA targets, NOT YOLO.** Zero labeling cost, gives orientation for free (sin/cos of shoe-pair vector). YOLO documented as a fallback if color thresholding fails the §5.5 validation gate. Note: shoes are dark/black on both teams — jersey color (red CHI vs white HOU) is the better discriminator.
2. **Train on the real Android APK, not on the Python clone.** Per teammate preference. Python clone retained only for §5.3 robustness perturbation eval (where we need engine-level access to gravity/restitution/mass that the closed APK doesn't expose).
3. **Per-user adb daemon port** (computed from UID) to coexist on shared workstation without colliding with other students' emulators.
4. **Airplane mode is part of the snapshot.** Ads break the training pipeline; this is the most robust fix.
5. **CHI vs HOU at default stats**, no CUSTOMIZE edits. Reproducibility matters more than balance — strength is held constant across all 9 experimental configs so it's not a confound on the OCA/DPR comparison.
6. **Action space corrected to Discrete(2): {NO_PRESS, PRESS}.** Bouncy Basketball is a one-button game (confirmed via web search and in-game observation). Tap-and-hold anywhere = jump (longer hold = higher), release in air = shoot, lateral movement is automatic. The previous 6-action design (NOOP/LEFT/RIGHT/JUMP/CHARGE/RELEASE) was based on incorrect assumptions about platformer-style controls. AdbBackend now chains 130ms `input swipe` calls to simulate sustained holds.
