# Research Proposal: Self-Supervised Auxiliary Tasks for Representation Learning in Chaotic Physics-Based RL

**Project Title:** Representation Learning via Physics-Aware Auxiliary Tasks in a Real Mobile Game
**Target Venue:** NTU-DRL-MiniConf 2026 (NeurIPS 2026 Template Format)
**Authors:** [Team Name / Anonymous]
**Last Updated:** 2026-05-21

---

## Abstract

Model-free DRL from raw pixels struggles in chaotic, physics-driven environments because the convolutional encoder must simultaneously learn visual feature extraction and policy optimization from a sparse scalar reward signal. We investigate whether self-supervised auxiliary tasks accelerate representation learning in *Bouncy Basketball*, an Android arcade game with chaotic ball dynamics (gravity, elastic collisions, charged shots). A shared CNN encoder is trained under three configurations: (1) a pure PPO baseline, (2) **Object-Centric Auxiliary (OCA)** regression to per-frame object pose (position + orientation) extracted by a label-free OpenCV pipeline that localizes each player's shoes via HSV color thresholding, and (3) **Dense Pixel Reconstruction (DPR)** via a transposed-convolution decoder predicting the next frame. We evaluate sample efficiency, asymptotic score, and policy robustness across all three. Training is performed on a parallel Android-emulator farm (8 headless instances sharing one AVD via read-only snapshots); robustness perturbations that require engine-level physics control are evaluated on a lightweight Python clone of the same game.

---

## 1. Introduction and Motivation

Vision-based DRL has achieved monumental success in complex environments, but chaotic physical systems — those governed by elastic collisions, parabolas, and continuous physical variables — remain a bottleneck. When training a PPO or DQN agent from raw pixels in games like *Bouncy Basketball*, the network must learn both feature representation and policy optimization concurrently from only a scalar reward. Consequently, the agent spends millions of steps discovering basic concepts like gravity and ball trajectory implicitly.

This work addresses a focused question: **Does an auxiliary self-supervised objective accelerate physics-aware representation learning, and at what level of abstraction (sparse object-centric vs. dense pixel-level) is the supervisory signal most useful?**

By constraining the latent feature representation to predict either future object coordinates or the next rendered frame, the visual encoder is forced to preserve critical dynamics (position, velocity, trajectory) early in training. We systematically contrast a sparse object-centric coordinate-regression head against a dense pixel-level reconstruction head, providing concrete empirical guidance on the optimal level of abstraction for physical reasoning in RL agents.

```
+-------------------------------------------------------------------------+
|                                  INPUT                                  |
|               Stacked Pixel Frames [I_{t-3}, ..., I_t]                  |
+-------------------------------------------------------------------------+
                                     |
                                     v
                       +---------------------------+
                       |   Convolutional Encoder   |
                       |       E_\theta(s_t)       |
                       +---------------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Latent Feature Vector  |
                       |            z_t            |
                       +---------------------------+
                                     |
           +-------------------------+-------------------------+
           |                         |                         |
           v                         v                         v
+---------------------+   +---------------------+   +---------------------+
|    Policy Heads     |   |    Object-Centric   |   |   Pixel Decoder     |
|   Actor / Critic    |   |  Auxiliary Head (2) |   |  Auxiliary Head (3) |
+---------------------+   +---------------------+   +---------------------+
           |                         |                         |
           v                         v                         v
   Action Selection          Future Coordinates        Reconstructed Frame
    a_t & Value V_t           \hat{y}_{t+1}               \hat{I}_{t+1}
```

---

## 2. Theoretical Framework

We model *Bouncy Basketball* as a Partially Observable Markov Decision Process (POMDP) $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma)$:

- $\mathcal{S}$: unobserved true physical state.
- $\mathcal{A}$: discrete action space — Bouncy Basketball is a one-button 2v2 game, so $\mathcal{A} = \{\textsc{no\_press}, \textsc{press}\}$. A single touch controls **both** players on the agent's team simultaneously: each player jumps on press (the ball-holder for a shot, the other for defense/block), the longer the hold the higher the jump, and a release in the air shoots. Lateral movement is automatic (each player auto-walks toward the ball). Consecutive $\textsc{press}$ actions hold the touch (charging the jump); a $\textsc{press} \to \textsc{no\_press}$ transition triggers the shoot.
- $\mathcal{R}(s,a)$: reward function (+1 on goal scored, 0 otherwise; see §4.1).
- $\Omega$: observation space, $I_t \in \mathbb{R}^{H \times W \times C}$ (raw RGB frame from the emulator).

The agent observes a frame-stack of the $k=4$ most recent observations:
$$s_t = \{I_{t-3}, I_{t-2}, I_{t-1}, I_t\}$$

A convolutional encoder $E_\theta$ maps this to a latent vector:
$$z_t = E_\theta(s_t) \in \mathbb{R}^{512}$$

The global training objective is a multi-task loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \lambda \cdot \mathcal{L}_{\text{aux}}$$

where $\mathcal{L}_{\text{RL}}$ is the standard PPO clipped surrogate (plus value loss and entropy bonus), $\mathcal{L}_{\text{aux}}$ is configuration-specific, and $\lambda$ is swept in $\{0.1, 0.5, 1.0\}$.

---

## 3. Methodology: Three Experimental Configurations

### Shared Encoder

NatureCNN backbone (Mnih et al., 2015): 84×84×4 grayscale input → conv(32, 8, 4) → conv(64, 4, 2) → conv(64, 3, 1) → flatten → fc(512) → $z_t$. Identical across all configurations and seeds.

### Configuration 1 — Pure PPO Baseline (B)

$\lambda = 0$. Encoder trained end-to-end via the policy and value gradients only.
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}}$$

**Hypothesis:** the agent suffers high sample inefficiency because the encoder has no incentive to learn physics-relevant features prior to reward arrival.

### Configuration 2 — Object-Centric Auxiliary (OCA)

A 2-layer MLP head $f_\psi$ (512 → 256 → 18) regresses the pose of the ball and **each of the four individual players** in the *next* frame:
$$\hat{y}_{t+1} = f_\psi(z_t) = [\hat{x}_b, \hat{y}_b,\; \{\hat{x}_i, \hat{y}_i, \hat{\theta}_i\}_{i=1}^4]_{t+1}$$

For each player $i$, $(x_i, y_i)$ is the jersey-centroid in the frame and $\theta_i$ is the **body axis** — the angle of the vector from the player's feet (a low-saturation dark blob in the y∈[660,760] feet band) to the player's jersey centroid. For an upright player, $\theta \approx -\pi/2$ (axis points up); for a player mid-jump diving forward, $\theta$ tilts toward the horizontal; a dunking flip can push it past horizontal. Orientation is encoded as $(\sin\theta, \cos\theta)$ pairs to avoid the wrap-around discontinuity, expanding the regression target to **18 dimensions** (2 ball + 4 players × 4 components). The two CHI players occupy slots [2:6] and [6:10]; the two HOU players occupy [10:14] and [14:18]. Players within a team are sorted by x (leftmost first) so the target dim assignment is stable across frames.

**Target source:** a label-free OpenCV pipeline. *Bouncy Basketball* uses saturated team jersey colors (CHI red, HOU white) that are easily separable by HSV thresholding; the two players on each team are detected as connected components of the team's jersey color in <1ms per frame. Each team's centroid is the mean of its two player blobs; orientation $\theta$ is the angle of the vector connecting them. The ball is localized by an analogous HSV threshold on its saturated orange sprite. (We initially planned to use shoe color, but shoes are dark/black on both teams and don't separate well by hue.) Targets are computed once per frame and cached into the rollout buffer; the CV pipeline is **not** re-run during gradient updates.

$$\mathcal{L}_{\text{aux}}^{\text{OCA}} = \frac{1}{N} \sum_{i=1}^{N} \| y^{(i)}_{t+1} - \hat{y}^{(i)}_{t+1} \|^2_2$$

**Hypothesis:** forcing the latent to encode object pose — including orientation, which directly correlates with shot trajectory in this game — yields sharper localized filters, accelerates convergence, and produces more accurate shot-timing behavior than dense pixel reconstruction.

**Why pose, not just position:** *Bouncy Basketball* players rotate during jumps and dunks; the orientation of the player's body when releasing the ball materially affects shot trajectory under the game's physics. Including $\theta$ in the auxiliary target makes the supervisory signal explicitly physics-relevant rather than merely spatial, which sharpens the contrast against DPR's spatially-uniform reconstruction signal.

**Acknowledged limitation:** because the APK is closed-source, OCA targets are derived from a visual CV pipeline rather than engine ground truth. We measure CV pipeline accuracy on a small hand-labeled validation set (§5.5) and report it as a confound. **Fallback:** if shoe colors prove indistinguishable between players or shoes merge under contact, we train a YOLOv8n detector on ~200 hand-labeled frames as a drop-in replacement (architecturally identical OCA target, different target source).

### Configuration 3 — Dense Pixel Reconstruction (DPR)

A transposed-conv decoder $D_\phi$ (512 → 7×7×64 → 14×14×32 → 28×28×16 → 84×84×1) reconstructs the next frame:
$$\hat{I}_{t+1} = D_\phi(z_t)$$
$$\mathcal{L}_{\text{aux}}^{\text{DPR}} = \frac{1}{HW} \sum_{h,w} \| I_{t+1}(h,w) - \hat{I}_{t+1}(h,w) \|^2$$

**Hypothesis:** dense reconstruction forces the latent to encode the entire frame, including physics-irrelevant content (background, scoreboard UI), wasting capacity and producing slower convergence than OCA.

### Common PPO Hyperparameters

| Hyperparameter | Value |
|---|---|
| Parallel workers (envs) | 8 (one per emulator instance) |
| Rollout length per worker | 128 |
| Batch size | 256 |
| Update epochs | 4 |
| Learning rate | 2.5e-4 (linear decay) |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| Clip $\epsilon$ | 0.1 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Total env steps | 1.0M per seed |
| Seeds per configuration | 3 |
| Frame skip | 4 (action repeated 4 frames) |

---

## 4. Environment Infrastructure

### 4.0 Opponent Distribution

We train against a *pool* of opposing teams rather than a single fixed opponent. The agent's team is held constant (CHI) but each `env.reset()` samples uniformly from a set of pre-saved snapshots, each one corresponding to a different opposing team (e.g. HOU, LAC, IND, DAL, …). Concretely we hold-out the team selection from the snapshot save, so each snapshot encodes "CHI vs $X$" for a different $X$, and the agent sees a different opponent AI each episode.

This is motivated by two concerns. First, training against a single opponent — especially one as weak as default-difficulty HOU — invites overfitting: the encoder could learn idiosyncratic features specific to HOU's play style rather than physics-aware features that generalize. Second, evaluating only against the same opponent would yield optimistic asymptotic scores that fail to generalize. By interleaving N opponents we reduce both risks at the cost of some increased variance in the learning curve (which we absorb by running 3 seeds per configuration).

In §5.2 we report mean score across *all* opponents in the pool; in §5.3 we extend the perturbation evaluation to also include held-out opponents not seen during training, as a generalization probe.

### 4.1 Primary Training Environment — Bouncy Basketball on Android

- **AVD:** Pixel 5, API 31, x86_64 with Google APIs (rooted not required for this path).
- **Parallelism:** 8 headless emulator instances sharing a single AVD via `-read-only` copy-on-write snapshots, on a single CSIE workstation. Precondition: `/dev/kvm` accessible (verified before commitment).
- **Snapshot reset:** episodic reset via `adb emu avd snapshot load clean_boot` (~10s vs. ~5min cold boot).
- **Frame capture:** `minicap` over TCP (~60fps raw frames), replacing `adb exec-out screencap` which is 30–80ms/frame.
- **Action injection:** `minitouch` persistent socket (~5ms/tap), replacing `adb shell input tap` (80–150ms).
- **Action space:** discrete, 2 actions {no_press, press}. Bouncy Basketball is a one-button 2v2 game: a single touch controls **both** of the agent's team players simultaneously. Tap-and-hold anywhere on screen jumps the players (longer hold = higher), release in air shoots, lateral movement is auto. Consecutive `press` actions keep the touch held, charging the jump.
- **Observation:** 84×84 grayscale, frame-stack $k=4$.
- **Reward:** +1 on goal, detected by template matching on the scoreboard region of the captured frame. Validated against manual counting on 10 episodes before training (target ≥98% agreement).
- **Process supervision:** each worker monitors `adb -s emulator-<port> get-state`; on failure, restarts the emulator from snapshot and resumes. Required — headless emulators crash under sustained load.

### 4.2 Auxiliary Evaluation Environment — Lightweight Python Clone

Used **only** for the §5.3 robustness perturbation study, which requires engine-level access to physics constants unavailable in the closed APK.

- **Stack:** pymunk (2D physics) + pygame (headless rendering).
- **Scope:** matches *Bouncy Basketball*'s physics regime (parabolic shots, elastic bounces, gravity) but not its visual fidelity.
- **Same observation and action space as the Android env**, ensuring policies trained on the Android env can be evaluated on the clone (with mild domain gap acknowledged).
- **Configurable physics constants:** gravity, ball restitution, ball mass exposed as `__init__` kwargs.

---

## 5. Evaluation Protocol

### 5.1 Sample Efficiency (primary metric)

Number of environment steps to reach 50% and 80% of asymptotic mean score, per configuration, averaged over 3 seeds. Reported with bootstrap 95% CI.

### 5.2 Asymptotic Performance

Mean score over 100 evaluation matches at training end. Reported with bootstrap 95% CI per configuration.

### 5.3 Robustness to Physics Perturbation (Python clone only)

After training on the Android env, transfer the policy (no fine-tuning) to the Python clone and evaluate under:

- Gravity: $g \in \{0.75 g_0, g_0, 1.25 g_0\}$
- Restitution: $e \in \{0.80 e_0, e_0, 1.20 e_0\}$
- Ball mass: $m \in \{0.70 m_0, m_0, 1.30 m_0\}$

Metric: score retention ratio relative to nominal-physics evaluation. Reports both raw scores and ratios.

### 5.4 Auxiliary Signal Quality (diagnostic)

- OCA: MSE of $\hat{y}_{t+1}$ vs. detector targets on held-out trajectories. Confirms the aux head is actually solving its task.
- DPR: PSNR and SSIM of $\hat{I}_{t+1}$ on held-out trajectories.

Without this check, a flat learning curve could be confused with "aux head failed to train" vs. "aux head trained but did not help RL."

### 5.5 CV Pipeline Validation (OCA prerequisite)

Held-out test set: ~30 hand-labeled frames spanning diverse game states (mid-air, ground, contact, occlusion). For each frame we manually annotate the two shoe centroids per player and the ball centroid. Reported metrics:

- **Detection rate:** fraction of frames in which both shoes of each player are recovered as exactly two connected components (target ≥ 95%).
- **Centroid error:** mean L2 pixel distance between predicted and annotated centroids, per object (target ≤ 2 px at 84×84 resolution, equivalent to ~3% of frame width).
- **Orientation error:** angular error of the shoe-pair vector vs. ground truth, on frames where both shoes are detected (target ≤ 10°).

If detection rate < 90% for any object, we fall back to training YOLOv8n on the same labeled set plus ~170 additional auto-labeled frames (bootstrapped with SAM2). CV pipeline accuracy is reported in the paper as a confound on OCA.

---

## 6. Execution Plan (May 21 – May 25, 2026)

Today: **May 21**. Submission deadline: **May 25, 23:59**.

| Phase | Date | Tasks | Owner |
|---|---|---|---|
| 0 | May 21 (today) | Verify `/dev/kvm` on workstation. Decide go/no-go on Android path. Set up 1 emulator + minicap + minitouch end-to-end. Capture 250 raw frames for labeling. | Person A |
| 0 | May 21 | Capture 50 diverse emulator frames. Tune HSV thresholds for ball + each player's shoes. Build OpenCV pipeline returning `(pos, theta)` per player + `(pos)` for ball. Validate on 30 hand-labeled frames against the §5.5 thresholds. If pass: ship. If fail: pivot to YOLOv8n fallback (still day-1 scope). | Person B |
| 0 | May 21 | Fork CleanRL `ppo_atari.py`. Refactor encoder into shared module; scaffold three configuration toggles (B / OCA / DPR). Smoke-test on Atari Pong. | Person C |
| 1 | May 22 | Stand up 8-emulator parallel farm with snapshot reset + supervisor + scoreboard reward detection. End-of-day target: random policy steps at ≥150 steps/sec aggregate. | Person A |
| 1 | May 22 | Wire CV pose pipeline into env step loop, caching `(pos, theta)` targets into rollout buffer. Stress-test on 5000 random-policy frames to confirm robustness (no NaNs, no dropped frames). Build Python clone (eval only) in parallel. | Person B |
| 1 | May 22 | Integrate vector env with PPO trainer. Run baseline (Configuration 1) shakedown at 100k steps to verify learning curve. | Person C |
| 2 | May 23 | Launch full training: 3 configs × 3 seeds × 1M steps. With 8 emulators @ ~20 steps/sec = ~160 aggregate steps/sec, ~1.75h pure stepping per run; ~9 runs total over ~16 GPU-hours. Monitor for emulator crashes. | All |
| 2 | May 23 | $\lambda$ sweep on the leading configuration (best-performing of B/OCA/DPR at 1M steps) at $\lambda \in \{0.1, 0.5, 1.0\}$ — single seed each, to identify if scaling matters. | Person C |
| 3 | May 24 | Robustness evaluation in Python clone for all 9 trained policies × 9 perturbation conditions. Generate sample-efficiency plots, asymptotic bar charts, diagnostic plots (§5.4), detector accuracy report (§5.5). | Persons A & B |
| 3 | May 24 | Begin paper draft: introduction, methodology, results scaffolding. | Person C |
| 4 | May 25 | Complete paper draft (NeurIPS LaTeX). Figures finalized. Limitations section written. Submit before 23:59. | All |

**Slack:** zero. **Critical-path risks:** KVM unavailable on workstation (kills Day 1), detector mAP < 0.85 (forces fallback to CV thresholding, +0.5 day), emulator instability (mitigated by supervisor).

**Fallback plan if Day 1 Android infrastructure slips:** abandon the APK as training environment, scale the Python clone up to full training quality (engine-true coordinates make OCA *cleaner*, not worse), and reframe the paper around the clone with the APK used only for a sim-to-real anecdote in §5. This preserves the science at the cost of the "real game" framing.

---

## 7. Limitations and Risks

1. **OCA supervisor noise.** Targets are CV-pipeline outputs (HSV thresholding on shoe colors), not engine state. The OCA-vs-DPR comparison is therefore between two visual processes (CV-supervised vs. self-supervised), not between symbolic and visual. We report CV pipeline accuracy (§5.5) as part of the result. The CV pipeline does, however, provide richer signal than typical bbox supervision because it recovers orientation $\theta$ — a quantity directly relevant to the game's physics.
2. **Throughput ceiling.** Android emulators cap at ~20 steps/sec/instance, so 8 parallel instances yield ~160 steps/sec — orders of magnitude slower than Atari-style envs. Total budget of 1M steps × 9 runs is feasible only with continuous training over Days 2–3 and zero crashes.
3. **Domain gap for §5.3.** Policies trained on the APK and evaluated on the Python clone face a visual + physics domain gap. We report nominal-physics clone scores as a baseline so that gravity/restitution/mass perturbations are measured *relative* to the clone's own nominal, isolating perturbation effects from transfer effects.
4. **Sparse rewards.** *Bouncy Basketball* rewards (goal scored) are sparse on the millisecond timescale. We rely on frame-skip 4 and PPO's entropy bonus to drive exploration. If learning stalls on Configuration 1, we add a small dense shaping term (e.g., reward proportional to ball-rim distance reduction) and apply it uniformly to all three configurations to preserve comparability.
5. **Single-game generalization.** All results are on one game. Generalizing claims about "physics-aware representation learning" beyond *Bouncy Basketball* is scoped accordingly in the paper.

---

## 8. Expected Scientific Contributions

1. A quantitative analysis of whether self-supervised predictive auxiliary tasks accelerate learning in a real chaotic physics game running on commodity Android emulation infrastructure.
2. A direct empirical comparison between sparse object-centric (CV-pose-supervised, including orientation) and dense pixel-level (self-supervised) auxiliary signals, with a recommendation on which paradigm yields better sample efficiency and robustness in physics-based RL.
3. A diagnostic methodology for separating "auxiliary task learned its target" (§5.4) from "auxiliary task helped the policy," addressing a frequent ambiguity in the auxiliary-task literature.
4. Open-source reproducible code: parallel Android-emulator RL pipeline, detector-supervised OCA implementation, and Python physics clone for perturbation studies.
