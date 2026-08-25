# PFSP League for Lux AI Season 1 — Results

Prioritized Fictitious Self-Play added to the Toad Brigade (1st place, Kaggle Lux AI 2021)
training loop, replacing pure self-play with opponents sampled from a pool of frozen
checkpoints, weighted toward those the agent is not reliably beating.

---

## Summary

**The mechanism was built and demonstrably works. It did not make the agent stronger.**

Two 1M-step runs (PFSP and a self-play baseline) were compared on 720 held-out games.
Neither run improved measurably on the model it started from. Three independent
diagnostics agree that the training itself made no directed progress, and a follow-up
experiment ruled out the learning rate as the cause.

This is a negative result, and the evidence for it is stronger than "we ran out of time".

---

## 1. What was built

| Component | Purpose |
|---|---|
| `lux_ai/league/` (~1,000 lines) | Pool, win-rate tracker, sampler, manager, actor client |
| `lux_ai/league/tests/` (~430 lines, 32 tests) | Unit tests + a baseline-equivalence proof |
| `conf/league_{sp,fsp,pfsp}.yaml` | Three comparison arms, differing by exactly one line each |
| `evaluation/league_eval.py` | Held-out evaluation across snapshots |
| `evaluation/compare_league_runs.py` | Paired statistical comparison |

**Design properties that were verified, not just intended:**

- **The league is isolated from the learner.** Nothing in `lux_ai/league/` imports from
  `lux_ai/torchbeast/`; a grep for IMPALA, V-trace, GAE or clipping inside it returns only
  a docstring saying so. Swapping the RL algorithm would not touch this code.
- **`league.enabled: false` provably reproduces the original learner.** Whole-run
  comparison cannot show this — two identical runs of this trainer already diverge more
  (5.2e-03 max parameter difference) than the league branch does (3.3e-03). So instead,
  `test_learn_equivalence.py` pulls the pre-league `learn()` out of git and runs it against
  the current one on the same batch and weights in one process. All seven loss terms match
  bit-for-bit.

### The algorithm, in three lines

```
weight[i] = (1 - winrate[i]) ** alpha        # alpha = 0.5
prob[i]   = weight[i] / sum(weight)
prob[i]   = (1 - eps) * prob[i] + eps / N    # eps = 0.1, uniform floor
```

Half of games are ordinary self-play (`mirror_frac: 0.5`); the rest draw an opponent from
the pool. The learning agent's seat is randomized per episode, and the frozen opponent's
transitions are masked out of every loss term so the agent never trains on its opponent's
moves.

---

## 2. Does the mechanism work? Yes

From the 1M-step PFSP run:

| Diagnostic | Result |
|---|---|
| Pool occupancy | filled to **10 / 10** |
| **Effective pool size** (`1 / Σp²`) | **9.72 of 10** |
| League games played | 861 (of 2,776 total) |
| Snapshots created | 31 |
| Masked experience fraction | 0.217 (expected ~0.25) |

**Sampling is correctly inverse to win rate** — the whole point of the priority function:

| Opponent | Win rate against it | Sampling probability |
|---|---|---|
| `11-24` (released final model) | 0.389 | **0.1120** ← played most |
| `10-02` | 0.951 | **0.0415** ← played least |

**Reservoir replacement kept history.** The retained snapshots span steps 64k, 193k, 289k,
353k, 674k, 706k, 867k and 963k — not just the most recent eight. Plain evict-oldest would
have filled the pool with near-identical copies of the current policy, which is the
degeneration the design was built to avoid.

Effective pool size never fell below ~9.7, so the league never collapsed into
"a frozen teacher with extra steps".

---

## 3. Does it improve the agent? No

### Held-out evaluation — 720 games

Three candidates, each playing the **same 240 matchups**: 2 held-out opponents × 15 seeds
× 4 map sizes × both seats. Neither held-out opponent was ever a pool member.

| Candidate | Win rate (95% CI) | Mean city-tile margin |
|---|---|---|
| **SP (self-play baseline)** | **77.9%** (72.3–82.7) | +49.3 |
| `11-24` — starting model, untrained | 74.8% (68.9–79.9) | +45.1 |
| **PFSP (league)** | 72.5% (66.5–77.8) | +33.6 |

The ordering is the opposite of the hypothesis, and all three intervals overlap heavily.

### Paired comparison

Because every candidate played identical seeds and maps, matches can be lined up
one-to-one — far more sensitive than comparing independent intervals.

| Comparison | Mean score difference | City-margin difference |
|---|---|---|
| PFSP − SP | −0.054 (−0.122, +0.014) | **−15.6 (−30.4, −0.8)** |
| PFSP − starting model | −0.023 (−0.095, +0.049) | −11.4 (−26.2, +3.3) |
| SP − starting model | +0.031 (−0.033, +0.096) | +4.2 (−8.4, +16.7) |

Only one interval excludes zero, and it is the city-tile margin **favouring the baseline**.

**The most informative number is not a win rate:** in the PFSP-vs-SP pairing,
**171 of 240 matchups ended identically** — same winner, same final board. After a million
steps the agents are still playing very nearly the same game.

---

## 4. Why not? Three diagnostics

### Diagnostic A — the policy barely moved, and moved undirectedly

Distance travelled in parameter space from the starting weights, excluding spectral-norm
buffers (which drift on every forward pass regardless of learning):

```
total drift after 1M steps : 3.08   (3.0% of the 101.47 weight norm)
first half of training     : +1.67
second half                : +0.83
```

Distance grew as **√t**, almost exactly. That is the signature of a random walk: successive
updates largely cancel. Directed learning would grow closer to linearly, because each step
would build on the last.

### Diagnostic B — the held-out learning curve is flat

Six checkpoints evaluated against held-out `11-09`, 40 games each, identical seeds/maps:

```
step 0        48.8%
step 193k     52.5%
step 386k     55.0%
step 578k     50.0%
step 771k     43.8%
step 1.00M    60.0%
```

Weighted linear fit: **slope +4.1 points over the whole run, 95% CI −14.5 to +22.6,
t = 0.43 — no significant trend.** Expected noise from 40 games alone is ±7.9 points per
point, so the scatter is entirely consistent with a flat line at ~52%. The curve is also
non-monotone (up to 55%, down to 43.8%, up to 60%) — a random walk, not learning.

### Diagnostic C — a higher learning rate made it worse, not better

Hypothesis tested: the run used `lr: 1e-5`, inherited from a config written for continuing
a ~900M-step run, whereas phase-5 training originally used `5e-5`. A 200k-step diagnostic
was run at `5e-5`, everything else identical.

| Measure | 1e-5 | 5e-5 |
|---|---|---|
| Drift at ~193k steps | 1.42 | **6.67** (4.7× — matches the 5× lr) |
| Drift shape | √t | **still √t** (halves 3.32 / 1.71) |
| Win rate vs anchor `11-24` | 0.389 | **0.222** |
| Win rate vs anchor `10-10` | 0.885 | **0.628** |
| Held-out vs `11-09` | 60.0% | 52.5% |
| Stability | fine | fine — `baseline_loss` bounded, no entropy collapse |

The optimizer responded exactly as expected — 5× the learning rate moved the weights 4.7×
further. But the agent got **substantially worse against fixed opponents, five times
faster**. This is not numerical instability; it is smooth optimization toward something
that makes the agent weaker.

**Conclusion: the learning rate was not the bottleneck.** The released checkpoint sits at a
strong local optimum, and the available fine-tuning signal pushes it downhill.

---

## 5. Methodological notes

Two things worth stating because they affect how the numbers should be read.

**A claimed confound that turned out not to exist.** It was initially argued that masking
the opponent's seat shrinks the summed loss by ~22% and therefore lowers PFSP's effective
learning rate. Direct measurement disproved this: `clip_grads: 10.0` rescales gradients
(measured norms ~7×10⁵, five orders of magnitude above the threshold) to exactly 10 before
Adam, and Adam is itself scale-invariant. Scaling the loss changed parameter movement by a
factor of **1.000000** with clipping and **0.999995** without it. The proposed "fix" would
have done nothing.

**A real asymmetry that does exist.** Self-play trains on both seats of every game; the
league trains on both seats only in mirror games. PFSP therefore learns from ~78% as much
experience per environment step. This does not shrink the step size (see above) — it means
each step's direction is estimated from fewer samples. To match SP on trainable transitions
rather than environment steps, PFSP would need ~28% more steps. This is inherent to the
method, not an implementation defect.

**An unrelated pre-existing quirk.** `teacher_kl_loss` logs as NaN in 100% of samples,
which makes `total_loss` NaN. This is a reporting artifact present in the original code:
the NaN comes from `0 × (log 0 − (−∞))` on masked actions, whose derivative is `−target`
= 0, so gradients stay finite and training is unaffected. Use `vtrace_pg_loss`,
`upgo_pg_loss`, `baseline_loss` and `entropy_loss` for monitoring instead.

---

## 6. Limitations

- **Two held-out opponents.** Only `11-09` and `09-17` were usable —
  `09-07` cannot be rebuilt by the current code (its config sets an `obs_space` kwarg that
  no longer exists), and every other available agent was needed in the pool.
- **240 games gives roughly ±6 points of resolution.** An effect smaller than that could
  not have been detected regardless of the result.
- **The FSP arm was not run.** GPU capacity allowed one run at a time; SP and PFSP were
  prioritized. Without FSP, "having a pool" cannot be separated from "prioritizing hard
  opponents" — though with neither arm improving, that separation is currently moot.
- **Single seed per arm.** Both runs used `seed: 42`. Run-to-run variance was not measured.

---

## 7. What we would do next

The remaining lever is **not** the learning rate. It is the training signal itself:

1. **Start from a less converged checkpoint.** The 09-17 or 10-10 models have more room to
   improve; fine-tuning a model already at a strong optimum may simply have no upside.
2. **Revisit the teacher KL.** `teacher_kl_cost: 0.001` anchors the policy to the released
   model. If the goal is to move away from it, that term is working against the objective.
3. **Reward shaping.** `GameResultReward` gives a single win/loss signal at turn 360.
   Phase-1 of the original training used a shaped reward for exactly this reason.
4. **Match on trainable transitions, not environment steps** — give PFSP ~28% more steps
   so both arms see the same amount of experience.

**Cheap kill-switches, now available.** Both would have flagged this run within one hour
instead of fourteen:

- **Parameter drift** — free, seconds, from snapshots. √t growth means updates are
  cancelling and nothing is being learned.
- **Anchor win rate** — already logged continuously to wandb against fixed opponents that
  never change. If it has not started climbing by 15–20% of a run, stop the run.

---

## 8. Reproduction

```bash
# three comparison arms, differing by one config line each
python run_monobeast.py --config-name league_sp   total_steps=1000000
python run_monobeast.py --config-name league_fsp  total_steps=1000000
python run_monobeast.py --config-name league_pfsp total_steps=1000000

# held-out evaluation and paired analysis
python -m evaluation.league_eval --snapshots league_agents/pfsp_final league_agents/sp_final \
       --n-seeds 15 --map-sizes 12 16 24 32
python -m evaluation.compare_league_runs --root evaluation_runs/league_final

# tests, including the baseline-equivalence proof
python -m pytest lux_ai/league/tests/ -q      # 32 tests, ~8 s
```

Run one training job at a time — three concurrent runs saturated the 16 GB GPU and slowed
each by roughly 30×.

| Artifact | Location |
|---|---|
| Trained agents | `league_agents/{pfsp_final, sp_final, diag_lr5e5_200k}/` |
| Evaluation results | `evaluation_runs/league_final/` |
| Training logs | `league_run_logs/`, `outputs/<date>/<time>/` |
| League design notes | `lux_ai/league/README.md` |

### Runtime

| Run | Steps | Wall time | Throughput |
|---|---|---|---|
| PFSP | 1,000,192 | 5 h 15 m | ~62 steps/s |
| SP | 1,000,192 | 3 h 15 m | ~86 steps/s |
| Diagnostic (5e-5) | 200,192 | 1 h 00 m | ~51 steps/s |
| Held-out evaluation | 720 games | 2 h 40 m | — |
