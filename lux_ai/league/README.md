# PFSP opponent league

Replaces pure self-play with Prioritized Fictitious Self-Play (AlphaStar-style):
a bounded pool of frozen checkpoints, sampled per episode with priority toward
opponents the current agent is *not* reliably beating. Motivation: pure
self-play causes strategic cycling and forgetting; the tournament format
(round robin vs. six unknown opponents) rewards robustness across a
distribution of opponents, not peak strength against one.

## Pipeline

```
winrate[i] -> priority fn -> weight[i] -> normalize -> prob[i]
priority fns: pfsp     weight = (1 - winrate) ** alpha
              variance weight = winrate * (1 - winrate)
              uniform  weight = 1                       (FSP)
prob = (1 - eps_uniform) * weight/sum(weight) + eps_uniform / N
```

Per episode: with probability `mirror_frac` the game is ordinary self-play
(both seats the current policy); otherwise an opponent is drawn from `prob`
and the learner's seat is randomized.

## Architecture

- The training loop touches the league through exactly three operations:
  opponent assignment per episode (actor side), outcome reporting per episode,
  and snapshotting every `snapshot_interval_updates` learner updates.
  **This package imports nothing from `lux_ai.torchbeast`**, so swapping the
  learner algorithm (IMPALA -> APPO/PPO) does not touch the league.
- **Cross-process state:** actors are spawned processes. Outcomes flow
  actor -> main through an `mp.SimpleQueue`; pool membership + sampling probs
  flow main -> actors through a versioned, atomically-written
  `league/state.json` in the hydra run dir (re-read at episode boundaries).
  The state file doubles as the resume mechanism and is human-inspectable.
- **Seat splitting:** the baseline network produces both players' actions in
  one forward pass. For league games the frozen opponent model runs a second
  forward on the same env output (grouped by opponent, one batched call per
  distinct opponent), and its seat's actions are spliced into the executed
  action tensors, so buffers record the actions actually taken.
- **Learner masking:** a `league_mask` buffer field ((T+1, N, 2), 1. for
  learner seats) zeroes the opponent seat out of every loss term (v-trace and
  UPGO advantages, baseline, entropy, teacher KL). All shipped configs use
  `reduction: sum`, so zero-masking is exact. Mirror games carry an all-ones
  mask, which is a float no-op — with `league.enabled: false` the code path
  is unchanged entirely.
- **Pool policy:** bounded (default 10), two permanent anchors (released final
  model + an earlier hall-of-fame checkpoint), remaining slots seeded with
  external agents and thereafter filled by reservoir replacement over the
  snapshot stream (keeps a uniform sample of history; never plain
  evict-oldest, which collapses the pool to recent near-clones).
- Opponent models are built against the *training env's* obs space (with the
  member's `student_`/`teacher_` prefix for MultiObs envs) and admitted only
  if their obs/act space matches; each actor keeps an LRU cache of at most
  `max_loaded_opponents` models on the actor device.

## Notes

- **Map size needs no randomization here**: `LuxEnv` never fixes
  width/height, so the engine picks from {12, 16, 24, 32} pseudo-randomly per
  seed, and the seed auto-increments every reset.
- **Old agent configs need patched defaults.** Configs written before a model
  kwarg existed (the 2021-10 agents predate `sum_player_embeddings`) crash
  `create_model`, and an actor process dies *silently* on an unhandled
  exception - training just stalls with no traceback. `member_config.py` fills
  those keys with the value the trainer used at the time (verified against the
  352-channel embedding merger in those checkpoints) and `validate_member()`
  builds and loads every candidate on CPU at admission, so a member that
  cannot load is rejected before an actor can ever touch it.
- `hall_of_fame/09-07_01-44-10_10088000` cannot be rebuilt at all: its config
  passes an `obs_space_kwarg` the current obs space no longer accepts. It is
  unusable as either a pool member or an eval opponent.
- Outcomes come from the terminal `GameResultReward` (+1/-1, ties (0, 0)) and
  score draws as 0.5; win rates are updated from training games only.
- Snapshots are written in the frozen-agent layout
  (`league/snapshots/<step>/lux_ai/rl_agent/{config.yaml, rl_agent_config.yaml,
  <step>_weights.pt}`), so `evaluation.harness.AgentSpec.from_directory` loads
  them directly.

## Diagnostics (wandb, `League/*`)

1. `prob/<member>` over time - shows the mechanism is doing something.
2. `effective_pool_size` (1 / sum(prob^2)) - if this collapses to 2-3, the
   league has degenerated into a frozen teacher. Main failure mode to watch.
3. Held-out win rate - `python -m evaluation.league_eval` pits snapshots
   against agents that were **never** pool members. This is the real claim.
4. `actual_snapshot_interval` + `winrate/<newest snapshot>` - if the win rate
   against a fresh snapshot sits near 50%, the interval is too short and the
   pool is filling with near-identical models.

## Anchor evaluation (`AnchorEval/*`)

The diagnostics above are computed from **training** games, which vary in map size,
seed and seat. That variance swamps the effect worth watching: on the 5M run,
resolving the 2.6-point decline against the starting checkpoint would have needed
~1,400 games (~7M steps). The rolling window is a blow-up alarm, not a progress
meter.

`evaluation/anchor_eval.py` replays a **fixed, paired** schedule against each
permanent anchor instead, so round-to-round change is far more sensitive than any
single absolute win rate. Enable it with the `anchor_eval_*` keys in the `league:`
block; `anchor_eval_enabled: false` (the default) makes it a complete no-op.

```
AnchorEval/win_rate_mean          <- read THIS, not the per-anchor lines
AnchorEval/win_rate/<anchor>      <- 20 games is +-15pp on its own
AnchorEval/ci95_low|ci95_high|city_margin|games/<anchor>
AnchorEval/round, eval_step, step_lag, wall_seconds, rounds_skipped, failures
```

Results also append to `<run_dir>/league/anchor_eval/anchor_eval.{txt,jsonl}`.

Three things to know before turning it on:

- **It is not free.** 20 games/anchor x 4 anchors is ~19 min per round on an idle
  machine and more under contention; at `every_n_games: 500` that is ~4 h added to
  a 22 h run.
- **It runs on a worker thread, not the main loop.** `league_outcome_queue` is an
  `mp.SimpleQueue`, measured to hold **136** outcomes (~29 min of games) before
  `put()` blocks the actors and starves the learner. A blocking round can exceed
  that. `anchor_eval_blocking: true` is for smoke tests only.
- **This package still imports nothing from the learner or the harness.** The
  dependency is owned by the trainer:
  `lux_ai.torchbeast -> evaluation -> lux_ai.{nns, lux_gym, league}`. The scheduler
  lives in `lux_ai/torchbeast/anchor_eval_hook.py` and the games in
  `evaluation/anchor_eval.py`, precisely so the rule at the top of `__init__.py`
  keeps holding.

## Tests

`python -m pytest lux_ai/league/tests/` (32 tests, ~8 s). Two are worth
knowing about:

- `test_pool_preset_members_are_loadable` - every agent the presets put in the
  pool must build and load, guarding the silent-actor-death failure above.
- `test_learn_equivalence.py` - proves `league.enabled: false` leaves the
  learner untouched. Note that comparing two whole training runs cannot show
  this: this trainer is not run-to-run reproducible (cuDNN/AMP plus
  actor/learner interleaving), and measured control divergence between two
  identical runs (5.2e-03 max parameter difference) exceeded the divergence
  between baseline and league-disabled (3.3e-03). So the test instead pulls
  the pre-league `learn()` out of git and runs it against the current one on
  the same batch and weights in a single process, requiring all seven loss
  terms to match bit-for-bit.

## Comparison runs

Same entry point, only the config differs:

```
python run_monobeast.py --config-name league_sp     # pure self-play baseline
python run_monobeast.py --config-name league_fsp    # uniform pool sampling
python run_monobeast.py --config-name league_pfsp   # prioritized sampling
```
