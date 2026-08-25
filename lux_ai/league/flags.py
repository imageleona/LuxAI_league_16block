import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LeagueFlags:
    """
    All league knobs, parsed from the `league:` block of the run config.
    Plain data only, so instances are picklable into spawned actor processes.
    """
    enabled: bool = False
    pool_size: int = 10
    # Directories of permanent pool members (never evicted).
    anchors: List[str] = field(default_factory=list)
    # Directories used to seed the remaining pool slots at startup.
    initial_members: List[str] = field(default_factory=list)
    # Fraction of episodes played against the current policy (ordinary self-play).
    mirror_frac: float = 0.5
    # Freeze a snapshot of the current policy into the pool every N learner updates.
    snapshot_interval_updates: int = 500
    # Priority function: pfsp -> (1 - winrate) ** alpha, variance -> winrate * (1 - winrate), uniform -> 1
    priority: str = "pfsp"
    alpha: float = 0.5
    # Uniform mixture floor so no member's win-rate estimate goes stale.
    eps_uniform: float = 0.1
    # Minimum share of league games given to EACH permanent anchor. Anchors never
    # change, so they are the only fixed yardstick during training; without a floor,
    # PFSP starves the ones the agent beats and the yardstick loses data exactly when
    # it is most wanted. 0 disables.
    anchor_floor: float = 0.15
    # Win rate against the permanent anchors is reported over a rolling window of
    # this many anchor games. Anchors never change, so it is the one fixed progress
    # yardstick during training. Counted in GAMES, not steps: games finish at
    # arbitrary step counts, so a step bucket would slice through them.
    anchor_report_every_n_games: int = 100
    # --- periodic fixed-seed evaluation against the permanent anchors ---
    # The rolling window above is built from TRAINING games, which vary in map size,
    # seed and seat; that variance swamps a few points of drift (measured: resolving
    # the 2.6-point decline of the 5M run would need ~1,400 games ~= 7M steps). These
    # knobs instead replay a fixed, paired schedule against each anchor, where
    # round-to-round change is far more sensitive than any absolute win rate.
    # Cost is real: 20 games/anchor x 4 anchors ~= 19 min/round on an idle machine,
    # more under contention with the actors. Disabled by default.
    anchor_eval_enabled: bool = False
    # Counted in league games (mirror games excluded, see actor_client.handle_dones).
    anchor_eval_every_n_games: int = 500
    # n_seeds x len(map_sizes) x 2 seats games per anchor.
    anchor_eval_n_seeds: int = 5
    # Must be a subset of (12, 16, 24, 32): MatchRequest rejects anything else.
    anchor_eval_map_sizes: List[int] = field(default_factory=lambda: [16, 32])
    anchor_eval_parallel_games: int = 8
    anchor_eval_device: str = "cuda:0"
    # Evaluate once before training starts, so later rounds compare against a
    # measured baseline rather than an assumed 0.5.
    anchor_eval_at_start: bool = True
    # Run the round on the main loop instead of a worker thread. Measured: the
    # league outcome queue holds 136 outcomes (~29 min of games), and a real round
    # under contention can exceed that - at which point actors block in put() and
    # the learner starves. Only safe when rounds are short, i.e. smoke tests.
    anchor_eval_blocking: bool = False
    # Give up permanently after this many consecutive failed rounds rather than
    # retrying a broken eval every 500 games for 20 hours.
    anchor_eval_max_consecutive_failures: int = 2

    # --- main exploiter (AlphaStar-style) ---
    # An exploiter is an ordinary run pointed at a single opponent: pool_size 1, one
    # anchor, mirror_frac 0.0. It trains until it beats that target, then gets frozen
    # into the main agent's pool. These two keys add the "until it beats it" part.
    # Stop when the rolling anchor win rate reaches this value. 0 disables the early
    # stop entirely, which is what every non-exploiter config wants.
    exploiter_target_winrate: float = 0.0
    # Minimum games in the rolling window before the stop may fire. Guards against
    # ending a run on a lucky streak; the window itself is anchor_report_every_n_games.
    exploiter_min_games: int = 100

    # --- win-rate-gated admission (second, independent admission path) ---
    # Interval snapshots keep entering the reservoir as normal. IN ADDITION, when the
    # current policy is reliably beating a fixed reference opponent, a snapshot is
    # force-admitted to the pool. Motivation: reservoir sampling admits snapshots by
    # coin flip regardless of strength, which is how the 1M run ended with a pool of
    # near-clones all sitting around 0.5 win rate.
    # 0. disables this path entirely, leaving admission exactly as before.
    winrate_admit_threshold: float = 0.0
    # Games against the reference opponent used for the estimate. 50 games is +-7
    # points (1 sigma), so a true 0.55 policy clears a 0.6 gate about a quarter of
    # the time - the cost of measuring this from training games at this rate.
    winrate_admit_window: int = 50
    # Substring identifying the reference opponent among the anchors. Empty = the
    # first configured anchor, which is the checkpoint the run started from.
    winrate_admit_reference: str = ""
    # Minimum learner updates between two win-rate admissions, so a sustained streak
    # does not admit a snapshot on every single main-loop tick.
    winrate_admit_cooldown_updates: int = 500

    # EMA parameters for the per-member win-rate estimate.
    winrate_lambda: float = 0.1
    winrate_init: float = 0.5
    # Max opponent models resident per actor process (LRU beyond this). Keep
    # this >= the number of distinct opponents typically assigned across the
    # actor's envs, or the cache thrashes with a ~0.5s disk reload per step.
    max_loaded_opponents: int = 8
    # Sample opponent actions (True) or take argmax (False).
    opponent_sample: bool = True
    # Absolute path of the published league state file. Set by the training
    # entrypoint before actors spawn; not meant to be set from yaml.
    state_path: Optional[str] = None

    @classmethod
    def from_dict(cls, config: Optional[Dict]) -> "LeagueFlags":
        if not config:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{key: val for key, val in config.items() if key in known})

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)
