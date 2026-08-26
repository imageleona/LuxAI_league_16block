"""
Fixed-seed evaluation of the current training policy against the league's
permanent anchors.

Why this exists: the league's own `anchor_winrate_rolling` is built from TRAINING
games, which vary in map size, seed and seat. That variance swamps the effect we
care about - on the 5M run, resolving the observed 2.6-point decline would have
needed ~1,400 games (~7M steps). Replaying a fixed, paired schedule instead holds
map/seed/seat constant, so round-to-round CHANGE is far more sensitive than any
single round's absolute win rate.

Lives in `evaluation/` rather than `lux_ai/league/` on purpose: `lux_ai.league`
must not import learner or harness code (see lux_ai/league/__init__.py), and a
league-side module importing `evaluation` would be a cycle, since `evaluation`
imports `lux_ai.league.member_config`. The dependency runs
`lux_ai.torchbeast -> evaluation -> lux_ai.{nns, lux_gym, league}`.

Usage without a trainer (2 games, ~30 s):
    python -m evaluation.anchor_eval
      --candidate internal_testing/hall_of_fame/11-24_12-56-23_062179520_must_research
      --anchors internal_testing/hall_of_fame/10-10_11-18-12_28576448
      --n-seeds 1 --map-sizes 12 --parallel-games 2
      --output evaluation_runs/anchor_eval_smoke
"""
import argparse
import datetime as _dt
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from .harness import DEFAULT_MAP_SIZES, AgentSpec
from .league_eval import evaluate_pair

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AnchorEvalConfig:
    """The `anchor_eval_*` subset of the league config block."""
    enabled: bool = False
    every_n_games: int = 500
    n_seeds: int = 5
    map_sizes: Tuple[int, ...] = (16, 32)
    parallel_games: int = 8
    device: str = "cuda:0"
    at_start: bool = True
    blocking: bool = False
    max_consecutive_failures: int = 2
    save_best: bool = False
    best_min_delta: float = 0.0

    def __post_init__(self) -> None:
        bad = [m for m in self.map_sizes if m not in DEFAULT_MAP_SIZES]
        if bad:
            # Fail here rather than 40 games in: MatchRequest.__post_init__ raises
            # on an unsupported map size, and by then we are inside a worker thread.
            raise ValueError(f"anchor_eval_map_sizes {bad} not in {list(DEFAULT_MAP_SIZES)}")
        if self.n_seeds < 1:
            raise ValueError(f"anchor_eval_n_seeds must be >= 1, was {self.n_seeds}")
        if self.every_n_games < 1:
            raise ValueError(f"anchor_eval_every_n_games must be >= 1, was {self.every_n_games}")

    @property
    def games_per_anchor(self) -> int:
        return self.n_seeds * len(self.map_sizes) * 2

    @property
    def seeds(self) -> List[int]:
        return list(range(self.n_seeds))

    @classmethod
    def from_league_flags(cls, league_flags: Any) -> "AnchorEvalConfig":
        if isinstance(league_flags, dict):
            def get(key, default=None):
                return league_flags.get(key, default)
        else:
            def get(key, default=None):
                return getattr(league_flags, key, default)
        return cls(
            enabled=bool(get("anchor_eval_enabled", False)),
            every_n_games=int(get("anchor_eval_every_n_games", 500)),
            n_seeds=int(get("anchor_eval_n_seeds", 5)),
            map_sizes=tuple(get("anchor_eval_map_sizes", (16, 32)) or (16, 32)),
            parallel_games=int(get("anchor_eval_parallel_games", 8)),
            device=str(get("anchor_eval_device", "cuda:0")),
            at_start=bool(get("anchor_eval_at_start", True)),
            blocking=bool(get("anchor_eval_blocking", False)),
            max_consecutive_failures=int(get("anchor_eval_max_consecutive_failures", 2)),
            save_best=bool(
                get("anchor_eval_save_best", False)
                or get("anchor_eval_promote_best", False)
            ),
            best_min_delta=float(get("anchor_eval_best_min_delta", 0.0)),
        )


def load_anchor_specs(
        anchor_dirs: Sequence[Union[str, Path]],
        device: str = "cuda:0",
) -> List[AgentSpec]:
    """
    Build the anchor AgentSpecs once, at startup.

    Each construction sha256s the whole agent directory (~80 MB), so this is not
    something to repeat per round. A directory that cannot be loaded is logged and
    skipped rather than raising: losing one anchor must not cost the whole run.
    """
    specs: List[AgentSpec] = []
    for d in anchor_dirs:
        try:
            specs.append(AgentSpec.from_directory(d, device=device))
        except Exception:
            logging.exception("AnchorEval: skipping unusable anchor %s", d)
    if not specs:
        logging.warning("AnchorEval: no usable anchors; evaluation will be a no-op")
    return specs


def run_anchor_round(
        candidate_dir: Union[str, Path],
        candidate_name: str,
        anchors: Sequence[AgentSpec],
        output_root: Union[str, Path],
        seeds: Sequence[int],
        map_sizes: Sequence[int],
        parallel_games: int,
        device: str = "cuda:0",
        should_stop: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Play len(seeds) * len(map_sizes) * 2 games against each anchor, SEQUENTIALLY.

    Sequential rather than one merged batch for two reasons. BatchedMatchRunner
    keys its results by MatchRequest.match_id, and build_paired_schedule emits the
    same ids for every opponent - merged, results would silently overwrite each
    other. And one runner per anchor keeps 2 models GPU-resident instead of
    1 + len(anchors), which matters on a card already shared with the trainer.

    Never raises. A per-anchor failure is recorded in ["errors"] and the remaining
    anchors still produce numbers.
    """
    output_root = Path(output_root)
    started = time.time()
    per_anchor: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    for anchor in anchors:
        if should_stop is not None and should_stop():
            logging.info("AnchorEval: stop requested, ending round early")
            break
        try:
            candidate = AgentSpec.from_directory(candidate_dir, name=candidate_name, device=device)
            # A stable per-anchor output dir: _prepare_diagnostic_engine caches the
            # patched Node engine there, so it is copied once per anchor for the
            # whole run rather than once per round.
            summary = evaluate_pair(
                candidate=candidate,
                opponent=anchor,
                output_dir=output_root / anchor.name,
                seeds=list(seeds),
                map_sizes=list(map_sizes),
                parallel_games=parallel_games,
            )
            per_anchor[anchor.name] = summary["aggregate"]
        except Exception as e:
            logging.exception("AnchorEval: anchor %s failed", anchor.name)
            errors[anchor.name] = "{}: {}".format(type(e).__name__, e)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    win_rates = [a["win_rate"] for a in per_anchor.values() if a.get("win_rate") is not None]
    return {
        "per_anchor": per_anchor,
        "errors": errors,
        "mean_win_rate": (sum(win_rates) / len(win_rates)) if win_rates else None,
        "games": sum(int(a.get("games", 0)) for a in per_anchor.values()),
        "wall_seconds": time.time() - started,
    }


def _ci(aggregate: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    ci = aggregate.get("ci95")
    if isinstance(ci, dict):
        return ci.get("low"), ci.get("high")
    if isinstance(ci, (list, tuple)) and len(ci) == 2:
        return ci[0], ci[1]
    return None, None


def _fmt(value: Optional[float], spec: str = "{:.4f}") -> str:
    return "nan" if value is None else spec.format(value)


def format_round_lines(record: Dict[str, Any]) -> List[str]:
    """
    One line per anchor plus a ROUND summary.

    Line-oriented rather than a fixed-width table so it survives a change to the
    anchor list and stays greppable.
    """
    ts = record.get("timestamp") or _dt.datetime.now().isoformat(timespec="seconds")
    head = "{} round={:03d} step={:012d}".format(ts, record.get("round", 0), record.get("step", 0))
    lines = []
    for name, agg in sorted(record.get("per_anchor", {}).items()):
        low, high = _ci(agg)
        lines.append(
            "{} anchor={} n={} wr={} ci95=[{},{}] wlt={}/{}/{} margin={}".format(
                head, name, int(agg.get("games", 0)),
                _fmt(agg.get("win_rate")),
                _fmt(low, "{:.2f}"), _fmt(high, "{:.2f}"),
                int(agg.get("wins", 0)), int(agg.get("losses", 0)), int(agg.get("ties", 0)),
                _fmt(agg.get("mean_city_tile_margin"), "{:+.2f}"),
            )
        )
    for name, err in sorted(record.get("errors", {}).items()):
        lines.append("{} anchor={} ERROR {}".format(head, name, err))
    lines.append(
        "{} ROUND mean_wr={} anchors={} n={} wall={:.1f}s".format(
            head, _fmt(record.get("mean_win_rate")),
            len(record.get("per_anchor", {})), int(record.get("games", 0)),
            record.get("wall_seconds", 0.),
        )
    )
    return lines


def wandb_payload(record: Dict[str, Any], current_step: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Nested under one top-level key, matching LeagueManager.wandb_stats()."""
    stats: Dict[str, Any] = {
        "win_rate_mean": record.get("mean_win_rate"),
        "round": record.get("round", 0),
        "games": record.get("games", 0),
        "wall_seconds": record.get("wall_seconds", 0.),
        "eval_step": record.get("step", 0),
        "league_games_at_eval": record.get("league_games", 0),
        "failures": len(record.get("errors", {})),
        "rounds_skipped": record.get("rounds_skipped", 0),
        "is_best": bool(record.get("is_best", False)),
        "previous_best_mean_win_rate": record.get("previous_best_mean_win_rate"),
        "baseline_mean_win_rate": record.get("baseline_mean_win_rate"),
        "delta_from_baseline": record.get("delta_from_baseline"),
    }
    if current_step is not None:
        # The round finishes ~150k steps after the weights were frozen; surfacing
        # the lag keeps that attribution visible on the chart instead of hidden.
        stats["step_lag"] = int(current_step) - int(record.get("step", 0))
    for name, agg in record.get("per_anchor", {}).items():
        low, high = _ci(agg)
        stats["win_rate/{}".format(name)] = agg.get("win_rate")
        stats["ci95_low/{}".format(name)] = low
        stats["ci95_high/{}".format(name)] = high
        stats["city_margin/{}".format(name)] = agg.get("mean_city_tile_margin")
        stats["games/{}".format(name)] = agg.get("games")
    return {"AnchorEval": stats}


def append_txt_line(path: Union[str, Path], line: str) -> None:
    """Open/append/close per call so a killed run keeps everything written so far."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def write_round(output_root: Union[str, Path], record: Dict[str, Any]) -> None:
    """Append the human-readable lines and the machine-readable record."""
    output_root = Path(output_root)
    for line in format_round_lines(record):
        append_txt_line(output_root / "anchor_eval.txt", line)
    append_txt_line(output_root / "anchor_eval.jsonl", json.dumps(record, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True, help="agent directory to evaluate")
    parser.add_argument("--anchors", nargs="+", required=True, help="anchor agent directories")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--map-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="evaluation_runs/anchor_eval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
    config = AnchorEvalConfig(n_seeds=args.n_seeds, map_sizes=tuple(args.map_sizes))

    anchors = load_anchor_specs(args.anchors, device=args.device)
    output_root = Path(args.output)
    record = run_anchor_round(
        candidate_dir=args.candidate,
        candidate_name=Path(args.candidate).name,
        anchors=anchors,
        output_root=output_root,
        seeds=config.seeds,
        map_sizes=list(config.map_sizes),
        parallel_games=args.parallel_games,
        device=args.device,
    )
    record["round"] = 0
    record["step"] = 0
    record["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
    write_round(output_root, record)
    for line in format_round_lines(record):
        print(line)


if __name__ == "__main__":
    main()
