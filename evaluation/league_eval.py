"""
Held-out evaluation for league training runs.

Pits league snapshots (outputs/<run>/league/snapshots/*) against a HELD-OUT
opponent set - agents that were never members of the training pool - which is
the actual robustness metric: evaluating only against our own pool would make
PFSP look good by construction.

Default held-out set (neither is ever admitted to the pool - keep it that way):
  - 09-17_22-05-30_20000128: trained on the old FixedShapeContinuousObs, which
    the league rejects on obs-space mismatch, so it can never leak into the pool.
  - 11-09_21-32-04_59822400: deliberately excluded from the pool presets and
    reserved for evaluation.
Rule-based/public agents (internal_testing/public_agents/) can be evaluated via
top_agents_tournament.py's subprocess path instead.
NB: 09-07_01-44-10_10088000 is NOT usable - its config sets an obs_space_kwarg
that the current obs space no longer accepts, so the model cannot be rebuilt.

Usage (from the repo root, with the lux_env37 python):
    python -m evaluation.league_eval --snapshots-root outputs/08-14/12-00-00/league/snapshots
    python -m evaluation.league_eval --snapshots latest_dir another_dir --n-seeds 10
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from .harness import (
    AgentSpec,
    BatchedMatchRunner,
    build_paired_schedule,
    summarize_results,
    write_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HELD_OUT = (
    REPO_ROOT / "internal_testing" / "hall_of_fame" / "09-17_22-05-30_20000128",
    REPO_ROOT / "internal_testing" / "hall_of_fame" / "11-09_21-32-04_59822400",
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")


def evaluate_pair(
        candidate: AgentSpec,
        opponent: AgentSpec,
        output_dir: Path,
        seeds: List[int],
        map_sizes: List[int],
        parallel_games: int,
) -> Dict:
    requests = build_paired_schedule(candidate, opponent, seeds=seeds, map_sizes=map_sizes)
    runner = BatchedMatchRunner(output_dir=str(output_dir), parallel_games=parallel_games)
    results = runner.run(requests)
    summary = summarize_results(results, runner.stats)
    # matches.csv / summary.md alongside each pairing, so the per-match record
    # does not have to be recovered from the result cache.
    write_report(
        output_dir,
        results,
        summary,
        metadata={"candidate": candidate.name, "opponent": opponent.name,
                  "seeds": list(seeds), "map_sizes": list(map_sizes)},
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-root", type=str, default=None,
                        help="Directory containing league snapshot dirs (league/snapshots of a run)")
    parser.add_argument("--snapshots", type=str, nargs="*", default=None,
                        help="Explicit snapshot/agent dirs to evaluate (overrides --snapshots-root)")
    parser.add_argument("--held-out", type=str, nargs="*",
                        default=[str(p) for p in DEFAULT_HELD_OUT],
                        help="Held-out opponent agent dirs (must never have been pool members)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Seeds per (map size, seat) pairing: total games per opponent "
                             "= n_seeds * len(map_sizes) * 2")
    parser.add_argument("--map-sizes", type=int, nargs="*", default=[12, 16, 24, 32])
    parser.add_argument("--parallel-games", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="evaluation_runs/league_eval")
    args = parser.parse_args()

    if args.snapshots:
        snapshot_dirs = [Path(p) for p in args.snapshots]
    elif args.snapshots_root:
        snapshot_dirs = sorted(p for p in Path(args.snapshots_root).iterdir() if p.is_dir())
    else:
        parser.error("Provide --snapshots-root or --snapshots")
        return
    if not snapshot_dirs:
        raise SystemExit("No snapshot directories found")

    seeds = list(range(args.n_seeds))
    output_root = Path(args.output)
    report: Dict[str, Dict] = {}

    for snapshot_dir in snapshot_dirs:
        candidate = AgentSpec.from_directory(snapshot_dir, device=args.device)
        report[candidate.name] = {}
        for held_out_dir in args.held_out:
            opponent = AgentSpec.from_directory(held_out_dir, device=args.device)
            pair_dir = output_root / candidate.name / opponent.name
            logging.info(f"Evaluating {candidate.name} vs held-out {opponent.name} "
                         f"({len(seeds) * len(args.map_sizes) * 2} games)")
            summary = evaluate_pair(candidate, opponent, pair_dir, seeds,
                                    args.map_sizes, args.parallel_games)
            aggregate = summary["aggregate"]
            logging.info(
                "  win rate {:.2%} (95% CI {:.2%}-{:.2%}), W/L/T {}/{}/{}".format(
                    aggregate["win_rate"], aggregate["ci95"][0], aggregate["ci95"][1],
                    aggregate["wins"], aggregate["losses"], aggregate["ties"])
            )
            report[candidate.name][opponent.name] = summary

    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "league_eval_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    lines = ["# League held-out evaluation", ""]
    for candidate_name in sorted(report):
        lines.append(f"## {candidate_name}")
        for opponent_name in sorted(report[candidate_name]):
            aggregate = report[candidate_name][opponent_name]["aggregate"]
            lines.append(
                "- vs {}: {:.2%} (95% CI {:.2%}-{:.2%}), {} games".format(
                    opponent_name, aggregate["win_rate"],
                    aggregate["ci95"][0], aggregate["ci95"][1], aggregate["games"])
            )
        lines.append("")
    (output_root / "league_eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    logging.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
