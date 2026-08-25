from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import torch

from .harness import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_MAP_SIZES,
    DEFAULT_SEEDS,
    AgentSpec,
    BatchedMatchRunner,
    build_paired_schedule,
    paired_repeat_mismatches,
    summarize_results,
    write_report,
)


def _parse_ints(value: str) -> List[int]:
    values: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_text, stop_text = part.split(":", 1)
            values.extend(range(int(start_text), int(stop_text)))
        else:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _common_parser(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument("--output", type=Path, required=True)
    subparser.add_argument("--device", default="cuda:0")
    subparser.add_argument("--parallel-games", type=int, default=8)
    subparser.add_argument(
        "--seeds",
        type=_parse_ints,
        default=list(DEFAULT_SEEDS),
        help="Comma-separated integers or half-open ranges such as 0:50",
    )
    subparser.add_argument(
        "--map-sizes",
        type=_parse_ints,
        default=list(DEFAULT_MAP_SIZES),
        help="Fixed map strata; default: 12,16,24,32",
    )


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lux AI Phase 1 deterministic evaluation")
    commands = parser.add_subparsers(dest="command", required=True)

    self_test = commands.add_parser("self-test", help="Run the mandatory baseline-vs-baseline test")
    _common_parser(self_test)
    self_test.add_argument("--agent", type=Path, default=DEFAULT_BASELINE_DIR)

    compare = commands.add_parser("compare", help="Compare a candidate with the fixed baseline")
    _common_parser(compare)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_DIR)
    compare.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("{} requested, but CUDA is not available".format(args.device))
    if sorted(args.map_sizes) != sorted(DEFAULT_MAP_SIZES) and len(args.seeds) == len(DEFAULT_SEEDS):
        print("WARNING: this is not the full four-map Phase 1 stratification", file=sys.stderr)

    baseline = AgentSpec.from_directory(
        args.agent if args.command == "self-test" else args.baseline,
        name="baseline",
        device=args.device,
    )
    if args.command == "self-test":
        candidate = baseline
        use_cache = False
    else:
        candidate = AgentSpec.from_directory(args.candidate, name="candidate", device=args.device)
        use_cache = not args.no_cache

    schedule = build_paired_schedule(candidate, baseline, args.seeds, args.map_sizes)
    print("Running {} games ({} seeds x {} map sizes x 2 seats)".format(
        len(schedule), len(args.seeds), len(args.map_sizes)
    ))
    print("Candidate hash: {}".format(candidate.content_hash))
    print("Baseline hash:  {}".format(baseline.content_hash))

    runner = BatchedMatchRunner(
        output_dir=args.output,
        parallel_games=args.parallel_games,
        use_cache=use_cache,
    )
    results = runner.run(schedule)
    summary = summarize_results(results, runner.stats)
    metadata = {
        "command": args.command,
        "candidate_hash": candidate.content_hash,
        "baseline_hash": baseline.content_hash,
        "seeds": list(args.seeds),
        "map_sizes": list(args.map_sizes),
        "parallel_games": args.parallel_games,
    }

    exit_code = 0
    if args.command == "self-test":
        mismatches = paired_repeat_mismatches(results)
        low, high = summary["aggregate"]["ci95"]
        contains_half = low <= 0.5 <= high
        metadata["repeat_mismatches"] = mismatches
        metadata["self_test_passed"] = contains_half and not mismatches
        if not metadata["self_test_passed"]:
            exit_code = 2

    write_report(args.output, results, summary, metadata)
    print(json.dumps({"metadata": metadata, "summary": summary}, indent=2))
    print("Report: {}".format((args.output / "summary.md").resolve()))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
