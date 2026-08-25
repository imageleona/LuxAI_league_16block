"""
Play one agent against itself N times and record which SEAT wins.

Both sides are the same network, so any systematic deviation from 50/50 is a
property of the game or of how the agent plays a given side - not of relative
agent strength. Worth knowing before reading anything into head-to-head results,
and it is why every evaluation elsewhere in this repo plays both seats.

Writes a CSV with one row per game plus a summary to stdout.

    python -m evaluation.seat_bias --agent league_agents/pfsp_5m_seed7_peak961k --games 100
"""
import argparse
import csv
import math
from pathlib import Path

from .harness import AgentSpec, BatchedMatchRunner, MatchRequest, wilson_interval


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", required=True, help="agent directory (plays BOTH seats)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--map-sizes", type=int, nargs="*", default=[12, 16, 24, 32])
    ap.add_argument("--parallel-games", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="evaluation_runs/seat_bias")
    ap.add_argument("--csv", default=None, help="defaults to <output>/seat_bias.csv")
    args = ap.parse_args()

    spec = AgentSpec.from_directory(args.agent, device=args.device)
    # Same spec on both sides: the harness keys runtimes by content hash, so one
    # model is loaded and one forward pass per turn serves both seats.
    requests = []
    for i in range(args.games):
        map_size = args.map_sizes[i % len(args.map_sizes)]
        requests.append(MatchRequest(
            agent_0=spec, agent_1=spec, seed=i, map_size=map_size,
            candidate_seat=0, match_id=f"selfplay_m{map_size:02d}_seed{i:05d}",
        ))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = BatchedMatchRunner(output_dir=str(out_dir), parallel_games=args.parallel_games)
    results = runner.run(requests)

    csv_path = Path(args.csv) if args.csv else out_dir / "seat_bias.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game", "seed", "map_size", "winner", "winner_label",
                    "p0_city_tiles", "p1_city_tiles", "p0_units", "p1_units",
                    "city_tile_margin_p0", "ending_turn", "wall_seconds"])
        for i, r in enumerate(results):
            ct = r.final_city_tiles
            un = r.final_units
            label = {0: "player_0", 1: "player_1"}.get(r.winner, "draw")
            w.writerow([i, r.seed, r.map_size, r.winner, label,
                        ct[0], ct[1], un[0], un[1], ct[0] - ct[1],
                        r.ending_turn, round(r.wall_seconds, 2)])

    n = len(results)
    p0 = sum(1 for r in results if r.winner == 0)
    p1 = sum(1 for r in results if r.winner == 1)
    draws = n - p0 - p1
    score0 = p0 + 0.5 * draws
    lo, hi = wilson_interval(score0, n)
    se = math.sqrt(0.25 / n) if n else float("nan")
    z = (score0 / n - 0.5) / se if n and se else float("nan")

    print(f"\n=== {n} self-play games: {spec.name} vs itself ===")
    print(f"  player 0 wins : {p0}")
    print(f"  player 1 wins : {p1}")
    print(f"  draws         : {draws}")
    print(f"  player 0 score: {100 * score0 / n:.1f}%  (95% CI {100 * lo:.1f}% - {100 * hi:.1f}%)")
    print(f"  z vs 50/50    : {z:+.2f}   -> {'SEAT BIAS' if abs(z) > 1.96 else 'no significant seat bias'}")
    print(f"\n  by map size:")
    for m in sorted({r.map_size for r in results}):
        sub = [r for r in results if r.map_size == m]
        w0 = sum(1 for r in sub if r.winner == 0)
        d = sum(1 for r in sub if r.winner not in (0, 1))
        print(f"    {m:>2}x{m:<2}: player 0 {w0}/{len(sub)}  (draws {d})")
    print(f"\n  csv: {csv_path}")


if __name__ == "__main__":
    main()
