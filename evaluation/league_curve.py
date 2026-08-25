"""
Held-out win rate as a function of training step.

Reads the per-match results cached by league_eval and assembles every candidate
whose name is a step number (league snapshots are named that way) into a curve,
then tests whether it trends.

A single endpoint cannot distinguish "never improved" from "improved and then
regressed". The curve can.

Usage:
    python -m evaluation.league_curve
    python -m evaluation.league_curve --baseline 11-24_12-56-23_062179520_must_research
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def load(root: str) -> List[Dict]:
    out = []
    for path in glob.glob(os.path.join(root, "**", "cache", "*.json"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("error"):
            continue
        seat = m["candidate_seat"]
        m["candidate"] = m["agent_0"] if seat == 0 else m["agent_1"]
        m["opponent"] = m["agent_1"] if seat == 0 else m["agent_0"]
        out.append(m)
    return out


def wilson(s: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p, d = s / n, 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - r), min(1.0, c + r)


def weighted_fit(xs: List[float], ys: List[float], ns: List[int]):
    # Weight by inverse variance, but smooth the proportion (add-2) first: a raw
    # 0% or 100% point has zero estimated variance and would otherwise receive
    # effectively infinite weight and dictate the whole fit.
    ws = []
    for y, n in zip(ys, ns):
        p = (y * n + 1.0) / (n + 2.0)
        ws.append(n / (p * (1.0 - p)))
    Sw = sum(ws)
    Sx = sum(w * x for w, x in zip(ws, xs))
    Sy = sum(w * y for w, y in zip(ws, ys))
    Sxx = sum(w * x * x for w, x in zip(ws, xs))
    Sxy = sum(w * x * y for w, x, y in zip(ws, xs, ys))
    den = Sw * Sxx - Sx * Sx
    if abs(den) < 1e-12:
        return 0.0, 0.0
    return (Sw * Sxy - Sx * Sy) / den, math.sqrt(Sw / den)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="evaluation_runs/league_final")
    ap.add_argument("--baseline", default="11-24_12-56-23_062179520_must_research",
                    help="candidate plotted as step 0 (the model training started from)")
    ap.add_argument("--final", default="pfsp_final", help="candidate plotted at --final-step")
    ap.add_argument("--final-step", type=int, default=1000192)
    ap.add_argument("--min-games", type=int, default=10,
                    help="ignore points with fewer games than this (an evaluation still in "
                         "progress leaves partial points behind)")
    ap.add_argument("--opponent", default=None,
                    help="restrict to one held-out opponent (default: the one shared by the "
                         "most step-labelled candidates)")
    args = ap.parse_args()

    matches = load(args.root)

    # Points MUST all be measured against the same opponent. Endpoints were run
    # against more opponents than the intermediate snapshots, and mixing them
    # makes an easy-opponent average look like a decline over training.
    def is_curve_point(name):
        return name.isdigit() or name in (args.baseline, args.final)

    if args.opponent is None:
        counts = defaultdict(set)
        for m in matches:
            if is_curve_point(m["candidate"]):
                counts[m["opponent"]].add(m["candidate"])
        if not counts:
            raise SystemExit(f"no step-labelled candidates found under {args.root}")
        args.opponent = max(counts, key=lambda o: len(counts[o]))
    matches = [m for m in matches if m["opponent"] == args.opponent]

    by_cand = defaultdict(list)
    for m in matches:
        by_cand[m["candidate"]].append(m)

    points = []
    for cand, rows in by_cand.items():
        if cand == args.baseline:
            step = 0
        elif cand == args.final:
            step = args.final_step
        elif cand.isdigit():
            step = int(cand)
        else:
            continue
        n = len(rows)
        s = sum(r["candidate_score"] for r in rows)
        margin = sum(r["candidate_city_margin"] for r in rows) / n
        points.append((step, s / n, n, s, margin, cand))
    points.sort()

    dropped = [p for p in points if p[2] < args.min_games]
    points = [p for p in points if p[2] >= args.min_games]
    for step, _, n, _, _, _ in dropped:
        print(f"  (skipping step {step}: only {n} game(s) so far, below --min-games)")

    if not points:
        raise SystemExit(f"no step-labelled candidates found under {args.root}")

    print(f"held-out opponent: {args.opponent}")
    print("all points measured against this one opponent, identical seeds and map sizes\n")
    print("%-10s %8s %7s %-22s %9s" % ("step", "score", "games", "95% CI", "city margin"))
    for step, y, n, s, margin, _ in points:
        lo, hi = wilson(s, n)
        bar = "#" * int(round(y * 40))
        print("%-10d %7.1f%% %7d  (%4.1f%% - %4.1f%%)   %+8.2f  %s"
              % (step, 100 * y, n, 100 * lo, 100 * hi, margin, bar))

    xs = [p[0] / 1e6 for p in points]
    ys = [p[1] for p in points]
    ns = [p[2] for p in points]
    slope, se = weighted_fit(xs, ys, ns)
    print("\nweighted linear fit of score vs step")
    print("  slope   : %+.1f percentage points per 1M steps (95%% CI %+.1f to %+.1f)"
          % (100 * slope, 100 * (slope - 1.96 * se), 100 * (slope + 1.96 * se)))
    print("  t       : %+.2f" % (slope / se if se else float("nan")))
    print("  verdict : %s" % ("SIGNIFICANT trend" if se and abs(slope) > 1.96 * se
                              else "no significant trend - consistent with a flat line"))

    base = next((p for p in points if p[0] == 0), None)
    best = max(points, key=lambda p: p[1])
    if base and best[0] != 0:
        lo, hi = wilson(best[3], best[2])
        blo, bhi = wilson(base[3], base[2])
        print("\n  best point : step %d at %.1f%% (CI %.1f-%.1f%%)"
              % (best[0], 100 * best[1], 100 * lo, 100 * hi))
        print("  vs step 0  : %.1f%% (CI %.1f-%.1f%%)" % (100 * base[1], 100 * blo, 100 * bhi))
        print("  -> %s" % ("intervals overlap; not distinguishable from the starting model"
                           if lo < bhi else "best point is above the starting model's interval"))
    print("\n  NB: each point's interval is wide by design (few games per point). Read the"
          "\n      shape across points, not any single value.")


if __name__ == "__main__":
    main()
