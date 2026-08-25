"""
Compare league training runs on held-out opponents.

Reads the per-match results the harness caches under <output>/<candidate>/
<opponent>/cache/*.json, so it works on any completed league_eval run.

Reports two things:

1. Per-candidate win rate with a Wilson interval - the headline number.
2. A PAIRED comparison between two candidates. Every candidate plays the same
   opponents on the same seeds, map sizes and seats, so matches can be lined up
   one-to-one and the difference taken per matchup. That removes the variance
   caused by some boards simply being easier than others, and detects a much
   smaller effect than comparing two independent intervals would.
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def wilson(successes: float, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - rad), min(1.0, centre + rad)


def load_matches(root: str) -> List[Dict]:
    matches = []
    for path in glob.glob(os.path.join(root, "**", "cache", "*.json"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        if m.get("error"):
            continue
        seat = m["candidate_seat"]
        m["candidate"] = m["agent_0"] if seat == 0 else m["agent_1"]
        m["opponent"] = m["agent_1"] if seat == 0 else m["agent_0"]
        matches.append(m)
    return matches


def key_of(m: Dict) -> Tuple:
    return (m["opponent"], m["seed"], m["map_size"], m["candidate_seat"])


def summarise(matches: List[Dict], label: str) -> None:
    by_cand = defaultdict(list)
    for m in matches:
        by_cand[m["candidate"]].append(m)
    print(f"\n=== {label} ===")
    print("%-26s %6s %6s %5s %5s %5s   %-22s %s" %
          ("candidate", "games", "score", "W", "L", "T", "win rate (95% CI)", "mean city margin"))
    for cand in sorted(by_cand, key=lambda c: -sum(x["candidate_score"] for x in by_cand[c]) / max(len(by_cand[c]), 1)):
        rows = by_cand[cand]
        n = len(rows)
        s = sum(r["candidate_score"] for r in rows)
        w = sum(1 for r in rows if r["candidate_score"] == 1.0)
        l = sum(1 for r in rows if r["candidate_score"] == 0.0)
        lo, hi = wilson(s, n)
        margin = sum(r["candidate_city_margin"] for r in rows) / n
        print("%-26s %6d %6.1f %5d %5d %5d   %6.1f%%  (%4.1f-%4.1f%%)   %+7.2f" %
              (cand[:26], n, s, w, l, n - w - l, 100 * s / n, 100 * lo, 100 * hi, margin))


def paired(matches: List[Dict], a: str, b: str) -> None:
    idx = {}
    for m in matches:
        idx.setdefault(m["candidate"], {})[key_of(m)] = m
    if a not in idx or b not in idx:
        print(f"\n(cannot pair {a} vs {b}: missing results)")
        return
    shared = sorted(set(idx[a]) & set(idx[b]))
    if not shared:
        print(f"\n(no shared matchups between {a} and {b})")
        return

    diffs = [idx[a][k]["candidate_score"] - idx[b][k]["candidate_score"] for k in shared]
    margins = [idx[a][k]["candidate_city_margin"] - idx[b][k]["candidate_city_margin"] for k in shared]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else 0.0
    mmean = sum(margins) / n
    mvar = sum((d - mmean) ** 2 for d in margins) / (n - 1) if n > 1 else 0.0
    mse = math.sqrt(mvar / n) if n > 1 else 0.0

    print(f"\n=== PAIRED: {a}  minus  {b} ===")
    print(f"  matched head-to-head matchups : {n}")
    print(f"  {a} better in {sum(1 for d in diffs if d > 0)}, "
          f"{b} better in {sum(1 for d in diffs if d < 0)}, "
          f"same in {sum(1 for d in diffs if d == 0)}")
    print(f"  mean score difference         : {mean:+.4f}  (95% CI {mean - 1.96 * se:+.4f} to {mean + 1.96 * se:+.4f})")
    print(f"  mean city-tile margin diff    : {mmean:+.2f}  (95% CI {mmean - 1.96 * mse:+.2f} to {mmean + 1.96 * mse:+.2f})")
    if se > 0:
        print(f"  t statistic                   : {mean / se:+.2f}")
    verdict = ("NO significant difference (interval spans 0)"
               if (mean - 1.96 * se) * (mean + 1.96 * se) <= 0
               else f"{a} significantly {'better' if mean > 0 else 'worse'} at the 95% level")
    print(f"  verdict                       : {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="evaluation_runs/league_final")
    ap.add_argument("--pair", nargs=2, action="append", default=None,
                    help="two candidate names to compare pairwise; repeatable")
    args = ap.parse_args()

    matches = load_matches(args.root)
    if not matches:
        raise SystemExit(f"no cached match results under {args.root}")
    print(f"loaded {len(matches)} completed matches from {args.root}")

    summarise(matches, "OVERALL, vs all held-out opponents")
    for opp in sorted({m["opponent"] for m in matches}):
        summarise([m for m in matches if m["opponent"] == opp], f"vs held-out: {opp}")

    pairs = args.pair or [["pfsp_final", "sp_final"]]
    for a, b in pairs:
        paired(matches, a, b)


if __name__ == "__main__":
    main()
