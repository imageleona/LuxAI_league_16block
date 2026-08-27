"""
Combine several evaluate_agent.py output JSON files into one comparison
table - the same shape as the team's procedure doc:

              vs_A  vs_B  vs_C  vs_D  vs_E  vs_F  vs_H  overall_win_rate
Cerberus baseline  6-4-0  3-7-0  8-2-0  5-5-0  7-3-0  4-6-0  6-4-0  55.7%
Tang (changed X)   ...
Wang (changed Y)   ...

Usage:
    python compile_comparison_table.py results_baseline.json results_tang.json results_wang.json ...

Everyone runs evaluate_agent.py on their own trained agent first (same
teams-dir, same map sizes, same act-timeout - keep these consistent across
everyone's runs or the table isn't a fair comparison), then hands their
--out JSON file back to be combined here.
"""
import json
import sys
from pathlib import Path


def main():
    paths = sys.argv[1:]
    if not paths:
        sys.exit("Usage: python compile_comparison_table.py result1.json result2.json ...")

    rows = [json.loads(Path(p).read_text()) for p in paths]

    all_teams = []
    for data in rows:
        for team in data["per_team"]:
            if team not in all_teams:
                all_teams.append(team)
    all_teams.sort()

    header = ["label"] + [f"vs_{t}" for t in all_teams] + ["overall_win_rate"]
    label_width = max([len(header[0])] + [len(d["label"]) for d in rows]) + 2
    col_width = 12

    def fmt_row(cells):
        return cells[0].ljust(label_width) + "".join(c.ljust(col_width) for c in cells[1:])

    print(fmt_row(header))
    print("-" * (label_width + col_width * (len(header) - 1)))
    for data in rows:
        cells = [data["label"]]
        for team in all_teams:
            r = data["per_team"].get(team)
            if not r or "error" in r:
                cells.append("N/A")
            else:
                cells.append(f"{r['wins']}-{r['losses']}-{r['ties']}")
        wr = data.get("overall_win_rate")
        cells.append(f"{wr:.1%}" if wr is not None else "N/A")
        print(fmt_row(cells))


if __name__ == "__main__":
    main()
