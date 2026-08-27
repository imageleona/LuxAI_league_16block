"""
Standardized evaluation script for comparing a Lux AI agent against the
other lab teams (A/B/C/D/E/F/H), matching the real inter-team competition
scoring format: for each opponent, play one game at each map size
(12, 16, 24, 32) and count wins/losses/ties.

Usage:
    python evaluate_agent.py --agent path/to/your_agent/main.py \
        --teams-dir other_teams_agents/ --label "唐さん(変更X)" --out my_results.json

Requires: pip install kaggle-environments (already in requirements.txt if
you're using this repo's environment).

Known gotchas this script already avoids - do not "simplify" these away:
- Uses kaggle_environments.evaluate() directly, NOT the `kaggle-environments
  run` CLI subcommand - `run` silently ignores episode count and only ever
  plays one game.
- Runs one match at a time, never in parallel. Running several
  kaggle_environments matches concurrently on one machine reliably crashes
  the Node.js game engine (BrokenPipeError) - this cost us real time to
  diagnose, don't reintroduce it.
- actTimeout defaults to 3 seconds in kaggle_environments, tuned for the
  original Kaggle competition's judged hardware. On a normal shared/laptop
  machine this times out almost every move. Default here is 30s; raise with
  --act-timeout if your agent is slow (a slow agent losing every game to
  timeouts is not a fair result).
- Map size is randomized by kaggle_environments by default (mapType:
  "random", width/height: -1). This script pins width/height explicitly to
  match the real 12/16/24/32 format instead.
- other_teams_agents/ as shared: A's original data turned out to be an
  unrelated public Kaggle 2021 solution and G's turned out to be our own
  team's data - both were pulled out. Only put verified other-team folders
  (each containing a working main.py) in the directory you point --teams-dir at.
"""
import argparse
import json
import sys
from pathlib import Path

from kaggle_environments import evaluate as ke_evaluate

MAP_SIZES = [12, 16, 24, 32]


def play_one_size(agent_path: str, opponent_path: str, size: int, act_timeout: int):
    config = {"width": size, "height": size, "actTimeout": act_timeout}
    rewards = ke_evaluate(
        "lux_ai_2021",
        [str(agent_path), str(opponent_path)],
        configuration=config,
        num_episodes=1,
    )
    return rewards[0]  # [agent_reward, opponent_reward]


def score_vs_team(agent_path: str, team_dir: Path, act_timeout: int):
    team_main = team_dir / "main.py"
    if not team_main.exists():
        return {"error": f"no main.py found in {team_dir}"}

    per_size = {}
    wins = losses = ties = errors = 0
    for size in MAP_SIZES:
        print(f"    size {size}x{size}...", end=" ", flush=True)
        try:
            agent_r, opp_r = play_one_size(agent_path, str(team_main), size, act_timeout)
        except Exception as e:  # noqa: BLE001 - one bad matchup shouldn't kill the whole run
            per_size[size] = {"error": str(e)}
            errors += 1
            print(f"ERROR: {e}")
            continue
        per_size[size] = {"agent_reward": agent_r, "opponent_reward": opp_r}
        if agent_r is None and opp_r is None:
            errors += 1
            outcome = "both errored"
        elif agent_r is None:
            losses += 1
            outcome = "loss (agent errored)"
        elif opp_r is None:
            wins += 1
            outcome = "win (opponent errored)"
        elif agent_r > opp_r:
            wins += 1
            outcome = f"win ({agent_r} vs {opp_r})"
        elif agent_r < opp_r:
            losses += 1
            outcome = f"loss ({agent_r} vs {opp_r})"
        else:
            ties += 1
            outcome = "tie"
        print(outcome)

    decided = wins + losses + ties
    win_rate = wins / decided if decided else None
    return {
        "wins": wins, "losses": losses, "ties": ties, "errors": errors,
        "win_rate": win_rate, "per_size": per_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, help="Path to your agent's main.py")
    parser.add_argument("--teams-dir", default="other_teams_agents",
                         help="Directory with one subfolder per opponent team, each containing a main.py")
    parser.add_argument("--out", default="eval_results.json")
    parser.add_argument("--act-timeout", type=int, default=30)
    parser.add_argument("--label", default=None, help="Name for this agent/run, shown in the comparison table")
    args = parser.parse_args()

    agent_path = Path(args.agent).resolve()
    if not agent_path.exists():
        sys.exit(f"Agent main.py not found: {agent_path}")

    teams_dir = Path(args.teams_dir)
    team_dirs = sorted(p for p in teams_dir.iterdir() if p.is_dir())
    if not team_dirs:
        sys.exit(f"No team subfolders found in {teams_dir}")

    label = args.label or agent_path.parent.name
    results = {}
    total_wins = total_losses = total_ties = total_errors = 0

    print(f"=== Evaluating '{label}' vs {len(team_dirs)} teams "
          f"(sizes {MAP_SIZES}, act_timeout={args.act_timeout}s) ===")
    for team_dir in team_dirs:
        print(f"--- vs {team_dir.name} ---", flush=True)
        r = score_vs_team(str(agent_path), team_dir, args.act_timeout)
        results[team_dir.name] = r
        if "error" not in r:
            total_wins += r["wins"]
            total_losses += r["losses"]
            total_ties += r["ties"]
            total_errors += r["errors"]
            wr = r["win_rate"]
            print(f"    => {r['wins']}-{r['losses']}-{r['ties']}"
                  + (f" (win_rate={wr:.1%})" if wr is not None else " (no decided games)"))
        else:
            print(f"    SKIPPED: {r['error']}")

    decided = total_wins + total_losses + total_ties
    overall_win_rate = total_wins / decided if decided else None

    summary = {
        "label": label,
        "agent_path": str(agent_path),
        "per_team": results,
        "total_wins": total_wins,
        "total_losses": total_losses,
        "total_ties": total_ties,
        "total_errors": total_errors,
        "overall_win_rate": overall_win_rate,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n=== {label}: {total_wins}-{total_losses}-{total_ties} overall",
          f"(win_rate={overall_win_rate:.1%})" if overall_win_rate is not None else "(no decided games)")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
