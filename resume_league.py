"""
Smart resume wrapper for run_monobeast.py league runs across repeated
debug-queue sessions.

Handles two things:
1. Chains load_dir/checkpoint_file/weights_only across invocations, so each
   new session continues from where the last one left off (model, optimizer,
   step count, and the league pool + win rates all resume automatically -
   see train()'s checkpoint_state handling and LeagueManager.load_state() in
   lux_ai/torchbeast/monobeast.py).
2. Regression guard: reads every prior session's own
   league/anchor_eval/anchor_eval.jsonl (the periodic fixed-seed win-rate
   evaluation against the anchor pool) across the whole chain. If the most
   recent eval round's mean_win_rate is more than --regression-threshold
   below the best round seen so far, it rolls back to the resumable
   checkpoint (<step>.pt, saved every checkpoint_freq minutes) closest to
   (at or before) that best round's step, instead of the literal latest one.

Usage (same flags every time - total_steps etc. are NOT remembered between
invocations, only load_dir/checkpoint_file/weights_only are):

    python resume_league.py --config-name league_cerberus_24block total_steps=200000

State lives in run_state_<config_name>.json one level above repo/ (NOT
inside outputs/, NOT git-tracked) - just bookkeeping of which
outputs/<date>/<time> dirs belong to this run's chain.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent  # repo root
STATE_DIR = SCRIPT_DIR.parent  # LuxAI2026/, sibling to repo/


def state_path_for(config_name: str) -> Path:
    return STATE_DIR / f"run_state_{config_name}.json"


def load_chain(config_name: str):
    p = state_path_for(config_name)
    if not p.exists():
        return []
    return json.loads(p.read_text()).get("sessions", [])


def save_chain(config_name: str, sessions):
    state_path_for(config_name).write_text(json.dumps({"sessions": sessions}, indent=2))


def collect_anchor_eval_rounds(sessions):
    """list of (step, mean_win_rate) across the whole chain, sorted by step."""
    rounds = []
    for s in sessions:
        jsonl = Path(s) / "league" / "anchor_eval" / "anchor_eval.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            step, wr = rec.get("step"), rec.get("mean_win_rate")
            if step is not None and wr is not None:
                rounds.append((int(step), float(wr)))
    rounds.sort(key=lambda r: r[0])
    return rounds


def collect_checkpoints(sessions):
    """step -> (session_dir, filename) for full resumable <step>.pt files (not _weights.pt)."""
    ckpts = {}
    for s in sessions:
        for f in Path(s).glob("*.pt"):
            if f.name.endswith("_weights.pt"):
                continue
            try:
                step = int(f.stem)
            except ValueError:
                continue
            ckpts[step] = (s, f.name)
    return ckpts


def pick_checkpoint(ckpts, target_step=None):
    if not ckpts:
        return None
    if target_step is None:
        best = max(ckpts.keys())
    else:
        at_or_before = [st for st in ckpts if st <= target_step]
        best = max(at_or_before) if at_or_before else min(ckpts.keys())
    return best, ckpts[best]


def newest_output_dir(repo_root: Path, before_dirs):
    candidates = sorted(
        (p for p in (repo_root / "outputs").glob("*/*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    for p in reversed(candidates):
        if str(p) not in before_dirs:
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument(
        "--regression-threshold", type=float, default=0.08,
        help="Roll back if latest mean_win_rate is this much below the best round seen (default 0.08).",
    )
    args, extra = parser.parse_known_args()
    config_name = args.config_name
    sessions = load_chain(config_name)

    cmd = [sys.executable, "-u", "run_monobeast.py", "--config-name", config_name]

    if not sessions:
        print(f"[resume_league] No prior sessions for '{config_name}' - starting fresh from the config's own load_dir.")
    else:
        rounds = collect_anchor_eval_rounds(sessions)
        ckpts = collect_checkpoints(sessions)
        if not ckpts:
            print("[resume_league] WARNING: no resumable <step>.pt checkpoints found in prior sessions "
                  "- starting fresh from the config's own load_dir instead.")
        else:
            target_step = None
            if len(rounds) >= 2:
                best_step, best_wr = max(rounds, key=lambda r: r[1])
                latest_step, latest_wr = rounds[-1]
                print(f"[resume_league] AnchorEval history: {len(rounds)} round(s). "
                      f"best mean_win_rate={best_wr:.4f} @ step {best_step}, "
                      f"latest mean_win_rate={latest_wr:.4f} @ step {latest_step}.")
                if latest_step != best_step and latest_wr < best_wr - args.regression_threshold:
                    target_step = best_step
                    print(f"[resume_league] Regression detected ({best_wr - latest_wr:.4f} below peak, "
                          f"threshold {args.regression_threshold}). Rolling back to step {best_step}.")
                else:
                    print("[resume_league] No regression. Resuming from the latest checkpoint.")
            else:
                print(f"[resume_league] Only {len(rounds)} AnchorEval round(s) so far - not enough "
                      f"history to judge regression. Resuming from the latest checkpoint.")

            step, (ckpt_dir, ckpt_file) = pick_checkpoint(ckpts, target_step)
            print(f"[resume_league] Resuming from step {step}: {ckpt_dir}/{ckpt_file}")
            cmd += [f"load_dir={ckpt_dir}", f"checkpoint_file={ckpt_file}", "weights_only=false"]

    cmd += extra
    print("[resume_league] Running:", " ".join(cmd))

    before = {str(p) for p in (SCRIPT_DIR / "outputs").glob("*/*") if p.is_dir()}
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))

    new_dir = newest_output_dir(SCRIPT_DIR, before)
    if new_dir:
        sessions.append(new_dir)
        save_chain(config_name, sessions)
        print(f"[resume_league] Recorded new session dir: {new_dir}")
    else:
        print("[resume_league] WARNING: could not detect a new output dir - chain not updated. "
              "The next invocation may re-resume from the same point as this one.")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
