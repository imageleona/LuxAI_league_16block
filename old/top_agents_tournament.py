import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
MAP_SIZES = (12, 16, 24, 32)
AGENTS = (
    ("09-07_10088000", "09-07_01-44-10_10088000"),
    ("09-17_20000128", "09-17_22-05-30_20000128"),
    ("10-10_28576448", "10-10_11-18-12_28576448"),
    ("10-10_28576448_must_research", "10-10_11-18-12_28576448_must_research"),
    ("11-09_59822400", "11-09_21-32-04_59822400"),
    ("11-24_62179520_must_research", "11-24_12-56-23_062179520_must_research"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run and rank the bundled Lux AI hall-of-fame agents."
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-timeout", type=int, default=3600,
                        help="Maximum wall-clock seconds per match")
    parser.add_argument("--agent-timeout", type=int, default=60000,
                        help="Maximum milliseconds per agent turn")
    return parser.parse_args()


def agent_path(folder):
    return (
        PROJECT_DIR
        / "internal_testing"
        / "hall_of_fame"
        / folder
        / "main.py"
    )


def read_replay(replay_path):
    with replay_path.open("r", encoding="utf-8") as replay_file:
        replay = json.load(replay_file)

    # Replays produced directly by the Lux CLI store the winner as ranked
    # agent IDs rather than Kaggle-style terminal rewards and statuses.
    cli_ranks = (replay.get("results") or {}).get("ranks") or []
    if len(cli_ranks) == 2:
        rank_by_agent = {
            int(rank["agentID"]): int(rank["rank"])
            for rank in cli_ranks
        }
        if set(rank_by_agent) != {0, 1}:
            raise ValueError("CLI replay contains unexpected agent IDs")
        if rank_by_agent[0] < rank_by_agent[1]:
            return (1, 0), ("DONE", "DONE"), "a"
        if rank_by_agent[1] < rank_by_agent[0]:
            return (0, 1), ("DONE", "DONE"), "b"
        return (0.5, 0.5), ("DONE", "DONE"), "tie"

    rewards = replay.get("rewards") or []
    statuses = replay.get("statuses") or []
    if len(rewards) != 2 or len(statuses) != 2:
        raise ValueError("Replay does not contain two rewards and statuses")

    if statuses[0] == "DONE" and statuses[1] != "DONE":
        outcome = "a"
    elif statuses[1] == "DONE" and statuses[0] != "DONE":
        outcome = "b"
    elif statuses[0] == "DONE" and statuses[1] == "DONE":
        reward_a = float(rewards[0])
        reward_b = float(rewards[1])
        if reward_a > reward_b:
            outcome = "a"
        elif reward_b > reward_a:
            outcome = "b"
        else:
            outcome = "tie"
    else:
        outcome = "error"

    return rewards, statuses, outcome


def sample_gpu():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            universal_newlines=True,
            timeout=5,
        )
        if completed.returncode == 0:
            return completed.stdout.strip().split(", ")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return ("", "", "", "", "")


def stop_process_tree(process):
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()


def run_match(match, cli, output_dir, match_timeout, agent_timeout):
    replay_path = output_dir / "replays" / (match["id"] + ".json")
    log_path = output_dir / "logs" / (match["id"] + ".log")
    telemetry_path = output_dir / "telemetry" / (match["id"] + ".csv")

    return_code = 0
    reused = False
    duration_seconds = 0.0
    if replay_path.exists() and replay_path.stat().st_size:
        try:
            read_replay(replay_path)
            reused = True
        except Exception:
            pass
    if not reused:
        command = [
            cli,
            str(match["path_a"]),
            str(match["path_b"]),
            "--python",
            sys.executable,
            "--width",
            str(match["map_size"]),
            "--height",
            str(match["map_size"]),
            "--seed",
            str(match["seed"]),
            "--memory",
            "8000",
            "--maxtime",
            str(agent_timeout),
            "--loglevel",
            "4",
            "--storeLogs=false",
            "--out",
            str(replay_path),
        ]
        with log_path.open("w", encoding="utf-8") as log_file:
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            child_env = dict(os.environ)
            preload = str(PROJECT_DIR / "lux_node_stream_patch.js")
            child_env["NODE_OPTIONS"] = "{} --require={}".format(
                child_env.get("NODE_OPTIONS", ""), preload
            ).strip()
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
                env=child_env,
            )
            started = time.monotonic()
            samples = []
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= match_timeout:
                    stop_process_tree(process)
                    process.wait()
                    log_file.write(
                        "\nTournament runner stopped this match after {} seconds.\n".format(
                            match_timeout
                        )
                    )
                    return_code = 124
                    break
                gpu = sample_gpu()
                samples.append((round(elapsed, 1),) + tuple(gpu))
                time.sleep(10)
            else:
                return_code = process.returncode
            duration_seconds = time.monotonic() - started

        with telemetry_path.open("w", newline="", encoding="utf-8") as telemetry_file:
            writer = csv.writer(telemetry_file)
            writer.writerow((
                "elapsed_seconds", "gpu_util_percent", "gpu_memory_util_percent",
                "gpu_memory_mb", "gpu_power_watts", "gpu_temperature_c",
            ))
            writer.writerows(samples)

    try:
        rewards, statuses, outcome = read_replay(replay_path)
        error = ""
    except Exception as exc:
        rewards = ("", "")
        statuses = ("", "")
        outcome = "error"
        error = str(exc)

    result = dict(match)
    result.update({
        "outcome": outcome,
        "reward_a": rewards[0] if len(rewards) > 0 else "",
        "reward_b": rewards[1] if len(rewards) > 1 else "",
        "status_a": statuses[0] if len(statuses) > 0 else "",
        "status_b": statuses[1] if len(statuses) > 1 else "",
        "return_code": return_code,
        "reused": reused,
        "duration_seconds": "{:.1f}".format(duration_seconds) if not reused else "",
        "error": error,
        "replay": str(replay_path),
        "log": str(log_path),
        "telemetry": str(telemetry_path) if not reused else "",
    })
    return result


def build_matches(rounds):
    matches = []
    pairings = list(combinations(AGENTS, 2))
    for round_index in range(1, rounds + 1):
        for pair_index, (agent_a, agent_b) in enumerate(pairings, 1):
            seed_base = round_index * 100000 + pair_index * 1000
            for map_size in MAP_SIZES:
                for side, (first, second) in enumerate(
                    ((agent_a, agent_b), (agent_b, agent_a))
                ):
                    match_id = "r{:02d}_p{:02d}_m{:02d}_s{}_{}_vs_{}".format(
                        round_index,
                        pair_index,
                        map_size,
                        side,
                        first[0],
                        second[0],
                    )
                    matches.append({
                        "id": match_id,
                        "round": round_index,
                        "map_size": map_size,
                        "seed": seed_base + map_size,
                        "agent_a": first[0],
                        "agent_b": second[0],
                        "path_a": agent_path(first[1]),
                        "path_b": agent_path(second[1]),
                    })
    return matches


def open_csv_for_write(path):
    try:
        return path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        fallback = path.with_name(path.stem + "_updated" + path.suffix)
        print("{} is locked; writing {} instead.".format(path.name, fallback.name))
        return fallback.open("w", newline="", encoding="utf-8")


def write_matches_csv(results, output_dir):
    fieldnames = (
        "id", "round", "map_size", "seed", "agent_a", "agent_b",
        "outcome", "reward_a", "reward_b", "status_a", "status_b",
        "return_code", "reused", "error", "replay", "log",
        "duration_seconds", "telemetry",
    )
    with open_csv_for_write(output_dir / "matches.csv") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda result: result["id"]))


def calculate_rankings(results):
    stats = {
        agent_name: {
            "agent": agent_name,
            "games": 0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "errors": 0,
            "points": 0.0,
        }
        for agent_name, _folder in AGENTS
    }

    for result in results:
        a = stats[result["agent_a"]]
        b = stats[result["agent_b"]]
        outcome = result["outcome"]
        if outcome == "error":
            a["errors"] += 1
            b["errors"] += 1
            continue

        a["games"] += 1
        b["games"] += 1
        if result["status_a"] != "DONE":
            a["errors"] += 1
        if result["status_b"] != "DONE":
            b["errors"] += 1

        if outcome == "a":
            a["wins"] += 1
            b["losses"] += 1
            a["points"] += 1.0
        elif outcome == "b":
            b["wins"] += 1
            a["losses"] += 1
            b["points"] += 1.0
        else:
            a["ties"] += 1
            b["ties"] += 1
            a["points"] += 0.5
            b["points"] += 0.5

    for values in stats.values():
        games = values["games"]
        values["point_rate"] = values["points"] / games if games else 0.0

    return sorted(
        stats.values(),
        key=lambda values: (
            values["point_rate"],
            values["points"],
            values["wins"],
            -values["errors"],
        ),
        reverse=True,
    )


def write_rankings(rankings, output_dir):
    fields = (
        "rank", "agent", "games", "wins", "losses", "ties", "errors",
        "points", "point_rate",
    )
    with open_csv_for_write(output_dir / "rankings.csv") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for rank, values in enumerate(rankings, 1):
            row = dict(values)
            row["rank"] = rank
            row["point_rate"] = "{:.4f}".format(row["point_rate"])
            writer.writerow(row)

    print("\nFinal ranking")
    print("{:<5} {:<38} {:>5} {:>4} {:>4} {:>4} {:>6} {:>8}".format(
        "Rank", "Agent", "Games", "W", "L", "T", "Errors", "Score"
    ))
    for rank, values in enumerate(rankings, 1):
        print("{:<5} {:<38} {:>5} {:>4} {:>4} {:>4} {:>6} {:>7.1%}".format(
            rank,
            values["agent"],
            values["games"],
            values["wins"],
            values["losses"],
            values["ties"],
            values["errors"],
            values["point_rate"],
        ))


def main():
    args = parse_args()
    if min(args.rounds, args.workers, args.match_timeout, args.agent_timeout) < 1:
        raise SystemExit("Tournament counts and timeouts must all be at least 1")

    cli = shutil.which("lux-ai-2021.cmd") or shutil.which("lux-ai-2021")
    if cli is None:
        raise SystemExit("lux-ai-2021 was not found on PATH")

    output_dir = args.output.resolve()
    (output_dir / "replays").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "telemetry").mkdir(parents=True, exist_ok=True)

    missing_agents = [
        str(agent_path(folder))
        for _name, folder in AGENTS
        if not agent_path(folder).is_file()
    ]
    if missing_agents:
        raise SystemExit("Missing agents:\n" + "\n".join(missing_agents))

    matches = build_matches(args.rounds)
    print("Scheduling {} games across {} agents.".format(len(matches), len(AGENTS)))
    print("Using Python: {}".format(sys.executable))

    results = []

    def record_result(result):
        results.append(result)
        print("[{}/{}] {} -> {}".format(
            len(results), len(matches), result["id"], result["outcome"]
        ), flush=True)

    if args.workers == 1:
        for match in matches:
            record_result(run_match(
                match, cli, output_dir, args.match_timeout, args.agent_timeout
            ))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_match, match, cli, output_dir,
                    args.match_timeout, args.agent_timeout
                ): match
                for match in matches
            }
            for future in as_completed(futures):
                result = future.result()
                record_result(result)

    write_matches_csv(results, output_dir)
    rankings = calculate_rankings(results)
    write_rankings(rankings, output_dir)


if __name__ == "__main__":
    main()
