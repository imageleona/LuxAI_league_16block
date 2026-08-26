"""
Scheduling side of the periodic anchor evaluation.

Owns the trigger counter, the worker thread, the result hand-off and the failure
policy. The evaluation itself lives in `evaluation.anchor_eval`; this module is
the only place that knows about the trainer.

Two design points worth keeping:

- **Thread, not a blocking call.** `league_outcome_queue` is an `mp.SimpleQueue`
  (one pipe). Measured on this machine, it holds **136** outcomes before `put()`
  blocks - about 29 minutes at the observed league-game rate. A real evaluation
  round is ~19 minutes idle and more under contention, so blocking the main loop
  can fill the pipe, block the actors inside `handle_dones`, and starve the
  learner. Running the round on a worker thread keeps the main loop draining.
  `blocking=True` remains available for smoke tests, where rounds are seconds.

- **The worker never calls wandb.** By the time a round finishes, `step` has moved
  on by ~150k, and wandb silently drops out-of-order steps. The worker pushes onto
  a queue; the main loop's `poll(step)` logs at the current step and reports
  `step_lag` so the attribution gap is visible rather than hidden.
"""
import datetime as _dt
import json
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

CANDIDATE_WEIGHTS_FILENAME = "candidate_weights.pt"


class AnchorEvalScheduler:
    """Fires a fixed-seed anchor evaluation every N league games."""

    def __init__(
            self,
            config: Any,
            run_dir: Union[str, Path],
            anchor_dirs: Sequence[Union[str, Path]],
            seed_games: int = 0,
    ) -> None:
        from evaluation.anchor_eval import load_anchor_specs

        self.config = config
        self.run_dir = Path(run_dir)
        self.output_root = self.run_dir / "league" / "anchor_eval"
        self.candidate_dir = self.output_root / "candidate"
        self.best_dir = self.output_root / "best"
        self.state_path = self.output_root / "scheduler_state.json"
        self.output_root.mkdir(parents=True, exist_ok=True)

        self._anchors = load_anchor_specs(anchor_dirs, device=config.device)
        self._results: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._games_since_round = 0
        self._games_total = int(seed_games)
        self._round = 0
        self._rounds_skipped = 0
        self._consecutive_failures = 0
        self._best_mean_win_rate: Optional[float] = None
        self._best_step: Optional[int] = None
        self._baseline_mean_win_rate: Optional[float] = None
        self._new_best_records: List[Dict[str, Any]] = []
        self._disabled = not config.enabled or not self._anchors
        self._pending_at_start = bool(config.at_start) and not self._disabled
        self._load_state()

        if not self._disabled:
            logging.info(
                "AnchorEval: enabled - %d anchors x %d games every %d league games (%s)",
                len(self._anchors), config.games_per_anchor, config.every_n_games,
                "blocking" if config.blocking else "threaded",
            )

    # ------------------------------------------------------------------ state

    def _load_state(self) -> None:
        if not self.state_path.is_file():
            return
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logging.warning("AnchorEval: could not read %s (%s); starting counters at zero",
                            self.state_path, e)
            return
        self._games_since_round = int(state.get("games_since_round", 0))
        self._games_total = int(state.get("games_total", self._games_total))
        self._round = int(state.get("round", 0))
        self._rounds_skipped = int(state.get("rounds_skipped", 0))
        best_mean = state.get("best_mean_win_rate")
        self._best_mean_win_rate = None if best_mean is None else float(best_mean)
        best_step = state.get("best_step")
        self._best_step = None if best_step is None else int(best_step)
        baseline = state.get("baseline_mean_win_rate")
        self._baseline_mean_win_rate = None if baseline is None else float(baseline)
        # A resumed run already has a baseline from before the interruption.
        self._pending_at_start = False
        logging.info("AnchorEval: resumed at round %d, %d league games seen",
                     self._round, self._games_total)

    def _save_state(self) -> None:
        payload = {
            "games_since_round": self._games_since_round,
            "games_total": self._games_total,
            "round": self._round,
            "rounds_skipped": self._rounds_skipped,
            "best_mean_win_rate": self._best_mean_win_rate,
            "best_step": self._best_step,
            "baseline_mean_win_rate": self._baseline_mean_win_rate,
        }
        try:
            self.state_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        except OSError as e:
            logging.warning("AnchorEval: could not write %s (%s)", self.state_path, e)

    # ---------------------------------------------------------------- trigger

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def note_games(self, n: int) -> None:
        """Add n finished league games (the return value of drain_outcomes)."""
        if self._disabled or n <= 0:
            return
        self._games_since_round += int(n)
        self._games_total += int(n)

    def due(self) -> bool:
        """True if a round should start now."""
        if self._disabled:
            return False
        if self._pending_at_start:
            return True
        if self._games_since_round < self.config.every_n_games:
            return False
        if self.running:
            # Single-flight: never start a second round on top of a slow one. Reset
            # the counter so the cadence stays aligned instead of firing immediately
            # again the moment the current round ends.
            self._rounds_skipped += 1
            self._games_since_round = 0
            logging.warning("AnchorEval: round %d still running, skipping this trigger "
                            "(%d skipped so far)", self._round, self._rounds_skipped)
            return False
        return True

    # ------------------------------------------------------------------- run

    def start(self, dump_fn: Callable[[Path, str], None], step: int) -> bool:
        """
        Freeze the current policy, then evaluate it.

        `dump_fn(target_dir, weights_filename)` is called on the CALLING thread so
        the weights match `step`. The evaluation then runs either here (blocking)
        or on a worker.
        """
        if self._disabled:
            return False

        # Belt and braces: this feature must only ever write inside its own scratch
        # directory. A future caller passing a different path would otherwise be
        # overwriting real checkpoints.
        candidate = self.candidate_dir.resolve()
        expected = (self.output_root / "candidate").resolve()
        if candidate != expected:
            raise RuntimeError(f"AnchorEval: refusing to write outside {expected} (got {candidate})")

        try:
            dump_fn(self.candidate_dir, CANDIDATE_WEIGHTS_FILENAME)
        except Exception:
            logging.exception("AnchorEval: could not write the candidate agent dir; skipping round")
            self._note_failure()
            self._pending_at_start = False
            self._games_since_round = 0
            return False

        self._pending_at_start = False
        self._games_since_round = 0
        self._round += 1
        self._save_state()

        args = (self._round, step, self._games_total)
        if self.config.blocking:
            self._run_round(*args)
        else:
            self._thread = threading.Thread(
                target=self._run_round, args=args,
                name="anchor-eval-%d" % self._round, daemon=True,
            )
            self._thread.start()
        return True

    def _run_round(self, round_idx: int, step: int, league_games: int) -> None:
        from evaluation.anchor_eval import run_anchor_round

        record: Dict[str, Any]
        try:
            record = run_anchor_round(
                candidate_dir=self.candidate_dir,
                candidate_name="candidate_%012d" % step,
                anchors=self._anchors,
                output_root=self.output_root,
                seeds=self.config.seeds,
                map_sizes=list(self.config.map_sizes),
                parallel_games=self.config.parallel_games,
                device=self.config.device,
                should_stop=self._stop.is_set,
            )
        except BaseException as e:  # noqa: BLE001 - a worker must never die silently
            logging.exception("AnchorEval: round %d failed outright", round_idx)
            record = {
                "per_anchor": {}, "errors": {"__round__": "{}: {}".format(type(e).__name__, e)},
                "mean_win_rate": None, "games": 0, "wall_seconds": 0.,
            }
        record["round"] = round_idx
        record["step"] = int(step)
        record["league_games"] = int(league_games)
        record["rounds_skipped"] = self._rounds_skipped
        record["timestamp"] = _dt.datetime.now().isoformat(timespec="seconds")
        self._results.put(record)

    # ------------------------------------------------------------------ drain

    def poll(self, step: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Write any finished rounds to disk and return the wandb payload, stamped
        with the CURRENT step. Returns None when nothing finished.
        """
        from evaluation.anchor_eval import wandb_payload, write_round

        payload = None
        while True:
            try:
                record = self._results.get_nowait()
            except queue.Empty:
                break
            try:
                self._consider_best(record)
                write_round(self.output_root, record)
            except OSError:
                logging.exception("AnchorEval: could not write the round log")
            if record.get("per_anchor"):
                self._consecutive_failures = 0
                logging.info("AnchorEval: round %d done - mean win rate %s over %d games (%.0fs)",
                             record["round"], record.get("mean_win_rate"),
                             record.get("games", 0), record.get("wall_seconds", 0.))
            else:
                self._note_failure()
            payload = wandb_payload(record, current_step=step)
        return payload

    def _consider_best(self, record: Dict[str, Any]) -> None:
        """Preserve the exact evaluated candidate when it sets a new best score."""
        mean = record.get("mean_win_rate")
        if not self.config.save_best or mean is None or not record.get("per_anchor"):
            record["is_best"] = False
            return
        mean = float(mean)
        if self._baseline_mean_win_rate is None:
            self._baseline_mean_win_rate = mean
        record["baseline_mean_win_rate"] = self._baseline_mean_win_rate
        record["delta_from_baseline"] = mean - self._baseline_mean_win_rate
        old_best = self._best_mean_win_rate
        improved = old_best is None or mean > old_best + self.config.best_min_delta
        record["is_best"] = improved
        record["previous_best_mean_win_rate"] = old_best
        if not improved:
            return

        source = self.candidate_dir / "lux_ai" / "rl_agent"
        destination = self.best_dir / "lux_ai" / "rl_agent"
        source_weights = source / CANDIDATE_WEIGHTS_FILENAME
        if not source_weights.is_file():
            raise FileNotFoundError(
                "AnchorEval: evaluated candidate weights disappeared before best-model save: {}".format(
                    source_weights
                )
            )
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("config.yaml", "rl_agent_config.yaml"):
            source_file = source / name
            if source_file.is_file():
                shutil.copy2(str(source_file), str(destination / name))
        # A constant name means repeated improvements overwrite one known file;
        # no globbing or deletion of older checkpoints is needed.
        shutil.copy2(str(source_weights), str(destination / "best_weights.pt"))
        metadata = {
            "step": int(record.get("step", 0)),
            "round": int(record.get("round", 0)),
            "mean_win_rate": mean,
            "per_anchor": record.get("per_anchor", {}),
        }
        (self.best_dir / "best_metadata.json").write_text(
            json.dumps(metadata, indent=1), encoding="utf-8"
        )
        self._best_mean_win_rate = mean
        self._best_step = metadata["step"]
        self._new_best_records.append(dict(record))
        self._save_state()
        logging.info(
            "AnchorEval: new best fixed-anchor mean %.4f at step %d (previous %s)",
            mean, self._best_step, old_best,
        )

    def pop_new_best_records(self) -> List[Dict[str, Any]]:
        """Return newly preserved best records once, on the trainer thread."""
        records = self._new_best_records
        self._new_best_records = []
        return records

    def _note_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.config.max_consecutive_failures:
            self._disabled = True
            logging.error("AnchorEval: DISABLED after %d consecutive failures - "
                          "training continues without anchor evaluation",
                          self._consecutive_failures)

    def shutdown(self, timeout: float = 60.) -> None:
        """Ask a running round to stop between anchors and wait briefly for it."""
        self._stop.set()
        if self.running:
            logging.info("AnchorEval: waiting up to %.0fs for the running round", timeout)
            self._thread.join(timeout=timeout)
        self._save_state()
