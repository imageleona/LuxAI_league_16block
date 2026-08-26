import json
import logging
import os
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from .flags import LeagueFlags
from .member_config import validate_member
from .pool import ANCHOR, EXTERNAL, SNAPSHOT, OpponentPool, PoolMember
from .sampler import PFSPSampler
from .state_io import read_state, replace_with_retry, write_state_atomic
from .winrate import WinRateTracker


def resolve_agent_files(dir_path: Union[str, Path]) -> Tuple[Path, Path]:
    """
    Resolve (config_path, checkpoint_path) for an agent directory.
    Accepts the frozen-agent layout (<dir>/lux_ai/rl_agent/{config.yaml, *.pt})
    and the flat training-output layout (<dir>/{config.yaml, *_weights.pt}).
    """
    dir_path = Path(dir_path).expanduser().resolve()
    rl_dir = dir_path / "lux_ai" / "rl_agent"
    if (rl_dir / "config.yaml").is_file():
        checkpoints = sorted(rl_dir.glob("*.pt"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected exactly one *.pt in {rl_dir}, found {len(checkpoints)}")
        return rl_dir / "config.yaml", checkpoints[0]
    if (dir_path / "config.yaml").is_file():
        checkpoints = sorted(dir_path.glob("*_weights.pt")) or sorted(dir_path.glob("*.pt"))
        if len(checkpoints) < 1:
            raise ValueError(f"No *.pt checkpoint found in {dir_path}")
        # Take the latest checkpoint of a training-output dir.
        return dir_path / "config.yaml", checkpoints[-1]
    raise FileNotFoundError(f"No agent config.yaml found under {dir_path}")


class LeagueManager:
    """
    Main-process owner of the opponent pool, win-rate tracker and sampler.
    Publishes {members, probs} to actor processes through a versioned JSON
    state file, and receives (member_id, outcome) tuples through a queue that
    the training main loop drains into report_outcome().
    """

    def __init__(
            self,
            league_flags: LeagueFlags,
            run_dir: Union[str, Path],
            student_spec: Dict,
            teacher_spec: Optional[Dict],
            seed: Optional[int] = None,
    ):
        self.flags = league_flags
        self.run_dir = Path(run_dir).resolve()
        self.league_dir = self.run_dir / "league"
        self.state_path = Path(league_flags.state_path) if league_flags.state_path \
            else self.league_dir / "state.json"
        # {"obs_space": name, "obs_space_kwargs": dict, "act_space": name}
        self.student_spec = student_spec
        self.teacher_spec = teacher_spec
        self.env_is_multiobs = (
            teacher_spec is not None
            and (teacher_spec["obs_space"], teacher_spec["obs_space_kwargs"])
            != (student_spec["obs_space"], student_spec["obs_space_kwargs"])
        )

        rng_seed = None if seed is None else int(seed) + 977
        self.rng = np.random.RandomState(rng_seed)
        self.pool = OpponentPool(league_flags.pool_size, rng=self.rng)
        self.tracker = WinRateTracker(ema_lambda=league_flags.winrate_lambda,
                                      init=league_flags.winrate_init)
        self.sampler = PFSPSampler(priority=league_flags.priority,
                                   alpha=league_flags.alpha,
                                   eps_uniform=league_flags.eps_uniform)
        # Rolling window of the most recent games against permanent anchors. The
        # EMA in WinRateTracker is short-memory on purpose (opponent selection must
        # react quickly); this is the progress measure, and it needs many games.
        self._anchor_window: Deque[Tuple[str, float]] = deque(
            maxlen=max(1, league_flags.anchor_report_every_n_games))
        # Rolling outcomes against ONE reference opponent, for the win-rate admission
        # gate. Separate from _anchor_window, which pools all anchors together and so
        # cannot answer "am I beating the checkpoint I started from?".
        self._reference_window: Deque[float] = deque(
            maxlen=max(1, league_flags.winrate_admit_window))
        self._reference_id: Optional[str] = None
        self.last_winrate_admit_updates = -10 ** 9
        self.version = 0
        self.last_snapshot_updates = 0
        self.last_snapshot_step = 0
        self.actual_snapshot_intervals: List[int] = []
        self._rl_agent_config_template = self._find_rl_agent_config_template()

    # ------------------------------------------------------------------ setup

    def seed_pool(self) -> None:
        for anchor_dir in self.flags.anchors:
            self._admit(anchor_dir, kind=ANCHOR)
        for member_dir in self.flags.initial_members:
            self._admit(member_dir, kind=EXTERNAL)
        if len(self.pool) == 0:
            logging.warning("League enabled but the pool is empty at startup: "
                            "all games will be mirror games until the first snapshot.")

    def _find_rl_agent_config_template(self) -> Optional[Path]:
        candidate = Path(__file__).resolve().parents[1] / "rl_agent" / "rl_agent_config.yaml"
        if candidate.is_file():
            return candidate
        for dir_path in list(self.flags.anchors) + list(self.flags.initial_members):
            candidate = Path(dir_path) / "lux_ai" / "rl_agent" / "rl_agent_config.yaml"
            if candidate.is_file():
                return candidate
        return None

    def _resolve_obs_space_prefix(self, member_conf: Dict) -> Optional[str]:
        member_obs = (member_conf.get("obs_space"), member_conf.get("obs_space_kwargs") or {})
        student_obs = (self.student_spec["obs_space"], self.student_spec["obs_space_kwargs"] or {})
        if member_obs == student_obs:
            return "student_" if self.env_is_multiobs else ""
        if self.env_is_multiobs and self.teacher_spec is not None:
            teacher_obs = (self.teacher_spec["obs_space"], self.teacher_spec["obs_space_kwargs"] or {})
            if member_obs == teacher_obs:
                return "teacher_"
        return None

    def _unique_member_id(self, base: str) -> str:
        member_id = base
        suffix = 2
        while member_id in self.pool:
            member_id = f"{base}-{suffix}"
            suffix += 1
        return member_id

    def _admit(self, dir_path: Union[str, Path], kind: str,
               force: bool = False) -> Optional[PoolMember]:
        dir_path = Path(dir_path).expanduser().resolve()
        try:
            config_path, checkpoint_path = resolve_agent_files(dir_path)
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"League: skipping {dir_path}: {e}")
            return None
        member_conf = OmegaConf.to_container(OmegaConf.load(config_path))

        if member_conf.get("act_space") != self.student_spec["act_space"]:
            logging.warning(f"League: skipping {dir_path}: act_space "
                            f"{member_conf.get('act_space')} != {self.student_spec['act_space']}")
            return None
        prefix = self._resolve_obs_space_prefix(member_conf)
        if prefix is None:
            logging.warning(f"League: skipping {dir_path}: obs_space {member_conf.get('obs_space')} "
                            f"matches neither the student's nor the teacher's")
            return None
        # Actually build the model and load its weights here: an actor process
        # dies silently on an unhandled exception, so a member that cannot load
        # must be rejected before it can ever be sampled.
        rejection = validate_member(config_path, checkpoint_path)
        if rejection is not None:
            logging.warning(f"League: skipping {dir_path}: {rejection}")
            return None

        member = PoolMember(
            member_id=self._unique_member_id(dir_path.name),
            dir_path=str(dir_path),
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            kind=kind,
            obs_space_prefix=prefix,
        )
        if kind == ANCHOR:
            self.pool.add_anchor(member)
            accepted, evicted = True, None
        elif force:
            # Evict whichever non-anchor the agent beats most easily: PFSP already
            # samples it least, so it is contributing the least gradient.
            others = [m for m in self.pool.members if m.kind != ANCHOR]
            evict_id = max(others, key=lambda m: self.tracker.winrate(m.member_id)).member_id                 if others else None
            evicted = self.pool.replace(member, evict_id)
            accepted = True
        else:
            accepted, evicted = self.pool.offer(member)
        if evicted is not None:
            self.tracker.remove_member(evicted.member_id)
            logging.info(f"League: evicted {evicted.member_id} from the pool")
        if accepted:
            self.tracker.add_member(member.member_id)
            logging.info(f"League: added {member.member_id} ({kind}, prefix='{prefix}') to the pool")
            return member
        return None

    # ------------------------------------------------------------- interface

    def report_outcome(self, member_id: str, outcome: float, step: Optional[int] = None) -> None:
        self.tracker.update(member_id, outcome)
        self._append_outcome_log(member_id, outcome, step)
        try:
            if self.pool.get(member_id).kind == ANCHOR:
                self._anchor_window.append((member_id, outcome))
                if member_id == self.reference_id():
                    self._reference_window.append(outcome)
        except KeyError:
            pass

    def _append_outcome_log(self, member_id: str, outcome: float, step: Optional[int]) -> None:
        """
        One line per finished league game.

        The EMA in WinRateTracker is deliberately short-memory because opponent
        SELECTION must react to the current policy - but that makes it useless as a
        progress measure, since it discards all but the last ~1/lambda games. Writing
        the raw outcomes means any windowed or step-bucketed statistic (and Elo
        against the permanent anchors) can be recovered afterwards. ~1 KB per 10
        games.
        """
        try:
            kind = self.pool.get(member_id).kind
        except KeyError:
            kind = "unknown"
        try:
            self.league_dir.mkdir(parents=True, exist_ok=True)
            with (self.league_dir / "outcomes.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "member_id": member_id,
                                    "kind": kind, "outcome": outcome}) + "\n")
        except OSError as e:
            logging.warning(f"League: could not append to the outcome log ({e})")

    def drain_outcomes(self, outcome_queue, step: Optional[int] = None) -> int:
        n = 0
        while not outcome_queue.empty():
            member_id, outcome = outcome_queue.get()
            self.report_outcome(member_id, outcome, step)
            n += 1
        return n

    def write_agent_dir(
            self,
            model_state_dict: Dict,
            target_dir: Union[str, Path],
            weights_filename: str,
    ) -> Path:
        """
        Write the AgentSpec layout into <target_dir>/lux_ai/rl_agent:
        config.yaml, rl_agent_config.yaml and one <weights_filename>.

        Pure I/O. Does NOT admit to the pool, touch the win-rate tracker, or update
        the snapshot bookkeeping - callers that want a pool member call snapshot().

        DELETES NOTHING, deliberately. AgentSpec.from_directory requires exactly one
        *.pt in the directory, and the tempting way to maintain that when reusing a
        directory is to glob and unlink the old checkpoint - which, pointed at the
        wrong path, would erase real model weights. Callers that reuse a directory
        across rounds must instead pass a CONSTANT weights_filename, so the
        invariant holds by overwriting one file rather than removing any.
        """
        target_dir = Path(target_dir)
        rl_dir = target_dir / "lux_ai" / "rl_agent"
        rl_dir.mkdir(parents=True, exist_ok=True)

        run_config = self.run_dir / "config.yaml"
        if run_config.is_file():
            shutil.copy(str(run_config), str(rl_dir / "config.yaml"))
        else:
            raise FileNotFoundError(f"Run config not found at {run_config}; cannot write agent dir")
        if self._rl_agent_config_template is not None:
            shutil.copy(str(self._rl_agent_config_template), str(rl_dir / "rl_agent_config.yaml"))
        else:
            logging.warning("League: no rl_agent_config.yaml template found; agent dir will not be "
                            "loadable by evaluation.harness.AgentSpec")
        cpu_state_dict = {key: val.detach().cpu() for key, val in model_state_dict.items()}
        # Write-then-rename: a crash mid-dump must not leave a truncated checkpoint
        # for the next round to hash and try to load. os.replace is atomic, and the
        # .tmp suffix keeps the partial file invisible to glob("*.pt").
        final_path = rl_dir / weights_filename
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save({"model_state_dict": cpu_state_dict}, str(tmp_path))
        # Retry the rename: on Windows os.replace fails while any other process
        # holds the destination open. Unlike publishing the league state, a failed
        # checkpoint write must NOT be swallowed - the caller would go on to
        # evaluate the previous round's weights and log them against this step.
        if not replace_with_retry(tmp_path, final_path):
            raise OSError("could not replace {}; a reader held it open".format(final_path))
        return target_dir

    def snapshot(self, model_state_dict: Dict, step: int, updates: int,
                 force_admit: bool = False) -> Optional[PoolMember]:
        """
        Freeze the current policy into an AgentSpec-compatible directory and add it
        to the pool.

        force_admit=False is the interval path: the snapshot is offered to the
        reservoir and usually rejected. force_admit=True is the win-rate path - the
        policy has beaten the reference opponent often enough to earn a guaranteed
        place, so it displaces an existing member. The two write to different
        directories so a snapshot taken by both paths on the same step cannot collide.
        """
        name = f"{step:012d}_wr" if force_admit else f"{step:012d}"
        snapshot_dir = self.write_agent_dir(
            model_state_dict,
            self.league_dir / "snapshots" / name,
            f"{name}_weights.pt",
        )
        if force_admit:
            self.last_winrate_admit_updates = updates
            return self._admit(snapshot_dir, kind=SNAPSHOT, force=True)
        if self.last_snapshot_step or self.last_snapshot_updates:
            self.actual_snapshot_intervals.append(updates - self.last_snapshot_updates)
        self.last_snapshot_updates = updates
        self.last_snapshot_step = step
        return self._admit(snapshot_dir, kind=SNAPSHOT)

    def admit_evaluated_snapshot(
            self,
            evaluated_agent_dir: Union[str, Path],
            step: int,
            round_idx: int,
    ) -> Optional[PoolMember]:
        """Force-admit the exact candidate that passed fixed-anchor evaluation.

        The evaluator reuses its candidate and best directories, so the pool may
        not point at either mutable location. Copy the known files into a unique,
        permanent snapshot directory. No file is deleted or globbed.
        """
        source_dir = Path(evaluated_agent_dir).expanduser().resolve()
        config_path, checkpoint_path = resolve_agent_files(source_dir)
        name = "{:012d}_eval_{:03d}".format(int(step), int(round_idx))
        target_dir = self.league_dir / "snapshots" / name
        rl_dir = target_dir / "lux_ai" / "rl_agent"
        rl_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(config_path), str(rl_dir / "config.yaml"))
        source_agent_config = config_path.parent / "rl_agent_config.yaml"
        if source_agent_config.is_file():
            shutil.copy2(str(source_agent_config), str(rl_dir / "rl_agent_config.yaml"))
        final_path = rl_dir / (name + "_weights.pt")
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        shutil.copy2(str(checkpoint_path), str(tmp_path))
        if not replace_with_retry(tmp_path, final_path):
            raise OSError("could not publish evaluated snapshot {}".format(final_path))
        return self._admit(target_dir, kind=SNAPSHOT, force=True)

    def reference_id(self) -> Optional[str]:
        """
        The anchor the win-rate admission gate measures against. Defaults to the first
        configured anchor, which is the checkpoint the run started from - so the gate
        asks "is the current policy actually better than where we began?".
        """
        if self._reference_id is not None:
            return self._reference_id
        anchors = [m for m in self.pool.members if m.kind == ANCHOR]
        if not anchors:
            return None
        wanted = self.flags.winrate_admit_reference
        match = next((m for m in anchors if wanted and wanted in m.member_id), None)
        self._reference_id = (match or anchors[0]).member_id
        return self._reference_id

    def reference_winrate(self) -> Optional[float]:
        """Win rate over the last N games against the reference opponent, or None
        until the window is full - the guard against admitting on a lucky streak."""
        window = self._reference_window
        if len(window) < window.maxlen:
            return None
        return sum(window) / len(window)

    def winrate_admission_due(self, updates: int) -> bool:
        """True if the current policy has earned a forced place in the pool."""
        if self.flags.winrate_admit_threshold <= 0.:
            return False
        if updates - self.last_winrate_admit_updates < self.flags.winrate_admit_cooldown_updates:
            return False
        rate = self.reference_winrate()
        return rate is not None and rate >= self.flags.winrate_admit_threshold

    def anchor_rolling_winrate(self) -> Optional[float]:
        """
        Win rate over the last N games against permanent anchors, or None until the
        window is full.

        Returning None rather than a partial mean is deliberate and is what makes this
        usable as a stopping rule: the number always rests on the same sample size, so
        it cannot look dramatic while resting on three games. Callers that stop a run
        on this value get the "don't stop on a lucky streak" guard for free.

        NB: not WinRateTracker.winrate() - that is an EMA with ~1/lambda games of
        memory (19 at the default), deliberately short so opponent SELECTION reacts to
        the current policy. It is far too noisy to terminate a run on.
        """
        window = self._anchor_window
        if len(window) < window.maxlen:
            return None
        return sum(outcome for _, outcome in window) / len(window)

    # -------------------------------------------------------------- publish

    def current_probs(self) -> List[float]:
        members = self.pool.members
        winrates = [self.tracker.winrate(m.member_id) for m in members]
        probs = self.sampler.probs(winrates)
        return self._apply_anchor_floor(members, probs)

    def _apply_anchor_floor(self, members: List[PoolMember], probs: List[float]) -> List[float]:
        """
        Guarantee each permanent anchor at least `anchor_floor` of the league games.

        The anchors are the only opponents that never change, so games against them
        are the one fixed yardstick available during training. Left to PFSP alone,
        an anchor the agent beats gets sampled rarely and the yardstick starves
        exactly when we most want to know whether the agent is improving.

        The floor is taken out of the non-anchor members proportionally, so the
        distribution still sums to 1 and the priority ordering among non-anchors is
        unchanged.
        """
        floor = self.flags.anchor_floor
        if floor <= 0. or not probs:
            return probs
        anchor_idx = [i for i, m in enumerate(members) if m.kind == ANCHOR]
        other_idx = [i for i, m in enumerate(members) if m.kind != ANCHOR]
        if not anchor_idx or not other_idx:
            return probs
        # Do not let the floors consume the entire distribution.
        floor = min(floor, 0.9 / len(anchor_idx))
        deficit = sum(max(0., floor - probs[i]) for i in anchor_idx)
        if deficit <= 0.:
            return probs
        other_total = sum(probs[i] for i in other_idx)
        if other_total <= 0.:
            return probs
        scale = max(0., 1. - deficit / other_total)
        out = list(probs)
        for i in anchor_idx:
            out[i] = max(probs[i], floor)
        for i in other_idx:
            out[i] = probs[i] * scale
        total = sum(out)
        return [p / total for p in out] if total > 0 else probs

    def effective_pool_size(self) -> float:
        probs = self.current_probs()
        denom = sum(p * p for p in probs)
        return 1. / denom if denom > 0 else 0.

    def publish(self) -> None:
        self.version += 1
        members = self.pool.members
        state = {
            "version": self.version,
            "members": [m.to_dict() for m in members],
            "probs": self.current_probs(),
            "winrates": {m.member_id: self.tracker.winrate(m.member_id) for m in members},
            "games": {m.member_id: self.tracker.games(m.member_id) for m in members},
            "pool": self.pool.to_dict(),
            "last_snapshot_updates": self.last_snapshot_updates,
            "last_snapshot_step": self.last_snapshot_step,
        }
        write_state_atomic(self.state_path, state)

    def load_state(self, state_path: Union[str, Path]) -> bool:
        """Resume league state from a previous run's state.json."""
        state = read_state(state_path)
        if state is None:
            return False
        pool = OpponentPool.from_dict(state["pool"], rng=self.rng)
        # Drop members whose directories have disappeared.
        pool._anchors = [m for m in pool._anchors if Path(m.checkpoint_path).is_file()]
        pool._others = [m for m in pool._others if Path(m.checkpoint_path).is_file()]
        self.pool = pool
        self.tracker = WinRateTracker.from_dict(
            {"winrates": state["winrates"], "games": state["games"]},
            ema_lambda=self.flags.winrate_lambda,
            init=self.flags.winrate_init,
        )
        for member in self.pool.members:
            self.tracker.add_member(member.member_id)
        self.last_snapshot_updates = state.get("last_snapshot_updates", 0)
        self.last_snapshot_step = state.get("last_snapshot_step", 0)
        # Re-admit configured anchors that are not in the resumed pool.
        for anchor_dir in self.flags.anchors:
            resolved = str(Path(anchor_dir).expanduser().resolve())
            if not any(m.dir_path == resolved for m in self.pool.members):
                self._admit(anchor_dir, kind=ANCHOR)
        logging.info(f"League: resumed pool with {len(self.pool)} members from {state_path}")
        return True

    # -------------------------------------------------------------- logging

    def wandb_stats(self) -> Dict:
        members = self.pool.members
        probs = self.current_probs()
        stats = {
            "pool_size": len(members),
            "effective_pool_size": self.effective_pool_size(),
        }
        if self.actual_snapshot_intervals:
            stats["actual_snapshot_interval"] = self.actual_snapshot_intervals[-1]
        for member, prob in zip(members, probs):
            stats[f"prob/{member.member_id}"] = prob
            stats[f"winrate/{member.member_id}"] = self.tracker.winrate(member.member_id)
            stats[f"games/{member.member_id}"] = self.tracker.games(member.member_id)

        # Progress yardstick: win rate over a rolling window of the last N games
        # against the permanent anchors. Only reported once the window is full, so
        # the number always rests on the same sample size and cannot look dramatic
        # while resting on three games.
        if self.flags.winrate_admit_threshold > 0.:
            stats["reference_winrate"] = self.reference_winrate()
            stats["reference_window_games"] = len(self._reference_window)
        window = list(self._anchor_window)
        rolling = self.anchor_rolling_winrate()
        if rolling is not None:
            stats["anchor_winrate_rolling"] = rolling
            stats["anchor_window_games"] = len(window)
            per = defaultdict(lambda: [0., 0])
            for member_id, outcome in window:
                per[member_id][0] += outcome
                per[member_id][1] += 1
            for member_id, (s, n) in per.items():
                stats[f"anchor_winrate_rolling/{member_id}"] = s / n
                stats[f"anchor_window_games/{member_id}"] = n
        return {"League": stats}
