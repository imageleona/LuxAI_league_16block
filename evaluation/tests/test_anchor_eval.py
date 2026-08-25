"""
Scheduling and formatting tests for the periodic anchor evaluation.

No GPU, no Node, no games: the round function is monkeypatched throughout, so
these run in milliseconds and cover the logic that is otherwise only exercised by
a multi-hour training run.
"""
import json
import threading
import time
from pathlib import Path

import pytest

from evaluation.anchor_eval import (
    AnchorEvalConfig,
    format_round_lines,
    wandb_payload,
)
from lux_ai.torchbeast.anchor_eval_hook import AnchorEvalScheduler


CONFIG_KWARGS = dict(enabled=True, every_n_games=10, n_seeds=1, map_sizes=(12,),
                     parallel_games=1, at_start=False, blocking=True,
                     max_consecutive_failures=2)


def _aggregate(win_rate=0.6, games=20):
    return {"games": games, "wins": 12, "losses": 7, "ties": 1, "win_rate": win_rate,
            "ci95": [0.4, 0.8], "mean_city_tile_margin": 1.5}


def _record(**kw):
    base = {"per_anchor": {"anchor_a": _aggregate()}, "errors": {}, "mean_win_rate": 0.6,
            "games": 20, "wall_seconds": 12.5, "round": 3, "step": 1234560,
            "league_games": 500, "rounds_skipped": 0, "timestamp": "2026-08-21T14:03:11"}
    base.update(kw)
    return base


def _scheduler(tmp_path, monkeypatch, round_fn, **overrides):
    """A scheduler with no real anchors and a fake round."""
    import evaluation.anchor_eval as ae

    monkeypatch.setattr(ae, "load_anchor_specs", lambda dirs, device="cpu": ["fake-anchor"])
    monkeypatch.setattr(ae, "run_anchor_round", round_fn)
    kwargs = dict(CONFIG_KWARGS)
    kwargs.update(overrides)
    return AnchorEvalScheduler(AnchorEvalConfig(**kwargs), tmp_path, ["fake-dir"])


def _dump(target_dir: Path, weights_filename: str) -> None:
    rl = Path(target_dir) / "lux_ai" / "rl_agent"
    rl.mkdir(parents=True, exist_ok=True)
    (rl / weights_filename).write_bytes(b"weights")


# ------------------------------------------------------------------ config

def test_config_rejects_unsupported_map_size():
    # MatchRequest would raise mid-round otherwise, inside a worker thread.
    with pytest.raises(ValueError, match="map_sizes"):
        AnchorEvalConfig(map_sizes=(12, 20))


def test_games_per_anchor_matches_paired_schedule():
    assert AnchorEvalConfig(n_seeds=5, map_sizes=(16, 32)).games_per_anchor == 20


# ----------------------------------------------------------------- trigger

def test_fires_every_n_games(tmp_path, monkeypatch):
    calls = []
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: calls.append(kw) or _record())
    for _ in range(3):
        sched.note_games(9)
        assert not sched.due()          # 9, 18-10=8, ... never reaches the threshold alone
        sched.note_games(1)
        assert sched.due()
        assert sched.start(_dump, step=1)
    assert len(calls) == 3


def test_at_start_fires_immediately_then_waits(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record(), at_start=True)
    assert sched.due(), "at_start must produce a baseline round before any games"
    sched.start(_dump, step=0)
    assert not sched.due()
    sched.note_games(10)
    assert sched.due()


def test_single_flight_skips_while_a_round_is_running(tmp_path, monkeypatch):
    release = threading.Event()

    def slow_round(**kw):
        release.wait(timeout=5)
        return _record()

    sched = _scheduler(tmp_path, monkeypatch, slow_round, blocking=False)
    sched.note_games(10)
    assert sched.due()
    sched.start(_dump, step=1)
    sched.note_games(10)
    assert not sched.due(), "must not start a second round on top of a running one"
    assert sched._rounds_skipped == 1
    release.set()
    sched.shutdown(timeout=5)


# ----------------------------------------------------------------- failure

def test_round_exception_does_not_propagate_and_disables_after_n(tmp_path, monkeypatch):
    def boom(**kw):
        raise RuntimeError("engine exploded")

    sched = _scheduler(tmp_path, monkeypatch, boom)
    for _ in range(2):
        sched.note_games(10)
        assert sched.due()
        sched.start(_dump, step=1)      # must not raise
        sched.poll(step=1)
    assert sched._disabled, "must stop retrying a broken eval for the rest of the run"
    sched.note_games(100)
    assert not sched.due()


def test_partial_failure_still_counts_as_success(tmp_path, monkeypatch):
    record = _record(errors={"anchor_b": "ValueError: nope"})
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: record)
    sched.note_games(10)
    sched.start(_dump, step=1)
    sched.poll(step=1)
    assert not sched._disabled
    assert sched._consecutive_failures == 0


# ------------------------------------------------------------------ output

def test_poll_writes_txt_and_jsonl_and_returns_payload(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record())
    sched.note_games(10)
    sched.start(_dump, step=1234560)
    payload = sched.poll(step=1300000)

    txt = (sched.output_root / "anchor_eval.txt").read_text(encoding="utf-8").splitlines()
    assert any("anchor=anchor_a" in line for line in txt)
    assert any("ROUND" in line for line in txt)
    rows = (sched.output_root / "anchor_eval.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[0])["per_anchor"]["anchor_a"]["win_rate"] == 0.6
    assert payload is not None and "AnchorEval" in payload


def test_poll_returns_none_when_nothing_finished(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record())
    assert sched.poll(step=1) is None


def test_wandb_payload_keys(tmp_path):
    stats = wandb_payload(_record(), current_step=1_300_000)["AnchorEval"]
    for key in ("win_rate_mean", "round", "games", "wall_seconds", "eval_step",
                "league_games_at_eval", "failures", "rounds_skipped", "step_lag",
                "win_rate/anchor_a", "ci95_low/anchor_a", "ci95_high/anchor_a",
                "city_margin/anchor_a", "games/anchor_a"):
        assert key in stats, key
    # The lag between freezing the weights and the round finishing must be visible.
    assert stats["step_lag"] == 1_300_000 - 1_234_560


def test_format_round_lines_roundtrips_the_numbers():
    lines = format_round_lines(_record())
    anchor_line = next(line for line in lines if "anchor=anchor_a" in line)
    fields = dict(tok.split("=", 1) for tok in anchor_line.split() if "=" in tok)
    assert fields["n"] == "20"
    assert float(fields["wr"]) == pytest.approx(0.6)
    assert fields["wlt"] == "12/7/1"
    assert float(fields["margin"]) == pytest.approx(1.5)


def test_errors_are_written_to_the_txt_log(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch,
                       lambda **kw: _record(errors={"anchor_b": "OSError: gone"}))
    sched.note_games(10)
    sched.start(_dump, step=1)
    sched.poll(step=1)
    txt = (sched.output_root / "anchor_eval.txt").read_text(encoding="utf-8")
    assert "anchor=anchor_b ERROR OSError: gone" in txt


# ------------------------------------------------------------------ resume

def test_counters_survive_a_restart(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record())
    sched.note_games(10)
    sched.start(_dump, step=1)
    assert sched._round == 1

    # A fresh scheduler on the same run dir continues the cadence rather than
    # restarting it - and does not re-fire the at_start baseline.
    resumed = _scheduler(tmp_path, monkeypatch, lambda **kw: _record(), at_start=True)
    assert resumed._round == 1
    assert not resumed.due()


# ------------------------------------------------------------------ safety

def test_disabled_config_is_inert(tmp_path, monkeypatch):
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record(), enabled=False)
    sched.note_games(1000)
    assert not sched.due()
    assert not sched.start(_dump, step=1)


def test_start_refuses_to_write_outside_its_scratch_dir(tmp_path, monkeypatch):
    """The one path that touches the filesystem must be pinned to its own dir."""
    sched = _scheduler(tmp_path, monkeypatch, lambda **kw: _record())
    sched.candidate_dir = tmp_path / "somewhere" / "else"
    with pytest.raises(RuntimeError, match="refusing to write outside"):
        sched.start(_dump, step=1)
