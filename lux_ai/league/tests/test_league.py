"""Unit tests for the league core. Pure python + numpy/torch, no engine or GPU."""
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from lux_ai.league.flags import LeagueFlags
from lux_ai.league.pool import ANCHOR, OpponentPool, PoolMember
from lux_ai.league.sampler import PFSPSampler
from lux_ai.league.state_io import read_state, read_state_if_newer, write_state_atomic
from lux_ai.league.winrate import WinRateTracker


def make_member(member_id, kind="snapshot"):
    return PoolMember(
        member_id=member_id,
        dir_path=f"/fake/{member_id}",
        config_path=f"/fake/{member_id}/config.yaml",
        checkpoint_path=f"/fake/{member_id}/weights.pt",
        kind=kind,
    )


# ---------------------------------------------------------------------- flags

def test_league_flags_from_dict_ignores_unknown_keys():
    flags = LeagueFlags.from_dict({"enabled": True, "alpha": 1.0, "not_a_knob": 3})
    assert flags.enabled is True
    assert flags.alpha == 1.0
    assert flags.mirror_frac == 0.5


def test_league_flags_from_none_is_disabled():
    assert LeagueFlags.from_dict(None).enabled is False


# ----------------------------------------------------------------------- pool

def test_pool_bounded_and_anchor_immune():
    rng = np.random.RandomState(0)
    pool = OpponentPool(pool_size=5, rng=rng)
    pool.add_anchor(make_member("anchor_a"))
    pool.add_anchor(make_member("anchor_b"))
    for i in range(200):
        pool.offer(make_member(f"snap_{i}"))
        assert len(pool) <= 5
        anchor_ids = [m.member_id for m in pool.members if m.kind == ANCHOR]
        assert sorted(anchor_ids) == ["anchor_a", "anchor_b"]
    assert pool.n_offered == 200


def test_pool_reservoir_is_roughly_uniform_over_stream():
    # Offer 60 members to 3 non-anchor slots many times; early and late members
    # should be retained at similar rates (i.e. not just the most recent ones).
    retained_early, retained_late = 0, 0
    n_trials = 400
    for trial in range(n_trials):
        pool = OpponentPool(pool_size=3, rng=np.random.RandomState(trial))
        for i in range(60):
            pool.offer(make_member(f"m_{i}"))
        ids = pool.member_ids
        retained_early += sum(1 for m in ids if int(m.split("_")[1]) < 30)
        retained_late += sum(1 for m in ids if int(m.split("_")[1]) >= 30)
    total = retained_early + retained_late
    assert total == n_trials * 3
    # Uniform expectation is 50/50; allow generous slack.
    assert 0.4 < retained_early / total < 0.6


def test_pool_roundtrip():
    pool = OpponentPool(pool_size=4, rng=np.random.RandomState(1))
    pool.add_anchor(make_member("a"))
    pool.offer(make_member("b"))
    restored = OpponentPool.from_dict(pool.to_dict())
    assert restored.member_ids == pool.member_ids
    assert restored.n_offered == pool.n_offered
    assert restored.pool_size == pool.pool_size


def test_pool_rejects_duplicates():
    pool = OpponentPool(pool_size=4)
    pool.add_anchor(make_member("a"))
    with pytest.raises(ValueError):
        pool.add_anchor(make_member("a"))
    with pytest.raises(ValueError):
        pool.offer(make_member("a"))


# -------------------------------------------------------------------- winrate

def test_winrate_ema_math():
    tracker = WinRateTracker(ema_lambda=0.1, init=0.5)
    tracker.add_member("x")
    assert tracker.winrate("x") == 0.5
    tracker.update("x", 1.0)
    assert math.isclose(tracker.winrate("x"), 0.9 * 0.5 + 0.1 * 1.0)
    tracker.update("x", 0.0)
    assert math.isclose(tracker.winrate("x"), 0.9 * 0.55)
    tracker.update("x", 0.5)  # draws count
    assert tracker.games("x") == 3


def test_winrate_unknown_member_outcome_is_dropped():
    tracker = WinRateTracker()
    tracker.update("ghost", 1.0)  # must not raise


def test_winrate_roundtrip():
    tracker = WinRateTracker(ema_lambda=0.2)
    tracker.add_member("x")
    tracker.update("x", 1.0)
    restored = WinRateTracker.from_dict(tracker.to_dict(), ema_lambda=0.2)
    assert restored.winrate("x") == tracker.winrate("x")
    assert restored.games("x") == 1


# -------------------------------------------------------------------- sampler

def test_pfsp_prioritizes_hard_opponents():
    sampler = PFSPSampler(priority="pfsp", alpha=0.5, eps_uniform=0.1)
    probs = sampler.probs([0.9, 0.5, 0.1])
    assert math.isclose(sum(probs), 1.0)
    assert probs[2] > probs[1] > probs[0]


def test_eps_floor_keeps_beaten_opponents_sampled():
    sampler = PFSPSampler(priority="pfsp", alpha=0.5, eps_uniform=0.1)
    probs = sampler.probs([1.0, 0.5])
    # Without the floor the fully-beaten opponent would get exactly 0.
    assert probs[0] >= 0.1 / 2


def test_variance_priority_favors_even_matchups():
    sampler = PFSPSampler(priority="variance", eps_uniform=0.)
    probs = sampler.probs([0.0, 0.5, 1.0])
    assert probs[1] > probs[0]
    assert probs[1] > probs[2]
    # Unbeatable and always-beaten opponents both give zero learning signal.
    assert math.isclose(probs[0], probs[2])


def test_uniform_priority_ignores_winrates():
    sampler = PFSPSampler(priority="uniform", eps_uniform=0.1)
    probs = sampler.probs([0.9, 0.1, 0.4])
    assert all(math.isclose(p, 1. / 3) for p in probs)


def test_all_weights_zero_falls_back_to_uniform():
    sampler = PFSPSampler(priority="pfsp", alpha=1.0, eps_uniform=0.)
    probs = sampler.probs([1.0, 1.0])
    assert probs == [0.5, 0.5]


def test_empty_pool_gives_empty_probs():
    assert PFSPSampler().probs([]) == []


# ------------------------------------------------------------- anchor floor

def _floor(members, probs, floor):
    """Call LeagueManager._apply_anchor_floor without constructing a manager."""
    from lux_ai.league.manager import LeagueManager

    mgr = LeagueManager.__new__(LeagueManager)
    mgr.flags = LeagueFlags(anchor_floor=floor)
    return LeagueManager._apply_anchor_floor(mgr, members, probs)


def test_anchor_floor_lifts_starved_anchors_and_still_sums_to_one():
    members = [make_member("anchor_a", ANCHOR), make_member("anchor_b", ANCHOR),
               make_member("s1"), make_member("s2")]
    probs = [0.02, 0.03, 0.60, 0.35]        # PFSP has starved both anchors
    out = _floor(members, probs, 0.15)
    assert math.isclose(sum(out), 1.0)
    assert out[0] >= 0.15 - 1e-9 and out[1] >= 0.15 - 1e-9
    # the shortfall comes out of the non-anchors, keeping their relative order
    assert out[2] > out[3]
    assert out[2] < probs[2] and out[3] < probs[3]


def test_anchor_floor_leaves_already_sampled_anchors_alone():
    members = [make_member("anchor_a", ANCHOR), make_member("s1")]
    probs = [0.40, 0.60]
    out = _floor(members, probs, 0.15)
    assert out == pytest.approx(probs)


def test_anchor_floor_disabled_is_a_no_op():
    members = [make_member("anchor_a", ANCHOR), make_member("s1")]
    probs = [0.01, 0.99]
    assert _floor(members, probs, 0.0) == pytest.approx(probs)


def test_anchor_floor_cannot_consume_the_whole_distribution():
    members = [make_member("a", ANCHOR), make_member("b", ANCHOR), make_member("s1")]
    out = _floor(members, [0.01, 0.01, 0.98], 0.9)   # absurd floor
    assert math.isclose(sum(out), 1.0)
    assert out[2] > 0.0                              # non-anchor still reachable


def test_anchor_floor_with_no_non_anchors_is_a_no_op():
    members = [make_member("a", ANCHOR), make_member("b", ANCHOR)]
    probs = [0.5, 0.5]
    assert _floor(members, probs, 0.15) == pytest.approx(probs)


# ------------------------------------------------------------------- state io

def test_state_roundtrip_and_versioning(tmp_path):
    path = tmp_path / "league" / "state.json"
    write_state_atomic(path, {"version": 1, "probs": [0.5, 0.5]})
    assert read_state(path)["version"] == 1
    assert read_state_if_newer(path, 0)["probs"] == [0.5, 0.5]
    assert read_state_if_newer(path, 1) is None
    write_state_atomic(path, {"version": 2})
    assert read_state_if_newer(path, 1)["version"] == 2


def test_publish_retries_when_a_reader_holds_the_file(tmp_path, monkeypatch):
    """On Windows os.replace fails while another process has the destination open,
    and the actors read this file constantly. A long run must not die because one
    publish lost that race."""
    import lux_ai.league.state_io as sio

    path = tmp_path / "state.json"
    write_state_atomic(path, {"version": 1})
    calls = {"n": 0}
    real_replace = os.replace

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(sio.os, "replace", flaky)
    assert sio.write_state_atomic(path, {"version": 2}, backoff=0.0) is True
    assert calls["n"] == 3
    assert read_state(path)["version"] == 2


def test_publish_gives_up_without_raising(tmp_path, monkeypatch):
    import lux_ai.league.state_io as sio

    path = tmp_path / "state.json"
    write_state_atomic(path, {"version": 1})

    def always_denied(src, dst):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(sio.os, "replace", always_denied)
    # Must return False rather than propagate: a lost publish is harmless, the
    # next one carries the same state, and crashing costs the whole run.
    assert sio.write_state_atomic(path, {"version": 2}, attempts=3, backoff=0.0) is False
    assert read_state(path)["version"] == 1


def test_read_missing_state_returns_none(tmp_path):
    assert read_state(tmp_path / "nope.json") is None
    assert read_state_if_newer(tmp_path / "nope.json", -1) is None


# ------------------------------------------------------------- member configs

REPO_ROOT = Path(__file__).resolve().parents[3]
FROZEN_AGENTS = sorted(
    (REPO_ROOT / "internal_testing").glob("*/*/lux_ai/rl_agent/config.yaml")
)


POOL_PRESET = REPO_ROOT / "conf" / "league_pfsp.yaml"


def _preset_member_dirs():
    from omegaconf import OmegaConf

    league = OmegaConf.load(POOL_PRESET)["league"]
    return [Path(d) for d in list(league["anchors"]) + list(league["initial_members"])]


@pytest.mark.skipif(not POOL_PRESET.is_file(), reason="preset missing")
@pytest.mark.parametrize("member_dir", _preset_member_dirs(), ids=lambda p: p.name)
def test_pool_preset_members_are_loadable(member_dir):
    """Every agent the presets put in the pool must build and load. An actor
    process dies silently on an unhandled exception, so a member that cannot
    load would stall training with no traceback - which is exactly what an
    old config missing `sum_player_embeddings` did."""
    from lux_ai.league.member_config import validate_member

    config_path = member_dir / "lux_ai" / "rl_agent" / "config.yaml"
    checkpoints = sorted(config_path.parent.glob("*.pt"))
    assert config_path.is_file(), f"missing {config_path}"
    assert len(checkpoints) == 1
    rejection = validate_member(config_path, checkpoints[0])
    assert rejection is None, f"{member_dir.name} rejected: {rejection}"


@pytest.mark.skipif(not FROZEN_AGENTS, reason="no frozen agents checked out")
@pytest.mark.parametrize("config_path", FROZEN_AGENTS, ids=lambda p: p.parents[2].name)
def test_validate_member_never_raises(config_path):
    """Validation must always answer None-or-reason, never propagate. Some
    frozen agents genuinely cannot be rebuilt (e.g. 09-07 sets an obs_space
    kwarg the current obs space dropped); those must be rejected, not crash."""
    from lux_ai.league.member_config import validate_member

    checkpoints = sorted(config_path.parent.glob("*.pt"))
    result = validate_member(config_path, checkpoints[0])
    assert result is None or isinstance(result, str)


def test_validate_member_rejects_mismatched_weights(tmp_path):
    from lux_ai.league.member_config import validate_member

    buildable = next(
        cfg for cfg in FROZEN_AGENTS
        if validate_member(cfg, sorted(cfg.parent.glob("*.pt"))[0]) is None
    )
    bad_checkpoint = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {"not_a_real_param": torch.zeros(3)}}, bad_checkpoint)
    reason = validate_member(buildable, bad_checkpoint)
    assert reason is not None and "weights do not match" in reason


# ------------------------------------------------------ learner-side masking

def test_masked_losses_ignore_opponent_seat():
    """With the opponent seat's mask at 0, corrupting that seat's advantages /
    values must not change the loss (zero-masking is exact under reduction=sum)."""
    from lux_ai.torchbeast.monobeast import compute_baseline_loss, compute_policy_gradient_loss

    torch.manual_seed(0)
    t, b = 4, 3
    mask = torch.ones((t, b, 2))
    mask[:, :, 1] = 0.

    log_probs = torch.randn((t, b, 2), requires_grad=True)
    advantages = torch.randn((t, b, 2))
    corrupted = advantages.clone()
    corrupted[:, :, 1] = 1e6

    loss_a = compute_policy_gradient_loss(log_probs, advantages * mask, reduction="sum")
    loss_b = compute_policy_gradient_loss(log_probs, corrupted * mask, reduction="sum")
    assert torch.isclose(loss_a, loss_b)

    grad_a, = torch.autograd.grad(loss_a, log_probs, retain_graph=True)
    grad_b, = torch.autograd.grad(loss_b, log_probs)
    assert torch.equal(grad_a, grad_b)
    assert torch.all(grad_a[:, :, 1] == 0.)

    values = torch.randn((t, b, 2), requires_grad=True)
    targets = torch.randn((t, b, 2))
    corrupted_targets = targets.clone()
    corrupted_targets[:, :, 1] = 1e6
    bl_a = compute_baseline_loss(values, targets, reduction="sum", mask=mask)
    bl_b = compute_baseline_loss(values, corrupted_targets, reduction="sum", mask=mask)
    assert torch.isclose(bl_a, bl_b)
    vgrad_a, = torch.autograd.grad(bl_a, values, retain_graph=True)
    assert torch.all(vgrad_a[:, :, 1] == 0.)

    # mask=None path is byte-identical to the pre-league behaviour
    bl_none = compute_baseline_loss(values, targets, reduction="sum")
    bl_ones = compute_baseline_loss(values, targets, reduction="sum", mask=torch.ones_like(mask))
    assert torch.equal(bl_none, bl_ones)


# ------------------------------------------------------- agent dir writing

def _bare_manager(run_dir):
    """A manager with just enough state to call write_agent_dir."""
    from lux_ai.league.manager import LeagueManager

    mgr = LeagueManager.__new__(LeagueManager)
    mgr.run_dir = Path(run_dir)
    mgr.league_dir = Path(run_dir) / "league"
    (Path(run_dir) / "config.yaml").write_text("obs_space: FixedShapeContinuousObsV2\n",
                                               encoding="utf-8")
    template = Path(run_dir) / "rl_agent_config.yaml"
    template.write_text("device: player_id\n", encoding="utf-8")
    mgr._rl_agent_config_template = template
    return mgr


def test_write_agent_dir_produces_the_agentspec_layout(tmp_path):
    import torch

    mgr = _bare_manager(tmp_path)
    target = tmp_path / "candidate"
    mgr.write_agent_dir({"w": torch.zeros(2)}, target, "candidate_weights.pt")

    rl = target / "lux_ai" / "rl_agent"
    assert (rl / "config.yaml").is_file()
    assert (rl / "rl_agent_config.yaml").is_file()
    # Exactly one checkpoint is what AgentSpec.from_directory requires.
    assert [p.name for p in sorted(rl.glob("*.pt"))] == ["candidate_weights.pt"]
    assert not list(rl.glob("*.tmp")), "the atomic-write temp file must be renamed away"


def test_write_agent_dir_reuse_overwrites_without_deleting(tmp_path):
    """
    Reusing the directory with a CONSTANT filename must keep the one-checkpoint
    invariant by overwriting - never by globbing and unlinking, which pointed at
    the wrong path would erase real model weights.
    """
    import torch

    mgr = _bare_manager(tmp_path)
    target = tmp_path / "candidate"
    rl = target / "lux_ai" / "rl_agent"

    mgr.write_agent_dir({"w": torch.zeros(2)}, target, "candidate_weights.pt")
    # A neighbouring file stands in for anything else living in the directory.
    sentinel = rl / "do_not_delete.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    mgr.write_agent_dir({"w": torch.ones(2)}, target, "candidate_weights.pt")

    assert [p.name for p in sorted(rl.glob("*.pt"))] == ["candidate_weights.pt"]
    assert sentinel.is_file(), "write_agent_dir must not remove anything"
    reloaded = torch.load(str(rl / "candidate_weights.pt"), map_location="cpu")
    assert torch.equal(reloaded["model_state_dict"]["w"], torch.ones(2))


def test_write_agent_dir_does_not_touch_the_pool(tmp_path):
    import torch

    mgr = _bare_manager(tmp_path)
    mgr.pool = OpponentPool(pool_size=3)
    mgr.pool.add_anchor(make_member("a", ANCHOR))
    before = list(mgr.pool.member_ids)

    mgr.write_agent_dir({"w": torch.zeros(2)}, tmp_path / "candidate", "candidate_weights.pt")

    assert list(mgr.pool.member_ids) == before


def test_admit_evaluated_snapshot_copies_exact_candidate_without_deleting(tmp_path):
    import torch

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    mgr = _bare_manager(run_dir)
    source = tmp_path / "evaluated"
    mgr.write_agent_dir({"w": torch.tensor([3.0])}, source, "best_weights.pt")
    sentinel = source / "lux_ai" / "rl_agent" / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    admitted = {}
    mgr._admit = lambda path, kind, force=False: admitted.update(
        path=Path(path), kind=kind, force=force
    ) or "member"

    result = mgr.admit_evaluated_snapshot(source, step=64000, round_idx=2)

    copied = (tmp_path / "run" / "league" / "snapshots" /
              "000000064000_eval_002" / "lux_ai" / "rl_agent" /
              "000000064000_eval_002_weights.pt")
    state = torch.load(str(copied), map_location="cpu")
    assert result == "member"
    assert torch.equal(state["model_state_dict"]["w"], torch.tensor([3.0]))
    assert admitted["force"] is True
    assert sentinel.read_text(encoding="utf-8") == "untouched"


# ------------------------------------------------- exploiter stop rule

def _window_manager(maxlen=10):
    """A manager with only the rolling anchor window wired up."""
    from collections import deque
    from lux_ai.league.manager import LeagueManager

    mgr = LeagueManager.__new__(LeagueManager)
    mgr._anchor_window = deque(maxlen=maxlen)
    return mgr


def test_anchor_rolling_winrate_is_none_until_the_window_is_full():
    """The None is the 'do not stop on a lucky streak' guard - a partial mean of
    three straight wins would read 1.0 and terminate the run immediately."""
    mgr = _window_manager(maxlen=10)
    for _ in range(9):
        mgr._anchor_window.append(("target", 1.0))
    assert mgr.anchor_rolling_winrate() is None
    mgr._anchor_window.append(("target", 1.0))
    assert mgr.anchor_rolling_winrate() == 1.0


def test_anchor_rolling_winrate_matches_the_window_mean():
    mgr = _window_manager(maxlen=4)
    for outcome in (1.0, 0.0, 1.0, 0.5):
        mgr._anchor_window.append(("target", outcome))
    assert mgr.anchor_rolling_winrate() == pytest.approx(0.625)


def test_anchor_rolling_winrate_follows_the_rolling_window():
    """Old games must age out, or a run that improved then regressed would still stop."""
    mgr = _window_manager(maxlen=4)
    for _ in range(4):
        mgr._anchor_window.append(("target", 1.0))
    assert mgr.anchor_rolling_winrate() == 1.0
    for _ in range(4):
        mgr._anchor_window.append(("target", 0.0))
    assert mgr.anchor_rolling_winrate() == 0.0


def test_exploiter_stop_threshold_semantics():
    """The condition used in monobeast: fire only when full AND at/above target."""
    target = 0.70
    mgr = _window_manager(maxlen=10)
    fires = lambda: (mgr.anchor_rolling_winrate() is not None
                     and mgr.anchor_rolling_winrate() >= target)

    for _ in range(5):                       # 5 straight wins, window not full
        mgr._anchor_window.append(("target", 1.0))
    assert not fires(), "must not stop on a short winning streak"

    for _ in range(5):                       # full, 10/10 -> 1.0
        mgr._anchor_window.append(("target", 1.0))
    assert fires()

    for _ in range(4):                       # 6/10 -> 0.6, below target
        mgr._anchor_window.append(("target", 0.0))
    assert mgr.anchor_rolling_winrate() == pytest.approx(0.6)
    assert not fires()


# ------------------------------------------------- exploiter pool shape

def test_pool_of_one_admits_nothing_beyond_its_single_anchor():
    """The exploiter config (pool_size 1 + one anchor) must never admit a snapshot of
    itself - otherwise it would start training against its own past, which is the one
    thing an exploiter must not do."""
    pool = OpponentPool(pool_size=1, rng=np.random.RandomState(0))
    pool.add_anchor(make_member("target", ANCHOR))
    for i in range(50):
        accepted, evicted = pool.offer(make_member("snap%d" % i))
        assert not accepted and evicted is None
    assert pool.member_ids == ["target"]


# --------------------------------------- win-rate-gated admission (pool side)

def test_replace_always_admits_and_never_evicts_an_anchor():
    pool = OpponentPool(pool_size=4, rng=np.random.RandomState(0))
    pool.add_anchor(make_member("anchor", ANCHOR))
    for i in range(3):
        pool.offer(make_member("s%d" % i))
    assert len(pool) == 4

    evicted = pool.replace(make_member("earned"), evict_id="s1")
    assert evicted.member_id == "s1"
    assert "earned" in pool
    assert "anchor" in pool, "anchors must never be displaced"
    assert len(pool) == 4


def test_replace_fills_a_free_slot_without_evicting():
    pool = OpponentPool(pool_size=4, rng=np.random.RandomState(0))
    pool.add_anchor(make_member("anchor", ANCHOR))
    assert pool.replace(make_member("earned")) is None
    assert "earned" in pool and len(pool) == 2


def test_replace_rejects_an_unknown_evict_target():
    pool = OpponentPool(pool_size=2, rng=np.random.RandomState(0))
    pool.add_anchor(make_member("anchor", ANCHOR))
    pool.offer(make_member("s0"))
    with pytest.raises(KeyError):
        pool.replace(make_member("earned"), evict_id="does_not_exist")


def test_replace_is_a_no_op_when_the_pool_is_all_anchors():
    """The exploiter config (pool_size 1, one anchor) has no evictable slot."""
    pool = OpponentPool(pool_size=1, rng=np.random.RandomState(0))
    pool.add_anchor(make_member("target", ANCHOR))
    assert pool.replace(make_member("earned")) is None
    assert pool.member_ids == ["target"]


# ------------------------------------ win-rate-gated admission (manager side)

def _gate_manager(threshold=0.6, window=10, cooldown=500):
    from collections import deque
    from lux_ai.league.manager import LeagueManager

    mgr = LeagueManager.__new__(LeagueManager)
    mgr.flags = LeagueFlags(winrate_admit_threshold=threshold,
                            winrate_admit_window=window,
                            winrate_admit_cooldown_updates=cooldown)
    mgr._reference_window = deque(maxlen=window)
    mgr._reference_id = "origin"
    mgr.last_winrate_admit_updates = -10 ** 9
    return mgr


def test_gate_stays_shut_until_the_window_is_full():
    mgr = _gate_manager(window=10)
    for _ in range(9):
        mgr._reference_window.append(1.0)
    assert mgr.reference_winrate() is None
    assert not mgr.winrate_admission_due(updates=10_000), "9 straight wins must not admit"
    mgr._reference_window.append(1.0)
    assert mgr.reference_winrate() == 1.0
    assert mgr.winrate_admission_due(updates=10_000)


def test_gate_respects_the_threshold():
    mgr = _gate_manager(threshold=0.6, window=10)
    # Losses first: the deque evicts oldest-first, so later wins displace them.
    for outcome in [0.0] * 5 + [1.0] * 5:          # 0.50, below
        mgr._reference_window.append(outcome)
    assert mgr.reference_winrate() == pytest.approx(0.5)
    assert not mgr.winrate_admission_due(updates=10_000)
    for _ in range(2):                              # two 0.0s roll out -> 0.70
        mgr._reference_window.append(1.0)
    assert mgr.reference_winrate() == pytest.approx(0.7)
    assert mgr.winrate_admission_due(updates=10_000)


def test_gate_cooldown_prevents_repeat_admission():
    """Without this a sustained streak would admit a snapshot on every main-loop tick."""
    mgr = _gate_manager(threshold=0.6, window=10, cooldown=500)
    for _ in range(10):
        mgr._reference_window.append(1.0)
    assert mgr.winrate_admission_due(updates=10_000)
    mgr.last_winrate_admit_updates = 10_000
    assert not mgr.winrate_admission_due(updates=10_100)
    assert not mgr.winrate_admission_due(updates=10_499)
    assert mgr.winrate_admission_due(updates=10_500)


def test_gate_disabled_by_default():
    mgr = _gate_manager(threshold=0.0, window=10)
    for _ in range(10):
        mgr._reference_window.append(1.0)
    assert not mgr.winrate_admission_due(updates=10 ** 9)
