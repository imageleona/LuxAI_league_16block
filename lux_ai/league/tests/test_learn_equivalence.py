"""
Regression guard for the league's central safety property: with
`league.enabled: false`, the learner must behave exactly as it did before the
league existed.

Whole-run comparison cannot show this - two identical runs of this trainer
already diverge (cuDNN/AMP nondeterminism plus actor/learner interleaving), by
more than the league branch does. So this test pins the thing that must not
move: it pulls the pre-league `learn()` out of git and runs it and the current
`learn()` on the same batch with the same weights, in one process, and
requires the losses to match bit-for-bit.

Needs CUDA and a git checkout; skipped otherwise.
"""
import copy
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
PRESET = REPO / "conf" / "league_sp.yaml"


def _git_baseline_source():
    """The last committed monobeast.py, i.e. the learner before the league."""
    return subprocess.check_output(
        ["git", "show", "HEAD:lux_ai/torchbeast/monobeast.py"],
        cwd=str(REPO), stderr=subprocess.DEVNULL,
    )


def _load_baseline_learn():
    """Execute the committed learner in memory without creating/deleting a file."""
    module_name = "lux_ai.torchbeast._baseline_monobeast_memory"
    module = types.ModuleType(module_name)
    module.__file__ = "<git:HEAD:lux_ai/torchbeast/monobeast.py>"
    module.__package__ = "lux_ai.torchbeast"
    previous_wandb = sys.modules.get("wandb")
    # The comparison disables W&B. A stub avoids importing its old SSL stack
    # after Torch has initialized CUDA DLLs on Windows.
    sys.modules["wandb"] = types.ModuleType("wandb")
    try:
        exec(compile(_git_baseline_source(), module.__file__, "exec"), module.__dict__)
    finally:
        if previous_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous_wandb
    return module.learn


def _build_flags(with_league_key: bool):
    from lux_ai.utils import flags_to_namespace

    conf = OmegaConf.load(PRESET)
    for key in ("defaults", "hydra"):
        conf.pop(key, None)
    conf = OmegaConf.to_container(conf)
    # Small and fully deterministic: no mixed precision, tiny network.
    conf.update(dict(
        n_actor_envs=4, unroll_length=4, batch_size=2, num_actors=1, num_buffers=2,
        n_blocks=2, hidden_dim=32, embedding_dim=16, total_steps=1000,
        seed=42, debug=True, disable_wandb=True, use_mixed_precision=False,
        use_teacher=True, teacher_kl_cost=0.005, teacher_baseline_cost=0.5,
        actor_device="cuda:0", learner_device="cuda:0", weights_only=True,
        load_dir=None, checkpoint_file=None, min_lr_mod=1.0, clip_grads=10.0,
        n_value_warmup_batches=0, checkpoint_freq=1000., num_learner_threads=1,
        model_log_freq=100, sharing_strategy="file_system",
    ))
    if with_league_key:
        conf["league"]["enabled"] = False
    else:
        conf.pop("league", None)
    return flags_to_namespace(conf)


def _make_batch(flags):
    from lux_ai.lux_gym import create_env
    from lux_ai.torchbeast.core.buffer_utils import buffers_apply, create_buffers

    env = create_env(flags, torch.device("cpu"))
    example_info = env.reset(force=True)["info"]
    buffers = create_buffers(flags, env.unwrapped[0].obs_space, example_info)
    env.close()

    torch.manual_seed(0)

    def fill(x):
        if x.dtype == torch.bool:
            return torch.zeros_like(x).bernoulli_(0.3)
        if x.dtype == torch.int64:
            return torch.zeros_like(x)
        return torch.randn_like(x).clamp(-1, 1)

    batch = buffers_apply(buffers[0], fill)
    # Actions are sampled without replacement, so overlapping slots must hold
    # distinct indices; repeats drive the conditional-probability denominator
    # negative and every policy-gradient loss becomes NaN.
    for key, actions in batch["actions"].items():
        idx = torch.arange(actions.shape[-1], dtype=actions.dtype)
        batch["actions"][key] = idx.expand_as(actions).contiguous()
    # Leave every action legal: a fully-masked tile sends logits to -inf and the
    # teacher KL evaluates 0 * -inf = NaN (in the baseline too), which would
    # make the comparison vacuous for that term.
    for key, mask in batch["info"]["available_actions_mask"].items():
        batch["info"]["available_actions_mask"][key] = torch.ones_like(mask)
    batch["done"] = torch.zeros_like(batch["done"])
    batch["done"][-1] = True
    batch["reward"] = batch["reward"].sign()
    return buffers_apply(
        batch, lambda x: x[:, :flags.batch_size].contiguous().to(flags.learner_device)
    )


def _run_learn(learn_fn, flags, batch, base_state):
    from lux_ai.nns import create_model

    torch.manual_seed(1234)
    learner_model = create_model(flags, flags.learner_device)
    learner_model.load_state_dict(base_state)
    learner_model.train()
    actor_model = create_model(flags, flags.actor_device)
    actor_model.load_state_dict(base_state)
    teacher_model = create_model(flags, flags.learner_device)
    teacher_model.load_state_dict(base_state)
    teacher_model.eval()

    optimizer = flags.optimizer_class(learner_model.parameters(), **flags.optimizer_kwargs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

    torch.manual_seed(1234)
    stats, _ = learn_fn(
        flags=flags, actor_model=actor_model, learner_model=learner_model,
        teacher_model=teacher_model, batch=copy.deepcopy(batch), optimizer=optimizer,
        grad_scaler=grad_scaler, lr_scheduler=scheduler, total_games_played=0,
    )
    return stats["Loss"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not (REPO / ".git").exists(), reason="needs a git checkout")
def test_disabled_league_matches_pre_league_learner():
    from lux_ai.nns import create_model

    baseline_learn = _load_baseline_learn()
    from lux_ai.torchbeast.monobeast import learn as league_learn

    flags_baseline = _build_flags(with_league_key=False)
    flags_disabled = _build_flags(with_league_key=True)

    torch.manual_seed(1234)
    base_state = copy.deepcopy(
        create_model(flags_baseline, flags_baseline.learner_device).state_dict()
    )
    batch = _make_batch(flags_baseline)

    baseline_losses = _run_learn(baseline_learn, flags_baseline, batch, base_state)
    league_losses = _run_learn(league_learn, flags_disabled, batch, base_state)

    assert set(baseline_losses) == set(league_losses)
    for key, expected in baseline_losses.items():
        actual = league_losses[key]
        assert actual == expected, f"{key}: {actual} != {expected}"
        assert actual == actual, f"{key} is NaN; the comparison would be vacuous"
