from types import SimpleNamespace

import pytest
import torch

from lux_ai.torchbeast.monobeast import _restore_optimizer, _restore_scheduler


def _adam_with_state(lr=1e-6):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=lr, eps=3e-4)
    parameter.grad = torch.tensor([0.5])
    optimizer.step()
    optimizer.zero_grad()
    return parameter, optimizer


def test_optimizer_only_restore_keeps_moments_but_resets_learning_rate():
    _, source = _adam_with_state(lr=1e-6)
    state = source.state_dict()
    state["param_groups"][0]["lr"] = 1e-8

    _, target = _adam_with_state(lr=1e-6)
    target.state.clear()
    flags = SimpleNamespace(
        load_optimizer_state=True,
        load_scheduler_state=False,
        optimizer_kwargs={"lr": 1e-6},
    )

    _restore_optimizer(target, {"optimizer_state_dict": state}, flags)

    assert len(target.state) == 1
    assert target.param_groups[0]["lr"] == pytest.approx(1e-6)
    assert target.param_groups[0]["initial_lr"] == pytest.approx(1e-6)


def test_requested_optimizer_state_must_exist():
    _, optimizer = _adam_with_state()
    flags = SimpleNamespace(
        load_optimizer_state=True,
        load_scheduler_state=False,
        optimizer_kwargs={"lr": 1e-6},
    )

    with pytest.raises(ValueError, match="full .pt checkpoint"):
        _restore_optimizer(optimizer, {"model_state_dict": {}}, flags)


def test_scheduler_state_can_be_restored_independently():
    _, optimizer = _adam_with_state()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    optimizer.step()
    scheduler.step()
    checkpoint = {"scheduler_state_dict": scheduler.state_dict()}

    _, restored_optimizer = _adam_with_state()
    restored = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda epoch: 1.0)
    flags = SimpleNamespace(load_scheduler_state=True)
    _restore_scheduler(restored, checkpoint, flags)

    assert restored.last_epoch == scheduler.last_epoch
