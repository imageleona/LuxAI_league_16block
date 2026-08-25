import torch

from lux_ai.torchbeast.monobeast import compute_teacher_kl_loss


def test_teacher_kl_ignores_all_infinite_logits_at_masked_locations():
    shape = (2, 3, 1, 2, 2, 2, 4)
    learner_logits = torch.full(shape, -float("inf"), requires_grad=True)
    teacher_logits = torch.full(shape, -float("inf"))
    actions_taken = torch.zeros(shape[:-1], dtype=torch.bool)

    loss = compute_teacher_kl_loss(learner_logits, teacher_logits, actions_taken)

    assert loss.shape == (2, 3, 2)
    assert torch.isfinite(loss).all()
    assert torch.equal(loss, torch.zeros_like(loss))


def test_teacher_kl_is_finite_with_mixed_valid_and_masked_locations():
    torch.manual_seed(0)
    shape = (2, 3, 1, 2, 2, 2, 4)
    actions_taken = torch.zeros(shape[:-1], dtype=torch.bool)
    actions_taken[..., 0, 0] = True

    learner_logits = torch.randn(shape)
    teacher_logits = torch.randn(shape)
    learner_logits = learner_logits.masked_fill(~actions_taken.unsqueeze(-1), -float("inf"))
    teacher_logits = teacher_logits.masked_fill(~actions_taken.unsqueeze(-1), -float("inf"))
    learner_logits.requires_grad_()

    loss = compute_teacher_kl_loss(learner_logits, teacher_logits, actions_taken)
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert learner_logits.grad is not None
    assert torch.isfinite(learner_logits.grad).all()


def test_teacher_kl_ignores_illegal_actions_inside_a_valid_location():
    shape = (2, 3, 1, 2, 2, 2, 4)
    actions_taken = torch.zeros(shape[:-1], dtype=torch.bool)
    actions_taken[..., 0, 0] = True

    learner_logits = torch.full(shape, -float("inf"))
    teacher_logits = torch.full(shape, -float("inf"))
    # Two actions are legal at each action-taking location. The remaining
    # dimensions reproduce the 0-probability actions that caused KL to become
    # NaN in a real training batch.
    learner_logits[..., 0, 0, :2] = torch.tensor([1.0, -0.5])
    teacher_logits[..., 0, 0, :2] = torch.tensor([-0.25, 0.75])
    learner_logits.requires_grad_()

    loss = compute_teacher_kl_loss(learner_logits, teacher_logits, actions_taken)
    loss.sum().backward()

    assert torch.isfinite(loss).all()
    assert (loss > 0).all()
    assert learner_logits.grad is not None
    assert torch.isfinite(learner_logits.grad).all()
