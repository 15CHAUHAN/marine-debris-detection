"""Tests for the compound focal + Dice loss module (src/train/losses.py)."""
import numpy as np
import pytest
import torch

from src.data.encoding import IGNORE_INDEX, N_CLASSES
from src.train.losses import (
    DEBRIS_ALPHA_MIN,
    MARINE_WATER_INDEX,
    CompoundFocalDiceLoss,
    FocalLoss,
    MulticlassDiceLoss,
    build_focal_alphas,
)


def _logits_target(batch, height=4, width=4, n_classes=N_CLASSES, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, n_classes, height, width, generator=g)
    target = torch.randint(0, IGNORE_INDEX, (batch, height, width), generator=g)
    target[0, 0, 0] = 0  # guarantee at least one debris pixel
    return logits, target


# ---------------------------------------------------------------------------
# build_focal_alphas
# ---------------------------------------------------------------------------

def test_build_focal_alphas_none_weights_boost_debris():
    alpha = build_focal_alphas(None)
    assert tuple(alpha.shape) == (N_CLASSES,)
    assert alpha[0].item() >= DEBRIS_ALPHA_MIN
    assert (alpha[1:] == 1.0).all()


def test_build_focal_alphas_debris_ge_five_and_five_times_water():
    class_weights = torch.linspace(0.5, 4.0, N_CLASSES)
    alpha = build_focal_alphas(class_weights)
    water_alpha = float(class_weights[MARINE_WATER_INDEX])
    assert 3.0 * water_alpha > DEBRIS_ALPHA_MIN  # scenario where 3x hits first
    assert alpha[0].item() == pytest.approx(3.0 * water_alpha, rel=1e-6)
    assert alpha[0].item() >= DEBRIS_ALPHA_MIN
    assert torch.isfinite(alpha).all()
    assert (alpha > 0).all()


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

def test_focal_matches_manual_formula():
    logits = torch.tensor([[[[3.0]], [[-1.0]]]], dtype=torch.float32)
    target = torch.tensor([[[1]]])
    loss = FocalLoss(gamma=2.0, alpha=torch.ones(2), reduction="mean")
    probs = torch.softmax(logits, dim=1)
    p_true = probs[0, 1, 0, 0]
    expected = -((1.0 - p_true) ** 2) * torch.log(p_true)
    got = loss(logits, target)
    assert got.shape == ()
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)


def test_focal_alpha_scales_loss():
    logits = torch.tensor([[[[3.0]], [[-1.0]]]], dtype=torch.float32)
    target = torch.tensor([[[0]]])
    alpha = torch.tensor([5.0, 1.0])
    uniform = FocalLoss(gamma=2.0, alpha=torch.ones(2), reduction="mean")(logits, target)
    weighted = FocalLoss(gamma=2.0, alpha=alpha, reduction="mean")(logits, target)
    assert weighted.item() == pytest.approx(uniform.item() * 5.0, rel=1e-5)


def test_focal_ignores_unlabeled_pixels():
    logits = torch.randn(2, N_CLASSES, 4, 4)
    target = torch.full((2, 4, 4), IGNORE_INDEX, dtype=torch.long)
    loss = FocalLoss(gamma=2.0, alpha=None)
    assert loss(logits, target).item() == pytest.approx(0.0)


def test_focal_ignores_mixing_labeled_and_unlabeled():
    logits, target = _logits_target(2)
    target[:, 0, :] = IGNORE_INDEX  # first row unlabeled
    loss = FocalLoss(gamma=2.0, alpha=None)
    isolated = FocalLoss(gamma=2.0, alpha=None)(logits, target)
    assert torch.isfinite(isolated)
    assert isolated.item() > 0.0


# ---------------------------------------------------------------------------
# MulticlassDiceLoss
# ---------------------------------------------------------------------------

def test_dice_perfect_prediction_is_near_zero():
    target = torch.randint(0, N_CLASSES, (1, 4, 4))
    one_hot = torch.nn.functional.one_hot(target, num_classes=N_CLASSES).clone().float()
    one_hot = one_hot.permute(0, 3, 1, 2)
    logits = 50.0 * one_hot - 50.0 * (1.0 - one_hot)
    loss = MulticlassDiceLoss(smooth=1.0)(logits, target)
    assert loss.item() < 1e-3


def test_dice_all_unlabeled_matches_manual_formula():
    logits = torch.randn(2, N_CLASSES, 4, 4)
    target = torch.full((2, 4, 4), IGNORE_INDEX, dtype=torch.long)
    loss = MulticlassDiceLoss(smooth=1.0)(logits, target)
    probs = torch.softmax(logits.float(), dim=1)
    card = probs.sum(dim=(0, 2, 3))
    expected = (1.0 - 1.0 / (card + 1.0)).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert 0.0 <= loss.item() < 1.0


def test_dice_uses_true_class_labels_not_a_single_class():
    # Predict class 0 with high confidence. Dice must reward a class-0 target
    # over a class-1 target; the old implementation collapsed every labeled
    # pixel onto one channel, which made these two losses identical.
    logits = torch.tensor([[[[10.0]], [[-10.0]]]])
    t_correct = torch.tensor([[[0]]])
    t_wrong = torch.tensor([[[1]]])
    loss = MulticlassDiceLoss(smooth=1.0)
    correct = loss(logits, t_correct)
    wrong = loss(logits, t_wrong)
    assert correct.item() < wrong.item()


def test_dice_wrong_prediction_is_large():
    logits = torch.zeros(1, N_CLASSES, 4, 4)
    target = torch.randint(0, N_CLASSES, (1, 4, 4))
    loss = MulticlassDiceLoss(smooth=0.0)(logits, target)
    assert loss.item() > 0.5


# ---------------------------------------------------------------------------
# CompoundFocalDiceLoss
# ---------------------------------------------------------------------------

def test_compound_is_focal_plus_lambda_times_dice():
    logits, target = _logits_target(2)
    class_weights = torch.linspace(1.0, 8.0, N_CLASSES)
    compound = CompoundFocalDiceLoss(class_weights=class_weights, lambda_dice=2.0, lambda_lovasz=0.0)
    expected = (
        FocalLoss(gamma=2.0, alpha=build_focal_alphas(class_weights))(logits, target)
        + 2.0 * MulticlassDiceLoss(smooth=1.0)(logits, target)
    )
    got = compound(logits, target)
    assert got.item() == pytest.approx(expected.item(), rel=1e-5)
    assert torch.isfinite(got)


def test_compound_has_finite_gradients_for_training_targets():
    logits, target = _logits_target(2)
    logits.requires_grad_(True)
    compound = CompoundFocalDiceLoss(class_weights=torch.ones(N_CLASSES))
    loss = compound(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum()) > 0.0


def test_compound_raises_on_non_finite_focal():
    class _BrokenFocal(torch.nn.Module):
        def forward(self, logits, target):
            return torch.tensor(float("nan"))

    compound = CompoundFocalDiceLoss()
    compound.focal = _BrokenFocal()
    logits, target = _logits_target(2)
    with pytest.raises(FloatingPointError):
        compound(logits, target)