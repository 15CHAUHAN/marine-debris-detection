"""Tests for the segmentation model zoo (unet, unet_attention, unet_cnn, effnet_transformer, yolo_vit_lstm, yolo_bilstm)."""
import pytest
import torch

from src.models.unet import build_unet
from src.models.unet_attention import build_attention_unet
from src.models.unet_cnn import build_cnn_unet
from src.models.unet_effnet_transformer import build_effnet_transformer_unet
from src.models.yolo_vit_lstm import build_yolo_vit_lstm
from src.models.yolo_bilstm import build_yolo_bilstm
from src.models.registry import VARIANTS, resolve_variant


def _count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_baseline_unet_output_shape():
    model = build_unet(pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 11, 256, 256))
    assert tuple(logits.shape) == (2, 11, 256, 256)


def test_attention_unet_output_shape():
    model = build_attention_unet(pretrained=True)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 11, 256, 256))
    assert tuple(logits.shape) == (2, 11, 256, 256)


def test_first_convolution_accepts_11_channels():
    model = build_unet(pretrained=True)
    assert model.encoder.conv1.in_channels == 11


def test_attention_variant_has_extra_parameters():
    baseline = build_unet(pretrained=True)
    attention = build_attention_unet(pretrained=True)
    assert _count_parameters(attention) > _count_parameters(baseline)


def test_attention_variant_uses_scse_blocks():
    model = build_attention_unet(pretrained=True)
    n_scse = len([m for m in model.modules() if m.__class__.__name__ == "SCSEModule"])
    assert n_scse >= 8, f"expected scSE modules in decoder, found {n_scse}"


def test_unet_without_pretrained_weights_also_works():
    model = build_unet(pretrained=False)
    logits = model(torch.randn(1, 11, 256, 256))
    assert tuple(logits.shape) == (1, 11, 256, 256)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_all_registered_variants_forward(variant):
    model = resolve_variant(variant, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 11, 256, 256))
    assert tuple(logits.shape) == (1, 11, 256, 256)


def test_cnn_unet_is_compact():
    model = build_cnn_unet()
    assert _count_parameters(model) < 2_000_000


def test_effnet_transformer_has_transformer_stack():
    model = build_effnet_transformer_unet(pretrained=False)
    n_transformer = len([m for m in model.modules() if m.__class__.__name__ == "TransformerEncoder"])
    assert n_transformer == 1


def test_effnet_transformer_encoder_pretrained_channels():
    model = build_effnet_transformer_unet(pretrained=False)
    assert model.unet.encoder._conv_stem.in_channels == 11


def test_yolo_vit_lstm_has_transformer_and_gru():
    model = build_yolo_vit_lstm(pretrained=False)
    has_transformer = any(
        m.__class__.__name__ == "TransformerEncoder" for m in model.modules()
    )
    has_gru = any(
        m.__class__.__name__ == "GRU" for m in model.modules()
    )
    assert has_transformer, "expected a TransformerEncoder in yolo_vit_lstm"
    assert has_gru, "expected a GRU in yolo_vit_lstm (replaced LSTM for parameter efficiency)"


def test_yolo_vit_lstm_output_shape():
    model = build_yolo_vit_lstm(pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 11, 256, 256))
    assert tuple(logits.shape) == (1, 11, 256, 256)


def test_yolo_bilstm_has_bidirectional_lstm():
    model = build_yolo_bilstm(pretrained=False)
    lstms = [m for m in model.modules() if m.__class__.__name__ == "LSTM"]
    assert len(lstms) >= 1, "expected at least one LSTM in yolo_bilstm"
    for lstm in lstms:
        assert lstm.bidirectional, "expected bidirectional=True"


def test_yolo_bilstm_output_shape():
    model = build_yolo_bilstm(pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 11, 256, 256))
    assert tuple(logits.shape) == (1, 11, 256, 256)


@pytest.mark.parametrize("builder", [build_yolo_vit_lstm, build_yolo_bilstm])
def test_yolo_variant_forward_pass_is_finite_with_debris_argmax(builder):
    model = builder(pretrained=False)
    model.eval()
    model_input = torch.randn(1, 11, 256, 256)
    with torch.no_grad():
        logits = model(model_input)
    assert tuple(logits.shape) == (1, 11, 256, 256)
    assert torch.isfinite(logits).all(), "forward pass produced non-finite logits"
    mask = torch.argmax(logits, dim=1)
    assert tuple(mask.shape) == (1, 256, 256)
    assert set(mask.unique().tolist()) <= set(range(11))