# Plan: Implement yolo_vit_lstm and yolo_bilstm Hybrid Architectures

## Context

The project has 4 registered segmentation variants (baseline, attention, cnn_unet, effnet_transformer) all sharing the same interface: input `(B, 11, 256, 256)`, output `(B, 11, 256, 256)` logits. The user requests two new hybrid models combining YOLO backbone features with transformer/LSTM components.

Key constraints discovered from codebase:
- All models must output `(B, 11, 256, 256)` to be trainable via `scripts/train_one.py` with CE+Dice loss
- `ultralytics==8.4.149` and `timm==1.0.29` are already in `requirements.txt`
- The registry pattern is `VARIANTS[str] = (builder_fn, label_string)` in `src/models/registry.py`
- No YOLO detector model exists yet; `src/detection/labels.py` is label extraction only
- Existing `cnn_lstm.py` uses bands-as-frames (11 "frames") for BiLSTM — new models should use spatial feature sequences instead

## Files to Create

### 1. `src/models/yolo_vit_lstm.py` — YOLO + ViT + LSTM

Architecture:
- **Backbone**: YOLOv8n via `ultralytics` (pretrained on COCO), frozen or fine-tuned. Extract multi-scale feature maps from early backbone stages (before the detection head). The YOLOv8 backbone produces features at strides 4, 8, 16, 32.
- **11-channel adapter**: Replace YOLOv8's first conv (3→ch) to accept 11 Sentinel-2 bands, replicating weights like existing `_adapt_first_convolution` pattern.
- **ViT spatial encoder**: Take the stride-8 feature map, project to transformer dim, add positional embedding, process through `nn.TransformerEncoder` (2 layers, 4 heads, dim=256 — same config as `unet_effnet_transformer.py`). This gives spatial self-attention over the feature grid.
- **LSTM sequence layer**: Reshape ViT output tokens (spatial grid → sequence of H×W tokens), pass through `nn.LSTM` (hidden=128, bidirectional=False) to capture sequential spatial context along raster-scan order.
- **Decoder**: Bilinear upsample + skip connections from YOLO backbone features to reconstruct full 256×256 resolution.
- **Head**: 1×1 conv → 11 classes.

Input: `(B, 11, 256, 256)` → Output: `(B, 11, 256, 256)`

### 2. `src/models/yolo_bilstm.py` — YOLO + BiLSTM

Architecture:
- **Backbone**: Same YOLOv8n backbone with 11-channel adapter as above.
- **Multi-scale feature collection**: Extract features from backbone stages at strides 4, 8, 16, 32.
- **Channel-wise BiLSTM neck**: For each spatial scale, flatten the channel dimension into a sequence (C channels → sequence length C, feature dim = H×W), process through `nn.LSTM(bidirectional=True, hidden=128)` to capture bidirectional channel/spectral dependencies. This follows the spirit of the user's request: "multi-scale spatial feature sequences through a BiLSTM layer to capture bidirectional spatial/channel context."
- **Feature aggregation**: Concatenate BiLSTM outputs from all scales, project to decoder channel dim.
- **Decoder**: Progressive bilinear upsampling with skip connections from backbone features.
- **Head**: 1×1 conv → 11 classes.

Input: `(B, 11, 256, 256)` → Output: `(B, 11, 256, 256)`

### 3. Modify `src/models/registry.py`

Add two new entries to `VARIANTS`:
```python
"yolo_vit_lstm": (lambda pretrained=True: build_yolo_vit_lstm(pretrained=pretrained), "YOLO + ViT + LSTM"),
"yolo_bilstm": (lambda pretrained=True: build_yolo_bilstm(pretrained=pretrained), "YOLO + BiLSTM"),
```

### 4. Modify `tests/test_models.py`

- Add imports for the two new builder functions
- The existing parametrized `test_all_registered_variants_forward` test will automatically pick up the new variants from the registry
- Add targeted tests: `test_yolo_vit_lstm_has_transformer`, `test_yolo_bilstm_has_lstm`

## Files to Modify

| File | Change |
|---|---|
| `src/models/yolo_vit_lstm.py` | **Create** — new module |
| `src/models/yolo_bilstm.py` | **Create** — new module |
| `src/models/registry.py` | Add 2 imports + 2 registry entries |
| `tests/test_models.py` | Add imports + 2 targeted tests |

## Implementation Details

### YOLOv8 Backbone Extraction

Use `ultralytics.nn.tasks.DetectionModel` or load via `YOLO("yolov8n.pt")` and access internal layers. The project already has `yolov8n.pt` in the root. Extract features by hooking or slicing the backbone before the Detect head.

Approach: subclass `ultralytics.nn.modules.Conv` / access `model.model[:n]` to get the backbone portion. YOLOv8n backbone has stages at indices roughly:
- Stage 0 (stride 2): Conv 3→16
- Stage 1 (stride 4): Conv 16→32  
- Stage 2 (stride 8): Conv 32→64 (C2f)
- Stage 3 (stride 16): Conv 64→128 (C2f)
- Stage 4 (stride 32): Conv 128→256 (C2f)

For a 256×256 input, the stride-8 feature map is 32×32 with 64 channels — suitable for ViT processing.

### ViT Configuration (matching existing effnet_transformer pattern)

From `unet_effnet_transformer.py:6-10`:
```python
_TRANSFORM_DIM = 256
# 2-layer, 4-head TransformerEncoder, dim_feedforward=1024, dropout=0.1
```

### LSTM Configuration

From `cnn_lstm.py` for reference, but adapted:
```python
# BiLSTM: hidden=128, bidirectional=True → output dim 256
# For yolo_vit_lstm: unidirectional LSTM, hidden=128
# For yolo_bilstm: bidirectional LSTM, hidden=128
```

### ConvTranspose2d Decoder Pattern

Both models use a lightweight decoder (not smp's decoder) to upsample from stride-8 (32×32) back to 256×256:
```python
# 32×32 → 64×64 (ConvTranspose2d, stride=2)
# 64×64 → 128×128 (ConvTranspose2d, stride=2)
# 128×128 → 256×256 (ConvTranspose2d, stride=2)
```

## Verification

1. `pytest tests/test_models.py -v` — all existing + new tests pass
2. `pytest tests/ -q` — full suite still passes (73 existing + new)
3. Smoke-test forward pass: `resolve_variant("yolo_vit_lstm", pretrained=False)(torch.randn(1, 11, 256, 256)).shape == (1, 11, 256, 256)`
4. Smoke-test forward pass: `resolve_variant("yolo_bilstm", pretrained=False)(torch.randn(1, 11, 256, 256)).shape == (1, 11, 256, 256)`

## Estimated Effort

- `yolo_vit_lstm.py`: ~120 lines (YOLO backbone adapter + ViT block + LSTM + decoder)
- `yolo_bilstm.py`: ~110 lines (YOLO backbone adapter + BiLSTM neck + decoder)
- Registry + tests: ~15 lines changed
- Total: ~245 new lines, ~15 modified lines
