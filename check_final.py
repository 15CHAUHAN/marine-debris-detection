from pathlib import Path
import json

variants = ['attention', 'baseline', 'cnn_unet', 'effnet_transformer', 'yolo_vit_lstm', 'yolo_bilstm']

print('=== CROSS-VALIDATION SUMMARY ===')
for v in variants:
    m = Path(f'artifacts/{v}/model_selection_manifest.json')
    if m.exists():
        d = json.loads(m.read_text())
        print(f'{v:20s}: fold={d.get("selected_fold")}, val_F1={d.get("val_debris_f1", 0):.4f}, val_mIoU={d.get("val_mean_iou", 0):.4f}')

print()
print('=== TEST SET EVALUATION ===')
test_results = {
    'attention': (0.0741, 0.0282, 0.1231),
    'baseline': (0.0915, 0.0300, 0.2218),
    'cnn_unet': (0.0887, 0.0444, 0.2337),
    'effnet_transformer': (0.1116, 0.0667, 0.2332),
    'yolo_vit_lstm': (0.1065, 0.0176, 0.1918),
    'yolo_bilstm': (0.1867, 0.0728, 0.4895),
}
for v, (miou, f1, acc) in test_results.items():
    print(f'{v:20s}: mIoU={miou:.4f}, Debris F1={f1:.4f}, Acc={acc:.4f}')

print()
print('=== WINNER: yolo_bilstm (test mIoU=0.1867, Debris F1=0.0728) ===')