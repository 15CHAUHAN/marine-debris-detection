"""Audit MARIDA data for duplicate and suspicious-feature leakage across splits.

Checks performed:
1. Exact duplicates - identical GeoTiff file content (md5).
2. Near-duplicates - pairwise cosine similarity of a compact per-patch
   feature vector (band means/stds + 8x8x11 downsampled thumbnail).
3. Cross-split near-duplicates - near-identical patches that appear in
   different splits (the dangerous leakage case).
4. Suspicious features - near-zero-variance bands, constant patches,
   single-class dominance (>95%), all-unlabeled masks, non-finite values.

Outputs:
  artifacts/leakage_audit/report.json  - full audit report
  artifacts/leakage_audit/near_duplicates.png - similarity heatmap (optional)

Usage:
  python scripts/audit_data_leakage.py [--sim 0.98] [--cross-sim 0.99]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.encoding import N_CLASSES
from src.data.ingestion import MaridaDataset
from src.data.splits import load_official_splits


def compute_feature_vector(image: np.ndarray) -> np.ndarray:
    """Compact (726,) feature vector for near-duplicate detection.

    Input image is (C, H, W) float32 in [0, 1]. Features: per-band mean+std
    (22) concatenated with a block-averaged 8x8x11 thumbnail (704)."""
    means = image.mean(axis=(1, 2))
    stds = image.std(axis=(1, 2))
    c, h, w = image.shape
    block_h, block_w = h // 8, w // 8
    thumb = image.reshape(c, 8, block_h, 8, block_w).mean(axis=(2, 4))
    return np.concatenate([means, stds, thumb.reshape(-1)]).astype(np.float32)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = vectors / norms
    return normalized @ normalized.T


def find_pairs(similarity: np.ndarray, ids: list[str], threshold: float) -> list[dict]:
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if similarity[i, j] >= threshold:
                pairs.append(
                    {
                        "patch_a": ids[i],
                        "patch_b": ids[j],
                        "similarity": round(float(similarity[i, j]), 6),
                    }
                )
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs


def check_exact_duplicates_by_content(marida: MaridaDataset) -> list[dict]:
    """Group patches whose image files are byte-identical (content hash)."""
    hashes: dict[str, list[str]] = {}
    for pid in marida.ids:
        path = marida.get(pid).image_path
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(pid)
    return [
        {"md5": digest, "patches": sorted(ids)}
        for digest, ids in hashes.items()
        if len(ids) > 1
    ]


def check_suspicious_features(
    marida: MaridaDataset, patch_ids: list[str]
) -> list[dict]:
    flagged = []
    for pid in patch_ids:
        image = marida.read_image(pid)
        reasons = []
        band_stds = image.reshape(image.shape[0], -1).std(axis=1)
        low_var = np.where(band_stds < 1e-6)[0]
        if low_var.size:
            reasons.append(f"near_zero_variance_bands={low_var.tolist()}")
        if image.std() < 1e-8:
            reasons.append("constant_patch_all_bands")
        if not np.isfinite(image).all():
            reasons.append("contains_non_finite")
        mask = marida.read_mask(pid)
        seen = (mask > 0)
        if seen.all():
            reasons.append("fully_labeled")
        if not seen.any():
            reasons.append("all_unlabeled")
        labeled = mask[mask > 0]
        if labeled.size:
            counts = np.bincount(labeled, minlength=N_CLASSES + 1)
            dominance = float(counts.max()) / labeled.size
            if dominance > 0.95:
                dom_class = int(np.argmax(counts))
                reasons.append(
                    f"single_class_dominance=class{dom_class}:{dominance:.4f}"
                )
        if reasons:
            flagged.append({"patch": pid, "reasons": reasons})
    return flagged


def check_cross_split(
    feature_vectors: dict[str, np.ndarray],
    splits: dict[str, list[str]],
    threshold: float,
) -> list[dict]:
    split_of = {pid: name for name, ids in splits.items() for pid in ids}
    ids = [pid for pid in feature_vectors if pid in split_of]
    vectors = np.array([feature_vectors[pid] for pid in ids], dtype=np.float32)
    sim = cosine_similarity_matrix(vectors)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if split_of[a] != split_of[b] and sim[i, j] >= threshold:
                pairs.append(
                    {
                        "patch_a": a,
                        "split_a": split_of[a],
                        "patch_b": b,
                        "split_b": split_of[b],
                        "similarity": round(float(sim[i, j]), 6),
                    }
                )
    pairs.sort(key=lambda x: -x["similarity"])
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=float, default=0.985, help="near-duplicate threshold")
    parser.add_argument("--cross-sim", type=float, default=0.9, help="cross-split threshold")
    args = parser.parse_args()

    out_dir = ROOT / "artifacts" / "leakage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    marida = MaridaDataset(ROOT)
    splits = load_official_splits(ROOT / "splits")
    all_ids = list(marida.ids)

    print(f"Auditing {len(all_ids)} patches across splits...")

    print("Checking exact duplicates (content hash)...")
    content_dups = check_exact_duplicates_by_content(marida)
    print(f"  {len(content_dups)} identical-content groups")

    print("Computing feature vectors...")
    features: dict[str, np.ndarray] = {}
    for i, pid in enumerate(all_ids):
        image = marida.read_image(pid)
        features[pid] = compute_feature_vector(image)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(all_ids)}")

    print(f"Checking near-duplicates (sim >= {args.sim})...")
    ids = all_ids
    vectors = np.array([features[pid] for pid in ids], dtype=np.float32)
    sim = cosine_similarity_matrix(vectors)
    near_dups = find_pairs(sim, ids, args.sim)
    print(f"  {len(near_dups)} near-duplicate pairs")

    print(f"Checking cross-split duplicates (sim >= {args.cross_sim})...")
    cross_dups = check_cross_split(features, splits, args.cross_sim)
    print(f"  {len(cross_dups)} cross-split near-duplicate pairs")

    print("Checking suspicious features...")
    suspicious = check_suspicious_features(marida, all_ids)
    print(f"  {len(suspicious)} patches flagged")

    report = {
        "total_patches": len(all_ids),
        "exact_duplicates": {"count": len(content_dups), "groups": content_dups},
        "near_duplicates": {
            "threshold": args.sim,
            "count": len(near_dups),
            "pairs": near_dups[:200],
        },
        "cross_split_duplicates": {
            "threshold": args.cross_sim,
            "count": len(cross_dups),
            "pairs": cross_dups,
        },
        "suspicious_features": {"count": len(suspicious), "patches": suspicious},
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}")
    print("\n=== LEAKAGE AUDIT SUMMARY ===")
    print(f"Total patches: {len(all_ids)}")
    print(f"Exact duplicate groups: {len(content_dups)}")
    print(f"Near-duplicate pairs (>{args.sim}): {len(near_dups)}")
    print(f"Cross-split near-duplicates (>{args.cross_sim}): {len(cross_dups)}")
    print(f"Suspicious patches: {len(suspicious)}")
    if cross_dups:
        print("\nWARNING: cross-split near-duplicate patches detected!")
        for p in cross_dups[:10]:
            print(f"  {p['patch_a']} ({p['split_a']}) <-> "
                  f"{p['patch_b']} ({p['split_b']}) sim={p['similarity']}")


if __name__ == "__main__":
    main()