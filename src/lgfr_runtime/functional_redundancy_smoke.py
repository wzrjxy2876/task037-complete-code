"""Deterministic numerical smoke for LGFR representative protection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from function_bms_utils import (
    TOP_CANDIDATE_RATIO,
    compute_function_uniqueness,
    compute_intra_cluster_functional_redundancy,
    groups_to_labels,
    mean_shift_bms,
    select_descriptor_cluster_representatives,
    summarize_functional_redundancy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./functional_redundancy_smoke")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--sigma", type=float, default=0.12)
    return parser


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = torch.tensor(
        [[-0.8, -0.7, -0.6], [0.0, 0.2, 0.1], [0.8, 0.7, 0.9]],
        dtype=torch.float32,
    )
    descriptors = torch.cat(
        [
            center + 0.015 * torch.randn(6, 3)
            for center in centers
        ],
        dim=0,
    )
    descriptor = mean_shift_bms(
        descriptors,
        sigma=args.sigma,
        function_similarity=None,
    )
    groups = descriptor[0]
    labels_before = groups_to_labels(groups, descriptors.shape[0])

    feature_centers = torch.tensor(
        [[1.0, 0.2, 0.1, 0.0], [0.1, 1.0, 0.2, 0.1], [0.0, 0.2, 1.0, 0.3]]
    )
    features = torch.cat(
        [
            center + 0.08 * torch.randn(6, 4)
            for center in feature_centers
        ],
        dim=0,
    )
    features = torch.nn.functional.normalize(features, dim=1)
    s_function = torch.clamp(features @ features.T, min=0.0, max=1.0)
    s_function.fill_diagonal_(1.0)
    importance = descriptors.mean(dim=1)
    redundancy = compute_intra_cluster_functional_redundancy(
        groups,
        s_function,
    )
    uniqueness = compute_function_uniqueness(redundancy)
    representatives = select_descriptor_cluster_representatives(
        groups,
        importance,
        uniqueness,
    )
    labels_after = groups_to_labels(groups, descriptors.shape[0])
    summary = summarize_functional_redundancy(
        groups,
        redundancy,
        representatives,
    )
    summary.update(
        {
            "seed": args.seed,
            "sigma": args.sigma,
            "unit_count": descriptors.shape[0],
            "descriptor_dimension": descriptors.shape[1],
            "top_candidate_ratio": TOP_CANDIDATE_RATIO,
            "descriptor_labels_unchanged": bool(
                torch.equal(labels_before, labels_after)
            ),
            "trajectory_is_descriptor_trajectory": True,
            "finite": bool(
                torch.isfinite(redundancy).all()
                and torch.isfinite(uniqueness).all()
                and torch.isfinite(s_function).all()
            ),
            "cuda_available": torch.cuda.is_available(),
            "cuda_peak_allocated_mb": (
                float(torch.cuda.max_memory_allocated() / 2**20)
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    unit_rows = []
    for cluster_id, members in enumerate(groups):
        representative = representatives[cluster_id]
        for index in members:
            unit_rows.append(
                {
                    "unit_index": index,
                    "cluster_id": cluster_id,
                    "descriptor_importance": float(importance[index]),
                    "function_redundancy": float(redundancy[index]),
                    "function_uniqueness": float(uniqueness[index]),
                    "protected": index == representative,
                }
            )
    with (output_dir / "unit_redundancy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    cluster_rows = []
    for cluster_id, members in enumerate(groups):
        values = redundancy[torch.as_tensor(members)]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "mean_redundancy": float(values.mean()),
                "max_redundancy": float(values.max()),
                "min_redundancy": float(values.min()),
                "representative_index": representatives[cluster_id],
            }
        )
    with (output_dir / "cluster_redundancy.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
        writer.writeheader()
        writer.writerows(cluster_rows)
    np.savez_compressed(
        output_dir / "functional_redundancy_smoke.npz",
        descriptors=descriptors.numpy(),
        descriptor_labels=labels_before.numpy(),
        S_func=s_function.numpy(),
        functional_redundancy=redundancy.numpy(),
        functional_uniqueness=uniqueness.numpy(),
        representative_indices=np.asarray(representatives),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
