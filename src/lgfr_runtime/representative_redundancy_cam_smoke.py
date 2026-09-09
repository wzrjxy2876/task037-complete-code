"""Synthetic end-to-end smoke for function-representative CAM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validate_representative_redundancy_cam import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="./representative_redundancy_cam_smoke_output",
    )
    parser.add_argument("--videos", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser


def write_inputs(input_dir: Path) -> tuple[Path, Path, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    unit_path = input_dir / "unit_redundancy.csv"
    cluster_path = input_dir / "cluster_redundancy.csv"
    formal_path = input_dir / "formal_descriptors.npz"
    unit_rows = []
    unit_ids = []
    for index in range(8):
        cluster_id = index // 4
        unit_id = f"block.mlp::neuron::{index}"
        unit_ids.append(unit_id)
        unit_rows.append(
            {
                "unit_index": index,
                "unit_id": unit_id,
                "descriptor_importance": (
                    1.0 if index % 4 == 0 else 0.1 + index
                ),
                "function_redundancy": (
                    0.1 if index % 4 == 0 else 0.9 - index * 0.02
                ),
                "function_uniqueness": (
                    0.9 if index % 4 == 0 else 0.1 + index * 0.02
                ),
                "protected": index % 4 == 0,
                "cluster_id": cluster_id,
                "stage": 0,
                "block": 0,
                "unit_type": "neuron",
                "layer": "block.mlp",
                "kept": index % 4 in {0, 1},
            }
        )
    with unit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(unit_rows[0]))
        writer.writeheader()
        writer.writerows(unit_rows)
    cluster_rows = [
        {
            "cluster_id": cluster_id,
            "size": 4,
            "mean_redundancy": 0.7 - cluster_id * 0.1,
            "mean_uniqueness": 0.3 + cluster_id * 0.1,
            "max_redundancy": 0.9,
            "min_redundancy": 0.1,
            "representative_id": unit_ids[cluster_id * 4],
            "protected_unit": unit_ids[cluster_id * 4],
            "stage": "0",
            "block": "0",
            "unit_type": "neuron",
            "head_count": 0,
            "neuron_count": 4,
            "layers": "block.mlp",
        }
        for cluster_id in range(2)
    ]
    with cluster_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cluster_rows[0]))
        writer.writeheader()
        writer.writerows(cluster_rows)
    np.savez_compressed(
        formal_path,
        unit_ids=np.asarray(unit_ids),
        descriptors_raw=np.zeros((8, 3), dtype=np.float32),
        descriptors_normalized=np.zeros((8, 3), dtype=np.float32),
        unit_info=np.asarray(
            [json.dumps({"unit_id": unit_id}) for unit_id in unit_ids]
        ),
    )
    return unit_path, cluster_path, formal_path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    unit_path, cluster_path, formal_path = write_inputs(
        output_dir / "_synthetic_inputs"
    )
    summary = run(
        argparse.Namespace(
            model_builder=(
                "paired_cluster_prototype_cam_smoke_adapter:"
                "build_model_for_probe"
            ),
            loader_builder=(
                "paired_cluster_prototype_cam_smoke_adapter:"
                "build_probe_loader"
            ),
            checkpoint=str(output_dir / "_synthetic_checkpoint"),
            unit_redundancy_csv=str(unit_path),
            cluster_redundancy_csv=str(cluster_path),
            formal_descriptors_npz=str(formal_path),
            output_dir=str(output_dir),
            num_clusters=2,
            cluster_ids="",
            members_per_cluster=2,
            videos=args.videos,
            response_mode="activation",
            target_mode="predicted",
            seed=args.seed,
            probe_batch_size=2,
            probe_split="val",
            num_workers=0,
            device="cpu",
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
