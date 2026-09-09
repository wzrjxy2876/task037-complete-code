"""Real-cache numerical gate for Task016 domain-total loss."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from functional_competition_pruning import (
    DomainState,
    build_functional_similarity,
    direct_summed_coverage_losses,
    domain_total_losses,
)
from task015_attention_ffn_diagnosis import rebuild_bms_domains


def run_preflight(
    descriptor_path: Path,
    vector_path: Path,
    mask_path: Path,
    output_path: Path,
    device: str,
) -> dict:
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("Task016 real preflight requires CUDA")
    with descriptor_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 36_378:
        raise ValueError(f"Descriptor rows={len(rows)}, expected 36,378")
    descriptors = np.asarray(
        [[float(row["D_abs"]), float(row["D_rel"]), float(row["D_dyn"])] for row in rows],
        dtype=np.float32,
    )  # [N,3]
    groups = rebuild_bms_domains(descriptors, str(target_device))
    vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if vectors.shape[0] != 36_378 or valid.shape != (36_378,):
        raise ValueError("Aligned Task014 cache has an unexpected unit axis")
    eligible = sorted(
        (members for members in groups if len(members) >= 2 and np.asarray(valid[members]).any()),
        key=lambda members: (len(members), members[0]),
    )
    if not eligible:
        raise RuntimeError("No active multi-unit BMS domain exists")
    members = eligible[0]
    domain_vectors = torch.from_numpy(
        np.asarray(vectors[members], dtype=np.float32).copy()
    ).to(target_device)  # [G,9*16*7*7]
    domain_valid = torch.from_numpy(
        np.asarray(valid[members], dtype=np.bool_).copy()
    ).to(target_device)  # Bool[G]
    similarity = build_functional_similarity(domain_vectors, domain_valid)  # [G,G]
    state = DomainState(0, list(members), similarity, domain_valid)
    total = domain_total_losses(state.losses, state.valid_function_mask)  # [G]
    direct = direct_summed_coverage_losses(
        state.similarity, state.retained, state.valid_function_mask
    )  # [G]
    retained = state.retained
    maximum_error = float((total[retained] - direct[retained]).abs().max().item())
    if not torch.allclose(total[retained], direct[retained], rtol=1e-6, atol=1e-7):
        raise RuntimeError(f"Task016 total-loss equivalence failed: {maximum_error}")
    payload = {
        "status": "passed",
        "descriptor_units": len(rows),
        "bms_domains": len(groups),
        "domain_size": len(members),
        "active_functional_demand_count": int(domain_valid.sum().item()),
        "null_functional_count": int((~domain_valid).sum().item()),
        "domain_size_semantics": "fixed non-null Task014 coverage demand count",
        "maximum_total_vs_direct_absolute_error": maximum_error,
        "device": str(target_device),
        "domain_average_redefined": False,
        "parameter_cost_used": False,
        "type_specific_rule_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--valid-mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_preflight(args.descriptor, args.vectors, args.valid_mask, args.output, args.device)
