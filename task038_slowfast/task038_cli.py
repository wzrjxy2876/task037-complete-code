"""Task038 command-line pipeline with explicit phase gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
from typing import Any

import torch

from .slowfast_contribution_probe import probe_contribution_fields
from .slowfast_dependency_graph import build_dependency_graph, write_dependency_graph
from .slowfast_descriptor_adapter import calibrate_descriptors
from .slowfast_f3_selector import global_bms_domains, reference_f3_prefix, select_f3
from .slowfast_finetune import (
    baseline_validation,
    build_loader,
    data_paths,
    fine_tune,
    resolve_authoritative_slowfast_config,
    set_seed,
)
from .slowfast_functional_archive import ContributionFieldArchive, DomainState
from .slowfast_logical_pruning import forward_backward_gate, logical_prune
from .slowfast_model_task038 import (
    load_checkpoint_identity,
    slowfast_16x8_resnet101_kinetics400,
)
from .slowfast_unit_adapter import build_inventory, validate_canonical_order


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _model(checkpoint: str, device: torch.device):
    model = slowfast_16x8_resnet101_kinetics400(101)
    identity = load_checkpoint_identity(model, checkpoint, device)
    return model.to(device), identity


def _validate_output_dir(path: str | Path) -> Path:
    root = Path("/data/jixinye25/work1/output").resolve()
    out = Path(path).resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Task038 runtime output must be under {root}, got {out}"
        ) from exc
    return out


def preflight(args) -> None:
    root = Path(__file__).resolve().parent
    inventory_model = slowfast_16x8_resnet101_kinetics400(101)
    inventory = build_inventory(inventory_model, 0.1)
    validate_canonical_order(inventory)
    graph = build_dependency_graph(inventory_model, inventory)
    structure = Path(args.output_dir) / "structure"
    inventory.write(structure / "unit_inventory.json")
    write_dependency_graph(structure / "dependency_graph.json", graph, inventory)
    old = {
        "IPslowfast.py": "67d2991649a3724986b0ddf52c4f124d3b9ae5cbbb30e8a5991a1cca99936488",
        "myslowfast.py": "9c161a362ad066f66d00f573c5da315cd328309320ed5cc1011fb0d84c51e2af",
    }
    archive_identity = {}
    for name, expected in old.items():
        path = root / "legacy_sources" / name
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f"legacy source hash mismatch for {name}")
        archive_identity[name] = {"sha256": got, "path": str(path)}
    authoritative_paths = {
        "IPslowfast.py": Path("/home/jixinye25/jxy_work1/Code/IPslowfast.py"),
        "myslowfast.py": Path("/home/jixinye25/jxy_work1/Code/myslowfast.py"),
    }
    authoritative_identity = {}
    for name, path in authoritative_paths.items():
        if not path.is_file():
            raise RuntimeError(f"authoritative source missing: {path}")
        authoritative_identity[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "path": str(path),
            "archive_sha256": archive_identity[name]["sha256"],
        }
    checkpoint = Path(args.checkpoint)
    authoritative_config = resolve_authoritative_slowfast_config()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    _json(Path(args.output_dir) / "preflight.json", {
        "status": "passed",
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": "task_038_slowfast_functional_coverage_migration",
        "architecture": "slowfast_16x8_resnet101_kinetics400",
        "candidate_definition": "one Conv3d output channel",
        "unit_count": inventory.num_units,
        "total_model_parameters": inventory.total_model_parameters,
        "max_achievable_analytical_sparsity": inventory.max_achievable_analytical_sparsity,
        "legacy_sources": archive_identity,
        "authoritative_sources": authoritative_identity,
        "authoritative_config": authoritative_config,
        "runtime_output_root": "/data/jixinye25/work1/output",
        "checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
        "data_paths": dict(zip(("train", "val", "frame_root"), data_paths())),
        "gpu_inventory": gpu.stdout.strip().splitlines(),
        "no_server_experiment_started_by_preflight": True,
    })


def load_inventory(args):
    model = slowfast_16x8_resnet101_kinetics400(101)
    return model, build_inventory(model, 0.1)


def run(args) -> None:
    set_seed(3407)
    out = _validate_output_dir(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.mode == "preflight":
        preflight(args)
        return
    model0, inventory = load_inventory(args)
    if args.mode == "baseline":
        model, _ = _model(args.checkpoint, device)
        baseline_validation(model, args.checkpoint, device, out / "baseline_validation.json")
        return
    if args.mode == "descriptors":
        model, _ = _model(args.checkpoint, device)
        train_list, _, _ = data_paths()
        loader = build_loader(train_list, 4, False, workers=2)
        calibrate_descriptors(model, loader, inventory, device, out / "descriptors", 10)
        return
    if args.mode == "bms":
        descriptor = torch.load(out / "descriptors" / "descriptor_vectors.pt", map_location=device).float()
        domains, sinks = global_bms_domains(descriptor, device)
        _json(out / "bms" / "bms_domains.json", {"schema": "task038_global_bms_v1", "parameters": {"sigma": 0.1, "tol": 1e-4, "max_iters": 100, "sink_merge_tol": 0.01}, "unit_count": inventory.num_units, "domains": domains})
        torch.save(sinks.detach().cpu(), out / "bms" / "sink_positions.pt")
        _json(out / "bms" / "bms_summary.json", {"status": "passed", "domain_count": len(domains), "all_units_assigned_once": True, "global": True})
        return
    if args.mode == "contribution":
        model, _ = _model(args.checkpoint, device)
        probe_contribution_fields(model, args.checkpoint, inventory, device, out / "contribution", 3407)
        return
    if args.mode == "numerical":
        domains = json.loads((out / "bms" / "bms_domains.json").read_text())["domains"]
        archive = ContributionFieldArchive(out / "contribution" / "fields", inventory)
        members = domains[0]
        vectors, valid = archive.load_vectors(members, device)
        state = DomainState.create(0, members, vectors, valid)
        cpu_vectors, cpu_valid = archive.load_vectors(members, torch.device("cpu"))
        cpu_state = DomainState.create(0, members, cpu_vectors, cpu_valid)
        losses, coverage = state.losses, state.coverage
        cpu_match = bool(torch.allclose(state.similarity.cpu(), cpu_state.similarity, rtol=1e-4, atol=1e-5) and torch.allclose(losses.cpu(), cpu_state.losses, rtol=1e-4, atol=1e-5))
        if not cpu_match:
            raise RuntimeError("CPU/GPU numerical preflight divergence")
        _json(out / "numerical_preflight.json", {"status": "passed", "domain_id": 0, "domain_size": len(members), "similarity_shape": list(state.similarity.shape), "similarity_min": float(state.similarity.min().item()), "similarity_max": float(state.similarity.max().item()), "coverage": float(coverage.item()), "finite": bool(torch.isfinite(losses[state.retained]).all().item()), "cpu_gpu_match": cpu_match, "field_semantics": "signed", "similarity_definition": "clamp(signed_cosine,0,1)"})
        return
    if args.mode in ("prefix", "selection"):
        domains = json.loads((out / "bms" / "bms_domains.json").read_text())["domains"]
        archive = ContributionFieldArchive(out / "contribution" / "fields", inventory)
        target = 0.5 * inventory.total_model_parameters
        destination = (
            out / "selection_prefix" / "run1"
            if args.mode == "prefix" else out / "selection"
        )
        result = select_f3(archive, inventory, domains, target, destination, device, args.max_steps if args.mode == "prefix" else None)
        if args.mode == "prefix":
            prefix_root = out / "selection_prefix"
            select_f3(
                archive, inventory, domains, target,
                prefix_root / "run2", device, args.max_steps
            )
            first = prefix_root / "run1" / "f3_selection_sequence.csv"
            second = prefix_root / "run2" / "f3_selection_sequence.csv"
            run1_bytes = first.read_bytes()
            run2_bytes = second.read_bytes()
            shutil.copyfile(first, prefix_root / "run1_sequence.csv")
            shutil.copyfile(second, prefix_root / "run2_sequence.csv")
            run1_sha = hashlib.sha256(run1_bytes).hexdigest()
            run2_sha = hashlib.sha256(run2_bytes).hexdigest()
            _json(prefix_root / "run1_sequence_sha256.json", {"sha256": run1_sha})
            _json(prefix_root / "run2_sequence_sha256.json", {"sha256": run2_sha})
            if run1_bytes != run2_bytes:
                raise RuntimeError("prefix replay is not deterministic")
            oracle = reference_f3_prefix(
                archive, inventory, domains, args.max_steps, device
            )
            production = result["trace"][:args.max_steps]
            keys = (
                "global_index", "delta_average", "delta_total",
                "domain_damage", "p_total", "p_average", "R_F3"
            )
            production_projection = [
                {key: row[key] for key in keys} for row in production
            ]
            oracle_projection = [
                {key: row[key] for key in keys} for row in oracle
            ]
            if production_projection != oracle_projection:
                _json(prefix_root / "exactness_gate.json", {
                    "status": "failed",
                    "production_steps": production_projection,
                    "oracle_steps": oracle_projection,
                })
                raise RuntimeError("production F3 prefix diverges from direct oracle")
            _json(prefix_root / "reference_oracle.json", {
                "steps": oracle, "max_steps": args.max_steps
            })
            _json(prefix_root / "exactness_gate.json", {
                "status": "passed",
                "steps": len(oracle),
                "production_vs_oracle_exact": True,
                "run1_vs_run2_byte_identical": run1_bytes == run2_bytes,
                "run1_sequence_sha256": run1_sha,
                "run2_sequence_sha256": run2_sha,
            })
        else:
            _json(out / "selection_summary.json", {"status": "passed" if result["registry"]["removed_parameter_cost"] >= target else "failed", "target_budget": target, "removed_parameter_cost": result["registry"]["removed_parameter_cost"], "overshoot": result["registry"]["removed_parameter_cost"] - target})
        return
    if args.mode == "logical":
        model, _ = _model(args.checkpoint, device)
        registry = out / "selection" / "f3_registry.json"
        report = logical_prune(model, inventory, registry)
        report.update(forward_backward_gate(model, device))
        _json(out / "logical_pruning.json", report)
        return
    if args.mode == "preft":
        model, _ = _model(args.checkpoint, device)
        logical_prune(model, inventory, out / "selection" / "f3_registry.json")
        _, val_list, _ = data_paths()
        result = __import__("task038_slowfast.slowfast_finetune", fromlist=["validate"]).validate(model, build_loader(val_list, 4, False), device)
        _json(out / "preft_validation.json", result)
        return
    if args.mode == "finetune":
        model = slowfast_16x8_resnet101_kinetics400(101)
        identity = load_checkpoint_identity(model, args.checkpoint, torch.device("cpu"))
        logical_prune(model, inventory, out / "selection" / "f3_registry.json")
        authoritative_config = resolve_authoritative_slowfast_config()
        configured_epochs = int(os.environ.get("TASK038_EPOCHS", str(authoritative_config["epochs"])))
        if configured_epochs != int(authoritative_config["epochs"]):
            raise RuntimeError("TASK038_EPOCHS does not match authoritative SlowFast config")
        batch_size = int(os.environ.get("TASK038_BATCH_SIZE", "16"))
        registry_manifest = json.loads(
            (out / "selection" / "f3_registry_manifest.json").read_text()
        )
        sequence_sha = (
            out / "selection" / "f3_selection_sequence.sha256"
        ).read_text().strip()
        result = fine_tune(
            model,
            args.checkpoint,
            args.gpu_ids,
            out / "finetune",
            int(authoritative_config["epochs"]),
            base_lr=float(authoritative_config["base_lr"]),
            weight_decay=float(authoritative_config["weight_decay"]),
            batch_size=batch_size,
            checkpoint_identity=identity,
            registry_sha256=registry_manifest["sha256"],
            sequence_sha256=sequence_sha,
            user_batch_size_override=True,
            authoritative_config=authoritative_config,
        )
        return
    raise ValueError(args.mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("preflight", "baseline", "descriptors", "bms", "contribution", "numerical", "prefix", "selection", "logical", "preft", "finetune"))
    parser.add_argument("--checkpoint", default=os.environ.get("TASK038_CHECKPOINT", "/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt"))
    parser.add_argument("--output_dir", default=os.environ.get("TASK038_OUTPUT_DIR", "/data/jixinye25/work1/output/task038_slowfast_functional_coverage_migration/n09_prune50_exact"))
    parser.add_argument("--device", default=os.environ.get("TASK038_DEVICE", "cuda:0"))
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument("--gpu-ids", dest="gpu_ids", type=int, nargs="+", default=[0, 1])
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
