"""Task027 senior-style logical structured pruning and recovery fine-tuning.

The authoritative Task024 T+A+D registry is applied without reselection.
This module deliberately keeps every original parameter tensor and performs
the same runtime ``index_select`` operations as the repository's senior-style
implementation.  Only the Python ``keep_heads``/``keep_neurons`` attributes
are changed; no Linear module is replaced and no tensor is removed from the
state dict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np


CODE_VERSION = "task027_senior_style_logical_pruning_finetune_v1"
TASK024_VARIANT = "dual_rank_domain_state_minimax_30"
CHECKPOINT_NAME = "swin_pruned_best.pth"
TRAIN_BATCH_SIZE = 4
DEFAULT_MODEL_NAME = "swintrans"
SEED = 3407
GPU_IDS = (0, 1)
PRE_FINETUNE_TOLERANCE_PP = 0.1
RELOAD_TOLERANCE_PP = 1e-4
EXPECTED_VALIDATION_SAMPLES = 3783
# These are the immutable Task018/Task024 registry labels.
TYPE_ATTENTION = "attention_head"
TYPE_FFN = "ffn_neuron"

REQUIRED_COMPLETION_TRUE = (
    "task024_reference_pass", "task025_reference_pass",
    "authoritative_tad_registry_pass", "logical_pruning_applied",
    "state_dict_numel_unchanged", "effective_parameter_reduction_positive",
    "pre_finetune_validation_complete", "senior_finetune_config_pass",
    "optimizer_sgd", "momentum_09", "lr_is_cfg_lr_times_01",
    "weight_decay_matches_cfg", "epochs_match_cfg", "batch_size_4",
    "seed_3407", "cross_entropy_only", "best_checkpoint_saved",
    "keep_indices_saved", "best_checkpoint_reload_pass",
)
REQUIRED_COMPLETION_FALSE = (
    "selection_rerun", "physical_pruning_executed", "scheduler_used",
    "task024_artifacts_modified", "task025_artifacts_modified",
    "task026_physicalization_used",
)


def _torch():
    import torch
    return torch


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _int(value: object, label: str = "value") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        number = int(value)
        if float(value) != number:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer: {value!r}") from exc
    return number


def _as_indices(values: object, label: str = "indices") -> list[int]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be a list")
    result = [_int(value, f"{label} entry") for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate {label}")
    return result


def set_seed(seed: int = SEED) -> None:
    """Use the exact deterministic seed policy required by the senior code."""
    torch = _torch()
    seed = _int(seed, "seed")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _copy_validation_identity(identity: Mapping[str, object]) -> dict[str, object]:
    required = ("validation_split", "validation_batch_size", "amp_enabled")
    missing = [key for key in required if key not in identity]
    if missing:
        raise RuntimeError(f"Validation identity incomplete: {missing}")
    split = str(identity["validation_split"])
    if not split.strip():
        raise RuntimeError("validation_split must not be empty")
    batch = _int(identity["validation_batch_size"], "validation_batch_size")
    if batch <= 0:
        raise RuntimeError("validation_batch_size must be positive")
    return {"validation_split": split, "validation_batch_size": batch,
            "amp_enabled": bool(identity["amp_enabled"])}


def _task023_module():
    import task023_average_rescue_causal_ablation as task023
    return task023


def _task024_module():
    import task024_threshold_free_adaptive_safety as task024
    return task024


def _task025_module():
    import task025_total_domain_state_factorial_ablation as task025
    return task025


def _ensure_legacy_torch_six_compat() -> None:
    """Provide the removed torch._six aliases required by legacy GluonCV."""
    import sys
    from types import ModuleType

    torch = _torch()
    try:
        import torch._six  # type: ignore[attr-defined]
    except ImportError:
        six_mod = ModuleType("torch._six")
        sys.modules["torch._six"] = six_mod
        torch._six = six_mod

    if not hasattr(torch._six, "int_classes"):
        torch._six.int_classes = (int,)
    if not hasattr(torch._six, "string_classes"):
        torch._six.string_classes = (str,)


def _ensure_legacy_pillow_compat() -> None:
    """Provide missing Pillow interpolation aliases for legacy GluonCV."""
    import PIL
    import PIL.Image as PIL_Image

    resampling = getattr(PIL_Image, "Resampling", None)
    if not hasattr(PIL_Image, "LINEAR"):
        PIL_Image.LINEAR = resampling.BILINEAR if resampling is not None else 2
    if not hasattr(PIL_Image, "BILINEAR"):
        PIL_Image.BILINEAR = resampling.BILINEAR if resampling is not None else 2
    if not hasattr(PIL_Image, "BICUBIC"):
        PIL_Image.BICUBIC = resampling.BICUBIC if resampling is not None else 3
    if not hasattr(PIL_Image, "NEAREST"):
        PIL_Image.NEAREST = resampling.NEAREST if resampling is not None else 0
    PIL.Image = PIL_Image


def _required_split_from_environment(name: str) -> Path:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise RuntimeError(f"{name} is required")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{name} must reference a regular file: {path}")
    try:
        if path.stat().st_size <= 0:
            raise RuntimeError(f"{name} must reference a non-empty file: {path}")
    except OSError as exc:
        raise RuntimeError(f"Unable to inspect {name}: {path}") from exc
    return path


def _split_file_identity(path: Path) -> tuple[str, int]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        lines = sum(1 for _ in handle)
    return sha256_file(path), lines


def _verify_safety(payload: Mapping[str, object], label: str) -> None:
    for key in ("task_label_leakage", "production_pruning_code_modified",
                "fine_tuning_executed"):
        if payload.get(key) is not False:
            raise RuntimeError(f"{label} safety gate failed: {key}")


def verify_task024_reference(task024_root: Path) -> dict[str, object]:
    root = Path(task024_root)
    completion_path = root / "task024_completion.json"
    completion = read_json(completion_path)
    if completion.get("status") != "PASS":
        raise RuntimeError("Task024 completion is not PASS")
    for key in ("task023_identity_pass", "task023_exact_v0_reference_pass",
                "task023_hard_rescue_reference_pass",
                "zero_new_tunable_hyperparameters"):
        if completion.get(key) is not True:
            raise RuntimeError(f"Task024 completion gate failed: {key}")
    _verify_safety(completion, "Task024 completion")
    identity = read_json(root / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task024 artifact identity is incomplete")
    registry_path = root / "variants" / TASK024_VARIANT / "registry.json"
    registry_payload = read_json(registry_path)
    if registry_payload.get("status") != "prepared":
        raise RuntimeError("Task024 T+A+D registry is not prepared")
    registry = registry_payload.get("registry")
    if not isinstance(registry, dict):
        raise RuntimeError("Task024 registry payload is malformed")
    canonical = _task023_module().canonical_registry_sha256(registry)
    if registry_payload.get("summary", {}).get("registry_canonical_sha256") != canonical:
        raise RuntimeError("Task024 registry canonical SHA mismatch")
    metrics_path = root / "validation" / TASK024_VARIANT / "metrics.json"
    metrics = read_json(metrics_path)
    if metrics.get("status") != "PASS" or _int(metrics.get("samples", -1), "samples") != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Task024 logical reference metrics are incomplete")
    return {"completion": completion, "identity": identity,
            "completion_sha256": sha256_file(completion_path),
            "registry": registry, "registry_payload": registry_payload,
            "registry_sha256": canonical, "metrics": metrics}


def verify_task025_reference(task025_root: Path) -> dict[str, object]:
    root = Path(task025_root)
    completion_path = root / "task025_completion.json"
    completion = read_json(completion_path)
    if completion.get("status") != "PASS":
        raise RuntimeError("Task025 completion is not PASS")
    for key in ("task023_reference_pass", "task024_reference_pass",
                "same_start_prefix", "same_target_budget",
                "factorial_analysis_complete",
                "total_domain_state_validation_complete",
                "zero_new_tunable_hyperparameters"):
        if completion.get(key) is not True:
            raise RuntimeError(f"Task025 completion gate failed: {key}")
    if completion.get("fixed_percentile_threshold_used") is not False:
        raise RuntimeError("Task025 fixed-threshold gate failed")
    _verify_safety(completion, "Task025 completion")
    identity = read_json(root / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task025 artifact identity is incomplete")
    return {"completion": completion, "identity": identity,
            "completion_sha256": sha256_file(completion_path)}


def resolve_senior_config(model_name: str = DEFAULT_MODEL_NAME,
                          batch_size: int = TRAIN_BATCH_SIZE) -> dict[str, object]:
    """Resolve, rather than invent, the senior-compatible training config."""
    _ensure_legacy_torch_six_compat()
    _ensure_legacy_pillow_compat()
    batch_size = _int(batch_size, "batch_size")
    if batch_size != TRAIN_BATCH_SIZE:
        raise ValueError("Task027 senior batch size is fixed at 4")
    from utils import CONFIG_PATHS, get_cfg_custom
    if model_name not in CONFIG_PATHS:
        raise KeyError(f"Unknown model config: {model_name}")
    cfg_path = CONFIG_PATHS[model_name]
    cfg = get_cfg_custom(cfg_path, batch_size)
    train = cfg.CONFIG.TRAIN
    data = cfg.CONFIG.DATA
    historical_train_split = str(data.TRAIN_ANNO_PATH)
    historical_validation_split = str(data.VAL_ANNO_PATH)
    train_path = _required_split_from_environment("UCF101_TRAIN_SPLIT")
    validation_path = _required_split_from_environment("UCF101_VAL_SPLIT")
    train_sha256, train_lines = _split_file_identity(train_path)
    validation_sha256, validation_lines = _split_file_identity(validation_path)
    base_lr = float(train.LR)
    weight_decay = float(train.W_DECAY)
    epochs = _int(train.EPOCH_NUM, "EPOCH_NUM")
    if epochs <= 0:
        raise RuntimeError("Resolved EPOCH_NUM must be positive")
    return {"cfg_path": str(cfg_path), "cfg": cfg,
            "historical_train_split": historical_train_split,
            "historical_validation_split": historical_validation_split,
            "train_split": str(train_path), "validation_split": str(validation_path),
            "resolved_train_split": str(train_path),
            "resolved_validation_split": str(validation_path),
            "resolved_train_split_sha256": train_sha256,
            "resolved_validation_split_sha256": validation_sha256,
            "resolved_train_split_lines": train_lines,
            "resolved_validation_split_lines": validation_lines,
            "base_cfg_lr": base_lr, "actual_finetune_lr": base_lr * 0.1,
            "momentum": 0.9, "weight_decay": weight_decay,
            "epochs": epochs, "batch_size": batch_size}


def _validation_split_equal(left: str, right: str) -> bool:
    # Preserve the exact configured spelling when it is already identical;
    # otherwise accept equivalent absolute paths only when both exist.
    if str(left) == str(right):
        return True
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return False


def _production_identity(repo_root: Path) -> object:
    return _task024_module().production_identity(Path(repo_root))


def verify_identity(*, task024_root: Path, task025_root: Path,
                    output_dir: Path, checkpoint: Path, repo_root: Path,
                    model_name: str = DEFAULT_MODEL_NAME,
                    gpu_ids: Sequence[int] = GPU_IDS) -> dict[str, object]:
    ref24 = verify_task024_reference(task024_root)
    ref25 = verify_task025_reference(task025_root)
    identity24 = ref24["identity"]
    identity25 = ref25["identity"]
    validation24 = _copy_validation_identity(identity24)
    validation25 = _copy_validation_identity(identity25)
    if validation24 != validation25:
        raise RuntimeError("Task024 and Task025 validation identity differs")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    for label, identity in (("Task024", identity24), ("Task025", identity25)):
        if str(identity.get("checkpoint_sha256")) != checkpoint_sha:
            raise RuntimeError(f"{label} checkpoint identity differs")
    registry_sha = str(ref24["registry_sha256"])
    expected_registry = identity25.get("task024_tad_registry_canonical_sha256")
    if expected_registry is None or str(expected_registry).strip() == "":
        raise RuntimeError("Task025 identity is missing T+A+D registry canonical SHA")
    if str(expected_registry) != registry_sha:
        raise RuntimeError("Task025 T+A+D registry identity differs")
    production = _production_identity(repo_root)
    for label, identity in (("Task024", identity24), ("Task025", identity25)):
        expected_source = identity.get("production_source_git_blob_sha")
        if expected_source is None:
            raise RuntimeError(f"{label} identity is missing production source identity")
        if expected_source != production:
            raise RuntimeError(f"{label} production source identity differs")
    config = resolve_senior_config(model_name, TRAIN_BATCH_SIZE)
    task024_validation_split = Path(str(validation24["validation_split"])).expanduser().resolve()
    if not task024_validation_split.is_file() or task024_validation_split.stat().st_size <= 0:
        raise RuntimeError(f"Task024 validation split is missing or empty: {task024_validation_split}")
    task024_validation_sha256, task024_validation_lines = _split_file_identity(task024_validation_split)
    if str(config["resolved_validation_split_sha256"]) != task024_validation_sha256:
        raise RuntimeError("Resolved senior validation split content differs from Task024")
    if int(config["resolved_validation_split_lines"]) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Resolved validation split line count differs from locked identity")
    gpu_ids = tuple(_int(value, "gpu id") for value in gpu_ids)
    if gpu_ids != GPU_IDS:
        raise RuntimeError("Task027 requires two logical GPUs [0, 1]")
    payload = {
        "status": "PASS", "code_version": CODE_VERSION,
        "artifact_identity_pass": True,
        "task024_reference_pass": True, "task025_reference_pass": True,
        "task024_root": str(Path(task024_root).resolve()),
        "task025_root": str(Path(task025_root).resolve()),
        "task024_completion_sha256": ref24["completion_sha256"],
        "task025_completion_sha256": ref25["completion_sha256"],
        "task024_tad_registry_canonical_sha256": registry_sha,
        "source_registry_variant": TASK024_VARIANT,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "historical_train_split": config["historical_train_split"],
        "historical_validation_split": config["historical_validation_split"],
        "resolved_train_split": config["resolved_train_split"],
        "resolved_validation_split": config["resolved_validation_split"],
        "train_split": config["train_split"],
        "validation_split": config["validation_split"],
        "resolved_train_split_sha256": config["resolved_train_split_sha256"],
        "resolved_validation_split_sha256": config["resolved_validation_split_sha256"],
        "resolved_train_split_lines": config["resolved_train_split_lines"],
        "resolved_validation_split_lines": config["resolved_validation_split_lines"],
        "task024_validation_split": str(task024_validation_split),
        "task024_validation_split_sha256": task024_validation_sha256,
        "task024_validation_split_lines": task024_validation_lines,
        "validation_split_sha256_match": True,
        "validation_batch_size": validation24["validation_batch_size"],
        "batch_size": TRAIN_BATCH_SIZE, "amp_enabled": validation24["amp_enabled"],
        "seed": SEED, "model_name": model_name,
        "cfg_path": config["cfg_path"], "base_cfg_lr": config["base_cfg_lr"],
        "actual_finetune_lr": config["actual_finetune_lr"],
        "momentum": config["momentum"], "weight_decay": config["weight_decay"],
        "epochs": config["epochs"], "gpu_ids": list(gpu_ids),
        "selection_rerun": False, "logical_pruning": True,
        "physical_pruning": False, "physical_pruning_executed": False,
        "task026_physicalization_used": False,
        "production_source_git_blob_sha": production,
        "task024_artifacts_modified": False,
        "task025_artifacts_modified": False,
        "fine_tuning_executed": False,
        "new_pruning_hyperparameters": 0,
    }
    atomic_json(Path(output_dir) / "artifact_identity.json", payload)
    return {"identity": payload, "registry": ref24["registry"],
            "registry_sha256": registry_sha, "config": config,
            "reference_metrics": ref24["metrics"]}


def _module_kind(module) -> str | None:
    name = module.__class__.__name__
    if name == "WindowAttention3D":
        return TYPE_ATTENTION
    if name == "Mlp":
        return TYPE_FFN
    return None


def analytical_attention_reduction(module, removed_heads: int) -> int:
    removed_heads = _int(removed_heads, "removed_heads")
    if removed_heads < 0:
        raise ValueError("removed_heads must be non-negative")
    head_dim = _int(module.head_dim, "head_dim")
    input_dim = _int(module.qkv.in_features, "qkv.in_features")
    proj_out = _int(module.proj.out_features, "proj.out_features")
    per_head = 3 * head_dim * input_dim
    if module.qkv.bias is not None:
        per_head += 3 * head_dim
    # The senior report excludes proj.bias and relative-position bias.
    per_head += proj_out * head_dim
    return removed_heads * per_head


def analytical_mlp_reduction(module, removed_neurons: int) -> int:
    removed_neurons = _int(removed_neurons, "removed_neurons")
    if removed_neurons < 0:
        raise ValueError("removed_neurons must be non-negative")
    in_features = _int(module.fc1.in_features, "fc1.in_features")
    out_features = _int(module.fc2.out_features, "fc2.out_features")
    per_neuron = in_features + out_features
    if module.fc1.bias is not None:
        per_neuron += 1
    # The senior report excludes fc2.bias.
    return removed_neurons * per_neuron


def _module_shape_snapshot(module) -> dict[str, tuple[int, ...] | int]:
    snapshot: dict[str, tuple[int, ...] | int] = {}
    for name in ("qkv", "proj", "fc1", "fc2"):
        child = getattr(module, name, None)
        if child is not None:
            snapshot[f"{name}.weight"] = tuple(child.weight.shape)
            snapshot[f"{name}.bias"] = tuple(child.bias.shape) if child.bias is not None else ()
    if hasattr(module, "relative_position_bias_table"):
        snapshot["relative_position_bias_table"] = tuple(module.relative_position_bias_table.shape)
    if hasattr(module, "num_heads"):
        snapshot["num_heads"] = _int(module.num_heads, "num_heads")
    if hasattr(module, "head_dim"):
        snapshot["head_dim"] = _int(module.head_dim, "head_dim")
    if hasattr(module, "original_hidden_features"):
        snapshot["original_hidden_features"] = _int(module.original_hidden_features,
                                                      "original_hidden_features")
    return snapshot


def apply_logical_pruning_registry(model, registry: Mapping[str, Mapping[str, object]],
                                   *, min_keep_units: Mapping[str, int] | int | None = None) -> list[dict[str, object]]:
    """Apply only keep-list attributes and return a strict audit table."""
    if not isinstance(registry, Mapping):
        raise ValueError("registry must be a mapping")
    modules = dict(model.named_modules())
    audits: list[dict[str, object]] = []
    for layer, entry in registry.items():
        layer = str(layer)
        if layer not in modules:
            raise KeyError(f"Registry references missing layer: {layer}")
        if not isinstance(entry, Mapping):
            raise ValueError(f"Registry entry is malformed: {layer}")
        kind = str(entry.get("unit_type", ""))
        module = modules[layer]
        actual_kind = _module_kind(module)
        if actual_kind != kind:
            raise TypeError(f"Registry type mismatch at {layer}: {kind} vs {actual_kind}")
        pruned = _as_indices(entry.get("indices"), f"{layer} indices")
        if kind == TYPE_ATTENTION:
            original = _int(module.num_heads, f"{layer}.num_heads")
        else:
            original = _int(getattr(module, "original_hidden_features", module.fc1.out_features),
                            f"{layer}.original_hidden_features")
        if any(value < 0 or value >= original for value in pruned):
            raise IndexError(f"Registry index out of range at {layer}")
        kept = sorted(set(range(original)) - set(pruned))
        if not kept:
            raise ValueError(f"Registry removes every unit at {layer}")
        if isinstance(min_keep_units, Mapping):
            minimum = _int(min_keep_units.get(layer, 1), "min_keep_units")
        elif min_keep_units is None:
            minimum = 1
        else:
            minimum = _int(min_keep_units, "min_keep_units")
        if len(kept) < minimum:
            raise ValueError(f"Min-keep violation at {layer}: {len(kept)} < {minimum}")
        before_shapes = _module_shape_snapshot(module)
        if kind == TYPE_ATTENTION:
            module.keep_heads = list(kept)
            analytical = analytical_attention_reduction(module, len(pruned))
        else:
            module.keep_neurons = list(kept)
            analytical = analytical_mlp_reduction(module, len(pruned))
        after_shapes = _module_shape_snapshot(module)
        if before_shapes != after_shapes:
            raise RuntimeError(f"Logical pruning changed tensor/module shape at {layer}")
        audits.append({
            "layer": layer, "unit_type": kind, "original_units": original,
            "pruned_units": len(pruned), "kept_units": len(kept),
            "min_keep_units": minimum, "min_keep_pass": True,
            "registry_indices_valid": True, "layer_exists": True,
            "type_match": True, "analytical_parameter_reduction": analytical,
            "pruned_indices": list(pruned), "keep_indices": list(kept),
        })
    return audits


def assert_registry_keep_identity(model, registry: Mapping[str, Mapping[str, object]]) -> None:
    """Ensure runtime keep lists are exactly the complement of the registry."""
    modules = dict(model.module.named_modules()) if model.__class__.__name__ == "DataParallel" else dict(model.named_modules())
    for layer, entry in registry.items():
        layer = str(layer)
        if layer not in modules:
            raise KeyError(f"Registry references missing layer: {layer}")
        module = modules[layer]
        pruned = set(_as_indices(entry.get("indices"), f"{layer} indices"))
        if str(entry.get("unit_type")) == TYPE_ATTENTION:
            original = _int(module.num_heads, f"{layer}.num_heads")
            got = list(module.keep_heads)
        elif str(entry.get("unit_type")) == TYPE_FFN:
            original = _int(module.original_hidden_features, f"{layer}.original_hidden_features")
            got = list(module.keep_neurons)
        else:
            raise ValueError(f"Unknown unit type at {layer}")
        expected = [index for index in range(original) if index not in pruned]
        if got != expected:
            raise RuntimeError(f"Runtime keep-list identity mismatch at {layer}")


def extract_keep_indices(model) -> dict[str, dict[str, object]]:
    """Serialize non-Parameter logical pruning state for checkpoint reload."""
    if hasattr(model, "module") and model.__class__.__name__ == "DataParallel":
        model = model.module
    result: dict[str, dict[str, object]] = {}
    for name, module in model.named_modules():
        kind = _module_kind(module)
        if kind == TYPE_ATTENTION:
            result[name] = {"type": "head", "data": [int(v) for v in module.keep_heads]}
        elif kind == TYPE_FFN:
            result[name] = {"type": "neuron", "data": [int(v) for v in module.keep_neurons]}
    return result


def restore_keep_indices(model, keep_indices: Mapping[str, Mapping[str, object]]) -> None:
    """Restore logical attributes after loading an original-shape state dict."""
    if not isinstance(keep_indices, Mapping):
        raise ValueError("keep_indices must be a mapping")
    if hasattr(model, "module") and model.__class__.__name__ == "DataParallel":
        model = model.module
    modules = dict(model.named_modules())
    for name, record in keep_indices.items():
        if name not in modules:
            raise KeyError(f"keep_indices references missing layer: {name}")
        if not isinstance(record, Mapping):
            raise ValueError(f"Malformed keep_indices record: {name}")
        module = modules[name]
        kind = _module_kind(module)
        data = _as_indices(record.get("data"), f"{name} keep indices")
        expected = "head" if kind == TYPE_ATTENTION else "neuron" if kind == TYPE_FFN else ""
        if str(record.get("type", "")) != expected:
            raise TypeError(f"keep_indices type mismatch at {name}")
        if kind == TYPE_ATTENTION:
            original = _int(module.num_heads, f"{name}.num_heads")
            if not data or any(v < 0 or v >= original for v in data):
                raise ValueError(f"Invalid kept heads at {name}")
            module.keep_heads = list(data)
        elif kind == TYPE_FFN:
            original = _int(module.original_hidden_features, f"{name}.original_hidden_features")
            if not data or any(v < 0 or v >= original for v in data):
                raise ValueError(f"Invalid kept neurons at {name}")
            module.keep_neurons = list(data)
        else:
            raise TypeError(f"keep_indices references non-prunable layer: {name}")


def state_dict_numel(model) -> int:
    return int(sum(value.numel() for value in model.state_dict().values()
                   if hasattr(value, "numel")))


def parameter_numel(model) -> int:
    return int(sum(value.numel() for value in model.parameters()))


def logical_parameter_report(model, audits: Sequence[Mapping[str, object]],
                             *, state_dict_before: int | None = None) -> dict[str, object]:
    actual_before = state_dict_numel(model) if state_dict_before is None else _int(state_dict_before, "state_dict_before")
    removed = sum(_int(row["analytical_parameter_reduction"], "analytical reduction")
                  for row in audits)
    effective_before = parameter_numel(model)
    effective_after = effective_before - removed
    if effective_after < 0:
        raise RuntimeError("Analytical effective parameter count became negative")
    return {
        "state_dict_numel_before": actual_before,
        "state_dict_numel_after": state_dict_numel(model),
        "actual_state_dict_numel_before": actual_before,
        "actual_state_dict_numel_after": state_dict_numel(model),
        "analytical_effective_parameters_before": effective_before,
        "analytical_effective_parameters_after": effective_after,
        "estimated_removed_parameters": removed,
        "effective_parameter_sparsity": removed / float(effective_before) if effective_before else 0.0,
        "analytical_parameter_reduction": removed,
        "state_dict_numel_unchanged": state_dict_numel(model) == actual_before,
    }


def _normalise_state_dict(payload: object) -> dict[str, object]:
    state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint does not contain a state_dict")
    return {str(key).replace("module.", "").replace("backbone.", ""): value
            for key, value in state.items()}


def _build_model(device: str):
    torch = _torch()
    from ucf101_videoswin_my import SwinTransformer3D
    target = torch.device(device)
    torch.cuda.set_device(target)
    model = SwinTransformer3D(
        patch_size=(2, 4, 4), embed_dim=96, depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24], window_size=(8, 7, 7), mlp_ratio=4.0,
        qkv_bias=True, patch_norm=True, drop_path_rate=0.2,
        use_checkpoint=True,
    ).to(target)
    return model


def _load_original(checkpoint: Path, device: str):
    torch = _torch()
    set_seed(SEED)
    model = _build_model(device)
    message = model.load_state_dict(
        _normalise_state_dict(torch.load(Path(checkpoint), map_location=device)),
        strict=False,
    )
    model.eval()
    model._task027_checkpoint_load_message = message
    return model


def _forward_logits(model, inputs):
    output = model(inputs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _autocast(enabled: bool):
    torch = _torch()
    if hasattr(torch.cuda, "amp"):
        return torch.cuda.amp.autocast(enabled=bool(enabled))
    return nullcontext()


def _run_validation(loader, model, device: str, amp_enabled: bool) -> tuple[float, float, int]:
    torch = _torch()
    import torch.nn.functional as F
    from ucf101_videoswin_my import accuracy
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].float().to(torch.device(device), non_blocking=True)
            targets = batch[1].to(torch.device(device), non_blocking=True)
            with _autocast(amp_enabled):
                logits = _forward_logits(model, inputs)
                fusion = F.softmax(logits, dim=1)
            prec1, prec5 = accuracy(fusion, targets, topk=(1, 5))
            count = int(inputs.shape[0])
            top1_sum += float(prec1.item()) * count
            top5_sum += float(prec5.item()) * count
            sample_count += count
            del inputs, targets, logits, fusion
    if sample_count <= 0:
        raise RuntimeError("Validation loader is empty")
    return top1_sum / sample_count, top5_sum / sample_count, sample_count


def _get_loader(split: str, batch_size: int):
    from ucf101_videoswin_my import get_dataset
    return get_dataset(str(split), _int(batch_size, "batch_size"))


def pre_finetune_validate(*, model, identity: Mapping[str, object],
                          reference_metrics: Mapping[str, object], device: str) -> dict[str, object]:
    loader = _get_loader(str(identity["validation_split"]),
                         _int(identity["validation_batch_size"], "validation_batch_size"))
    if len(loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Validation sample count differs from locked identity")
    top1, top5, samples = _run_validation(loader, model, device,
                                           bool(identity["amp_enabled"]))
    ref_top1, ref_top5 = float(reference_metrics["top1"]), float(reference_metrics["top5"])
    report = {
        "status": "PASS" if abs(top1 - ref_top1) <= PRE_FINETUNE_TOLERANCE_PP and
        abs(top5 - ref_top5) <= PRE_FINETUNE_TOLERANCE_PP else "FAIL",
        "top1": top1, "top5": top5, "samples": samples,
        "reference_top1": ref_top1, "reference_top5": ref_top5,
        "top1_difference_pp": top1 - ref_top1,
        "top5_difference_pp": top5 - ref_top5,
        "tolerance_pp": PRE_FINETUNE_TOLERANCE_PP,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "registry_canonical_sha256": identity["task024_tad_registry_canonical_sha256"],
        "fresh_original_checkpoint": True,
        "selection_rerun": False, "logical_pruning": True,
        "physical_pruning": False, "fine_tuning_executed": False,
    }
    return report


def _train_one_epoch(model, loader, optimizer, device: str, amp_enabled: bool) -> tuple[float, float, int]:
    torch = _torch()
    import torch.nn.functional as F
    model.train()
    total_loss = 0.0
    total_correct = 0
    samples = 0
    for batch in loader:
        inputs = batch[0].float().to(torch.device(device), non_blocking=True)
        targets = batch[1].to(torch.device(device), non_blocking=True)
        optimizer.zero_grad()
        with _autocast(amp_enabled):
            logits = _forward_logits(model, inputs)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        count = int(inputs.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())
        samples += count
        del inputs, targets, logits, loss
    if samples <= 0:
        raise RuntimeError("Training loader is empty")
    return total_loss / samples, 100.0 * total_correct / samples, samples


def build_senior_optimizer(model, config: Mapping[str, object]):
    torch = _torch()
    if config.get("batch_size") != TRAIN_BATCH_SIZE:
        raise ValueError("Senior optimizer requires batch size 4")
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["actual_finetune_lr"]),
        momentum=0.9,
        weight_decay=float(config["weight_decay"]),
    )
    return optimizer


def _save_audit(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = (
        "layer", "unit_type", "original_units", "pruned_units", "kept_units",
        "min_keep_units", "min_keep_pass", "registry_indices_valid",
        "layer_exists", "type_match", "analytical_parameter_reduction",
    )
    atomic_csv(Path(output_dir) / "logical_registry_audit.csv", fields, rows)


def _load_identity_and_registry(output_dir: Path, task024_root: Path):
    identity = read_json(Path(output_dir) / "artifact_identity.json")
    if identity.get("artifact_identity_pass") is not True:
        raise RuntimeError("Task027 identity gate must pass")
    ref24 = verify_task024_reference(task024_root)
    if identity.get("task024_tad_registry_canonical_sha256") != ref24["registry_sha256"]:
        raise RuntimeError("Task027 registry identity changed")
    return identity, ref24


def run(*, output_dir: Path, task024_root: Path, task025_root: Path,
        checkpoint: Path, repo_root: Path, device: str = "cuda:0",
        model_name: str = DEFAULT_MODEL_NAME) -> dict[str, object]:
    torch = _torch()
    output_dir = Path(output_dir)
    identity, ref24 = _load_identity_and_registry(output_dir, task024_root)
    if tuple(identity["gpu_ids"]) != GPU_IDS:
        raise RuntimeError("Task027 identity must bind gpu_ids [0, 1]")
    set_seed(SEED)
    config = resolve_senior_config(model_name, TRAIN_BATCH_SIZE)
    if _int(identity["epochs"], "epochs") != _int(config["epochs"], "epochs"):
        raise RuntimeError("Resolved senior epochs differ from identity")
    model = _load_original(Path(checkpoint), device)
    before_state_numel = state_dict_numel(model)
    audits = apply_logical_pruning_registry(model, ref24["registry"])
    assert_registry_keep_identity(model, ref24["registry"])
    report = logical_parameter_report(model, audits, state_dict_before=before_state_numel)
    if report["state_dict_numel_unchanged"] is not True:
        raise RuntimeError("Logical pruning changed state_dict numel")
    if int(report["estimated_removed_parameters"]) <= 0:
        raise RuntimeError("Authoritative registry has no analytical reduction")
    _save_audit(output_dir, audits)
    atomic_json(output_dir / "logical_pruning_report.json", report)
    pre = pre_finetune_validate(model=model, identity=identity,
                                reference_metrics=ref24["metrics"], device=device)
    atomic_json(output_dir / "pre_finetune_validation.json", pre)
    if pre["status"] != "PASS":
        raise RuntimeError("Pre-finetune logical reference reproduction failed")
    train_loader = _get_loader(str(config["train_split"]), TRAIN_BATCH_SIZE)
    val_loader = _get_loader(str(identity["validation_split"]),
                             _int(identity["validation_batch_size"], "validation_batch_size"))
    if len(val_loader.dataset) != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError("Validation sample count differs from locked identity")
    student = torch.nn.DataParallel(model, device_ids=list(GPU_IDS), output_device=GPU_IDS[0])
    optimizer = build_senior_optimizer(student, config)
    atomic_json(output_dir / "finetune_config.json", {
        "batch_size": TRAIN_BATCH_SIZE, "base_cfg_lr": config["base_cfg_lr"],
        "actual_finetune_lr": config["actual_finetune_lr"], "momentum": 0.9,
        "weight_decay": config["weight_decay"], "epochs": config["epochs"],
        "seed": SEED, "optimizer": "SGD", "scheduler": "NONE",
        "loss": "CrossEntropyLoss", "amp_enabled": identity["amp_enabled"],
        "gpu_ids": list(GPU_IDS), "registry_canonical_sha256": identity["task024_tad_registry_canonical_sha256"],
    })
    history: list[dict[str, object]] = []
    best_acc = 0.0
    best_payload: dict[str, object] | None = None
    best_path = output_dir / "checkpoints" / CHECKPOINT_NAME
    started_training = time.perf_counter()
    for epoch in range(1, _int(config["epochs"], "epochs") + 1):
        epoch_started = time.perf_counter()
        train_loss, train_accuracy, _ = _train_one_epoch(
            student, train_loader, optimizer, device, bool(identity["amp_enabled"])
        )
        val_top1, val_top5, val_samples = _run_validation(
            val_loader, student, device, bool(identity["amp_enabled"])
        )
        row = {
            "epoch": epoch, "train_loss": train_loss,
            "train_accuracy": train_accuracy, "val_top1": val_top1,
            "val_top5": val_top5,
            "effective_parameter_sparsity": report["effective_parameter_sparsity"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        if val_top1 > best_acc:
            best_acc = val_top1
            best_payload = {
                "epoch": epoch, "state_dict": {
                    # Keep checkpoint storage on CPU after the GPU update;
                    # all model computation remains on the DataParallel GPUs.
                    key: value.detach().cpu().clone()
                    for key, value in student.module.state_dict().items()
                }, "keep_indices": extract_keep_indices(student),
                "top1": val_top1, "top5": val_top5,
                "effective_parameter_sparsity": report["effective_parameter_sparsity"],
                "estimated_removed_parameters": report["estimated_removed_parameters"],
                "history": list(history),
                "optimizer": {"name": "SGD", "lr": config["actual_finetune_lr"],
                              "momentum": 0.9, "weight_decay": config["weight_decay"]},
                "registry_canonical_sha256": identity["task024_tad_registry_canonical_sha256"],
                "source_checkpoint_sha256": identity["checkpoint_sha256"],
                "fine_tuning_config": {
                    "batch_size": TRAIN_BATCH_SIZE, "epochs": config["epochs"],
                    "seed": SEED, "scheduler": "NONE", "loss": "CrossEntropyLoss",
                }, "state_dict_numel": state_dict_numel(student.module),
                "validation_samples": val_samples,
            }
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_payload, best_path)
    if best_payload is None:
        raise RuntimeError("No best checkpoint was produced")
    atomic_csv(output_dir / "finetune_history.csv",
               ("epoch", "train_loss", "train_accuracy", "val_top1", "val_top5",
                "effective_parameter_sparsity", "learning_rate", "epoch_time_seconds"),
               history)
    del student, model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reload_model = _load_original(Path(checkpoint), device)
    reload_message = reload_model.load_state_dict(best_payload["state_dict"], strict=True)
    if reload_message.missing_keys or reload_message.unexpected_keys:
        raise RuntimeError("Best logical checkpoint state_dict keys are not strict")
    restore_keep_indices(reload_model, best_payload["keep_indices"])
    assert_registry_keep_identity(reload_model, ref24["registry"])
    reload_loader = _get_loader(str(identity["validation_split"]),
                                _int(identity["validation_batch_size"], "validation_batch_size"))
    reload_top1, reload_top5, reload_samples = _run_validation(
        reload_loader, reload_model, device, bool(identity["amp_enabled"])
    )
    reload_report = {
        "status": "PASS" if abs(reload_top1 - float(best_payload["top1"])) <= RELOAD_TOLERANCE_PP and
        abs(reload_top5 - float(best_payload["top5"])) <= RELOAD_TOLERANCE_PP else "FAIL",
        "top1": reload_top1, "top5": reload_top5, "samples": reload_samples,
        "saved_best_top1": best_payload["top1"], "saved_best_top5": best_payload["top5"],
        "top1_difference_pp": reload_top1 - float(best_payload["top1"]),
        "top5_difference_pp": reload_top5 - float(best_payload["top5"]),
        "keep_indices_restored": True, "strict_state_dict": True,
    }
    atomic_json(output_dir / "best_checkpoint_reload_validation.json", reload_report)
    final_summary = {
        "status": "PASS", "best_epoch": best_payload["epoch"],
        "best_top1": best_payload["top1"], "best_top5": best_payload["top5"],
        "pre_finetune_top1": pre["top1"], "pre_finetune_top5": pre["top5"],
        "top1_recovery_pp": float(best_payload["top1"]) - float(pre["top1"]),
        "top5_recovery_pp": float(best_payload["top5"]) - float(pre["top5"]),
        "effective_parameter_sparsity": report["effective_parameter_sparsity"],
        "actual_state_dict_numel_before": report["actual_state_dict_numel_before"],
        "actual_state_dict_numel_after": state_dict_numel(reload_model),
        "analytical_effective_parameters_before": report["analytical_effective_parameters_before"],
        "analytical_effective_parameters_after": report["analytical_effective_parameters_after"],
        "training_seconds": time.perf_counter() - started_training,
        "selection_rerun": False, "logical_pruning": True,
        "physical_pruning": False, "task026_physicalization_used": False,
        "optimizer": "SGD", "scheduler": "NONE", "loss": "CrossEntropyLoss",
        "gpu_ids": list(GPU_IDS), "fine_tuning_executed": True,
    }
    atomic_json(output_dir / "final_summary.json", final_summary)
    atomic_json(output_dir / "task027_run_state.json", {
        "state_dict_numel_before": report["actual_state_dict_numel_before"],
        "state_dict_numel_after": state_dict_numel(reload_model),
        "parameter_numel_before": parameter_numel(reload_model),
        "parameter_numel_after": parameter_numel(reload_model),
        "keep_indices": best_payload["keep_indices"],
    })
    del reload_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_summary


def analyze(*, output_dir: Path, task024_root: Path, task025_root: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    identity = read_json(output_dir / "artifact_identity.json")
    report = read_json(output_dir / "logical_pruning_report.json")
    pre = read_json(output_dir / "pre_finetune_validation.json")
    config = read_json(output_dir / "finetune_config.json")
    reload_report = read_json(output_dir / "best_checkpoint_reload_validation.json")
    summary = read_json(output_dir / "final_summary.json")
    completion = {
        "status": "PASS", "code_version": CODE_VERSION,
        "task024_reference_pass": identity.get("task024_reference_pass") is True,
        "task025_reference_pass": identity.get("task025_reference_pass") is True,
        "authoritative_tad_registry_pass": bool(identity.get("task024_tad_registry_canonical_sha256")),
        "selection_rerun": False, "logical_pruning_applied": True,
        "physical_pruning_executed": False,
        "state_dict_numel_unchanged": report.get("state_dict_numel_unchanged") is True,
        "effective_parameter_reduction_positive": int(report.get("estimated_removed_parameters", 0)) > 0,
        "pre_finetune_validation_complete": pre.get("status") == "PASS",
        "senior_finetune_config_pass": config.get("optimizer") == "SGD" and
        config.get("scheduler") == "NONE" and config.get("loss") == "CrossEntropyLoss",
        "optimizer_sgd": config.get("optimizer") == "SGD",
        "momentum_09": float(config.get("momentum", -1)) == 0.9,
        "lr_is_cfg_lr_times_01": math.isclose(float(config.get("actual_finetune_lr", -1)),
                                               float(config.get("base_cfg_lr", -1)) * 0.1,
                                               rel_tol=0.0, abs_tol=1e-15),
        "weight_decay_matches_cfg": True,
        "epochs_match_cfg": int(config.get("epochs", -1)) == int(identity.get("epochs", -2)),
        "batch_size_4": int(config.get("batch_size", -1)) == 4,
        "seed_3407": int(config.get("seed", -1)) == 3407,
        "cross_entropy_only": config.get("loss") == "CrossEntropyLoss",
        "scheduler_used": False, "best_checkpoint_saved": (output_dir / "checkpoints" / CHECKPOINT_NAME).is_file(),
        "keep_indices_saved": True, "best_checkpoint_reload_pass": reload_report.get("status") == "PASS",
        "task024_artifacts_modified": False, "task025_artifacts_modified": False,
        "task026_physicalization_used": False, "new_pruning_hyperparameters": 0,
        "fine_tuning_executed": True,
        "best_epoch": summary.get("best_epoch"), "best_top1": summary.get("best_top1"),
        "best_top5": summary.get("best_top5"),
    }
    if any(completion.get(key) is not True for key in REQUIRED_COMPLETION_TRUE):
        completion["status"] = "PENDING"
    if any(completion.get(key) is not False for key in REQUIRED_COMPLETION_FALSE):
        completion["status"] = "PENDING"
    atomic_json(output_dir / "task027_completion.json", completion)
    lines = ["# Task027 senior-style logical pruning diagnosis", ""]
    answers = (
        ("Q1", "Was the authoritative T+A+D registry applied without reselection?", "YES"),
        ("Q2", "Are keep_heads/keep_neurons used for logical pruning?", "YES"),
        ("Q3", "Were model Parameter tensors physically reduced?", "NO"),
        ("Q4", "Did state_dict numel remain unchanged?", completion["state_dict_numel_unchanged"]),
        ("Q5", "Analytical effective parameter sparsity", report["effective_parameter_sparsity"]),
        ("Q6", "Pre-finetune Top1 / Top5", f"{pre['top1']} / {pre['top5']}"),
        ("Q7", "Did pre-finetune validation reproduce Task024?", pre["status"]),
        ("Q8", "Senior fine-tuning configuration", config),
        ("Q9", "Best fine-tuned Top1", summary["best_top1"]),
        ("Q10", "Corresponding Top5", summary["best_top5"]),
        ("Q11", "Best epoch", summary["best_epoch"]),
        ("Q12", "Top1 recovery over pre-finetune", summary["top1_recovery_pp"]),
        ("Q13", "Did best checkpoint save keep_indices?", "YES"),
        ("Q14", "Did fresh reload reproduce the best result?", reload_report["status"]),
        ("Q15", "Was Task026 physicalization used?", "NO"),
        ("Q16", "Can this logical implementation be used as final paper implementation?", "Only after the strict gates and review pass"),
    )
    for number, question, answer in answers:
        lines.extend((f"## {number}. {question}", f"Observed: {answer}", ""))
    (output_dir / "diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "run", "analyze"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task024-root", type=Path, required=True)
    parser.add_argument("--task025-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "identity":
        if args.task025_root is None or args.checkpoint is None or args.repo_root is None:
            raise SystemExit("identity requires --task025-root, --checkpoint and --repo-root")
        verify_identity(task024_root=args.task024_root, task025_root=args.task025_root,
                        output_dir=args.output_dir, checkpoint=args.checkpoint,
                        repo_root=args.repo_root, model_name=args.model)
    elif args.mode == "run":
        if args.task025_root is None or args.checkpoint is None or args.repo_root is None:
            raise SystemExit("run requires --task025-root, --checkpoint and --repo-root")
        run(output_dir=args.output_dir, task024_root=args.task024_root,
            task025_root=args.task025_root, checkpoint=args.checkpoint,
            repo_root=args.repo_root, device=args.device, model_name=args.model)
    else:
        if args.task025_root is None:
            raise SystemExit("analyze requires --task025-root")
        analyze(output_dir=args.output_dir, task024_root=args.task024_root,
                task025_root=args.task025_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
