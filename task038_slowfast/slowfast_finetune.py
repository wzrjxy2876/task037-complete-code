"""Task038 data, checkpoint, validation, and fine-tuning protocol."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
import tqdm

from .slowfast_model_task038 import load_checkpoint_identity


DEFAULT_TRAIN_LIST = "/data/jixinye25/UCF101_Frame/train_rgb_split1.txt"
DEFAULT_VAL_LIST = "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
DEFAULT_FRAME_ROOT = "/data/jixinye25/UCF101_Frame/frames"
DEFAULT_CHECKPOINT = "/home/jixinye25/jxy_work1/pretrained/slowfast-teacher-ucf101.ckpt"
AUTHORITATIVE_UTILS_PATH = Path("/home/jixinye25/jxy_work1/Code/utils.py")


def _authoritative_config_path() -> tuple[Path, Path, str]:
    """Resolve CONFIG_PATHS['slowfast_resnet101'] from the real utils.py."""
    utils_path = AUTHORITATIVE_UTILS_PATH
    if not utils_path.is_file():
        raise RuntimeError(f"authoritative utils.py missing: {utils_path}")
    source = utils_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(utils_path))
    config_root = None
    config_filename = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if "CONFIG_ROOT" in names and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                config_root = node.value.value
        if "CONFIG_PATHS" in names and isinstance(node.value, ast.Dict):
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or key.value != "slowfast_resnet101":
                    continue
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "join"
                    and len(value.args) == 2
                    and isinstance(value.args[0], ast.Name)
                    and value.args[0].id == "CONFIG_ROOT"
                    and isinstance(value.args[1], ast.Constant)
                    and isinstance(value.args[1].value, str)
                ):
                    config_filename = value.args[1].value
    if config_root is None or config_filename is None:
        raise RuntimeError(
            "could not resolve CONFIG_PATHS['slowfast_resnet101'] from "
            f"{utils_path}"
        )
    return utils_path, (utils_path.parent / config_root / config_filename).resolve(), config_filename


def resolve_authoritative_slowfast_config() -> dict[str, Any]:
    """Read and validate the historical SlowFast training configuration."""
    utils_path, config_path, config_filename = _authoritative_config_path()
    if not config_path.is_file():
        raise RuntimeError(f"resolved SlowFast config missing: {config_path}")
    config_text = config_path.read_text(encoding="utf-8")
    train_indent = None
    train_values: dict[str, str] = {}
    in_train = False
    for line in config_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "TRAIN:":
            in_train = True
            train_indent = indent
            continue
        if in_train and train_indent is not None and indent <= train_indent:
            break
        if in_train:
            field = stripped.split("#", 1)[0].strip()
            if ":" not in field:
                continue
            key, value = field.split(":", 1)
            if key in {"LR", "W_DECAY", "EPOCH_NUM"}:
                train_values[key] = value.strip().strip("'\"")
    missing = {"LR", "W_DECAY", "EPOCH_NUM"} - set(train_values)
    if missing:
        raise RuntimeError(f"authoritative config TRAIN fields missing: {sorted(missing)}")
    base_lr = float(train_values["LR"])
    weight_decay = float(train_values["W_DECAY"])
    epochs = int(train_values["EPOCH_NUM"])
    expected = (0.005, 1e-5, 100)
    if (base_lr, weight_decay, epochs) != expected:
        raise RuntimeError(
            "authoritative SlowFast config values differ from the formal Task038 gate: "
            f"LR={base_lr}, W_DECAY={weight_decay}, EPOCH_NUM={epochs}"
        )
    return {
        "config_key": "slowfast_resnet101",
        "utils_path": str(utils_path),
        "utils_sha256": hashlib.sha256(utils_path.read_bytes()).hexdigest(),
        "config_path": str(config_path),
        "config_filename": config_filename,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_lr": base_lr,
        "effective_ft_lr": base_lr * 0.1,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "train_lr": base_lr,
        "train_weight_decay": weight_decay,
        "train_epoch_num": epochs,
        "resolution": "CONFIG_PATHS['slowfast_resnet101'] resolved from authoritative Code/utils.py",
    }


def set_seed(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_paths() -> tuple[str, str, str]:
    return (
        os.environ.get("UCF101_TRAIN_LIST", DEFAULT_TRAIN_LIST),
        os.environ.get("UCF101_VAL_LIST", DEFAULT_VAL_LIST),
        os.environ.get("UCF101_FRAME_ROOT", DEFAULT_FRAME_ROOT),
    )


def _dataset_module():
    from PIL import Image
    if not hasattr(Image, "LINEAR"):
        Image.LINEAR = getattr(Image, "BILINEAR", 2)
    six = __import__('torch._six', fromlist=['x'])
    if not hasattr(six, "int_classes"):
        six.int_classes = (int,)
    if not hasattr(six, "string_classes"):
        six.string_classes = (str,)
    runtime = Path(__file__).resolve().parent.parent / "src" / "lgfr_runtime"
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    module = importlib.import_module("dataset.ucf101")
    _utils = importlib.import_module("utils")
    _train, _val, root = data_paths()
    module.UCF_DATA_ROOT = root
    _utils.UCF_DATA_ROOT = root
    return module


def build_dataset(split_path: str, indices: Sequence[int] | None = None):
    module = _dataset_module()
    spatial, temporal = module.test_transform()
    dataset = module.attack_ucf101(
        split_path, spatial_transform=spatial, temporal_transform=temporal
    )
    if indices is not None:
        dataset = Subset(dataset, list(indices))
    return dataset


def build_loader(
    split_path: str,
    batch_size: int = 4,
    shuffle: bool = False,
    indices: Sequence[int] | None = None,
    workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    dataset = build_dataset(split_path, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def sample_identity_payload(
    split_path: str, selected: Sequence[dict[str, Any]], seed: int
) -> dict[str, Any]:
    ordered = [
        {
            "split": str(row["split"]),
            "split_index": int(row["split_index"]),
            "video_id": str(row["video_id"]),
            "label": int(row["label"]),
        }
        for row in selected
    ]
    canonical = json.dumps(
        ordered, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "seed": int(seed),
        "sample_count": len(ordered),
        "class_labels": sorted({int(row["label"]) for row in ordered}),
        "annotation_entries": ordered,
        "indices": [int(row["split_index"]) for row in ordered],
        "ordered_sample_list": ordered,
        "selection_algorithm": (
            "random.Random(seed); sorted eligible class ids; "
            "sorted(rng.sample(classes, 3)); per-class rng.sample(3); "
            "selected rows sorted by split_index"
        ),
        "sample_identity_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def balanced_n9_indices(
    split_path: str = DEFAULT_VAL_LIST,
    seed: int = 3407,
    num_classes: int = 3,
    videos_per_class: int = 3,
) -> list[dict[str, Any]]:
    by_class: dict[int, list[tuple[int, str]]] = {}
    with open(split_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 3:
                continue
            label = int(fields[2])
            by_class.setdefault(label, []).append((index, fields[0]))
    eligible = sorted(
        label for label, rows in by_class.items() if len(rows) >= videos_per_class
    )
    if len(eligible) < num_classes:
        raise RuntimeError("not enough eligible classes for balanced probe")
    rng = random.Random(int(seed))
    chosen_classes = sorted(rng.sample(eligible, num_classes))
    selected: list[dict[str, Any]] = []
    for label in chosen_classes:
        chosen = rng.sample(by_class[label], videos_per_class)
        for index, name in sorted(chosen, key=lambda row: row[0]):
            selected.append(
                {
                    "split": str(Path(split_path).resolve()),
                    "split_index": int(index),
                    "video_id": name,
                    "label": int(label),
                }
            )
    if (
        len(selected) != num_classes * videos_per_class
        or len({row["label"] for row in selected}) != num_classes
    ):
        raise RuntimeError("deterministic balanced N=9 selection failed")
    return selected


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, :k].any(1).float().sum().item() for k in topk]


def validate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    description: str = "Validating",
) -> dict[str, Any]:
    model.eval()
    total = top1 = top5 = 0
    losses: list[float] = []
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, ncols=0, desc=description):
            x, target = batch[0].to(device).float(), batch[1].to(device).long()
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise RuntimeError("validation logits are non-finite")
            loss = F.cross_entropy(logits, target)
            a1, a5 = accuracy(logits, target)
            total += int(target.numel())
            top1 += a1
            top5 += a5
            losses.append(float(loss.item()))
    if total == 0:
        raise RuntimeError("validation loader has zero samples")
    return {
        "sample_count": total,
        "top1": top1 / total,
        "top5": top5 / total,
        "mean_cross_entropy": float(np.mean(losses)),
        "finite": True,
    }


def baseline_validation(
    model: nn.Module, checkpoint: str | Path, device: torch.device, output: str | Path
) -> dict[str, Any]:
    identity = load_checkpoint_identity(model, checkpoint, device)
    _, val_list, _ = data_paths()
    result = validate(
        model, build_loader(val_list, 4, False), device, "Task038 baseline validation"
    )
    result["checkpoint"] = identity
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def fine_tune(
    model: nn.Module,
    checkpoint: str | Path,
    device_ids: Sequence[int],
    output_dir: str | Path,
    epochs: int,
    base_lr: float | None = None,
    weight_decay: float | None = None,
    batch_size: int = 16,
    checkpoint_identity: dict[str, Any] | None = None,
    registry_sha256: str | None = None,
    sequence_sha256: str | None = None,
    user_batch_size_override: bool = True,
    authoritative_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if int(batch_size) != 16:
        raise ValueError("Task038 authoritative formal fine-tuning batch_size is 16")
    resolved_config = dict(authoritative_config or resolve_authoritative_slowfast_config())
    resolved_base_lr = float(resolved_config["base_lr"])
    resolved_weight_decay = float(resolved_config["weight_decay"])
    resolved_epochs = int(resolved_config["epochs"])
    if base_lr is None:
        base_lr = resolved_base_lr
    elif float(base_lr) != resolved_base_lr:
        raise ValueError("base_lr does not match authoritative SlowFast config")
    if weight_decay is None:
        weight_decay = resolved_weight_decay
    elif float(weight_decay) != resolved_weight_decay:
        raise ValueError("weight_decay does not match authoritative SlowFast config")
    if int(epochs) != resolved_epochs:
        raise ValueError("epochs does not match authoritative SlowFast config")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(3407)
    identity = checkpoint_identity or load_checkpoint_identity(
        model, checkpoint, torch.device("cpu"), allow_runtime_mask_buffers=True
    )
    train_list, val_list, _ = data_paths()
    workers = int(os.environ.get("TASK038_WORKERS", "9"))
    train_loader = build_loader(
        train_list, batch_size, True, workers=workers, drop_last=True
    )
    val_loader = build_loader(
        val_list, batch_size, True, workers=workers, drop_last=True
    )
    device = torch.device(
        f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=list(device_ids))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("fine-tuning has no trainable parameters")
    optimizer = torch.optim.SGD(
        trainable, lr=base_lr * 0.1, momentum=0.9, weight_decay=weight_decay
    )
    protocol = {
        "seed": 3407,
        "criterion": "cross_entropy",
        "optimizer": "SGD",
        "lr": base_lr * 0.1,
        "momentum": 0.9,
        "weight_decay": weight_decay,
        "amp": False,
        "scheduler": "NONE",
        "batch_size": int(batch_size),
        "myslowfast_default_batch_size": 4,
        "user_batch_size_override": bool(user_batch_size_override),
        "workers": workers,
        "train_shuffle": True,
        "val_shuffle": True,
        "drop_last": True,
        "device_ids": list(device_ids),
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    config = {
        "schema": "task038_finetune_config_v2",
        "architecture": "slowfast_16x8_resnet101_kinetics400",
        "protocol": protocol,
        "checkpoint": identity,
        "registry_sha256": registry_sha256,
        "sequence_sha256": sequence_sha256,
        "authoritative_config": resolved_config,
        "source_protocol": (
            "myslowfast.py helper semantics retained; formal batch_size=16 "
            "is an explicit user override of its legacy default=4"
        ),
    }
    (out / "finetune_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def state_dict():
        return (
            model.module.state_dict()
            if isinstance(model, nn.DataParallel)
            else model.state_dict()
        )

    best = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        with tqdm.tqdm(
            train_loader, ncols=0, desc=f"Task038 F3 epoch {epoch:03d}/{int(epochs):03d}"
        ) as progress:
            for batch in progress:
                x = batch[0].float().to(next(model.parameters()).device, non_blocking=True)
                target = batch[1].long().to(next(model.parameters()).device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("fine-tuning loss is non-finite")
                loss.backward()
                optimizer.step()
                predictions = torch.argmax(logits, dim=1)
                correct = int((predictions == target).sum().item())
                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{correct / max(int(target.numel()), 1):.4f}",
                )
                loss_sum += float(loss.item()) * int(target.numel())
                seen += int(target.numel())
        validation = validate(
            model,
            val_loader,
            next(model.parameters()).device,
            f"Task038 F3 validation {epoch:03d}/{int(epochs):03d}",
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            **validation,
        }
        history.append(row)
        common_payload = {
            "epoch": epoch,
            "model": state_dict(),
            "state_dict": state_dict(),
            "optimizer": optimizer.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": None if best is None else best["top1"],
            "history": history,
            "registry_sha256": registry_sha256,
            "sequence_sha256": sequence_sha256,
            "protocol": protocol,
        }
        torch.save(common_payload, out / "slowfast_Task038_F3_latest.pth")
        if best is None or row["top1"] > best["top1"]:
            best = row
            best_payload = dict(common_payload)
            best_payload["best_acc"] = best["top1"]
            best_payload["best_validation"] = best
            torch.save(best_payload, out / "slowfast_Task038_F3_best.pth")
        (out / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if best is None:
        raise RuntimeError("fine-tuning produced no validation result")
    result = {
        "protocol": protocol,
        "checkpoint": identity,
        "registry_sha256": registry_sha256,
        "sequence_sha256": sequence_sha256,
        "best": best,
        "history": history,
        "best_checkpoint": str(out / "slowfast_Task038_F3_best.pth"),
        "latest_checkpoint": str(out / "slowfast_Task038_F3_latest.pth"),
    }
    (out / "finetune_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
