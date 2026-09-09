"""Task038 data, checkpoint, validation, and fine-tuning protocol."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

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
) -> DataLoader:
    dataset = build_dataset(split_path, indices)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def balanced_n9_indices(split_path: str = DEFAULT_VAL_LIST) -> list[dict[str, Any]]:
    by_class: dict[int, list[tuple[int, str]]] = {}
    with open(split_path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 3:
                continue
            label = int(fields[2])
            by_class.setdefault(label, []).append((index, fields[0]))
    selected: list[dict[str, Any]] = []
    for label in sorted(by_class)[:3]:
        for index, name in by_class[label][:3]:
            selected.append(
                {
                    "split": str(Path(split_path).resolve()),
                    "split_index": index,
                    "video_id": name,
                    "label": label,
                }
            )
    if len(selected) != 9 or len({row["label"] for row in selected}) != 3:
        raise RuntimeError("deterministic balanced N=9 selection failed")
    return selected


def accuracy(logits: torch.Tensor, target: torch.Tensor, topk=(1, 5)):
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return [correct[:, :k].any(1).float().sum().item() for k in topk]


def validate(
    model: nn.Module, loader: Iterable, device: torch.device
) -> dict[str, Any]:
    model.eval()
    total = top1 = top5 = 0
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
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
    result = validate(model, build_loader(val_list, 4, False), device)
    result["checkpoint"] = identity
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def fine_tune(
    model: nn.Module,
    checkpoint: str | Path,
    device_ids: Sequence[int],
    output_dir: str | Path,
    epochs: int,
    base_lr: float = 0.005,
    weight_decay: float = 1e-5,
    batch_size: int = 4,
    checkpoint_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(3407)
    identity = checkpoint_identity or load_checkpoint_identity(model, checkpoint, torch.device("cpu"), allow_runtime_mask_buffers=True)
    train_list, val_list, _ = data_paths()
    workers = int(os.environ.get("TASK038_WORKERS", "2"))
    train_loader = build_loader(train_list, batch_size, True, workers=workers)
    val_loader = build_loader(val_list, batch_size, False, workers=workers)
    model = model.to(torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available() and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=list(device_ids))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr * 0.1, momentum=0.9, weight_decay=weight_decay
    )
    best = None
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        with tqdm.tqdm(train_loader, ncols=0) as progress:
            for batch in progress:
                x, target = batch[0].float().to(next(model.parameters()).device), batch[1].long().to(next(model.parameters()).device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise RuntimeError("fine-tuning loss is non-finite")
                loss.backward()
                optimizer.step()
                predictions = torch.argmax(logits, dim=1)
                correct = int((predictions == target).sum().item())
                progress.set_description(f"Task038 F3 epoch {epoch:03d}/{int(epochs):03d} | Loss: {loss.item():.4f} | Acc: {correct / max(int(target.numel()), 1):.4f}")
                loss_sum += float(loss.item()) * int(target.numel())
                seen += int(target.numel())
        validation = validate(model, val_loader, next(model.parameters()).device)
        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), **validation}
        history.append(row)
        if best is None or row["top1"] > best["top1"]:
            best = row
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"state_dict": state, "epoch": epoch, "validation": row}, out / "best.ckpt")
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if best is None:
        raise RuntimeError("fine-tuning produced no validation result")
    result = {"protocol": {"seed": 3407, "criterion": "cross_entropy", "optimizer": "SGD", "lr": base_lr * 0.1, "momentum": 0.9, "weight_decay": weight_decay, "amp": False, "scheduler": "NONE"}, "checkpoint": identity, "best": best, "history": history}
    (out / "finetune_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
