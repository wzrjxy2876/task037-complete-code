from __future__ import annotations

import argparse
import os
import random
import sys
import time
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


try:
    import PIL
    import PIL.Image as PIL_Image

    if not hasattr(PIL_Image, "LINEAR"):
        resampling = getattr(PIL_Image, "Resampling", None)
        PIL_Image.LINEAR = resampling.BILINEAR if resampling else 2
        PIL_Image.BILINEAR = resampling.BILINEAR if resampling else 2
        PIL_Image.BICUBIC = resampling.BICUBIC if resampling else 3
        PIL_Image.NEAREST = resampling.NEAREST if resampling else 0
        PIL.Image = PIL_Image
        sys.modules["PIL.Image"] = PIL_Image
except ImportError:
    pass

try:
    import torch._six
except ImportError:
    torch._six = ModuleType("torch._six")
    sys.modules["torch._six"] = torch._six
if not hasattr(torch._six, "int_classes"):
    torch._six.int_classes = (int,)
if not hasattr(torch._six, "string_classes"):
    torch._six.string_classes = (str,)


from dataset.ucf101 import get_dataset
from MC import InteractionPruner, SwinTransformer3D
from utils import CONFIG_PATHS, OPT_PATH, get_cfg_custom


PRETRAINED_PATH = "/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt"
TRAIN_LIST = "/data/jixinye25/UCF101_Frame/train_rgb_split1.txt"
VAL_LIST = "/data/jixinye25/UCF101_Frame/val_rgb_split1.txt"
CHECKPOINT_PATH = "/data/jixinye25/AAAwork1/outlog/swin_transformer/"
TRAIN_STATE_PATH = "/data/jixinye25/AAAwork1/outlog/swin_transformer/"


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, count=1):
        self.val = value
        self.sum += value * count
        self.count += count
        self.avg = self.sum / max(self.count, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Descriptor-BMS pruning with intra-cluster LG-FRF redundancy"
        )
    )
    parser.add_argument("--gpu", type=str, default="0", help="Visible CUDA index list")
    parser.add_argument("--batch_size", type=int, default=4, metavar="N")
    parser.add_argument("--model", type=str, default="swintrans")
    parser.add_argument("--file_prefix", type=str, default="")
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--gamma_decay", type=float, default=0.5)
    parser.add_argument("--min_keep_ratio", type=float, default=0.1)
    parser.add_argument("--calib_batches", type=int, default=10)
    parser.add_argument(
        "--analyze_functional_redundancy",
        action="store_true",
        help="Export LGFR representatives and protection metadata without pruning",
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    args.adv_path = os.path.join(
        OPT_PATH, f"UCF-{args.model}{args.file_prefix}"
    )
    os.makedirs(args.adv_path, exist_ok=True)
    return args


def parse_gpu_ids(gpu_argument):
    ids = [int(item.strip()) for item in gpu_argument.split(",") if item.strip()]
    if not ids:
        raise ValueError("--gpu must contain at least one CUDA index")
    return ids


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def accuracy(output, target, topk=(1,)):
    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)
    _, prediction = output.topk(maxk, dim=1, largest=True, sorted=True)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))
    results = []
    for requested_k in topk:
        k = min(requested_k, output.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / batch_size))
    return results


def run_one_epoch(epoch, net, optimizer, data_loader, device):
    net.train()
    total_loss = 0.0
    total_correct = 0
    with tqdm.tqdm(data_loader, ncols=0) as progress:
        for videos, targets, _ in progress:
            videos = videos.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = net(videos)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            progress.set_description(
                f"Epoch {epoch} | Loss {loss.item():.4f} | "
                f"Acc {(logits.argmax(dim=1) == targets).float().mean().item():.4f}"
            )
    average_loss = total_loss / max(len(data_loader), 1)
    average_accuracy = total_correct / max(len(data_loader.dataset), 1)
    return average_loss, average_accuracy


def validate_rgb(val_loader, net, top1, top5, device):
    net.eval()
    with torch.no_grad():
        for videos, targets, _ in tqdm.tqdm(
            val_loader, desc="Validating", ncols=0
        ):
            videos = videos.float().to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            output, _ = net(videos)
            precision1, precision5 = accuracy(output, targets, topk=(1, 5))
            top1.update(precision1.item(), videos.size(0))
            top5.update(precision5.item(), videos.size(0))


def get_rng_states():
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def resume_training(resume_epoch, model, optimizer, device):
    start_epoch = 1
    if resume_epoch <= 0:
        return start_epoch
    start_epoch = resume_epoch + 1
    model_path = os.path.join(CHECKPOINT_PATH, f"checkpoint-{resume_epoch}.ckpt")
    train_path = os.path.join(
        TRAIN_STATE_PATH, f"checkpoint-{resume_epoch}_optimizer.ckpt"
    )
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(torch.load(model_path, map_location=device))
    optimizer_state = torch.load(train_path, map_location=device)
    optimizer.load_state_dict(optimizer_state["optimizer"])
    return start_epoch


def extract_keep_indices(model):
    indices = {}
    target_model = model.module if isinstance(model, nn.DataParallel) else model
    for name, module in target_model.named_modules():
        if hasattr(module, "keep_heads") and module.keep_heads is not None:
            indices[name] = {"type": "head", "data": module.keep_heads}
        elif hasattr(module, "keep_neurons") and module.keep_neurons is not None:
            indices[name] = {"type": "neuron", "data": module.keep_neurons}
    return indices


def _normalize_checkpoint_key(key):
    normalized = key
    for prefix in ("module.", "backbone."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


def load_compatible_checkpoint(model, checkpoint_path, device, logger):
    if not os.path.exists(checkpoint_path):
        logger(f"Warning: pretrained checkpoint not found: {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    normalized = {
        _normalize_checkpoint_key(key): value for key, value in state_dict.items()
    }
    model_state = model.state_dict()
    compatible = {}
    shape_mismatches = []
    unexpected = []
    for key, value in normalized.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
        else:
            compatible[key] = value
    load_result = model.load_state_dict(compatible, strict=False)
    non_head_missing = [
        key for key in load_result.missing_keys if not key.startswith("cls_head.")
    ]
    logger(f"successfully loaded: {len(compatible)} tensors")
    logger(f"shape mismatches: {shape_mismatches}")
    logger(f"unexpected: {sorted(set(unexpected + list(load_result.unexpected_keys)))}")
    logger(f"non-head missing: {non_head_missing}")


def smoke_test(model, loader, device, logger):
    model.eval()
    batch = next(iter(loader))
    videos = batch[0] if isinstance(batch, (list, tuple)) else batch
    videos = videos[:1].float().to(device, non_blocking=True)
    with torch.no_grad():
        output, _ = model(videos)
    if output.ndim != 2 or output.shape[0] != 1:
        raise AssertionError(f"unexpected smoke-test output shape: {tuple(output.shape)}")
    logger(f"smoke test output shape: {tuple(output.shape)}")


def main():
    args = parse_args()
    set_seed(3407)
    gpu_ids = parse_gpu_ids(args.gpu)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cpu")

    log_name = f"prune_report_{args.model}_{time.strftime('%m%d_%H%M')}.txt"
    log_path = os.path.join(args.adv_path, log_name)

    def logger(message):
        print(message)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")

    logger(
        f"Task start: {args.model} | device={device} | "
        "clustering=descriptor | lgfr_role=function_representative_protection"
    )
    student = SwinTransformer3D(
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        patch_norm=True,
        drop_path_rate=0.2,
    ).to(device)
    load_compatible_checkpoint(student, PRETRAINED_PATH, device, logger)

    cfg = get_cfg_custom(CONFIG_PATHS[args.model], args.batch_size)
    train_loader = get_dataset(TRAIN_LIST, args.batch_size)
    val_loader = get_dataset(VAL_LIST, args.batch_size)

    pruner = InteractionPruner(
        student,
        target_sparsity=args.sparsity,
        sigma=args.sigma,
        gamma_decay=args.gamma_decay,
        min_keep_ratio=args.min_keep_ratio,
    )
    logger("Starting calibration and online frame-relation collection")
    pruner.run_calibration(
        train_loader, device=device, num_batches=args.calib_batches
    )
    analysis_dir = os.path.join(
        args.adv_path,
        "function_representative_protection_analysis",
    )
    analysis = pruner.analyze_functional_redundancy(output_dir=analysis_dir)
    if args.analyze_functional_redundancy:
        logger(
            "Analysis-only mode complete: no pruning, fine-tuning, or model "
            f"mutation was performed. Results: {analysis_dir}"
        )
        return

    pruner.prune(prepared_analysis=analysis)
    actual_report = student.get_detailed_pruning_report()
    logger(
        f"Pruning complete: target={args.sparsity:.2%}, "
        f"estimated parameter reduction={actual_report['sparsity']:.2%}"
    )
    smoke_test(student, val_loader, device, logger)

    if torch.cuda.is_available() and len(gpu_ids) > 1:
        student = nn.DataParallel(student, device_ids=gpu_ids)
    optimizer = torch.optim.SGD(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=cfg.CONFIG.TRAIN.LR * 0.1,
        momentum=0.9,
        weight_decay=cfg.CONFIG.TRAIN.W_DECAY,
    )

    best_accuracy = 0.0
    training_history = []
    for epoch in range(1, cfg.CONFIG.TRAIN.EPOCH_NUM + 1):
        train_loss, train_accuracy = run_one_epoch(
            epoch, student, optimizer, train_loader, device
        )
        top1 = AverageMeter()
        top5 = AverageMeter()
        validate_rgb(val_loader, student, top1, top5, device)
        current_sparsity = actual_report["sparsity"]
        logger(
            f"Epoch [{epoch:02d}] | loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | Top1={top1.avg:.3f}% | "
            f"Top5={top5.avg:.3f}% | sparsity={current_sparsity:.2%}"
        )
        training_history.append(
            {
                "epoch": epoch,
                "top1": top1.avg,
                "top5": top5.avg,
                "sparsity": current_sparsity,
            }
        )
        if top1.avg > best_accuracy:
            best_accuracy = top1.avg
            target_model = (
                student.module if isinstance(student, nn.DataParallel) else student
            )
            save_checkpoint = {
                "epoch": epoch,
                "state_dict": target_model.state_dict(),
                "keep_indices": extract_keep_indices(student),
                "top1": top1.avg,
                "top5": top5.avg,
                "sparsity": current_sparsity,
                "history": training_history,
                "optimizer": optimizer.state_dict(),
                "rng_state": get_rng_states(),
            }
            save_path = os.path.join(args.adv_path, "swin_pruned_best.pth")
            torch.save(save_checkpoint, save_path)
            logger(
                f"Saved best model at epoch {epoch}: "
                f"Top1={top1.avg:.2f}% | Top5={top5.avg:.2f}%"
            )
    logger(f"Task complete. Best Top1={best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
