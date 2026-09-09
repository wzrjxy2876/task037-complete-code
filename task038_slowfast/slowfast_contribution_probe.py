"""Fresh N=9 signed contribution-field probe for Task038."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .slowfast_finetune import balanced_n9_indices, build_loader, data_paths, sample_identity_payload, set_seed
from .slowfast_unit_adapter import UnitInventory


def _capture_names(inventory: UnitInventory) -> dict[str, str]:
    result = {}
    for unit in inventory.units:
        result.setdefault(
            unit.module_name,
            unit.module_name.rsplit(".", 1)[0] if unit.conv_position == "conv3" else unit.module_name,
        )
    return result




def probe_contribution_fields(
    model: nn.Module,
    checkpoint: str | Path,
    inventory: UnitInventory,
    device: torch.device,
    output_dir: str | Path,
    seed: int = 3407,
) -> dict[str, Any]:
    del checkpoint
    if seed != 3407:
        raise ValueError("Task038 contribution seed is frozen at 3407")
    set_seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    field_root = out / "fields"
    field_root.mkdir(exist_ok=True)
    _, val_list, _ = data_paths()
    identity = balanced_n9_indices(val_list, seed=seed)
    loader = build_loader(
        val_list, batch_size=1, shuffle=False,
        indices=[int(row["split_index"]) for row in identity], workers=0
    )
    model.eval().to(device)
    if int(model.fc.out_features) != 101:
        raise RuntimeError("SlowFast UCF101 probe requires 101 output classes")

    names = _capture_names(inventory)
    modules = dict(model.named_modules())
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(hook_name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            captures[hook_name] = output
            return output
        return hook

    for hook_name in sorted(set(names.values())):
        if hook_name not in modules:
            raise RuntimeError(f"capture module missing: {hook_name}")
        handles.append(modules[hook_name].register_forward_hook(make_hook(hook_name)))

    memmaps: dict[str, np.memmap] = {}
    valmaps: dict[str, np.memmap] = {}
    entries = []
    try:
        for layer_name in sorted({u.module_name for u in inventory.units}):
            units = [u for u in inventory.units if u.module_name == layer_name]
            stem = layer_name.replace(".", "__")
            memmaps[layer_name] = np.lib.format.open_memmap(
                field_root / f"{stem}.fields.npy", mode="w+", dtype=np.float32,
                shape=(9, len(units), 16, 7, 7)
            )
            valmaps[layer_name] = np.lib.format.open_memmap(
                field_root / f"{stem}.valid.npy", mode="w+", dtype=np.bool_,
                shape=(9, len(units))
            )
            entries.append(
                {
                    "layer_name": layer_name,
                    "global_start": units[0].global_index,
                    "global_end": units[-1].global_index + 1,
                    "shape": [9, len(units), 16, 7, 7],
                    "fields_path": f"{stem}.fields.npy",
                    "valid_path": f"{stem}.valid.npy",
                    "normalization": "raw signed pooled float32; one L2 after cross-video concatenation",
                }
            )
        for sample_index, batch in enumerate(loader):
            x = batch[0].to(device).float()
            target = batch[1].to(device).long()
            captures.clear()
            with torch.enable_grad():
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("non-finite contribution logits")
                score = logits[0, target[0]]
                hook_names = sorted(captures)
                outputs = tuple(captures[name] for name in hook_names)
                gradients = torch.autograd.grad(
                    score, outputs, retain_graph=False, allow_unused=True
                )
                for hook_name, activation, gradient in zip(hook_names, outputs, gradients):
                    if gradient is None:
                        raise RuntimeError(f"missing gradient for capture site {hook_name}")
                    field = activation.float() * gradient.float()
                    pooled = F.adaptive_avg_pool3d(field, (16, 7, 7))[0].detach().cpu().numpy()
                    for layer_name, mapped_hook in names.items():
                        if mapped_hook != hook_name:
                            continue
                        units = [u for u in inventory.units if u.module_name == layer_name]
                        if pooled.shape[0] != len(units):
                            raise RuntimeError(f"field width mismatch at {layer_name}")
                        memmaps[layer_name][sample_index] = pooled.astype(np.float32, copy=False)
                        valmaps[layer_name][sample_index] = (
                            np.linalg.norm(pooled.reshape(pooled.shape[0], -1), axis=1) > 0.0
                        )
            model.zero_grad(set_to_none=True)
            del x, logits, score, outputs, gradients
        for array in memmaps.values():
            array.flush()
        for array in valmaps.values():
            array.flush()
    finally:
        for handle in handles:
            handle.remove()
    manifest = {
        "schema": "task038_signed_contribution_field_v1",
        "field_semantics": "signed true-class raw logit X*d z_y/dX",
        "capture_sites": {
            "conv1_conv2": "activation site before downstream consumer",
            "conv3": "semantic residual block output after ReLU",
            "lateral": "lateral output before Slow concatenation",
        },
        "sample_count": 9,
        "sample_identity": identity,
        "pooled_shape": [16, 7, 7],
        "feature_dimension": 9 * 16 * 7 * 7,
        "unit_count": inventory.num_units,
        "entries": entries,
        "no_per_video_normalization": True,
        "disk_backed_memmap": True,
        "seed": seed,
        "device": str(device),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out / "field_manifest.json").write_text(text, encoding="utf-8")
    (field_root / "field_manifest.json").write_text(text, encoding="utf-8")
    identity_payload = sample_identity_payload(val_list, identity, seed)
    (out / "n09_sample_identity.json").write_text(
        json.dumps(identity_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "sample_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
