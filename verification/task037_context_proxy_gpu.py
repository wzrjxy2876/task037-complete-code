#!/usr/bin/env python3
"""Task040 exact HTOR diagnosis for the existing Video Swin model.

The probe is intentionally diagnostic-only.  It uses the validated Task037
adapter for model construction, checkpoint loading, balanced UCF101 sampling,
and pruning-unit discovery, but computes a new score from raw true-class
logits and temporary whole-unit masks.  It never physically prunes or trains.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SelectedUnit:
    global_index: int
    layer_name: str
    unit_type: str
    unit_index: int
    spec: Any


def ensure_project_importable(project_root: Path) -> None:
    """Add only package-local source roots; no historical checkout is needed."""

    candidates = [
        project_root,
        project_root / "src" / "lgfr_runtime",
        project_root / "src",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Task040 requested CUDA, but CUDA is unavailable")
    if requested.type == "cuda":
        torch.cuda.set_device(requested)
    return requested


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(project_root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def normalize_checkpoint_key(key: str) -> str:
    normalized = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "backbone.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                changed = True
    return normalized


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must contain a mapping")
    state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state_dict/model is not a mapping")
    return {
        normalize_checkpoint_key(key): value
        for key, value in state.items()
        if torch.is_tensor(value)
    }


def make_checkpoint_identity(
    model: nn.Module,
    checkpoint_path: Path,
    adapter_metadata: Mapping[str, Any],
    all_specs: Sequence[Any],
) -> dict[str, Any]:
    """Audit the already-loaded adapter result without changing load semantics."""

    state = checkpoint_state(checkpoint_path)
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
        elif tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
        else:
            compatible[key] = value

    missing = sorted(key for key in model_state if key not in compatible)
    non_head_missing = [key for key in missing if not key.startswith("cls_head.")]
    non_head_mismatches = [
        item for item in shape_mismatches if not item["key"].startswith("cls_head.")
    ]
    if non_head_missing or non_head_mismatches:
        raise RuntimeError(
            "Task037 adapter loading semantics were not satisfied: "
            f"non_head_missing={non_head_missing[:8]}, "
            f"non_head_mismatches={non_head_mismatches[:8]}"
        )

    head_state_keys = sorted(
        key for key in state if key.startswith("cls_head.")
    )
    head_loaded_keys = sorted(
        key for key in compatible if key.startswith("cls_head.")
    )
    head_missing_keys = sorted(
        key for key in missing if key.startswith("cls_head.")
    )
    head_mismatch_keys = sorted(
        item["key"] for item in shape_mismatches if item["key"].startswith("cls_head.")
    )
    if head_mismatch_keys:
        head_status = "shape_mismatch_allowed_by_task037_adapter"
    elif head_missing_keys:
        head_status = "missing_allowed_by_task037_adapter"
    else:
        head_status = "loaded"

    head_count = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "head")
    neuron_count = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "neuron")
    return {
        "task": "task040",
        "checkpoint_absolute_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "loaded_tensor_count": len(compatible),
        "loaded_parameter_count": int(sum(value.numel() for value in compatible.values())),
        "missing_keys": missing,
        "unexpected_keys": sorted(unexpected),
        "shape_mismatches": shape_mismatches,
        "non_head_missing_keys": non_head_missing,
        "non_head_shape_mismatches": non_head_mismatches,
        "classifier_head": {
            "status": head_status,
            "checkpoint_keys": head_state_keys,
            "loaded_keys": head_loaded_keys,
            "missing_keys": head_missing_keys,
            "shape_mismatch_keys": head_mismatch_keys,
        },
        "model_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "model_trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "discovered_pruning_layer_count": len(all_specs),
        "discovered_attention_head_count": head_count,
        "discovered_ffn_neuron_count": neuron_count,
        "adapter_metadata": dict(adapter_metadata),
    }


def normalize_unit_type(value: str) -> str:
    value = value.strip().lower()
    aliases = {"head": "head", "attention": "head", "neuron": "neuron", "ffn": "neuron"}
    if value not in aliases:
        raise ValueError(f"unsupported unit type: {value!r}")
    return aliases[value]


def representative_indices(count: int, limit: int) -> list[int]:
    if count <= 0 or limit <= 0:
        return []
    return sorted(set(np.linspace(0, count - 1, min(count, limit)).round().astype(int).tolist()))


def select_units(
    all_specs: Sequence[Any],
    layer_selector: str,
    explicit_units: Sequence[str],
    units_per_layer: int,
    ctfrs_module: Any,
) -> list[SelectedUnit]:
    selected_specs = ctfrs_module.filter_layers(all_specs, layer_selector)
    head_total = sum(int(spec.num_units) for spec in all_specs if spec.unit_type == "head")
    offsets: dict[str, int] = {}
    seen_by_type = {"head": 0, "neuron": 0}
    for spec in all_specs:
        offsets[spec.name] = seen_by_type[spec.unit_type]
        seen_by_type[spec.unit_type] += int(spec.num_units)

    def global_unit_index(spec: Any, unit_index: int) -> int:
        type_offset = 0 if spec.unit_type == "head" else head_total
        return type_offset + offsets[spec.name] + unit_index

    selected: list[SelectedUnit] = []

    if explicit_units:
        for expression in explicit_units:
            try:
                layer_pattern, unit_type, unit_text = expression.rsplit(":", 2)
                pattern = re.compile(layer_pattern)
                unit_type = normalize_unit_type(unit_type)
                unit_index = int(unit_text)
            except ValueError as exc:
                raise ValueError(
                    "--unit must have the form LAYER_REGEX:head|neuron:INDEX"
                ) from exc
            matches = [spec for spec in selected_specs if pattern.search(spec.name)]
            if not matches:
                raise ValueError(f"--unit layer regex matched no selected layer: {layer_pattern!r}")
            for spec in matches:
                if spec.unit_type != unit_type:
                    continue
                if not 0 <= unit_index < int(spec.num_units):
                    raise ValueError(
                        f"unit index {unit_index} is outside {spec.name} size {spec.num_units}"
                    )
                selected.append(
                    SelectedUnit(
                        global_index=global_unit_index(spec, unit_index),
                        layer_name=spec.name,
                        unit_type=unit_type,
                        unit_index=unit_index,
                        spec=spec,
                    )
                )
    else:
        for spec in selected_specs:
            for unit_index in representative_indices(int(spec.num_units), units_per_layer):
                selected.append(
                    SelectedUnit(
                        global_index=global_unit_index(spec, unit_index),
                        layer_name=spec.name,
                        unit_type=spec.unit_type,
                        unit_index=unit_index,
                        spec=spec,
                    )
                )

    unique: dict[tuple[str, str, int], SelectedUnit] = {}
    for unit in selected:
        unique[(unit.layer_name, unit.unit_type, unit.unit_index)] = unit
    selected = list(unique.values())
    selected.sort(key=lambda item: (item.global_index, item.unit_type, item.unit_index))
    if not selected:
        raise ValueError("no representative units were selected")
    selected_types = {unit.unit_type for unit in selected}
    if selected_types != {"head", "neuron"}:
        raise ValueError(
            "Task040 representative selection must include both attention heads and FFN neurons"
        )
    return selected


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_video_manifest(loader: Any, selected_indices: Sequence[int], output_path: Path) -> None:
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    rows: list[dict[str, Any]] = []
    for order, index in enumerate(selected_indices):
        directory, duration, label = dataset.clips[int(index)]
        rows.append(
            {
                "video_index": order,
                "dataset_index": int(index),
                "video_id": str(directory),
                "duration": int(duration),
                "label": int(label),
            }
        )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["video_index"])
        writer.writeheader()
        writer.writerows(rows)


@contextlib.contextmanager
def temporary_unit_mask(spec: Any, unit_index: int):
    """Match the validated Task037 whole-head/whole-neuron hook semantics."""

    def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]):
        values = inputs[0].clone()
        if spec.unit_type == "head":
            head_dim = int(spec.module.head_dim)
            expected = int(spec.num_units) * head_dim
            if values.shape[-1] != expected:
                raise ValueError(
                    f"attention projection input shape {tuple(values.shape)} does not match "
                    f"{spec.num_units} heads x {head_dim}"
                )
            reshaped = values.reshape(*values.shape[:-1], int(spec.num_units), head_dim)
            reshaped[..., unit_index, :] = 0.0
            values = reshaped.reshape_as(values)
        elif spec.unit_type == "neuron":
            values[..., unit_index] = 0.0
        else:
            raise ValueError(f"unsupported pruning unit type: {spec.unit_type!r}")
        return (values,) + tuple(inputs[1:])

    handle = spec.hook_module.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def unwrap_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in ("logits", "output", "pred"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    raise TypeError(f"could not extract logits from {type(output)!r}")


def true_class_logit(model: nn.Module, videos: torch.Tensor, label: int) -> float:
    with torch.no_grad():
        logits = unwrap_logits(model(videos))
    if logits.ndim != 2 or logits.shape[0] != videos.shape[0]:
        raise ValueError(f"unexpected model logits shape: {tuple(logits.shape)}")
    if not 0 <= int(label) < logits.shape[1]:
        raise ValueError(f"ground-truth label {label} is outside logits dimension {logits.shape[1]}")
    value = logits[:, int(label)].detach().to(dtype=torch.float64)
    if not torch.isfinite(value).all():
        raise ValueError("non-finite true-class raw logit")
    return float(value[0].item())


def infer_temporal_length(videos: torch.Tensor) -> int:
    if videos.ndim != 5:
        raise ValueError(
            "Task040 expected the existing UCF101 loader layout [B,C,T,H,W]; "
            f"received {tuple(videos.shape)}"
        )
    return int(videos.shape[2])


def intervention_batches(
    clip: torch.Tensor,
    interventions: Sequence[Any],
    batch_size: int,
    core: Any,
) -> list[tuple[list[Any], torch.Tensor]]:
    batches: list[tuple[list[Any], torch.Tensor]] = []
    for start in range(0, len(interventions), batch_size):
        group = list(interventions[start : start + batch_size])
        values = core.apply_temporal_interventions(clip, group, time_dim=1)
        batches.append((group, values))
    return batches


def write_raw_header(handle: Any) -> None:
    fieldnames = [
        "video_index", "video_id", "label", "unit_global_index", "layer_name",
        "unit_type", "unit_index", "level", "block_size", "pair_index",
        "z_true_original", "z_true_original_masked", "z_true_intervened",
        "z_true_intervened_masked", "d_original", "d_intervened", "tau",
    ]
    handle.write(",".join(fieldnames) + "\n")



import pandas as pd

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description='Task037 context-conditioned first-order deletion proxy')
    for n in ['project_root','checkpoint','output_dir','val_list','context_manifest','confirmation_output','unit_manifest','domain_manifest','video_manifest']: p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--adapter',default='ucf101_videoswin_probe_adapter_v2'); p.add_argument('--frame_root',default=''); p.add_argument('--num_classes',type=int,default=5); p.add_argument('--videos_per_class',type=int,default=3); p.add_argument('--num_workers',type=int,default=2); p.add_argument('--seed',type=int,default=3407); p.add_argument('--max_videos',type=int,default=0); p.add_argument('--max_contexts',type=int,default=0); return p.parse_args()

def _native_slice(x,spec,ui):
    if spec.unit_type=='head':
        h,hd=int(spec.num_units),int(spec.module.head_dim)
        if x.shape[-1]!=h*hd: raise RuntimeError(f'bad attention shape {tuple(x.shape)}')
        return x.reshape(*x.shape[:-1],h,hd)[...,ui,:]
    if spec.unit_type=='neuron':
        if x.shape[-1]<=ui: raise RuntimeError(f'bad FFN shape {tuple(x.shape)}')
        return x[...,ui]
    raise ValueError(spec.unit_type)

def _load_frozen_units(path,specs):
    rows=pd.read_csv(path)
    if len(rows)!=36: raise AssertionError(f'unit manifest {len(rows)}')
    by={str(s.name):s for s in specs}; out=[]
    for r in rows.sort_values('global_index').itertuples(index=False):
        name=str(r.layer); typ='head' if 'attention' in str(r.unit_type).lower() else 'neuron'; spec=by.get(name)
        if spec is None or spec.unit_type!=typ: raise AssertionError(f'unit identity mismatch {name}')
        ui=int(r.unit_index)
        if ui<0 or ui>=int(spec.num_units): raise AssertionError(f'unit index mismatch {name}:{ui}')
        out.append(SelectedUnit(int(r.global_index),name,typ,ui,spec))
    if len({u.global_index for u in out})!=36: raise AssertionError('duplicate global indices')
    return out

def _contexts(path,actual_t,core):
    cm=pd.read_csv(path); expected={(s,p) for s in (1,2,4,8,16) for p in (0,4,8,12)}; got={(int(r.span),int(r.canonical_pair_index)) for r in cm.itertuples(index=False)}
    if len(cm)!=20 or got!=expected: raise AssertionError('context manifest identity')
    allit=core.enumerate_fixed_cardinality_temporal_pairs(actual_t); by={(int(i.block_size),int(i.pair_index)):i for i in allit}; out=[{'context_id':0,'span':0,'pair_index':-1,'frame_pair':'original','intervention':None}]
    for r in cm.sort_values(['span','canonical_pair_index']).itertuples(index=False):
        k=(int(r.span),int(r.canonical_pair_index)); it=by.get(k)
        if it is None or (int(it.left_start),int(it.right_start))!=(int(r.frame_a),int(r.frame_b)): raise AssertionError(f'context pair mismatch {k}')
        out.append({'context_id':len(out),'span':k[0],'pair_index':k[1],'frame_pair':f'{int(r.frame_a)}-{int(r.frame_b)}','intervention':it})
    return out

def _oracle_lookup(d):
    ori=pd.read_csv(d/'task_bms_confirm30_original_damage.csv'); rel=pd.read_csv(d/'task_bms_confirm30_conditioned_damage.csv'); od={(str(r.video_key),int(r.global_index)):float(r.D_ori) for r in ori.itertuples(index=False)}; rd={(str(r.video_key),int(r.global_index),int(r.span),int(r.pair_index)):float(r.D_rel) for r in rel.itertuples(index=False)}; dm={(str(r.video_key),int(r.span),int(r.pair_index)):float(r.Delta_model) for r in rel.drop_duplicates(['video_key','span','pair_index']).itertuples(index=False)}
    if len(od)!=1080 or len(rd)!=21600: raise AssertionError(f'oracle sizes {len(od)}/{len(rd)}')
    return od,rd,dm

def _install_native_hooks(selected,captures):
    by={id(u.spec.hook_module):u.spec for u in selected}; hs=[]
    for key,spec in by.items():
        def make(k,sp):
            def hook(_m,inputs):
                if not inputs or not torch.is_tensor(inputs[0]): raise RuntimeError(f'hook input missing {sp.name}')
                x=inputs[0]
                if not x.requires_grad: x.requires_grad_(True)
                x.retain_grad(); captures[k]={'activation':x,'module':sp}
            return hook
        hs.append(spec.hook_module.register_forward_pre_hook(make(key,spec)))
    return hs

def run_probe(args):
    project_root=Path(args.project_root).resolve(); checkpoint=Path(args.checkpoint).resolve(); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); ensure_project_importable(project_root); set_seed(int(args.seed)); device=resolve_device(args.device)
    core=importlib.import_module('task040_htor_core'); ctfrs=importlib.import_module('probe_ctfrs_dynamic_function'); adapter=importlib.import_module(args.adapter); model,meta=adapter.build_model_for_probe(checkpoint=str(checkpoint),device=device); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    specs=ctfrs.discover_unit_layers(model); selected=_load_frozen_units(Path(args.unit_manifest),specs); od,rd,dm=_oracle_lookup(Path(args.confirmation_output)); loader,selected_indices,chosen_classes=ctfrs.build_balanced_loader(project_root=project_root,val_list=args.val_list,frame_root=args.frame_root,num_classes=int(args.num_classes),videos_per_class=int(args.videos_per_class),num_workers=int(args.num_workers),seed=int(args.seed))
    vm_by={str(r.video_path):r for r in pd.read_csv(args.video_manifest).itertuples(index=False)}; base_dataset=loader.dataset
    while hasattr(base_dataset,'dataset'): base_dataset=base_dataset.dataset
    first=next(iter(loader)); actual_t=infer_temporal_length(first[0]); contexts=_contexts(Path(args.context_manifest),actual_t,core); contexts=contexts[:int(args.max_contexts)+1] if args.max_contexts else contexts; maxv=int(args.max_videos) if args.max_videos else 0; um=pd.read_csv(args.unit_manifest); dom={int(r.global_index):int(r.domain_id) for r in um.itertuples(index=False)}; cat={int(k):str(v.iloc[0].category) for k,v in um.groupby('domain_id')}
    captures={}; handles=_install_native_hooks(selected,captures); rows=[]; caprows=[]; nvideo=0; neval=0
    try:
        for batch in loader:
            if maxv and nvideo>=maxv: break
            videos,targets,indices=batch[:3]; base=videos[0].float().to(device,non_blocking=True).detach(); label=int(targets[0].item()); dsidx=int(indices[0].item()); video_id=str(base_dataset.clips[dsidx][0]); key=Path(video_id).name
            if key not in vm_by: raise AssertionError(f'video not in manifest {key}')
            v=vm_by[key]
            for c in contexts:
                captures.clear(); model.zero_grad(set_to_none=True); inp=base.unsqueeze(0).clone() if c['intervention'] is None else core.apply_temporal_interventions(base,[c['intervention']],time_dim=1)[0];
                if inp.ndim==4: inp=inp.unsqueeze(0)
                inp=inp.float().to(device,non_blocking=True).clone().requires_grad_(True); logits=unwrap_logits(model(inp));
                if logits.ndim!=2 or logits.shape[0]!=1 or not 0<=label<logits.shape[1]: raise RuntimeError(f'logit shape {tuple(logits.shape)}')
                target=logits[0,label]; target.backward(); neval+=1
                if len(captures)!=len({id(u.spec.hook_module) for u in selected}): raise RuntimeError(f'hooks {len(captures)}')
                for cap in captures.values():
                    x=cap['activation']; g=x.grad
                    if g is None or tuple(x.shape)!=tuple(g.shape) or x.numel()!=g.numel(): raise RuntimeError(f'gradient mismatch {cap["module"].name}')
                    caprows.append({'video_key':key,'context_id':c['context_id'],'hook_layer':cap['module'].name,'unit_type':cap['module'].unit_type,'native_shape':str(list(x.shape)),'gradient_shape':str(list(g.shape)),'native_numel':int(x.numel()),'gradient_numel':int(g.numel()),'grad_finite':bool(torch.isfinite(g).all().item()),'one_backward':True})
                for u in selected:
                    cap=captures[id(u.spec.hook_module)]; x=cap['activation']; g=x.grad; signed=float((_native_slice(x,u.spec,u.unit_index)*_native_slice(g,u.spec,u.unit_index)).sum().detach().to(dtype=torch.float64).cpu().item()); oracle=od[(key,u.global_index)] if c['context_id']==0 else rd[(key,u.global_index,int(c['span']),int(c['pair_index']))]
                    rows.append({'class':str(v.class_name),'class_name':str(v.class_name),'class_index':int(v.class_index),'category':cat[dom[u.global_index]],'Delta_model':0.0 if c['context_id']==0 else dm[(key,int(c['span']),int(c['pair_index']))],'video_id':video_id,'video_key':key,'manifest_order':int(v.manifest_order),'context_id':int(c['context_id']),'span':int(c['span']),'pair_index':int(c['pair_index']),'frame_pair':str(c['frame_pair']),'domain_id':dom[u.global_index],'global_index':int(u.global_index),'unit_type':'attention_head' if u.unit_type=='head' else 'ffn_neuron','layer':u.layer_name,'stage':int(u.spec.stage),'unit_index':int(u.unit_index),'true_class_logit':float(target.detach().cpu().item()),'proxy_signed_damage':signed,'proxy_absolute_damage':abs(signed),'oracle_signed_damage':float(oracle),'oracle_absolute_damage':abs(float(oracle)),'native_activation_shape':str(list(x.shape)),'native_gradient_shape':str(list(g.shape)),'native_activation_numel':int(x.numel()),'native_gradient_numel':int(g.numel())})
                del logits,target,inp; model.zero_grad(set_to_none=True); captures.clear()
            nvideo+=1
    finally:
        for h in handles: h.remove()
    fields=['class','class_name','class_index','category','Delta_model','video_id','video_key','manifest_order','context_id','span','pair_index','frame_pair','domain_id','global_index','unit_type','layer','stage','unit_index','true_class_logit','proxy_signed_damage','proxy_absolute_damage','oracle_signed_damage','oracle_absolute_damage','native_activation_shape','native_gradient_shape','native_activation_numel','native_gradient_numel']; pd.DataFrame(rows,columns=fields).to_csv(out/'task_context_proxy_raw.csv',index=False); pd.DataFrame(caprows).to_csv(out/'task_context_proxy_capture_audit.csv',index=False); vm=pd.read_csv(args.video_manifest); vm['source_manifest_sha256']=sha256_file(Path(args.video_manifest)); vm.to_csv(out/'task_context_proxy_manifest.csv',index=False)
    ident=make_checkpoint_identity(model,checkpoint,meta,specs); ident.update({'task':'task037_context_proxy','checkpoint_sha256':sha256_file(checkpoint),'expected_checkpoint_sha256':'4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63','device':str(device),'dtype':'torch.float32','amp':False,'git_branch':git_value(project_root,'rev-parse','--abbrev-ref','HEAD'),'git_commit':git_value(project_root,'rev-parse','HEAD'),'videos_processed':nvideo,'contexts_per_video':len(contexts),'context_evaluations':neval,'expected_context_evaluations':nvideo*len(contexts),'selected_unit_count':len(selected),'raw_row_count':len(rows),'expected_raw_row_count':nvideo*len(contexts)*len(selected),'native_hook_count':len({id(u.spec.hook_module) for u in selected}),'one_forward_one_backward_per_context':True,'gradient_cleared_each_context':True,'native_tensor_authoritative':True}); write_json(out/'task_context_proxy_runtime_summary.json',ident)
    audit=[{'check':'checkpoint_sha256','expected':ident['expected_checkpoint_sha256'],'observed':ident['checkpoint_sha256'],'status':'PASS' if ident['checkpoint_sha256']==ident['expected_checkpoint_sha256'] else 'FAIL'},{'check':'frozen_unit_count','expected':36,'observed':len(selected),'status':'PASS' if len(selected)==36 else 'FAIL'},{'check':'contexts_per_video','expected':21,'observed':len(contexts),'status':'PASS' if len(contexts)==21 else 'FAIL'},{'check':'context_evaluations','expected':nvideo*len(contexts),'observed':neval,'status':'PASS' if neval==nvideo*len(contexts) else 'FAIL'},{'check':'raw_rows','expected':nvideo*len(contexts)*len(selected),'observed':len(rows),'status':'PASS' if len(rows)==nvideo*len(contexts)*len(selected) else 'FAIL'}]; pd.DataFrame(audit).to_csv(out/'task_context_proxy_identity_audit.csv',index=False); return ident

def main(): print(json.dumps(run_probe(parse_args()),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
