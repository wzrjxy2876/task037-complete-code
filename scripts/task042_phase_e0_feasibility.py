#!/usr/bin/env python3
"""Task042 Phase E.0: temporal-relation preservation-loss feasibility only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_DEFAULT = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
OUTPUT_DEFAULT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy/phase_e0")
PROJECT_ROOT = Path("/home/jixinye25/jxy_work1/swintrans_task035")
CHECKPOINT = Path("/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
BASE_OUTPUT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
BRANCH = "task_042_post_bms_frame_relation_redundancy"
EXPECTED_CHECKPOINT_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
EXPECTED_PHASE_D_HEAD = "a8135a59c7b81ba978c158a56baae2b4f0b6a651"
SAME_TYPE_DOMAIN = "269"
MIXED_DOMAIN = "271"
GATE_TARGETS = (328, 774)  # Phase-D F3 global order: FFN in mixed 271, then head in 269.
GATE_VALUES = (1.00, 0.75, 0.50, 0.25, 0.00)
SPANS = (1, 2, 4, 8, 16)
VIDEO_INDICES = (0, 3)  # frozen Task042 manifest: one clip from each of two classes.
EPS_SENSITIVITY = 1e-12
EPS_NORMALIZATION = 1e-8
EPS_PEARSON = 1e-12
OPTIMIZER_STEPS = 3
OPTIMIZER_LR = 1e-4
REQUIRED_OUTPUTS = (
    "task042_phase_e0_gate_audit.csv",
    "task042_phase_e0_relation_loss.csv",
    "task042_phase_e0_gradient_audit.csv",
    "task042_phase_e0_relation_drift.csv",
    "task042_phase_e0_optimization_sanity.csv",
    "task042_phase_e0_summary.json",
    "task042_phase_e0_report.md",
)
PRIMITIVES = None


def require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError("Task042 Phase-E.0 gate failed: " + message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha_for_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    require(not path.exists(), "refusing to overwrite existing output " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), "refusing to overwrite existing output " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def model_parameter_sha(model: Any) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        value = parameter.detach().contiguous().cpu().numpy()
        digest.update(value.tobytes())
    for name, buffer in model.named_buffers():
        digest.update(name.encode("utf-8"))
        digest.update(buffer.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


class PhaseE0Capture:
    """Differentiable captures plus temporary output-only attention/FFN gates."""

    def __init__(self, model: Any, units: Sequence[Mapping[str, Any]], gate_state: Any,
                 torch: Any, detach: bool = False):
        self.model, self.units, self.gate_state = model, list(units), gate_state
        self.torch, self.detach = torch, bool(detach)
        self.values: Dict[int, Any] = {}
        self.handles = []
        self.by_layer: Dict[str, Dict[str, Any]] = {}
        self.qkv_v: Dict[str, Any] = {}
        for row in self.units:
            layer = str(row["layer"])
            group = self.by_layer.setdefault(layer, {"heads": [], "neurons": []})
            key = "heads" if row["capture_kind"] == "head" else "neurons"
            group[key].append(row)
        modules = dict(model.named_modules())
        for layer, spec in self.by_layer.items():
            require(layer in modules, "frozen activation layer missing from model: " + layer)
            module = modules[layer]
            if spec["heads"]:
                self._attach_attention(layer, module, spec["heads"])
            if spec["neurons"]:
                self._attach_ffn(layer, module, spec["neurons"])

    def _save(self, uid: int, value: Any) -> None:
        if self.detach:
            value = value.detach()
        self.values[int(uid)] = value

    def clear(self) -> None:
        self.values.clear()

    def _attach_attention(self, layer: str, module: Any,
                          rows: Sequence[Mapping[str, Any]]) -> None:
        require(hasattr(module, "qkv") and hasattr(module, "attn_drop") and
                hasattr(module, "num_heads"), "unexpected attention module: " + layer)
        heads = int(module.num_heads)
        head_rows = list(rows)
        head_by_index = {int(row["unit_index"]): row for row in head_rows}
        require(len(head_by_index) == len(head_rows), "duplicate frozen head in layer " + layer)

        def qkv_hook(_module: Any, _inputs: Any, output: Any) -> None:
            require(output.ndim == 3 and int(output.shape[-1]) % (3 * heads) == 0,
                    "unexpected qkv shape in " + layer)
            head_dim = int(output.shape[-1]) // (3 * heads)
            tokens = int(output.shape[1])
            self.qkv_v[layer] = output.reshape(int(output.shape[0]), tokens, 3,
                                               heads, head_dim).permute(2, 0, 3, 1, 4)[2]

        def attn_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            require(layer in self.qkv_v, "attention V cache missing in " + layer)
            local_gates = {}
            for head, row in head_by_index.items():
                uid = int(row["task037_global_index"])
                gate = float(self.gate_state.values.get(uid, 1.0))
                if gate != 1.0:
                    local_gates[head] = gate
            gated = output
            if local_gates:
                    gated = PRIMITIVES.scale_attention_probabilities(output, local_gates)
            per_head = self.torch.matmul(gated, self.qkv_v[layer])
            for head, row in head_by_index.items():
                self._save(int(row["task037_global_index"]), per_head[:, head, :, :])
            return gated

        self.handles.append(module.qkv.register_forward_hook(qkv_hook))
        self.handles.append(module.attn_drop.register_forward_hook(attn_hook))

    def _attach_ffn(self, layer: str, module: Any,
                    rows: Sequence[Mapping[str, Any]]) -> None:
        require(hasattr(module, "fc1") and hasattr(module, "act") and hasattr(module, "fc2"),
                "unexpected FFN module: " + layer)
        hidden = int(module.fc1.out_features)
        neurons = {int(row["unit_index"]): row for row in rows}
        require(len(neurons) == len(rows), "duplicate frozen FFN neuron in layer " + layer)

        def act_hook(_module: Any, _inputs: Any, output: Any) -> Any:
            require(output.ndim >= 2 and int(output.shape[-1]) == hidden,
                    "unexpected post-GELU FFN activation shape in " + layer)
            local_gates = {}
            for neuron, row in neurons.items():
                uid = int(row["task037_global_index"])
                gate = float(self.gate_state.values.get(uid, 1.0))
                if gate != 1.0:
                    local_gates[neuron] = gate
            gated = output
            if local_gates:
                gated = PRIMITIVES.scale_ffn_activations(output, local_gates)
            for neuron, row in neurons.items():
                self._save(int(row["task037_global_index"]), gated[..., neuron])
            return gated

        self.handles.append(module.act.register_forward_hook(act_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.values.clear()
        self.qkv_v.clear()


def verify_phase_d_inputs(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                                               Dict[str, Any], Dict[str, Any]]:
    unit_path = BASE_OUTPUT / "task042_unit_manifest.csv"
    phase_d_dir = BASE_OUTPUT / "phase_d"
    candidate_path = phase_d_dir / "task042_phase_d_candidate_provenance.csv"
    summary_path = phase_d_dir / "task042_phase_d_summary.json"
    identity_path = phase_d_dir / "task042_phase_d_input_identity.json"
    for path in (unit_path, candidate_path, summary_path, identity_path, CHECKPOINT):
        require(path.is_file(), "frozen input missing: " + str(path))
    require(git_value(repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH,
            "must remain on existing Task042 branch")
    require(not git_value(repo, "status", "--porcelain"),
            "code checkout must be clean before Phase E.0 inference")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    require(summary.get("git_head") == EXPECTED_PHASE_D_HEAD and
            summary.get("decision") == "C. TEMPORAL_COVERAGE_PROGRESSIVE_PATH_REJECTED",
            "Phase-D authority or scientific state differs from the frozen record")
    require(sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA,
            "authoritative checkpoint SHA changed")
    candidate_rows = read_csv(candidate_path)
    expected_candidate_hash = summary["outputs"]["task042_phase_d_candidate_provenance.csv"]["sha256"]
    require(sha256_file(candidate_path) == expected_candidate_hash,
            "Phase-D candidate provenance hash differs from summary")
    input_hashes = identity["frozen_input_hashes"]
    require(sha256_file(unit_path) == input_hashes["task042_unit_manifest.csv"]["sha256"],
            "frozen Task042 unit manifest hash changed")
    units_all = read_csv(unit_path)
    by_uid = {int(row["task037_global_index"]): dict(row) for row in units_all}
    require(len(by_uid) == len(units_all), "duplicate frozen unit identity")
    units = [row for row in units_all if str(row["domain_id"]) in
             (SAME_TYPE_DOMAIN, MIXED_DOMAIN)]
    units.sort(key=lambda row: (int(row["domain_id"]), int(row["task037_global_index"])))
    domain_types: Dict[str, set] = {}
    for row in units:
        domain_types.setdefault(str(row["domain_id"]), set()).add(row["unit_type"])
    require(domain_types.get(SAME_TYPE_DOMAIN) == {"attention_head"},
            "selected same-type BMS domain is not attention-only")
    require(domain_types.get(MIXED_DOMAIN) == {"attention_head", "ffn_neuron"},
            "selected mixed BMS domain no longer mixes heads and FFN neurons")
    provenance_by_uid = {int(row["task037_global_index"]): row for row in candidate_rows}
    target_rows = []
    for uid in GATE_TARGETS:
        require(uid in by_uid and uid in provenance_by_uid,
                "frozen target candidate is absent from Task042/Phase-D manifests")
        row = dict(by_uid[uid])
        provenance = provenance_by_uid[uid]
        require(provenance["selected_for_removal"] == "True" and
                int(provenance["within_domain_removal_rank"]) == 1,
                "target is not the frozen first F3 removal candidate in its domain")
        expected_domain = SAME_TYPE_DOMAIN if uid == 774 else MIXED_DOMAIN
        expected_type = "attention_head" if uid == 774 else "ffn_neuron"
        require(str(row["domain_id"]) == expected_domain and row["unit_type"] == expected_type,
                "target BMS type/domain changed from deterministic Phase-E.0 design")
        row["frozen_removal_rank"] = int(provenance["within_domain_removal_rank"])
        row["f3_global_step"] = int(provenance["f3_global_step"])
        row["domain_group"] = provenance["domain_group"]
        target_rows.append(row)
    require(len(units) == 7, "Phase-E.0 domain subset should contain the frozen 7 units")
    return units, target_rows, summary, identity


def build_pair_subset(core: Any) -> Tuple[List[Any], List[Dict[str, int]]]:
    all_rows = []
    intervention_by_key = {}
    for intervention in core.enumerate_fixed_cardinality_temporal_pairs(32):
        intervention_by_key[(int(intervention.block_size), int(intervention.pair_index))] = intervention
    for span in SPANS:
        key = (span, 0)
        require(key in intervention_by_key, "frozen Task040 span/pair identity missing: " + repr(key))
        intervention = intervention_by_key[key]
        all_rows.append({"span": span, "pair_index": 0,
                         "frame_a": int(intervention.left_start),
                         "frame_b": int(intervention.right_start)})
    require(len(all_rows) == 5 and len({row["span"] for row in all_rows}) == 5,
            "minimal frame-pair diagnostic subset is not exact")
    return [intervention_by_key[(row["span"], row["pair_index"])] for row in all_rows], all_rows


def load_clips(ctfrs: Any, config: Mapping[str, Any], video_manifest: Sequence[Mapping[str, Any]],
               torch: Any, device: Any) -> Tuple[List[Any], List[int], List[Dict[str, Any]]]:
    by_name = {Path(str(row["video_id"])).name: dict(row) for row in video_manifest}
    selected = {int(row["video_index"]): dict(row) for row in video_manifest
                if int(row["video_index"]) in VIDEO_INDICES}
    require(set(selected) == set(VIDEO_INDICES), "deterministic video subset is absent")
    loader = ctfrs.build_balanced_loader(
        project_root=Path(config["project_root"]),
        val_list=str(config["exact_video_list"]),
        frame_root=str(config["frame_root"]), num_classes=10,
        videos_per_class=3, num_workers=2, seed=3407)[0]
    dataset = loader.dataset
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    clips_by_index, labels_by_index = {}, {}
    for batch in loader:
        videos, labels, indices = batch[0], batch[1], batch[2]
        require(int(videos.shape[0]) == 1 and int(videos.shape[2]) == 32,
                "frozen calibration loader must emit one 32-frame clip")
        local_index = int(indices[0].item())
        name = Path(str(dataset.clips[local_index][0])).name
        require(name in by_name, "calibration loader produced a video outside the frozen manifest")
        row = by_name[name]
        video_index = int(row["video_index"])
        if video_index in selected:
            require(int(labels[0].item()) == int(row["label"]),
                    "deterministic video label differs from the frozen manifest")
            clips_by_index[video_index] = videos[0].to(device=device, dtype=torch.float32,
                                                        non_blocking=True)
            labels_by_index[video_index] = int(labels[0].item())
        if set(clips_by_index) == set(VIDEO_INDICES):
            break
    require(set(clips_by_index) == set(VIDEO_INDICES),
            "deterministic mini-subset clips could not be loaded")
    ordered_rows = [selected[index] for index in VIDEO_INDICES]
    return ([clips_by_index[index] for index in VIDEO_INDICES],
            [labels_by_index[index] for index in VIDEO_INDICES], ordered_rows)


def model_forward(model: Any, capture: PhaseE0Capture, clip: Any,
                  phase_d: Any) -> Tuple[Any, Dict[int, Any]]:
    capture.clear()
    logits = phase_d.unwrap_logits(model(clip.unsqueeze(0)))
    expected = {int(row["task037_global_index"]) for row in capture.units}
    require(set(capture.values) == expected,
            "forward did not capture every frozen Phase-E.0 unit")
    return logits, dict(capture.values)


def collect_no_grad(model: Any, capture: PhaseE0Capture, gate_state: Any,
                    clips: Sequence[Any], labels: Sequence[int], pair_ops: Sequence[Any],
                    unit_rows: Sequence[Mapping[str, Any]], core: Any, phase_d: Any,
                    torch: Any, functional: Any) -> Tuple[List[List[List[float]]], float,
                                                          List[Dict[int, Any]], List[Any]]:
    sensitivities, base_activations, base_logits, ce_values = [], [], [], []
    gate_state_values = dict(gate_state.values)
    with torch.no_grad():
        for clip, label in zip(clips, labels):
            logits, h0 = model_forward(model, capture, clip, phase_d)
            base_activations.append(h0)
            base_logits.append(logits.detach())
            target = torch.tensor([int(label)], dtype=torch.long, device=clip.device)
            ce_values.append(float(functional.cross_entropy(logits, target).item()))
            by_unit = [[] for _ in unit_rows]
            for intervention in pair_ops:
                swapped = core.apply_temporal_interventions(
                    clip, [intervention], time_dim=1)[0]
                _logits, h1 = model_forward(model, capture, swapped, phase_d)
                for index, row in enumerate(unit_rows):
                    uid = int(row["task037_global_index"])
                    value = PRIMITIVES.relative_sensitivity(h0[uid], h1[uid], EPS_SENSITIVITY)
                    by_unit[index].append(float(value.item()))
            sensitivities.append(by_unit)
    require(dict(gate_state.values) == gate_state_values,
            "gate state changed during a read-only forward")
    return sensitivities, statistics.mean(ce_values), base_activations, base_logits


def finite_gradient_metrics(model: Any, torch: Any) -> Dict[str, Any]:
    total_sq = 0.0
    stage_sq = {"stage0": 0.0, "stage1": 0.0, "stage2": 0.0, "stage3": 0.0,
                "other": 0.0}
    finite = True
    nonzero = False
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        is_finite = bool(torch.isfinite(grad).all().item())
        finite = finite and is_finite
        norm_sq = float(grad.detach().float().square().sum().item())
        total_sq += norm_sq
        nonzero = nonzero or norm_sq > 0.0
        match = re.match(r"^layers\.(\d+)\.", name)
        stage = "stage" + match.group(1) if match else "other"
        if stage not in stage_sq:
            stage = "other"
        stage_sq[stage] += norm_sq
    result = {"total_grad_norm": math.sqrt(total_sq), "finite": finite,
              "nonzero": nonzero}
    for stage, norm_sq in stage_sq.items():
        result[stage + "_grad_norm"] = math.sqrt(norm_sq)
    return result


def ce_backward(model: Any, capture: PhaseE0Capture, clips: Sequence[Any],
                labels: Sequence[int], phase_d: Any, torch: Any,
                functional: Any) -> float:
    total = 0.0
    for clip, label in zip(clips, labels):
        logits, _values = model_forward(model, capture, clip, phase_d)
        target = torch.tensor([int(label)], dtype=torch.long, device=clip.device)
        ce = functional.cross_entropy(logits, target)
        total += float(ce.detach().item()) / len(clips)
        (ce / float(len(clips))).backward()
    return total


def ltr_backward(model: Any, capture: PhaseE0Capture, clips: Sequence[Any],
                 pair_ops: Sequence[Any], student_sens: Sequence[Sequence[Sequence[float]]],
                 teacher_sens: Sequence[Sequence[Sequence[float]]],
                 unit_rows: Sequence[Mapping[str, Any]], alive: Sequence[bool],
                 core: Any, phase_d: Any, torch: Any,
                 relation_loss_fn: Any) -> Tuple[float, List[Dict[str, Any]]]:
    domains = [str(row["domain_id"]) for row in unit_rows]
    uids = [int(row["task037_global_index"]) for row in unit_rows]
    # Compute dL/de on a compact sensitivity tensor, then apply its exact VJP
    # to differentiable, recomputed student activations one pair at a time.
    # This avoids retaining six full Swin graphs while never detaching student
    # activations in the gradient path.
    meta_student = torch.tensor(student_sens, dtype=torch.float32, requires_grad=True)
    meta_teacher = torch.tensor(teacher_sens, dtype=torch.float32)
    meta_loss, meta_details = relation_loss_fn(
        meta_student, meta_teacher, domains, alive, uids,
        EPS_NORMALIZATION, EPS_PEARSON)
    coefficients = torch.autograd.grad(meta_loss, meta_student)[0].detach()
    total_loss = float(meta_loss.detach().item())
    detail_rows = []
    for item in meta_details:
        detail_rows.append({
            "video_position": int(item["video_position"]),
            "unit_i": int(item["unit_i"]), "unit_j": int(item["unit_j"]),
            "domain_id": str(item["domain_id"]),
            "d_student": float(item["d_student"].detach().item()),
            "d_teacher": float(item["d_teacher"].detach().item()),
            "squared_error": float(item["squared_error"].detach().item()),
            "absolute_drift": float(item["absolute_drift"].detach().item()),
        })
    del meta_student, meta_teacher, meta_loss, meta_details
    for video_pos, clip in enumerate(clips):
        _base_logits, h0 = model_forward(model, capture, clip, phase_d)
        del _base_logits
        capture.qkv_v.clear()
        for pair_pos, intervention in enumerate(pair_ops):
            swapped = core.apply_temporal_interventions(clip, [intervention], time_dim=1)[0]
            _pair_logits, h1 = model_forward(model, capture, swapped, phase_d)
            e_q = torch.stack([PRIMITIVES.relative_sensitivity(
                h0[uid], h1[uid], EPS_SENSITIVITY) for uid in uids])
            coeff = coefficients[video_pos, :, pair_pos].to(
                device=e_q.device, dtype=e_q.dtype)
            surrogate = (coeff * e_q).sum()
            del _pair_logits
            surrogate.backward(retain_graph=pair_pos < len(pair_ops) - 1)
            del h1, swapped, e_q, coeff, surrogate
            capture.clear()
            capture.qkv_v.clear()
        del h0
    return total_loss, detail_rows


def no_grad_relation(sens: Sequence[Sequence[Sequence[float]]],
                     teacher_sens: Sequence[Sequence[Sequence[float]]],
                     unit_rows: Sequence[Mapping[str, Any]], alive: Sequence[bool],
                     torch: Any, relation_loss_fn: Any) -> Tuple[float, List[Dict[str, Any]]]:
    student_tensor = torch.tensor(sens, dtype=torch.float32)
    teacher_tensor = torch.tensor(teacher_sens, dtype=torch.float32)
    domains = [str(row["domain_id"]) for row in unit_rows]
    uids = [int(row["task037_global_index"]) for row in unit_rows]
    loss, details = relation_loss_fn(student_tensor, teacher_tensor, domains, alive, uids,
                                     EPS_NORMALIZATION, EPS_PEARSON)
    numeric = []
    for item in details:
        numeric.append({
            "video_position": int(item["video_position"]),
            "unit_i": int(item["unit_i"]), "unit_j": int(item["unit_j"]),
            "domain_id": str(item["domain_id"]),
            "d_student": float(item["d_student"].detach().item()),
            "d_teacher": float(item["d_teacher"].detach().item()),
            "squared_error": float(item["squared_error"].detach().item()),
            "absolute_drift": float(item["absolute_drift"].detach().item()),
        })
    value = float(loss.detach().item())
    return value, numeric


def alive_mask(unit_rows: Sequence[Mapping[str, Any]], gate_state: Any) -> List[bool]:
    return [float(gate_state.values.get(int(row["task037_global_index"]), 1.0)) > 0.0
            for row in unit_rows]


def restore_exact(student: Any, teacher_logits: Sequence[Any], clips: Sequence[Any],
                  capture: PhaseE0Capture, gate_state: Any, phase_d: Any,
                  torch: Any) -> bool:
    gate_state.reset()
    with torch.no_grad():
        for clip, expected in zip(clips, teacher_logits):
            logits, _values = model_forward(student, capture, clip, phase_d)
            if not torch.equal(logits, expected):
                return False
    return all(float(value) == 1.0 for value in gate_state.values.values())


def target_row_by_id(rows: Sequence[Mapping[str, Any]], uid: int) -> Mapping[str, Any]:
    return next(row for row in rows if int(row["task037_global_index"]) == int(uid))


def report_relation_rows(target: Mapping[str, Any], gate: float,
                         details: Sequence[Mapping[str, Any]], ltr: float,
                         drift: float, video_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for item in details:
        pos = int(item["video_position"])
        video = video_rows[pos]
        result.append({
            "target_task037_global_index": int(target["task037_global_index"]),
            "target_domain_id": str(target["domain_id"]),
            "target_unit_type": target["unit_type"], "gate_value": float(gate),
            "video_index": int(video["video_index"]), "video_id": video["video_id"],
            "domain_id": str(item["domain_id"]), "unit_i": int(item["unit_i"]),
            "unit_j": int(item["unit_j"]), "d_teacher": item["d_teacher"],
            "d_student": item["d_student"], "squared_error": item["squared_error"],
            "absolute_drift": item["absolute_drift"], "config_L_TR": ltr,
            "config_Drift": drift,
        })
    return result


def mean_field(details: Sequence[Mapping[str, Any]], field: str) -> float:
    return statistics.mean(float(item[field]) for item in details)


def run_optimizer_sanity(teacher: Any, teacher_sens: Sequence[Sequence[Sequence[float]]],
                         clips: Sequence[Any], labels: Sequence[int], video_rows: Sequence[Mapping[str, Any]],
                         unit_rows: Sequence[Mapping[str, Any]], pair_ops: Sequence[Any],
                         target_id: int, core: Any, phase_d: Any, torch: Any,
                         functional: Any, relation_loss_fn: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    trace_rows: List[Dict[str, Any]] = []
    results: Dict[str, Any] = {}
    init_hashes = {}
    for method in ("CE_ONLY", "CE_PLUS_LTR"):
        model = copy.deepcopy(teacher)
        model.eval()
        model.requires_grad_(True)
        state = PRIMITIVES.GateState([int(r["task037_global_index"]) for r in unit_rows
                           if int(r["task037_global_index"]) in GATE_TARGETS])
        state.set_one(target_id, 0.5)
        capture = PhaseE0Capture(model, unit_rows, state, torch, detach=False)
        init_hashes[method] = model_parameter_sha(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=OPTIMIZER_LR)
        alive = alive_mask(unit_rows, state)
        last_grad_metrics = {"total_grad_norm": 0.0, "finite": True, "nonzero": False}
        initial_drift = None
        for step in range(OPTIMIZER_STEPS + 1):
            sens, ce_value, _base_acts, _base_logits = collect_no_grad(
                model, capture, state, clips, labels, pair_ops, unit_rows, core, phase_d,
                torch, functional)
            ltr_value, details = no_grad_relation(sens, teacher_sens, unit_rows, alive,
                                                 torch, relation_loss_fn)
            drift = mean_field(details, "absolute_drift")
            if step == 0:
                initial_drift = drift
            trace_rows.append({
                "method": method, "step": step, "fixed_gate_target": target_id,
                "fixed_gate_value": 0.5, "optimizer": "SGD", "learning_rate": OPTIMIZER_LR,
                "CE": ce_value, "L_TR": ltr_value, "relation_drift": drift,
                "gradient_norm": last_grad_metrics["total_grad_norm"],
                "gradient_finite": last_grad_metrics["finite"],
                "trainable_gate_parameters": 0, "video_indices": json.dumps(list(VIDEO_INDICES)),
                "clip_count": len(clips), "batch_reused_each_step": True,
            })
            if step == OPTIMIZER_STEPS:
                results[method] = {"initial_drift": initial_drift,
                                   "final_drift": drift, "final_CE": ce_value,
                                   "final_L_TR": ltr_value,
                                   "final_gradient_finite": last_grad_metrics["finite"]}
                break
            model.zero_grad(set_to_none=True)
            ce_backward(model, capture, clips, labels, phase_d, torch, functional)
            if method == "CE_PLUS_LTR":
                ltr_backward(model, capture, clips, pair_ops, sens, teacher_sens, unit_rows,
                             alive, core, phase_d, torch, relation_loss_fn)
            grad_metrics = finite_gradient_metrics(model, torch)
            require(grad_metrics["finite"], "non-finite optimization sanity gradient")
            optimizer.step()
            require(all(bool(torch.isfinite(p).all().item()) for p in model.parameters()),
                    "optimizer sanity produced a non-finite parameter")
            last_grad_metrics = grad_metrics
        capture.close()
        state.reset()
        results[method]["initial_parameter_sha256"] = init_hashes[method]
        del capture, state, optimizer, model
        torch.cuda.empty_cache()
    same_initialization = init_hashes["CE_ONLY"] == init_hashes["CE_PLUS_LTR"]
    ce_rows = [r for r in trace_rows if r["method"] == "CE_ONLY"]
    plus_rows = [r for r in trace_rows if r["method"] == "CE_PLUS_LTR"]
    results["same_initialization"] = same_initialization
    results["ce_only_initial_drift"] = float(ce_rows[0]["relation_drift"])
    results["ce_plus_ltr_initial_drift"] = float(plus_rows[0]["relation_drift"])
    results["ce_only_final_drift"] = float(ce_rows[-1]["relation_drift"])
    results["ce_plus_ltr_final_drift"] = float(plus_rows[-1]["relation_drift"])
    results["ltr_reduces_own_initial_drift"] = (
        results["ce_plus_ltr_final_drift"] < results["ce_plus_ltr_initial_drift"])
    results["ltr_beats_ce_only_final_drift"] = (
        results["ce_plus_ltr_final_drift"] < results["ce_only_final_drift"])
    return trace_rows, results


def make_report(summary: Mapping[str, Any]) -> str:
    decision = str(summary["decision"])
    return "\n".join([
        "# Task042 Phase E.0：时序关系保持损失可行性审计",
        "",
        "本阶段仅验证可行性，没有进行 50% 剪枝、全量微调、物理剪枝或 Lambda 搜索。候选集合及 F3 顺序保持 Phase D 冻结状态。",
        "",
        "## 决策",
        "",
        "**%s**" % decision,
        "",
        "- 检查点：`%s`（SHA-256 `%s`）。" % (summary["checkpoint"], summary["checkpoint_sha256"]),
        "- 结构门控目标：same-type 域 269 的 Attention head 774；mixed 域 271 的 FFN neuron 328；二者均为 Phase D 域内冻结 F3 rank 1。",
        "- 同域关系参照：两个冻结域中的 7 个 Task042 unit；包含 269 的纯 head 域和 271 的 head/FFN 混合域。",
        "- 校准数据：Task042 冻结视频清单中 video_index 0 与 3；帧对为各 span 的固定 pair_index 0，具体 frame identity 记录在 summary JSON。",
        "- 关系定义：对 5 个帧对敏感度向量做可微 RMS 标准化；`d=(1-rho)/2`，Pearson 分母使用 `sqrt(||x_c||²||y_c||²+1e-12)`；`L_TR=mean((d_S-d_T)^2)`。梯度按精确链式 VJP `dL/de · de/dθ` 分帧对重算；学生激活保留计算图，只释放已反传的帧对图。",
        "- 门控为临时 Python 浮点乘法常数，不是参数；Attention 在 `A@V` 前按 head 缩放，FFN 在 post-GELU/pre-fc2 按 neuron 缩放。",
        "- 梯度与训练 sanity 使用相同两段视频、固定 g=0.5、等权 `CE + L_TR`，SGD %d 步，lr=%g；没有调节 Lambda。" %
        (OPTIMIZER_STEPS, OPTIMIZER_LR),
        "",
        "## 核心结果",
        "",
        "- 全部 gate/梯度/损失数值有限：`%s`。" % summary["all_numeric_finite"],
        "- gate 复位后原模型 logits 精确恢复：`%s`；只读门控阶段参数哈希保持：`%s`。" %
        (summary["all_restorations_exact"], summary["parameters_unchanged_during_gate_audit"]),
        "- 非平凡门控下的非零 L_TR 梯度：`%s`；g=0 时 surviving unit 仍获得梯度：`%s`。" %
        (summary["nonzero_ltr_gradients_at_nontrivial_gates"], summary["zero_gate_survivors_receive_gradient"]),
        "- 初始 relation drift：CE-only `%0.8g`，CE+L_TR `%0.8g`；3 步后分别 `%0.8g` 与 `%0.8g`。" %
        (summary["optimization"]["ce_only_initial_drift"],
         summary["optimization"]["ce_plus_ltr_initial_drift"],
         summary["optimization"]["ce_only_final_drift"],
         summary["optimization"]["ce_plus_ltr_final_drift"]),
        "- L_TR 相对初始 drift 有改善：`%s`；最终 drift 低于 CE-only：`%s`。" %
        (summary["optimization"]["ltr_reduces_own_initial_drift"],
         summary["optimization"]["ltr_beats_ce_only_final_drift"]),
        "",
        "正结果仅说明损失在这个 2-video、7-unit、5-pair 诊断子集上可反传并通过小步 sanity；不授权完整渐进剪枝实验。gate=0 时按规范排除被移除 unit 的关系对，因此 pair 集在零门控点发生预期的结构性缩减，连续性需结合 drift CSV 中的 shared-pair 对照读取。",
        "",
        "## 输出",
        "",
        *["- `%s`" % name for name in REQUIRED_OUTPUTS],
        "",
    ])


def run(args: argparse.Namespace) -> Dict[str, Any]:
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    require(args.physical_gpu in (0, 1), "only physical GPU 0 or 1 may be used")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(args.physical_gpu),
            "CUDA_VISIBLE_DEVICES must expose only the chosen physical GPU")
    for name in REQUIRED_OUTPUTS:
        require(not (output / name).exists(), "refusing to overwrite Phase-E.0 output " + name)
    units, targets, phase_d_summary, phase_d_identity = verify_phase_d_inputs(repo)

    runtime = str(repo / "src" / "lgfr_runtime")
    scripts = str(repo / "scripts")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    global PRIMITIVES
    import task042_phase_e0_temporal_preservation as primitives
    PRIMITIVES = primitives
    GateState = primitives.GateState
    relation_loss = primitives.relation_loss
    import task042_frame_relation_redundancy as task042
    import task042_phase_c_joint_mask_oracle as phase_c
    import task041_phase_d_fullval_oracle as phase_d

    import torch
    import torch.nn.functional as functional
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "worker must see exactly one isolated CUDA device")
    torch.set_num_threads(2)
    torch.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")

    config = json.loads((BASE_OUTPUT / "task042_run_config.json").read_text(encoding="utf-8"))
    video_manifest = read_csv(BASE_OUTPUT / "task042_video_manifest.csv")
    clips, labels, video_rows = load_clips(task042._runtime_modules(repo)[2], config,
                                           video_manifest, torch, device)
    core, probe, ctfrs = task042._runtime_modules(repo)
    pair_ops, pair_rows = build_pair_subset(core)
    teacher, adapter, identity = task042._model_and_identity(
        PROJECT_ROOT, CHECKPOINT, device, probe, ctfrs)
    require(identity.get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA and
            not identity.get("missing_keys") and not identity.get("unexpected_keys") and
            not identity.get("shape_mismatches") and
            identity.get("classifier_head", {}).get("status") == "loaded",
            "teacher checkpoint/classifier identity is not authoritative")
    teacher.eval()
    teacher.requires_grad_(False)
    target_ids = [int(row["task037_global_index"]) for row in targets]
    teacher_state = GateState([])
    teacher_capture = PhaseE0Capture(teacher, units, teacher_state, torch, detach=True)
    teacher_sens, teacher_ce, teacher_base_acts, teacher_logits = collect_no_grad(
        teacher, teacher_capture, teacher_state, clips, labels, pair_ops, units,
        core, phase_d, torch, functional)
    teacher_capture.close()
    require(all(not value.requires_grad for sample in teacher_base_acts
                for value in sample.values()), "teacher activations were not detached")

    student = copy.deepcopy(teacher)
    student.eval()
    student.requires_grad_(True)
    audit_state = GateState(target_ids)
    capture = PhaseE0Capture(student, units, audit_state, torch, detach=False)
    parameter_sha_before = model_parameter_sha(student)
    gate_rows, gradient_rows, relation_rows, drift_rows = [], [], [], []
    all_gradients_finite = True
    nonzero_nontrivial = True
    zero_gate_survivor_grad = True
    all_restore = True
    previous_drift: Dict[int, Tuple[float, Dict[Tuple[int, int, int], float],
                                   Dict[Tuple[int, int, int], float]]] = {}

    for target in targets:
        uid = int(target["task037_global_index"])
        for gate in GATE_VALUES:
            audit_state.set_one(uid, gate)
            sens, ce_value, base_acts, _student_logits = collect_no_grad(
                student, capture, audit_state, clips, labels, pair_ops, units,
                core, phase_d, torch, functional)
            alive = alive_mask(units, audit_state)
            ltr_value, details = no_grad_relation(sens, teacher_sens, units, alive,
                                                 torch, relation_loss)
            drift_value = mean_field(details, "absolute_drift")
            scale_error = relative_error(base_acts[0][uid],
                                         float(gate) * teacher_base_acts[0][uid], torch)
            pair_keys = {(int(item["video_position"]), int(item["unit_i"]), int(item["unit_j"]))
                         for item in details}
            per_pair_drift = {(int(item["video_position"]), int(item["unit_i"]), int(item["unit_j"])):
                              float(item["absolute_drift"]) for item in details}
            per_pair_d_student = {(int(item["video_position"]), int(item["unit_i"]), int(item["unit_j"])):
                                  float(item["d_student"]) for item in details}
            prev = previous_drift.get(uid)
            common_keys = pair_keys.intersection(prev[1]) if prev else set()
            shared_drift_change = (statistics.mean(abs(per_pair_drift[k] - prev[1][k])
                                                    for k in common_keys)
                                   if common_keys else 0.0)
            max_shared_relation_change = (max(abs(per_pair_d_student[k] - prev[2][k])
                                              for k in common_keys)
                                          if common_keys else 0.0)
            pair_set_changed = bool(prev and pair_keys != set(prev[1]))
            previous_drift[uid] = (drift_value, per_pair_drift, per_pair_d_student)

            student.zero_grad(set_to_none=True)
            ce_backward_value = ce_backward(student, capture, clips, labels,
                                            phase_d, torch, functional)
            ce_grad = finite_gradient_metrics(student, torch)
            student.zero_grad(set_to_none=True)
            vjp_ltr, vjp_details = ltr_backward(
                student, capture, clips, pair_ops, sens, teacher_sens, units, alive,
                core, phase_d, torch, relation_loss)
            ltr_grad = finite_gradient_metrics(student, torch)
            gradient_finite = bool(ce_grad["finite"] and ltr_grad["finite"] and
                                   math.isfinite(ltr_value) and math.isfinite(ce_value))
            all_gradients_finite = all_gradients_finite and gradient_finite
            if gate < 1.0:
                nonzero_nontrivial = nonzero_nontrivial and bool(ltr_grad["nonzero"])
            if gate == 0.0:
                zero_gate_survivor_grad = zero_gate_survivor_grad and bool(ltr_grad["nonzero"])
            all_restore = restore_exact(student, teacher_logits, clips, capture,
                                        audit_state, phase_d, torch) and all_restore
            alive_pair_count = len(details) // len(clips)
            gate_rows.append({
                "target_task037_global_index": uid, "domain_id": target["domain_id"],
                "domain_group": target["domain_group"], "unit_type": target["unit_type"],
                "layer": target["layer"], "unit_index": target["unit_index"],
                "frozen_removal_rank": target["frozen_removal_rank"],
                "f3_global_step": target["f3_global_step"], "gate_value": gate,
                "gate_trainable": False, "activation_gate_exact_relative_error": scale_error,
                "alive_relation_pairs_per_video": alive_pair_count,
                "CE": ce_value, "L_TR": ltr_value,
                "chain_rule_vjp_L_TR": vjp_ltr,
                "vjp_vs_readonly_L_TR_abs_error": abs(vjp_ltr - ltr_value),
                "gate_reset_to_one": all(float(v) == 1.0 for v in audit_state.values.values()),
                "restored_logits_exact": all_restore,
            })
            gradient_row = {
                "target_task037_global_index": uid, "domain_id": target["domain_id"],
                "unit_type": target["unit_type"], "gate_value": gate,
                "CE": ce_value, "L_TR": ltr_value,
                "CE_gradient_norm": ce_grad["total_grad_norm"],
                "L_TR_student_gradient_norm": ltr_grad["total_grad_norm"],
                "L_TR_gradient_finite": ltr_grad["finite"],
                "CE_gradient_finite": ce_grad["finite"],
                "L_TR_gradient_nonzero": ltr_grad["nonzero"],
                "alive_relation_pairs_per_video": alive_pair_count,
                "gate_zero_removed_unit_required_to_receive_gradient": gate != 0.0,
                "surviving_parameter_gradients_verified": ltr_grad["nonzero"],
            }
            for stage in ("stage0", "stage1", "stage2", "stage3", "other"):
                gradient_row[stage + "_gradient_norm_from_L_TR"] = ltr_grad[stage + "_grad_norm"]
            gradient_rows.append(gradient_row)
            relation_rows.extend(report_relation_rows(target, gate, details,
                                                       ltr_value, drift_value, video_rows))
            # The shared-pair drift change is reported separately from pair-set changes at g=0.
            drift_rows.append({
                "target_task037_global_index": uid, "domain_id": target["domain_id"],
                "unit_type": target["unit_type"], "gate_value": gate,
                "Drift_mean_abs_d": drift_value,
                "Drift_max_abs_d": max(float(item["absolute_drift"]) for item in details),
                "L_TR": ltr_value, "relation_pair_count_across_videos": len(details),
                "alive_relation_pairs_per_video": alive_pair_count,
                "shared_pair_mean_abs_drift_change_from_previous_gate": shared_drift_change,
                "shared_pair_max_abs_relation_change_from_previous_gate": max_shared_relation_change,
                "pair_set_changed_at_this_gate": pair_set_changed,
                "zero_gate_structural_pair_exclusion": bool(gate == 0.0),
                "pair_identity_set": json.dumps(sorted([list(key) for key in pair_keys]),
                                                 separators=(",", ":")),
            })
            student.zero_grad(set_to_none=True)
            del sens, base_acts, details, vjp_details, _student_logits
            torch.cuda.empty_cache()

    capture.close()
    audit_state.reset()
    bare_restore_exact = True
    with torch.no_grad():
        for clip, expected in zip(clips, teacher_logits):
            actual = phase_d.unwrap_logits(student(clip.unsqueeze(0)))
            bare_restore_exact = bare_restore_exact and bool(torch.equal(actual, expected))
    all_restore = all_restore and bare_restore_exact
    parameter_sha_after = model_parameter_sha(student)
    parameters_unchanged = parameter_sha_before == parameter_sha_after
    require(parameters_unchanged, "read-only gate audit changed student model parameters")
    del capture, audit_state, student
    del teacher_base_acts
    torch.cuda.empty_cache()

    optimization_rows, optimization = run_optimizer_sanity(
        teacher, teacher_sens, clips, labels, video_rows, units, pair_ops,
        328, core, phase_d, torch, functional, relation_loss)
    if not optimization["same_initialization"]:
        optimization["same_initialization"] = False
    gate_nontrivial_rows = [row for row in gradient_rows if float(row["gate_value"]) < 1.0]
    at_zero_rows = [row for row in gradient_rows if float(row["gate_value"]) == 0.0]
    nonzero_nontrivial = all(bool(row["L_TR_gradient_nonzero"]) for row in gate_nontrivial_rows)
    zero_gate_survivor_grad = all(bool(row["surviving_parameter_gradients_verified"])
                                  for row in at_zero_rows)
    all_numeric_finite = (all_gradients_finite and
        all(math.isfinite(float(row[key])) for row in gate_rows
            for key in ("CE", "L_TR", "activation_gate_exact_relative_error")) and
        all(math.isfinite(float(row[key])) for row in gradient_rows
            for key in ("CE", "L_TR", "CE_gradient_norm", "L_TR_student_gradient_norm")) and
        all(math.isfinite(float(row[key])) for row in relation_rows
            for key in ("d_teacher", "d_student", "squared_error", "absolute_drift",
                        "config_L_TR", "config_Drift")) and
        all(math.isfinite(float(row[key])) for row in drift_rows
            for key in ("Drift_mean_abs_d", "Drift_max_abs_d", "L_TR")) and
        all(math.isfinite(float(row[key])) for row in optimization_rows
            for key in ("CE", "L_TR", "relation_drift", "gradient_norm")))
    no_restore_violations = all_restore and parameters_unchanged
    optimization_finite = all(bool(row["gradient_finite"]) for row in optimization_rows)
    a_conditions = (all_numeric_finite and nonzero_nontrivial and zero_gate_survivor_grad and
                    no_restore_violations and optimization_finite and
                    optimization["ltr_reduces_own_initial_drift"] and
                    optimization["ltr_beats_ce_only_final_drift"] and
                    optimization["same_initialization"])
    if a_conditions:
        decision = "A. TEMPORAL_RELATION_PRESERVATION_LOSS_FEASIBLE"
    elif not all_numeric_finite or not no_restore_violations:
        decision = "C. TEMPORAL_RELATION_PRESERVATION_LOSS_REJECTED"
    elif not optimization_finite:
        decision = "C. TEMPORAL_RELATION_PRESERVATION_LOSS_REJECTED"
    elif not nonzero_nontrivial or not zero_gate_survivor_grad:
        decision = "B. TEMPORAL_RELATION_PRESERVATION_LOSS_NUMERICALLY_UNRESOLVED"
    else:
        decision = "C. TEMPORAL_RELATION_PRESERVATION_LOSS_REJECTED"

    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "task": "Task042 Phase E.0 temporal-relation preservation-loss feasibility only",
        "branch": BRANCH, "git_head": git_value(repo, "rev-parse", "HEAD"),
        "physical_gpu": int(args.physical_gpu), "gpu_name": torch.cuda.get_device_name(0),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "phase_d_head": EXPECTED_PHASE_D_HEAD,
        "phase_d_candidate_provenance_sha256": sha256_file(
            BASE_OUTPUT / "phase_d/task042_phase_d_candidate_provenance.csv"),
        "candidate_gate_targets_in_frozen_order": [
            {k: row[k] for k in ("task037_global_index", "domain_id", "unit_type", "layer",
                                 "unit_index", "frozen_removal_rank", "f3_global_step")}
            for row in targets],
        "relation_unit_ids_by_domain": {
            domain: [int(row["task037_global_index"]) for row in units
                     if str(row["domain_id"]) == domain]
            for domain in (SAME_TYPE_DOMAIN, MIXED_DOMAIN)},
        "videos": [{"video_index": int(row["video_index"]),
                    "dataset_index": int(row["dataset_index"]),
                    "video_id": row["video_id"], "label": int(row["label"])}
                   for row in video_rows],
        "pair_interventions": pair_rows,
        "gate_values": list(GATE_VALUES), "only_gpu_0_or_1": True,
        "dtype": "float32", "amp": False,
        "teacher_frozen": True, "student_checkpoint_copy": True,
        "gate_trainable_parameters": 0,
        "relation_definition": {
            "sensitivity": "||H_i(S_q X_v)-H_i(X_v)||_F/(||H_i(X_v)||_F+1e-12)",
            "normalization": "center by selected-pair mean; divide by sqrt(population variance + 1e-8)",
            "pearson": "center normalized vectors; rho=sum(xc*yc)/sqrt(sum(xc^2)*sum(yc^2)+1e-12); clamp[-1,1]",
            "distance": "(1-rho)/2", "loss": "mean same-domain alive-pair (d_student-d_teacher)^2",
            "gate_zero": "unit and its relation pairs excluded exactly at gate=0",
        },
        "student_gradient_path": "exact chain-rule vector-Jacobian product over frame-pair sensitivity scalars; per-pair student activations are recomputed with autograd enabled and are not detached",
        "optimization": {**optimization, "steps": OPTIMIZER_STEPS,
                         "optimizer": "SGD", "learning_rate": OPTIMIZER_LR,
                         "objective": "CE + 1.0 * L_TR; no lambda search"},
        "teacher_mean_CE_on_diagnostic_subset": teacher_ce,
        "student_parameter_sha256_before_gate_audit": parameter_sha_before,
        "student_parameter_sha256_after_gate_audit": parameter_sha_after,
        "parameters_unchanged_during_gate_audit": parameters_unchanged,
        "all_restorations_exact": all_restore,
        "all_numeric_finite": all_numeric_finite,
        "nonzero_ltr_gradients_at_nontrivial_gates": nonzero_nontrivial,
        "zero_gate_survivors_receive_gradient": zero_gate_survivor_grad,
        "optimization_finite": optimization_finite,
        "decision": decision,
        "scope": "No physical pruning, no full fine-tuning, no 50% pruning, no lambda tuning, no Task043.",
    }
    # Save the required audit files only after every experiment and restoration check succeeds.
    write_csv_new(output / REQUIRED_OUTPUTS[0], gate_rows,
                  ("target_task037_global_index", "domain_id", "domain_group", "unit_type",
                   "layer", "unit_index", "frozen_removal_rank", "f3_global_step", "gate_value",
                   "gate_trainable", "activation_gate_exact_relative_error",
                   "alive_relation_pairs_per_video", "CE", "L_TR", "chain_rule_vjp_L_TR",
                   "vjp_vs_readonly_L_TR_abs_error", "gate_reset_to_one", "restored_logits_exact"))
    write_csv_new(output / REQUIRED_OUTPUTS[1], relation_rows,
                  ("target_task037_global_index", "target_domain_id", "target_unit_type", "gate_value",
                   "video_index", "video_id", "domain_id", "unit_i", "unit_j", "d_teacher",
                   "d_student", "squared_error", "absolute_drift", "config_L_TR", "config_Drift"))
    grad_fields = ["target_task037_global_index", "domain_id", "unit_type", "gate_value", "CE", "L_TR",
                   "CE_gradient_norm", "L_TR_student_gradient_norm", "L_TR_gradient_finite",
                   "CE_gradient_finite", "L_TR_gradient_nonzero", "alive_relation_pairs_per_video",
                   "gate_zero_removed_unit_required_to_receive_gradient", "surviving_parameter_gradients_verified"]
    grad_fields += [stage + "_gradient_norm_from_L_TR" for stage in
                    ("stage0", "stage1", "stage2", "stage3", "other")]
    write_csv_new(output / REQUIRED_OUTPUTS[2], gradient_rows, grad_fields)
    write_csv_new(output / REQUIRED_OUTPUTS[3], drift_rows,
                  ("target_task037_global_index", "domain_id", "unit_type", "gate_value",
                   "Drift_mean_abs_d", "Drift_max_abs_d", "L_TR", "relation_pair_count_across_videos",
                   "alive_relation_pairs_per_video", "shared_pair_mean_abs_drift_change_from_previous_gate",
                   "shared_pair_max_abs_relation_change_from_previous_gate",
                   "pair_set_changed_at_this_gate", "zero_gate_structural_pair_exclusion", "pair_identity_set"))
    write_csv_new(output / REQUIRED_OUTPUTS[4], optimization_rows,
                  ("method", "step", "fixed_gate_target", "fixed_gate_value", "optimizer",
                   "learning_rate", "CE", "L_TR", "relation_drift", "gradient_norm",
                   "gradient_finite", "trainable_gate_parameters", "video_indices", "clip_count",
                   "batch_reused_each_step"))
    summary = {
        **identity,
        "outputs": {name: {"path": str(output / name),
                           "sha256": sha256_file(output / name)}
                    for name in REQUIRED_OUTPUTS
                    if name not in (REQUIRED_OUTPUTS[5], REQUIRED_OUTPUTS[6])},
        "gate_audit_row_count": len(gate_rows),
        "gradient_audit_row_count": len(gradient_rows),
        "relation_pair_row_count": len(relation_rows),
        "relation_drift_row_count": len(drift_rows),
        "optimization_sanity_row_count": len(optimization_rows),
        "mean_CE_by_gate": statistics.mean(float(row["CE"]) for row in gate_rows),
        "mean_L_TR_by_gate": statistics.mean(float(row["L_TR"]) for row in gate_rows),
        "mean_L_TR_gradient_norm_by_gate": statistics.mean(
            float(row["L_TR_student_gradient_norm"]) for row in gradient_rows),
        "max_vjp_vs_readonly_L_TR_abs_error": max(
            float(row["vjp_vs_readonly_L_TR_abs_error"]) for row in gate_rows),
    }
    report = make_report(summary)
    report_path = output / REQUIRED_OUTPUTS[6]
    require(not report_path.exists(), "refusing to overwrite Phase-E.0 report")
    report_path.write_text(report, encoding="utf-8")
    summary["outputs"][REQUIRED_OUTPUTS[6]] = {"path": str(report_path),
                                               "sha256": sha256_file(report_path)}
    summary_path = output / REQUIRED_OUTPUTS[5]
    write_json_new(summary_path, summary)
    print("PHASE_E0_DECISION=" + decision, flush=True)
    print("PHASE_E0_OUTPUT=" + str(output), flush=True)
    print("PHASE_E0_ALL_RESTORATIONS_EXACT=" + str(all_restore), flush=True)
    print("PHASE_E0_LTR_REDUCES_DRIFT=" +
          str(optimization["ltr_reduces_own_initial_drift"]), flush=True)
    return summary


def relative_error(observed: Any, expected: Any, torch: Any) -> float:
    expected = expected.detach()
    return float(((observed.detach() - expected).reshape(-1).norm() /
                  (expected.reshape(-1).norm() + EPS_SENSITIVITY)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
