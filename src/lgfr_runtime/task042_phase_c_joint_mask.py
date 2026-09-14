"""Task042 Phase-C joint temporary-mask invariants and evaluation identities."""
from __future__ import annotations
import contextlib, hashlib, json
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

class JointMaskError(RuntimeError):
    pass

class RestorationError(JointMaskError):
    pass

def canonical_ids(values: Iterable[Any]) -> List[int]:
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise JointMaskError("duplicate Task037 unit ID")
    return sorted(result)

def deterministic_mask_id(domain_id: Any, keep_count: int, masked_ids: Iterable[Any]) -> str:
    payload = json.dumps({"domain_id": str(domain_id), "keep_count": int(keep_count),
                          "masked_task037_ids": canonical_ids(masked_ids)},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "JM-" + hashlib.sha256(payload).hexdigest()[:20]

def build_pair_evaluations(domain_id: Any, domain_group: str, unit_ids: Iterable[Any],
                           keep_count: int, temporal_ids: Iterable[Any],
                           descriptor_ids: Iterable[Any], unit_types: Mapping[int, str],
                           temporal_orientation_status: str):
    universe = canonical_ids(unit_ids)
    temporal, descriptor = canonical_ids(temporal_ids), canonical_ids(descriptor_ids)
    k = int(keep_count)
    if not 0 < k < len(universe):
        raise JointMaskError("keep_count must be within the tested reduction range")
    if len(temporal) != k or len(descriptor) != k:
        raise JointMaskError("representative set size does not match keep_count")
    if not set(temporal).issubset(universe) or not set(descriptor).issubset(universe):
        raise JointMaskError("representative set contains a unit outside its BMS domain")
    if set(universe) != set(int(x) for x in unit_types):
        raise JointMaskError("unit type table does not exactly cover the BMS domain")
    tmask, dmask = sorted(set(universe)-set(temporal)), sorted(set(universe)-set(descriptor))
    differs = temporal != descriptor
    tied = temporal_orientation_status == "TEMPORAL_ORIENTATION_TIED"
    relation = "A_TEMPORAL_DIFFERS" if differs else "B_SETS_EQUAL"
    informative = bool(differs and not tied)

    def make_eval(method, retained, masked, shared=""):
        return {
            "mask_evaluation_id": deterministic_mask_id(domain_id, k, masked),
            "domain_id": str(domain_id), "domain_group": str(domain_group),
            "unit_count": len(universe), "keep_count": k, "method": method,
            "shared_methods": shared,
            "retained_task037_ids": json.dumps(retained, separators=(",", ":")),
            "masked_task037_ids": json.dumps(masked, separators=(",", ":")),
            "masked_attention_count": sum(unit_types[i] in ("head", "attention_head") for i in masked),
            "masked_ffn_count": sum(unit_types[i] in ("neuron", "ffn_neuron") for i in masked),
            "set_relation": relation, "temporal_orientation_status": temporal_orientation_status,
        }
    if differs:
        evaluations = [make_eval("temporal", temporal, tmask),
                       make_eval("descriptor", descriptor, dmask)]
    else:
        evaluations = [make_eval("shared_temporal_descriptor", temporal, tmask,
                                 "temporal|descriptor")]
    pair = {
        "domain_id": str(domain_id), "domain_group": str(domain_group),
        "unit_count": len(universe), "keep_count": k,
        "temporal_retained_task037_ids": json.dumps(temporal, separators=(",", ":")),
        "descriptor_retained_task037_ids": json.dumps(descriptor, separators=(",", ":")),
        "temporal_mask_evaluation_id": deterministic_mask_id(domain_id, k, tmask),
        "descriptor_mask_evaluation_id": deterministic_mask_id(domain_id, k, dmask),
        "set_relation": relation, "temporal_orientation_status": temporal_orientation_status,
        "informative_directional_pair": informative,
        "mixed_domain_secondary": str(domain_group) == "mixed",
    }
    return pair, evaluations

def _version_signature(model):
    values = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            version = int(tensor._version)
        except Exception as exc:
            raise JointMaskError("cannot read tensor version counter: " + name) from exc
        values.append((name, version))
    return tuple(values)

def _hook_signature(modules):
    result = []
    for module in modules:
        hooks = getattr(module, "_forward_pre_hooks", None)
        if hooks is None:
            raise JointMaskError("mask target lacks a forward pre-hook table")
        result.append((id(module), tuple((int(key), id(fn)) for key, fn in hooks.items())))
    return tuple(result)

def _capture_observer(spec, indices, calls, torch):
    kind = str(spec.unit_type)
    if kind not in ("head", "neuron"):
        raise JointMaskError("unsupported validated Task040 unit type: " + kind)
    def observe(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise JointMaskError("mask target did not receive an activation tensor")
        values = inputs[0]
        if kind == "head":
            head_dim, num_units = int(spec.module.head_dim), int(spec.num_units)
            if int(values.shape[-1]) != num_units * head_dim:
                raise JointMaskError("attention activation width differs from Task040 mask semantics")
            view = values.reshape(*values.shape[:-1], num_units, head_dim)
            for index in indices:
                if int(torch.count_nonzero(view[..., index, :]).item()) != 0:
                    raise JointMaskError("intended attention head was not fully masked")
                calls[(id(spec.hook_module), int(index))] += 1
        else:
            if int(values.shape[-1]) != int(spec.num_units):
                raise JointMaskError("FFN activation width differs from Task040 mask semantics")
            for index in indices:
                if int(torch.count_nonzero(values[..., index]).item()) != 0:
                    raise JointMaskError("intended FFN neuron was not fully masked")
                calls[(id(spec.hook_module), int(index))] += 1
    return observe

@contextlib.contextmanager
def joint_temporary_unit_masks(model, records, mask_factory, torch):
    """Apply existing Task040 whole-unit masks together and enforce exact restore."""
    if not records:
        raise JointMaskError("joint mask request is empty")
    normalized, seen_ids, seen_coordinates, grouped = [], set(), set(), {}
    for raw in records:
        spec, unit_id, unit_index = raw["spec"], int(raw["task037_global_index"]), int(raw["unit_index"])
        kind = str(spec.unit_type)
        coordinate = (str(spec.name), kind, unit_index)
        if unit_id in seen_ids or coordinate in seen_coordinates:
            raise JointMaskError("duplicate-mask rejection: duplicate ID or unit coordinate")
        if kind not in ("head", "neuron"):
            raise JointMaskError("unsupported Task040 mask type: " + kind)
        if unit_index < 0 or unit_index >= int(spec.num_units):
            raise JointMaskError("unit_index is outside the validated whole-unit width")
        seen_ids.add(unit_id); seen_coordinates.add(coordinate)
        normalized.append((spec, unit_index, unit_id))
        group = grouped.setdefault(id(spec.hook_module), {"spec": spec, "indices": []})
        if group["spec"] is not spec and (str(group["spec"].name), str(group["spec"].unit_type)) != (str(spec.name), kind):
            raise JointMaskError("conflicting unit specs share one hook module")
        group["indices"].append(unit_index)
    modules = [g["spec"].hook_module for g in grouped.values()]
    hooks_before, versions_before = _hook_signature(modules), _version_signature(model)
    stack, observers, calls = contextlib.ExitStack(), [], defaultdict(int)
    try:
        for spec, unit_index, _uid in normalized:
            stack.enter_context(mask_factory(spec, unit_index))
        hooks_masked = _hook_signature(modules)
        delta = sum(len(r[1]) for r in hooks_masked) - sum(len(r[1]) for r in hooks_before)
        if delta != len(normalized):
            raise JointMaskError("not every requested whole-unit mask installed exactly one hook")
        for group in grouped.values():
            spec = group["spec"]
            observers.append(spec.hook_module.register_forward_pre_hook(
                _capture_observer(spec, group["indices"], calls, torch)))
    except Exception:
        stack.close()
        raise
    try:
        yield calls
        expected = [(id(g["spec"].hook_module), index)
                    for g in grouped.values() for index in g["indices"]]
        if any(calls.get(key, 0) == 0 for key in expected):
            raise JointMaskError("one or more intended units were never verified masked")
    finally:
        for handle in reversed(observers):
            handle.remove()
        stack.close()
        if _hook_signature(modules) != hooks_before or _version_signature(model) != versions_before:
            raise RestorationError("joint mask did not restore exact hook tables and parameter/buffer version counters")
