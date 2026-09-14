#!/usr/bin/env python3
"""Task042 Phase D: frozen F3 within-domain order with temporal-coverage allocation."""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess, sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

REPO_DEFAULT = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
OUTPUT_DEFAULT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy/phase_d")
PROJECT_ROOT = Path("/home/jixinye25/jxy_work1/swintrans_task035")
CHECKPOINT = Path("/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
VAL_LIST = Path("/data/jixinye25/UCF101_Frame/val_rgb_split1.txt")
FRAME_ROOT = Path("/data/jixinye25/UCF101_Frame/frames")
PHASE_C_OUTPUT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
TASK037_ROOT = Path("/home/jixinye25/jxy_work1/swintrans_task037_complete")
TASK037_SOURCE = TASK037_ROOT / "pruning/task034_mid_veto_50_logical_finetune.py"
TASK037_TRACE = OUTPUT_DEFAULT / "task042_phase_d_f3_authoritative_global_trace.csv"
TRACE_PREFLIGHT = OUTPUT_DEFAULT / "task042_phase_d_f3_trace_preflight.json"
EXPECTED_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
EXPECTED_CLIPS = 3783
EXPECTED_TASK037_COMMIT_PREFIX = "5524625"
EXPECTED_SOURCE_SHA = "af792c112a9a8ae92562aab21e582ac22a92459f2f692918c71e7e5fd427f593a"
BRANCH = "task_042_post_bms_frame_relation_redundancy"
OUTPUT_NAMES = (
 "task042_phase_d_candidate_provenance.csv","task042_phase_d_baseline_path.csv",
 "task042_phase_d_temporal_progressive_path.csv","task042_phase_d_checkpoint_manifest.csv",
 "task042_phase_d_joint_mask_damage.csv","task042_phase_d_matched_budget_comparison.csv",
 "task042_phase_d_domain_allocation.csv","task042_phase_d_mixed_domain_analysis.csv",
 "task042_phase_d_summary.json","task042_phase_d_report.md")
sys.path.insert(0, str(REPO_DEFAULT / "src" / "lgfr_runtime"))
sys.path.insert(0, str(REPO_DEFAULT / "scripts"))
import task042_phase_d_progressive_path as pathlib_d
import task042_phase_c_joint_mask_oracle as phase_c


def req(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError("Task042 Phase-D gate failed: " + message)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    req(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require_clean(repo: Path) -> str:
    req(git_value(repo, "rev-parse", "--abbrev-ref", "HEAD") == BRANCH,
        "must stay on the existing Task042 branch")
    req(not git_value(repo, "status", "--porcelain"),
        "Task042 checkout must be clean before prepare/preflight/inference")
    return git_value(repo, "rev-parse", "HEAD")


def unit_map(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {}
    for raw in rows:
        row = dict(raw)
        uid = int(row["task037_global_index"])
        req(uid not in result, "duplicate exact Task042 unit identity")
        row["task037_global_index"] = uid
        row["domain_id"] = str(row["domain_id"])
        result[uid] = row
    return result


def check_trace_authority() -> tuple[dict[str, Any], list[dict[str, str]]]:
    preflight = read_json(TRACE_PREFLIGHT)
    req(preflight.get("status") == "PASS", "Task037 exact-order trace preflight is not PASS")
    req(preflight.get("original_sequence_exact_prefix_reproduced") is True,
        "F3 continuation did not reproduce the original production prefix")
    req(preflight.get("selector_source_sha256") == EXPECTED_SOURCE_SHA and
        sha256(TASK037_SOURCE) == EXPECTED_SOURCE_SHA,
        "authoritative Task037 selector source hash changed")
    req(git_value(TASK037_ROOT, "rev-parse", "HEAD").startswith(EXPECTED_TASK037_COMMIT_PREFIX),
        "Task037 complete-code source checkout changed")
    req(sha256(TASK037_TRACE) == preflight.get("trace_sha256"),
        "extended authoritative F3 trace hash differs from preflight")
    rows = phase_c.read_csv(TASK037_TRACE)
    req(len(rows) == int(preflight.get("extended_trace_rows", -1)),
        "extended Task037 trace row count differs from preflight")
    original_path = Path(preflight["original_trace"])
    req(original_path.is_file() and sha256(original_path) == preflight["original_trace_sha256"],
        "original Task037 50-percent trace identity changed")
    original = phase_c.read_csv(original_path)
    req(len(original) == int(preflight["original_trace_rows"]) == 26520,
        "original F3 prefix length changed")
    fields = tuple(original[0])
    req(all(field in rows[0] for field in fields), "extended trace schema changed")
    for index, old in enumerate(original):
        req(all(old[field] == rows[index][field] for field in fields),
            "extended F3 trace does not reproduce exact original row " + str(index + 1))
    return preflight, rows


def eligible_distances(unit_rows: Sequence[Mapping[str, Any]],
                       distance_rows: Sequence[Mapping[str, Any]]) -> dict:
    by_domain = {domain: [] for domain in pathlib_d.DOMAIN_IDS}
    for row in unit_rows:
        domain = str(row["domain_id"])
        if domain in by_domain:
            by_domain[domain].append(int(row["task037_global_index"]))
    expected = set()
    for domain, members in by_domain.items():
        members.sort()
        req(len(members) >= 2, "BMS domain has fewer than two tested units: " + domain)
        for pos, left in enumerate(members):
            for right in members[pos + 1:]:
                expected.add((domain, left, right))
    observed = set()
    for row in distance_rows:
        domain = str(row["domain_id"])
        if domain not in by_domain:
            continue
        left, right = int(row["task037_global_index_i"]), int(row["task037_global_index_j"])
        key = (domain, min(left, right), max(left, right))
        req(key not in observed, "duplicate frozen d_temp pair: " + repr(key))
        observed.add(key)
    req(observed == expected and len(distance_rows) == len(expected),
        "frozen full_10x3 d_temp matrix is incomplete or contains extraneous pairs")
    return pathlib_d.distance_index(distance_rows)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    head = require_clean(repo)
    output.mkdir(parents=True, exist_ok=True)
    extra_names = ["task042_phase_d_prepared.json", "task042_phase_c_prepared.json",
                   "task042_phase_d_preflight.json", "task042_phase_d_temporal_proposal_table.csv",
                   "task042_phase_d_unique_mask_manifest.csv", "task042_phase_d_input_identity.json"]
    req(not any((output / name).exists() for name in (*OUTPUT_NAMES, *extra_names)),
        "refusing to overwrite Phase-D outputs from an earlier attempt")
    trace_identity, trace_rows = check_trace_authority()
    input_paths = {
        "task042_unit_manifest.csv": PHASE_C_OUTPUT / "task042_unit_manifest.csv",
        "task042_temporal_pair_distance.csv": PHASE_C_OUTPUT / "task042_temporal_pair_distance.csv"}
    for path in input_paths.values():
        req(path.is_file(), "frozen Task042 input missing: " + str(path))
    raw_units = phase_c.read_csv(input_paths["task042_unit_manifest.csv"])
    units = unit_map(raw_units)
    req(len(units) == 51, "frozen Task042 unit manifest must contain 51 units")
    eligible_count = sum(row["domain_id"] in pathlib_d.DOMAIN_IDS for row in units.values())
    req(eligible_count == 31, "eligible 10-domain cohort must contain exactly 31 tested units")
    distance_rows = phase_c.read_csv(input_paths["task042_temporal_pair_distance.csv"])
    distances = eligible_distances(raw_units, distance_rows)
    provenance, removal_ids, candidates = pathlib_d.candidate_provenance(raw_units, trace_rows)
    req(len(provenance) == 31 and sum(map(len, removal_ids.values())) == 21,
        "exact Task037 F3 projection must yield 21 removals and 10 final representatives")
    total_parameters = int(float(trace_identity["total_parameters"]))
    req(float(total_parameters) == float(trace_identity["total_parameters"]),
        "Task037 model parameter total is not an exact integer")
    baseline_path, temporal_path, proposals = pathlib_d.build_progressive_paths(
        raw_units, provenance, removal_ids, candidates, distances, total_parameters)
    removable_parameters = sum(int(row["DeltaP"]) for row in candidates.values())
    checkpoint_rows, evals = pathlib_d.build_checkpoint_manifest(
        baseline_path, temporal_path, raw_units, removal_ids, distances, removable_parameters)

    eligible_ids = {uid for uid, row in units.items() if row["domain_id"] in pathlib_d.DOMAIN_IDS}
    worker_rows, unique_rows = [], []
    for index, (eid, item) in enumerate(sorted(evals.items())):
        masked = json.loads(item["masked_task037_ids"])
        retained = sorted(eligible_ids - set(masked))
        row = {
            "evaluation_index": index, "mask_evaluation_id": eid,
            "domain_id": "ALL_ELIGIBLE_BMS_DOMAINS", "domain_group": "all",
            "unit_count": len(eligible_ids), "keep_count": len(retained),
            "method": "joint_path_checkpoint", "shared_methods": "",
            "retained_task037_ids": json.dumps(retained, separators=(",", ":")),
            "masked_task037_ids": json.dumps(masked, separators=(",", ":")),
            "masked_attention_count": int(item["masked_attention_count"]),
            "masked_ffn_count": int(item["masked_ffn_count"]),
            "set_relation": "PATH_STATE", "temporal_orientation_status": "NOT_APPLICABLE",
            "informative_directional_pair": False,
            "removed_parameters": int(item["removed_parameters"]),
            "cohort_parameter_ratio": float(item["cohort_parameter_ratio"]),
            "domain_J_by_domain": item["domain_J_by_domain"],
            "current_total_temporal_coverage": float(item["current_total_temporal_coverage"]),
            "max_domain_J": float(item["max_domain_J"]), "contexts": item["contexts"]}
        worker_rows.append(row)
        unique_rows.append(dict(row))
    req(bool(worker_rows), "no nonempty joint-mask states exist")
    req(all(set(json.loads(row["masked_task037_ids"])).issubset(eligible_ids) for row in worker_rows),
        "path state contains a unit outside the frozen tested cohort")

    baseline = phase_c.baseline_cache()
    req(sha256(CHECKPOINT) == EXPECTED_SHA, "authoritative checkpoint SHA changed")
    baseline_identity = baseline["identity"]
    hashes = {name: {"path": str(path), "sha256": sha256(path)}
              for name, path in input_paths.items()}
    hashes.update({
        "task037_f3_extended_trace": {"path": str(TASK037_TRACE), "sha256": sha256(TASK037_TRACE)},
        "task037_f3_trace_preflight": {"path": str(TRACE_PREFLIGHT), "sha256": sha256(TRACE_PREFLIGHT)}})
    identity = {
        "task042_branch": BRANCH, "task042_head": head,
        "task037_branch": trace_identity["task037_branch"],
        "task037_commit": trace_identity["task037_code_commit"],
        "task037_selector_source": str(TASK037_SOURCE),
        "task037_selector_source_sha256": sha256(TASK037_SOURCE),
        "task037_ordering_key": trace_identity["ordering_key"],
        "task037_trace_sha256": sha256(TASK037_TRACE), "task037_trace_rows": len(trace_rows),
        "original_50pct_trace_sha256": trace_identity["original_trace_sha256"],
        "original_50pct_rows_exactly_reproduced": True,
        "directional_order_interpretation": (
            "Project the exact selected sequence from the authoritative dynamic Task037 F3 global trace "
            "onto each frozen BMS domain. Freeze each domain's first n-1 projected IDs and their recorded "
            "F3 score provenance; baseline uses their original global trace order."),
        "eligible_domain_ids": list(pathlib_d.DOMAIN_IDS),
        "eligible_unit_count": eligible_count, "removal_candidate_count": len(candidates),
        "one_final_representative_per_domain": True, "frozen_input_hashes": hashes,
        "checkpoint_sha256": EXPECTED_SHA, "validation_list_sha256": sha256(VAL_LIST),
        "baseline_cache_sha256": baseline["sha256"], "baseline_identity": baseline_identity,
        "total_model_parameters_from_task037_trace": total_parameters,
        "removable_cohort_parameters": removable_parameters}
    phase_c.write_csv_new(output / OUTPUT_NAMES[0], provenance)
    phase_c.write_csv_new(output / OUTPUT_NAMES[1], baseline_path)
    phase_c.write_csv_new(output / OUTPUT_NAMES[2], temporal_path)
    phase_c.write_csv_new(output / "task042_phase_d_temporal_proposal_table.csv", proposals)
    phase_c.write_csv_new(output / OUTPUT_NAMES[3], checkpoint_rows)
    phase_c.write_csv_new(output / "task042_phase_d_unique_mask_manifest.csv", unique_rows)
    phase_c.write_json_new(output / "task042_phase_d_input_identity.json", identity)

    unit_config = {str(uid): {key: row.get(key, "") for key in (
        "task037_global_index", "layer", "unit_type", "capture_kind", "unit_index", "domain_id", "stage")}
        for uid, row in units.items()}
    baseline_config = {key: baseline[key] for key in (
        "path", "identity_path", "sha256", "mean_ce", "top1", "top5")}
    common = {
        "task": "Task042 Phase D temporal-coverage progressive path pilot",
        "required_branch": BRANCH, "prepared_head": head, "repo": str(repo),
        "output": str(output), "project_root": str(PROJECT_ROOT),
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_SHA,
        "val_list": str(VAL_LIST), "val_list_sha256": sha256(VAL_LIST),
        "frame_root": str(FRAME_ROOT), "baseline_cache": baseline_config,
        "baseline_identity": baseline_identity, "unit_rows": unit_config,
        "frozen_inputs": hashes, "evaluation_count": len(worker_rows),
        "checkpoint_rows": checkpoint_rows, "evaluation_rows": worker_rows,
        "input_identity": identity, "cohort_removable_parameters": removable_parameters,
        "total_model_parameters": total_parameters, "dtype": "torch.float32", "amp": False,
        "physical_gpus": [0, 1], "batch_size": 4, "workers_per_gpu": 2,
        "mask_manifest_path": str(output / "task042_phase_d_unique_mask_manifest.csv"),
        "mask_manifest_sha256": sha256(output / "task042_phase_d_unique_mask_manifest.csv")}
    phase_c.write_json_new(output / "task042_phase_d_prepared.json", common)
    phase_c_config = dict(common)
    phase_c_config.update({
        "manifest_path": common["mask_manifest_path"],
        "manifest_sha256": common["mask_manifest_sha256"], "pair_rows": [],
        "fixed_protocol": {"validation_clips": EXPECTED_CLIPS, "dtype": "float32",
                           "amp": False, "physical_gpus": [0, 1], "batch_size": 4,
                           "workers_per_gpu": 2}})
    phase_c.write_json_new(output / "task042_phase_c_prepared.json", phase_c_config)
    return {"status": "PREPARE_OK", "task042_head": head, "eligible_units": eligible_count,
            "removal_candidates": len(candidates), "cohort_removable_parameters": removable_parameters,
            "baseline_steps": len(baseline_path), "temporal_steps": len(temporal_path),
            "checkpoint_rows": len(checkpoint_rows), "unique_joint_masks": len(worker_rows),
            "baseline_cache_reused": True, "phase_d_output": str(output)}


def current_config(repo: Path, output: Path) -> dict[str, Any]:
    config = read_json(output / "task042_phase_d_prepared.json")
    phase_c.current_head(repo, config)
    req(sha256(output / "task042_phase_d_unique_mask_manifest.csv") ==
        config["mask_manifest_sha256"], "unique-mask manifest changed")
    for key, item in config["frozen_inputs"].items():
        req(sha256(item["path"]) == item["sha256"], "frozen input changed: " + key)
    return config


def actual_parameter_cost(spec: Any, unit_type: str) -> int:
    module = spec.module
    if unit_type == pathlib_d.ATTENTION:
        return pathlib_d.attention_head_parameter_cost(
            int(module.qkv.in_features), int(module.head_dim), module.qkv.bias is not None)
    if unit_type == pathlib_d.FFN:
        return pathlib_d.ffn_neuron_parameter_cost(
            int(module.fc1.in_features), int(module.fc2.out_features), module.fc1.bias is not None)
    raise RuntimeError("unknown validated unit type: " + str(unit_type))


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    config = current_config(repo, output)
    phase_c.validate_cache(config)
    req(sha256(config["checkpoint"]) == EXPECTED_SHA, "checkpoint changed before preflight")
    if str(repo / "src" / "lgfr_runtime") not in sys.path:
        sys.path.insert(0, str(repo / "src" / "lgfr_runtime"))
    import task041_phase_d_fullval_oracle as phase_d
    model, identity, specs = phase_d.load_model(
        Path(config["project_root"]), Path(config["checkpoint"]), torch.device("cpu"))
    req(identity.get("checkpoint_sha256") == EXPECTED_SHA and
        identity.get("classifier_head", {}).get("status") == "loaded" and
        not identity.get("missing_keys") and not identity.get("unexpected_keys"),
        "checkpoint/classifier identity gate failed")
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    req(parameter_count == int(config["total_model_parameters"]),
        "Video Swin parameter total differs from authoritative Task037 total")
    by_spec = phase_c.spec_index(specs)
    candidate_rows = phase_c.read_csv(output / OUTPUT_NAMES[0])
    checked = []
    for row in candidate_rows:
        if row.get("selected_for_removal") not in (True, "True"):
            continue
        uid = str(row["task037_global_index"])
        unit = config["unit_rows"][uid]
        kind = pathlib_d.canonical_unit_type(unit["unit_type"])
        spec_kind = "head" if kind == pathlib_d.ATTENTION else "neuron"
        key = (str(unit["layer"]), spec_kind)
        req(key in by_spec, "model missing exact candidate layer " + repr(key))
        spec, index = by_spec[key], int(unit["unit_index"])
        req(0 <= index < int(spec.num_units), "candidate unit index exceeds model width")
        measured, recorded = actual_parameter_cost(spec, kind), int(float(row["DeltaP"]))
        req(measured == recorded, "Task037 DeltaP differs from exact model effect for unit " + uid)
        checked.append({"task037_global_index": int(uid), "domain_id": row["domain_id"],
                        "unit_type": kind, "layer": unit["layer"], "unit_index": index,
                        "DeltaP": measured})
    req(len(checked) == 21, "not all 21 candidate parameter effects were validated")
    req(sum(row["DeltaP"] for row in checked) == int(config["cohort_removable_parameters"]),
        "structural parameter reductions do not sum to frozen cohort budget")
    ident = config["baseline_identity"]
    req(ident.get("checkpoint_sha256") == EXPECTED_SHA and ident.get("valid_clips") == EXPECTED_CLIPS and
        ident.get("dtype") == "torch.float32" and ident.get("amp") is False and
        ident.get("same_videoswin_ucf101_preprocessing") is True and
        ident.get("same_classification_head_semantics") is True,
        "cached unmasked baseline protocol is incompatible")
    result = {
        "status": "PREFLIGHT_OK", "task042_head": config["prepared_head"],
        "task037_trace_sha256": config["input_identity"]["task037_trace_sha256"],
        "checkpoint_sha256": EXPECTED_SHA, "validation_clips": EXPECTED_CLIPS,
        "baseline_cache_sha256": config["baseline_cache"]["sha256"], "baseline_cache_reused": True,
        "model_parameter_count": parameter_count, "candidate_parameter_effects_checked": len(checked),
        "cohort_removable_parameters": config["cohort_removable_parameters"],
        "unique_joint_masks": config["evaluation_count"], "dtype": "torch.float32",
        "amp": False, "physical_gpus_allowed": [0, 1],
        "attention_candidates": sum(r["unit_type"] == pathlib_d.ATTENTION for r in checked),
        "ffn_candidates": sum(r["unit_type"] == pathlib_d.FFN for r in checked),
        "candidate_costs": checked}
    phase_c.write_json_new(output / "task042_phase_d_preflight.json", result)
    result.pop("candidate_costs")
    return result


def worker(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    current_config(repo, output)
    return phase_c.worker(SimpleNamespace(
        repo=str(repo), output=str(output), num_shards=2,
        shard_index=args.shard_index, physical_gpu=args.physical_gpu))


def auc(points: Sequence[tuple[float, float]]) -> float:
    compact = []
    for x, y in sorted(points):
        if compact and math.isclose(x, compact[-1][0], rel_tol=0.0, abs_tol=1e-15):
            compact[-1] = (x, y)
        else:
            compact.append((x, y))
    if not compact or compact[0][0] > 0:
        compact.insert(0, (0.0, 0.0))
    return sum((x1-x0)*(y0+y1)/2 for (x0,y0),(x1,y1) in zip(compact,compact[1:]))


def json_set(value: Any) -> set[int]:
    return {int(v) for v in json.loads(str(value))}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    config = current_config(repo, output)
    preflight_path = output / "task042_phase_d_preflight.json"
    req(preflight_path.is_file() and read_json(preflight_path).get("status") == "PREFLIGHT_OK",
        "GPU run requires a passing model-identity preflight")
    for name in OUTPUT_NAMES[4:]:
        req(not (output / name).exists(), "refusing to overwrite final artifact " + name)
    by_eval = {}
    for shard in (0, 1):
        csv_path, json_path = output / f"task042_phase_c_shard_{shard}.csv", output / f"task042_phase_c_shard_{shard}.json"
        req(csv_path.is_file() and json_path.is_file(), "both GPU workers must complete")
        meta = read_json(json_path)
        req(meta.get("status") == "WORKER_OK" and meta.get("physical_gpu") == shard and
            meta.get("all_masks_verified") is True and meta.get("all_masks_restored_exact") is True and
            meta.get("validation_clips_per_evaluation") == EXPECTED_CLIPS,
            "worker identity/full-validation/restoration gate failed")
        for row in phase_c.read_csv(csv_path):
            eid = row["mask_evaluation_id"]
            req(eid not in by_eval, "duplicate mask evaluation across worker shards")
            req(int(row["validation_clips"]) == EXPECTED_CLIPS and
                row["mask_restored_exact"] == "True" and row["restored_logits_exact"] == "True",
                "mask result lacks full-validation or exact-restoration evidence")
            by_eval[eid] = row
    expected = {row["mask_evaluation_id"] for row in config["evaluation_rows"]}
    req(set(by_eval) == expected, "GPU shards do not exactly cover unique-mask manifest")
    baseline = config["baseline_cache"]
    indexed, damage_rows = {}, []
    for row in config["checkpoint_rows"]:
        eid = row["mask_set_id"]
        if eid == "BASELINE_CACHE":
            metric = {"mean_ce_increase":0.0,"top1_change_percentage_points":0.0,
                      "top5_change_percentage_points":0.0,"prediction_flip_rate":0.0,
                      "validation_clips":EXPECTED_CLIPS,"mask_restored_exact":True,
                      "restored_logits_exact":True,"worker_physical_gpu":"baseline_cache"}
        else:
            req(eid in by_eval, "checkpoint refers to an unevaluated mask " + eid)
            metric = by_eval[eid]
        combined = dict(row)
        for key in ("mean_ce_increase","top1_change_percentage_points",
                    "top5_change_percentage_points","prediction_flip_rate"):
            combined[key] = float(metric[key])
        for key in ("validation_clips","mask_restored_exact","restored_logits_exact","worker_physical_gpu"):
            combined[key] = metric[key]
        damage_rows.append(combined)
        indexed[(row["comparison_id"],row["path"])] = combined

    comparisons = []
    comparison_ids = sorted({row["comparison_id"] for row in config["checkpoint_rows"]})
    for cid in comparison_ids:
        base, temporal = indexed[(cid,"baseline")], indexed[(cid,"temporal_progressive")]
        comparisons.append({
            "comparison_id":cid,"checkpoint_kind":base["checkpoint_kind"],
            "requested_cohort_ratio":base["requested_cohort_ratio"],
            "requested_parameter_budget":int(base["requested_parameter_budget"]),
            "baseline_prefix_steps":int(base["prefix_steps"]),
            "temporal_prefix_steps":int(temporal["prefix_steps"]),
            "baseline_removed_parameters":int(base["achieved_removed_parameters"]),
            "temporal_removed_parameters":int(temporal["achieved_removed_parameters"]),
            "achieved_parameter_gap_temporal_minus_baseline":
                int(temporal["achieved_removed_parameters"])-int(base["achieved_removed_parameters"]),
            "baseline_achieved_cohort_ratio":float(base["achieved_cohort_parameter_ratio"]),
            "temporal_achieved_cohort_ratio":float(temporal["achieved_cohort_parameter_ratio"]),
            "baseline_mask_set_id":base["mask_set_id"],"temporal_mask_set_id":temporal["mask_set_id"],
            "informative_distinct_mask_pair":base["mask_set_id"]!=temporal["mask_set_id"],
            "baseline_mean_ce_increase":float(base["mean_ce_increase"]),
            "temporal_mean_ce_increase":float(temporal["mean_ce_increase"]),
            "temporal_minus_baseline_mean_ce":float(temporal["mean_ce_increase"])-float(base["mean_ce_increase"]),
            "baseline_top1_change_percentage_points":float(base["top1_change_percentage_points"]),
            "temporal_top1_change_percentage_points":float(temporal["top1_change_percentage_points"]),
            "temporal_minus_baseline_top1_percentage_points":
                float(temporal["top1_change_percentage_points"])-float(base["top1_change_percentage_points"]),
            "baseline_top5_change_percentage_points":float(base["top5_change_percentage_points"]),
            "temporal_top5_change_percentage_points":float(temporal["top5_change_percentage_points"]),
            "temporal_minus_baseline_top5_percentage_points":
                float(temporal["top5_change_percentage_points"])-float(base["top5_change_percentage_points"]),
            "baseline_prediction_flip_rate":float(base["prediction_flip_rate"]),
            "temporal_prediction_flip_rate":float(temporal["prediction_flip_rate"]),
            "temporal_minus_baseline_prediction_flip_rate":
                float(temporal["prediction_flip_rate"])-float(base["prediction_flip_rate"]),
            "temporal_lower_ce_damage":float(temporal["mean_ce_increase"])<float(base["mean_ce_increase"])})
    unique_pairs = {}
    for row in comparisons:
        if row["informative_distinct_mask_pair"]:
            unique_pairs.setdefault((row["baseline_mask_set_id"],row["temporal_mask_set_id"]),row)
    pairs = list(unique_pairs.values())
    wins_t = sum(bool(row["temporal_lower_ce_damage"]) for row in pairs)
    wins_b = sum(float(row["temporal_minus_baseline_mean_ce"])>0 for row in pairs)
    maj_t = bool(pairs and wins_t>len(pairs)/2)
    maj_b = bool(pairs and wins_b>len(pairs)/2)
    points = {"baseline":[(0.0,0.0)],"temporal_progressive":[(0.0,0.0)]}
    for row in damage_rows:
        points[row["path"]].append((float(row["achieved_cohort_parameter_ratio"]),
                                    float(row["mean_ce_increase"])))
    auc_b, auc_t = auc(points["baseline"]), auc(points["temporal_progressive"])
    unit_rows = unit_map([dict(row,task037_global_index=uid)
                          for uid,row in config["unit_rows"].items()])
    nonmixed_win = False
    for row in pairs:
        if not row["temporal_lower_ce_damage"]:
            continue
        base, temporal = indexed[(row["comparison_id"],"baseline")],indexed[(row["comparison_id"],"temporal_progressive")]
        changed = json_set(base["masked_task037_ids"]) ^ json_set(temporal["masked_task037_ids"])
        if any(unit_rows[uid]["domain_id"] not in pathlib_d.MIXED_DOMAIN_IDS for uid in changed):
            nonmixed_win = True
    if maj_t and auc_t<auc_b and nonmixed_win:
        decision="A. TEMPORAL_COVERAGE_PROGRESSIVE_PATH_PROMISING"
    elif maj_b and auc_b<auc_t:
        decision="C. TEMPORAL_COVERAGE_PROGRESSIVE_PATH_REJECTED"
    else:
        decision="B. TEMPORAL_COVERAGE_PROGRESSIVE_PATH_WEAK_OR_UNRESOLVED"

    provenance=phase_c.read_csv(output/OUTPUT_NAMES[0])
    base_path=phase_c.read_csv(output/OUTPUT_NAMES[1])
    temporal_path=phase_c.read_csv(output/OUTPUT_NAMES[2])
    proposals=phase_c.read_csv(output/"task042_phase_d_temporal_proposal_table.csv")
    alloc_rows=[]
    for domain in pathlib_d.DOMAIN_IDS:
        prov=[r for r in provenance if r["domain_id"]==domain]
        base=[r for r in base_path if r["domain_id"]==domain]
        temp=[r for r in temporal_path if r["domain_id"]==domain]
        alloc_rows.append({
            "domain_id":domain,"domain_group":"mixed" if domain in pathlib_d.MIXED_DOMAIN_IDS else "nonmixed",
            "original_tested_unit_count":len(prov),
            "frozen_within_domain_candidate_order":json.dumps(
                [int(r["task037_global_index"]) for r in prov if r["selected_for_removal"]=="True"],separators=(",",":")),
            "baseline_removal_steps":json.dumps([int(r["step"]) for r in base],separators=(",",":")),
            "temporal_accepted_steps":json.dumps([int(r["step"]) for r in temp],separators=(",",":")),
            "baseline_removed_ids":json.dumps([int(r["candidate_task037_global_index"]) for r in base],separators=(",",":")),
            "temporal_removed_ids":json.dumps([int(r["candidate_task037_global_index"]) for r in temp],separators=(",",":")),
            "baseline_removed_attention_count":sum(r["unit_type"]==pathlib_d.ATTENTION for r in base),
            "baseline_removed_ffn_count":sum(r["unit_type"]==pathlib_d.FFN for r in base),
            "temporal_removed_attention_count":sum(r["unit_type"]==pathlib_d.ATTENTION for r in temp),
            "temporal_removed_ffn_count":sum(r["unit_type"]==pathlib_d.FFN for r in temp),
            "baseline_final_J":json.loads(base[-1]["domain_J_by_domain"])[domain] if base else None,
            "temporal_final_J":json.loads(temp[-1]["domain_J_by_domain"])[domain] if temp else None,
            "candidate_directional_global_steps":json.dumps(
                [int(r["f3_global_step"]) for r in prov if r["selected_for_removal"]=="True"],separators=(",",":"))})
    proposal_by_id=defaultdict(list)
    for row in proposals:
        proposal_by_id[int(row["candidate_task037_global_index"])].append(row)
    base_step={int(r["candidate_task037_global_index"]):int(r["step"]) for r in base_path}
    temp_step={int(r["candidate_task037_global_index"]):int(r["step"]) for r in temporal_path}
    mixed_rows=[]
    for row in provenance:
        if row["domain_id"] not in pathlib_d.MIXED_DOMAIN_IDS or row["selected_for_removal"]!="True":
            continue
        gid=int(row["task037_global_index"])
        mixed_rows.append({
            "domain_id":row["domain_id"],"within_domain_removal_rank":int(row["within_domain_removal_rank"]),
            "task037_global_index":gid,"unit_type":row["unit_type"],"layer":row["layer"],
            "unit_index":int(row["unit_index"]),"Task037_f3_global_step":int(row["f3_global_step"]),
            "baseline_accepted_step":base_step[gid],"temporal_accepted_step":temp_step[gid],
            "temporal_proposal_observations":json.dumps([
                {"temporal_step":int(r["temporal_step"]),"DeltaJ":float(r["DeltaJ"]),
                 "DeltaP":int(r["DeltaP"]),"Gamma":float(r["Gamma"]),"accepted":r["accepted"]=="True"}
                for r in proposal_by_id[gid]],sort_keys=True,separators=(",",":"))})
    phase_c.write_csv_new(output/OUTPUT_NAMES[4],damage_rows)
    phase_c.write_csv_new(output/OUTPUT_NAMES[5],comparisons)
    phase_c.write_csv_new(output/OUTPUT_NAMES[6],alloc_rows)
    phase_c.write_csv_new(output/OUTPUT_NAMES[7],mixed_rows)
    all_masks=all(int(by_eval[e]["verified_mask_unit_count"])==len(json_set(by_eval[e]["masked_task037_ids"]))
                  for e in by_eval)
    restored=all(by_eval[e]["mask_restored_exact"]=="True" and
                 by_eval[e]["restored_logits_exact"]=="True" for e in by_eval)
    summary={
        "task":"Task042 Phase D temporal-coverage-driven progressive pruning path pilot",
        "decision":decision,"branch":BRANCH,"git_head":config["prepared_head"],
        "server":"jixinye25@192.168.0.12","checkpoint_sha256":EXPECTED_SHA,
        "baseline_cache_sha256":baseline["sha256"],"baseline_cache_reused":True,
        "baseline_mean_ce":baseline["mean_ce"],"baseline_top1":baseline["top1"],"baseline_top5":baseline["top5"],
        "validation_clips":EXPECTED_CLIPS,"dtype":"torch.float32","amp":False,"physical_gpus":[0,1],
        "eligible_domains":list(pathlib_d.DOMAIN_IDS),"eligible_units":31,"removal_candidates":21,
        "one_final_representative_per_domain":True,
        "cohort_removable_parameters":config["cohort_removable_parameters"],
        "total_model_parameters":config["total_model_parameters"],
        "unique_joint_mask_evaluations":len(by_eval),
        "primary_budget_checkpoint_count":sum(r["checkpoint_kind"]=="PRIMARY_BUDGET" for r in comparisons),
        "informative_distinct_mask_pair_count":len(pairs),"temporal_lower_ce_count":wins_t,
        "baseline_lower_ce_count":wins_b,"strict_majority_favors_temporal":maj_t,
        "baseline_damage_auc_ce_times_parameter_fraction":auc_b,
        "temporal_damage_auc_ce_times_parameter_fraction":auc_t,
        "aggregate_trajectory_favors_temporal":auc_t<auc_b,
        "nonmixed_domain_difference_at_temporal_win":nonmixed_win,
        "decision_gate":{"A_strict_majority_temporal":maj_t,"A_lower_temporal_trajectory_auc":auc_t<auc_b,
                         "A_temporal_win_involves_nonmixed_domain":nonmixed_win,
                         "C_strict_majority_baseline":maj_b,"C_lower_baseline_trajectory_auc":auc_b<auc_t},
        "directional_path_definition":config["input_identity"]["directional_order_interpretation"],
        "all_full_validation_counts":all(int(r["validation_clips"])==EXPECTED_CLIPS for r in damage_rows),
        "all_joint_masks_verified":all_masks,"all_restorations_exact":restored,
        "scope":"Temporary joint masking only; no physical pruning, fine-tuning, training, threshold, learned temporal score, quotas, or Task043.",
        "outputs":{}}
    for name in OUTPUT_NAMES[:8]:
        summary["outputs"][name]={"path":str(output/name),"sha256":sha256(output/name)}
    lines=[
        "# Task042 Phase D - Temporal-Coverage Progressive Path Pilot","",
        "## Decision","",f"**{decision}**","",
        "This tests whether frozen temporal-relation coverage can allocate compression across existing BMS domains. It does not claim temporal distance identifies unimportant individual units.","",
        "## Protocol and identity","",
        f"- Task042 branch/head: {BRANCH} / {config['prepared_head']}.",
        f"- Task037 F3 dynamic trace SHA-256: {config['input_identity']['task037_trace_sha256']}; original 26,520-row 50% prefix reproduced exactly.",
        "- Candidate order inside each domain is the exact projection of the Task037 production F3 global trace. Baseline preserves its global order; the temporal path changes only which domain is consumed next.",
        "- Frozen cohort: 31 units in 10 BMS domains; 21 removals while retaining one representative per domain.",
        f"- Removable parameter amount: {config['cohort_removable_parameters']:,} / {config['total_model_parameters']:,} model parameters.",
        f"- Full validation: {EXPECTED_CLIPS}/{EXPECTED_CLIPS}; checkpoint {EXPECTED_SHA}; FP32, AMP=False; baseline cache reused ({baseline['sha256']}).",
        f"- Unique joint masks: {len(by_eval)}; physical GPU 0/1 only; all masks verified/restored exactly: {all_masks} / {restored}.","",
        "## Preregistered damage comparison","",
        f"- Informative distinct-mask pairs: {len(pairs)}; temporal lower CE: {wins_t}; baseline lower CE: {wins_b}.",
        f"- CE-damage trajectory AUC (lower is better): baseline {auc_b:.10g}; temporal {auc_t:.10g}.",
        f"- At least one temporal-win pair changes a non-mixed domain: {nonmixed_win}.",
        "- A requires a strict majority of informative matched/nearest-lower budget pairs favoring temporal, lower trajectory AUC, and at least one temporal-win pair with a non-mixed-domain difference. Exact cumulative-parameter intersections are also listed.","",
        "|Budget|Baseline removed P|Temporal removed P|Baseline CE increase|Temporal CE increase|Temporal - baseline|Distinct masks|",
        "|---|---:|---:|---:|---:|---:|---|"]
    for row in comparisons:
        if row["checkpoint_kind"]!="PRIMARY_BUDGET":
            continue
        lines.append(f"|{row['requested_cohort_ratio']}|{row['baseline_removed_parameters']}|{row['temporal_removed_parameters']}|{row['baseline_mean_ce_increase']:.8g}|{row['temporal_mean_ce_increase']:.8g}|{row['temporal_minus_baseline_mean_ce']:.8g}|{row['informative_distinct_mask_pair']}|")
    lines += ["","## Domain allocation","","|Domain|Group|Baseline steps|Temporal accepted steps|Baseline IDs|Temporal IDs|","|---:|---|---|---|---|---|"]
    for row in alloc_rows:
        lines.append(f"|{row['domain_id']}|{row['domain_group']}|{row['baseline_removal_steps']}|{row['temporal_accepted_steps']}|{row['baseline_removed_ids']}|{row['temporal_removed_ids']}|")
    lines += ["","## Mixed domains 271/297","","Descriptive only; no Attention/FFN quotas were imposed and these domains alone do not establish type fairness.","","|Domain|Order|Unit|Type|Baseline step|Temporal step|","|---:|---:|---:|---|---:|---:|"]
    for row in mixed_rows:
        lines.append(f"|{row['domain_id']}|{row['within_domain_removal_rank']}|{row['task037_global_index']}|{row['unit_type']}|{row['baseline_accepted_step']}|{row['temporal_accepted_step']}|")
    lines += ["","## Scope","","Both paths use the same exact tested units, frozen within-domain Task037 F3 order, frozen d_temp matrix, and exact structural parameter benefits. Only domain allocation differs. Joint temporary masks were restored and checked; no physical pruning, global 50% pruning, or fine-tuning was performed.",""]
    report=output/OUTPUT_NAMES[9]
    req(not report.exists(),"refusing to overwrite Phase-D report")
    report.write_text("\n".join(lines),encoding="utf-8")
    summary["outputs"][OUTPUT_NAMES[9]]={"path":str(report),"sha256":sha256(report)}
    phase_c.write_json_new(output/OUTPUT_NAMES[8],summary)
    return {"status":"FINALIZE_OK","decision":decision,"unique_joint_masks":len(by_eval),
            "informative_pairs":len(pairs),"temporal_wins":wins_t,"baseline_wins":wins_b,
            "baseline_auc":auc_b,"temporal_auc":auc_t,"nonmixed_win":nonmixed_win,
            "report":str(report)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("prepare","preflight","worker","finalize"),required=True)
    parser.add_argument("--repo",default=str(REPO_DEFAULT))
    parser.add_argument("--output",default=str(OUTPUT_DEFAULT))
    parser.add_argument("--shard-index",type=int,choices=(0,1),default=0)
    parser.add_argument("--physical-gpu",type=int,choices=(0,1),default=0)
    args=parser.parse_args()
    if args.phase=="prepare": result=prepare(args)
    elif args.phase=="preflight": result=preflight(args)
    elif args.phase=="worker": result=worker(args)
    else: result=finalize(args)
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()

