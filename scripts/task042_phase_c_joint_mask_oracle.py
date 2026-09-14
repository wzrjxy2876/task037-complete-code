#!/usr/bin/env python3
"""Task042 Phase C: full-validation oracle for simultaneous post-BMS unit masks."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, statistics, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_DEFAULT = Path("/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy")
OUTPUT_DEFAULT = Path("/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy")
PROJECT_DEFAULT = Path("/home/jixinye25/jxy_work1/swintrans_task035")
CHECKPOINT = Path("/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt")
VAL_LIST = Path("/data/jixinye25/UCF101_Frame/val_rgb_split1.txt")
FRAME_ROOT = Path("/data/jixinye25/UCF101_Frame/frames")
PHASE_D = Path("/data/jixinye25/work1/output/task041_phase_d_fullval_resolution_audit")
EXPECTED_SHA = "4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63"
EXPECTED_CLIPS = 3783
BRANCH = "task_042_post_bms_frame_relation_redundancy"
EXISTING_STATUS = ("UNAVAILABLE: Task037 uses an adaptive global functional-score budget trace; "
                   "no exact fixed-domain/fixed-keep representative rule exists for these sets")
STRONG, CONTROLS, MIXED = {"415", "400", "103", "113"}, {"11", "76"}, {"271", "297"}
PAIRS = [("415",1),("415",2),("400",1),("400",2),("103",1),("103",2),("113",1),("113",2),
         ("11",1),("76",1),("76",2),("271",1),("271",2),("271",3),("297",1),("297",2),("297",3)]
OUTPUT_NAMES = ("task042_phase_c_evaluation_manifest.csv",
 "task042_phase_c_joint_mask_damage.csv","task042_phase_c_temporal_vs_descriptor.csv",
 "task042_phase_c_strong_vs_diverse.csv","task042_phase_c_radius_damage.csv",
 "task042_phase_c_coverage_regret.csv","task042_phase_c_mixed_domain.csv",
 "task042_phase_c_summary.json","task042_phase_c_report.md")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "lgfr_runtime"))
from task042_phase_c_joint_mask import build_pair_evaluations, canonical_ids, joint_temporary_unit_masks

def require(ok, message):
    if not ok:
        raise RuntimeError("Task042 Phase-C gate failed: " + message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def write_csv_new(path, rows):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def write_json_new(path, value):
    path = Path(path)
    require(not path.exists(), "refusing to overwrite " + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")

def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

def parse_ids(value):
    result = json.loads(str(value))
    require(isinstance(result, list), "representative IDs are not a JSON list")
    return canonical_ids(result)

def norm_int(value):
    return int(float(str(value).strip()))

def group_for(domain):
    if domain in STRONG: return "strong_structured"
    if domain in CONTROLS: return "diverse_control"
    if domain in MIXED: return "mixed"
    raise RuntimeError("unexpected Phase-C domain " + domain)

def canonical_type(value):
    aliases = {"head":"attention_head", "attention_head":"attention_head",
               "neuron":"ffn_neuron", "ffn_neuron":"ffn_neuron"}
    value = str(value).strip()
    require(value in aliases, "unknown Task037 unit type " + value)
    return aliases[value]

def baseline_cache():
    identity_path = PHASE_D / "task041_phase_d_baseline_identity.json"
    sample_path = PHASE_D / "task041_fullval_baseline_per_sample.csv"
    require(identity_path.is_file() and sample_path.is_file(), "Phase-D baseline cache is missing")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    require(identity.get("checkpoint_sha256") == EXPECTED_SHA and
            Path(identity.get("checkpoint_absolute_path","")).resolve() == CHECKPOINT.resolve(),
            "cached baseline checkpoint identity mismatch")
    require(identity.get("val_list_sha256") == sha256(VAL_LIST) and
            Path(identity.get("val_list_absolute_path","")).resolve() == VAL_LIST.resolve(),
            "cached baseline validation-list identity mismatch")
    require(Path(identity.get("frame_root","")).resolve() == FRAME_ROOT.resolve(),
            "cached baseline frame-root mismatch")
    require(identity.get("valid_clips") == EXPECTED_CLIPS and
            identity.get("baseline_rows") == EXPECTED_CLIPS,
            "cached baseline clip count mismatch")
    require(identity.get("same_videoswin_ucf101_preprocessing") is True and
            identity.get("same_classification_head_semantics") is True and
            identity.get("dtype") == "torch.float32" and identity.get("amp") is False,
            "cached baseline protocol/precision is not authoritative")
    rows = read_csv(sample_path)
    required = {"dataset_index","label","true_class_logit","cross_entropy",
                "top1_correct","top5_correct","predicted_class"}
    require(len(rows) == EXPECTED_CLIPS and bool(rows) and required.issubset(rows[0]),
            "baseline row count or schema mismatch")
    rows.sort(key=lambda row: norm_int(row["dataset_index"]))
    lines = [line.split() for line in VAL_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == EXPECTED_CLIPS, "validation list is not exactly 3783 clips")
    ces, top1, top5 = [], [], []
    for index, (row, parts) in enumerate(zip(rows, lines)):
        require(norm_int(row["dataset_index"]) == index, "baseline indices are not exact 0..3782")
        require(len(parts) >= 3 and norm_int(row["label"]) == norm_int(parts[-1]),
                "baseline label differs from validation-list row " + str(index))
        values = [float(row[x]) for x in ("true_class_logit","cross_entropy")]
        require(all(math.isfinite(x) for x in values), "baseline has non-finite logits/CE")
        ces.append(float(row["cross_entropy"]))
        top1.append(int(row["top1_correct"]))
        top5.append(int(row["top5_correct"]))
    return {"identity":identity,"path":str(sample_path),"identity_path":str(identity_path),
            "sha256":sha256(sample_path),"rows":rows,"mean_ce":statistics.mean(ces),
            "top1":statistics.mean(top1),"top5":statistics.mean(top5)}

def build_plan(output):
    units_path = output / "task042_unit_manifest.csv"
    rep_path, desc_path, margin_path = (output/"task042_phase_b_representative_sets.csv",
        output/"task042_phase_b_descriptor_baseline.csv", output/"task042_phase_b_margin_audit.csv")
    for path in (units_path, rep_path, desc_path, margin_path, CHECKPOINT, VAL_LIST):
        require(path.is_file(), "required frozen input missing: " + str(path))
    units, domains = {}, {}
    for row in read_csv(units_path):
        uid = norm_int(row["task037_global_index"])
        require(uid not in units, "duplicate frozen Task042 unit ID")
        row = dict(row); row["uid"] = uid; row["canonical_type"] = canonical_type(row["unit_type"])
        row["domain_id"] = str(row["domain_id"])
        units[uid] = row; domains.setdefault(row["domain_id"], []).append(uid)
    require(len(units) == 51, "frozen Phase-A unit manifest must contain 51 units")
    reps, descs, margins = {}, {}, {}
    for row in read_csv(rep_path):
        if row.get("subset") == "full_10x3":
            key = (str(row["domain_id"]), norm_int(row["keep_count"]))
            require(key not in reps, "duplicate Phase-B full_10x3 selection")
            reps[key] = row
    for row in read_csv(desc_path):
        key = (str(row["domain_id"]), norm_int(row["keep_count"]))
        require(key not in descs, "duplicate Phase-B descriptor selection")
        descs[key] = row
    for row in read_csv(margin_path):
        margins[(str(row["domain_id"]), norm_int(row["keep_count"]))] = row
    require(all(key in reps and key in descs and key in margins for key in PAIRS),
            "Phase-B exact sets do not cover all preregistered domain/budgets")
    pairs, evaluations, seen_evals = [], [], {}
    for domain, k in PAIRS:
        group = group_for(domain); members = canonical_ids(domains.get(domain, []))
        expected_n = 4 if domain in MIXED else (2 if domain == "11" else 3)
        require(len(members) == expected_n, "frozen BMS group size changed for " + domain)
        tr, dr, mr = reps[(domain,k)], descs[(domain,k)], margins[(domain,k)]
        temporal, descriptor = parse_ids(tr["retained_task037_ids"]), parse_ids(dr["descriptor_selected_ids"])
        require(len(temporal) == k and len(descriptor) == k and
                set(temporal).issubset(members) and set(descriptor).issubset(members),
                "Phase-B set identity/budget mismatch in domain " + domain)
        tie_count = norm_int(mr["geometry_optimal_set_count_before_ID_tiebreak"])
        orientation = "TEMPORAL_ORIENTATION_TIED" if tie_count > 1 else "TEMPORAL_ORIENTATION_UNIQUE"
        jmax_t, jmax_d = float(tr["J_max"]), float(dr["descriptor_selection_J_max_under_temporal_distance"])
        regret = jmax_d - jmax_t
        require(math.isclose(regret, jmax_d-float(dr["temporal_selection_J_max_under_temporal_distance"]),
                             rel_tol=0.0, abs_tol=1e-12), "Phase-B coverage values are inconsistent")
        type_table = {uid:units[uid]["canonical_type"] for uid in members}
        pair, candidates = build_pair_evaluations(domain, group, members, k, temporal,
                                                  descriptor, type_table, orientation)
        pair.update({"phase_b_temporal_J_max":jmax_t,
                     "phase_b_descriptor_J_max_under_temporal_distance":jmax_d,
                     "phase_b_temporal_coverage_regret_descriptor":regret,
                     "temporal_geometry_optimal_set_count":tie_count,
                     "existing_selection_baseline_status":EXISTING_STATUS})
        pairs.append(pair)
        for candidate in candidates:
            candidate.update({"temporal_selected_ids":json.dumps(temporal,separators=(",",":")),
                "descriptor_selected_ids":json.dumps(descriptor,separators=(",",":")),
                "phase_b_temporal_J_max":jmax_t,
                "phase_b_descriptor_J_max_under_temporal_distance":jmax_d,
                "phase_b_temporal_coverage_regret_descriptor":regret,
                "temporal_geometry_optimal_set_count":tie_count,
                "informative_directional_pair":pair["informative_directional_pair"],
                "existing_selection_baseline_status":EXISTING_STATUS,
                "requires_evaluation":True})
            eid = candidate["mask_evaluation_id"]
            if eid in seen_evals:
                require(seen_evals[eid]["masked_task037_ids"] == candidate["masked_task037_ids"]
                        and seen_evals[eid]["domain_id"] == candidate["domain_id"],
                        "mask evaluation identity collision")
            else:
                seen_evals[eid] = candidate
                evaluations.append(candidate)
    for index,row in enumerate(evaluations): row["evaluation_index"] = index
    return {"pairs":pairs,"evaluations":evaluations,"units":units}

def prepare(args):
    repo, output = Path(args.repo).resolve(), Path(args.output).resolve()
    require(git(repo,"rev-parse","--abbrev-ref","HEAD") == BRANCH,"stay on Task042 branch")
    require(sha256(CHECKPOINT) == EXPECTED_SHA,"authoritative checkpoint hash mismatch")
    require(not any((output/name).exists() for name in OUTPUT_NAMES),
            "Phase-C outputs already exist; refusing overwrite")
    config_path = output/"task042_phase_c_prepared.json"
    require(not config_path.exists(),"prepared Phase-C config already exists")
    plan, baseline = build_plan(output), baseline_cache()
    manifest_path = output/OUTPUT_NAMES[0]
    write_csv_new(manifest_path, plan["evaluations"])
    config = {"task":"Task042 Phase C temporal representative-set joint masking oracle",
        "required_branch":BRANCH,"prepared_head":git(repo,"rev-parse","HEAD"),
        "repo":str(repo),"output":str(output),"project_root":str(Path(args.project_root).resolve()),
        "checkpoint":str(CHECKPOINT),"checkpoint_sha256":EXPECTED_SHA,
        "val_list":str(VAL_LIST),"val_list_sha256":sha256(VAL_LIST),"frame_root":str(FRAME_ROOT),
        "frozen_inputs":{name:{"path":str(output/name),"sha256":sha256(output/name)}
          for name in ("task042_unit_manifest.csv","task042_phase_b_representative_sets.csv",
                       "task042_phase_b_descriptor_baseline.csv","task042_phase_b_margin_audit.csv")},
        "baseline_cache":{k:baseline[k] for k in ("path","identity_path","sha256","mean_ce","top1","top5")},
        "baseline_identity":baseline["identity"],"manifest_path":str(manifest_path),
        "manifest_sha256":sha256(manifest_path),"evaluation_rows":plan["evaluations"],
        "pair_rows":plan["pairs"],"unit_rows":{str(uid):{k:row[k] for k in
          ("task037_global_index","layer","unit_type","capture_kind","unit_index","domain_id")}
          for uid,row in plan["units"].items()},
        "evaluation_count":len(plan["evaluations"]),"pair_count":len(plan["pairs"]),
        "existing_selection_baseline_status":EXISTING_STATUS,
        "fixed_protocol":{"validation_clips":EXPECTED_CLIPS,"dtype":"float32","amp":False,
                          "physical_gpus":[0,1],"batch_size":4,"workers_per_gpu":2}}
    write_json_new(config_path,config)
    return {"status":"PREPARE_OK","pairs":len(plan["pairs"]),
            "unique_mask_evaluations":len(plan["evaluations"]),
            "informative_non_tied_pairs":sum(bool(r["informative_directional_pair"]) for r in plan["pairs"]),
            "prepared_head":config["prepared_head"],"manifest":str(manifest_path)}

def validate_cache(config):
    path=Path(config["baseline_cache"]["path"])
    require(sha256(path)==config["baseline_cache"]["sha256"],"baseline cache changed")
    require(sha256(config["val_list"])==config["val_list_sha256"],"validation list changed")
    require(sha256(config["checkpoint"])==EXPECTED_SHA,"checkpoint changed")
    ident=json.loads(Path(config["baseline_cache"]["identity_path"]).read_text(encoding="utf-8"))
    require(ident.get("checkpoint_sha256")==EXPECTED_SHA and
            ident.get("val_list_sha256")==config["val_list_sha256"] and
            ident.get("valid_clips")==EXPECTED_CLIPS,"baseline identity changed")
    rows=read_csv(path); rows.sort(key=lambda r:norm_int(r["dataset_index"]))
    require(len(rows)==EXPECTED_CLIPS,"baseline row count changed")
    return rows

def spec_index(specs):
    aliases={"head":"head","attention_head":"head","neuron":"neuron","ffn_neuron":"neuron"}
    result={}
    for spec in specs:
        key=(str(spec.name),aliases.get(str(spec.unit_type),str(spec.unit_type)))
        require(key not in result,"duplicate discovered unit layer "+repr(key))
        result[key]=spec
    return result

def current_head(repo,config):
    require(git(repo,"rev-parse","--abbrev-ref","HEAD")==BRANCH,"checkout moved off Task042")
    require(git(repo,"rev-parse","HEAD")==config["prepared_head"],"Task042 checkout changed after prepare")
    require(not git(repo,"status","--porcelain"),"worktree must be clean for preflight/inference")

def preflight(args):
    import torch
    repo,output=Path(args.repo).resolve(),Path(args.output).resolve()
    config=json.loads((output/"task042_phase_c_prepared.json").read_text(encoding="utf-8"))
    current_head(repo,config)
    require(sha256(config["manifest_path"])==config["manifest_sha256"],"evaluation manifest changed")
    for name,item in config["frozen_inputs"].items():
        require(sha256(item["path"])==item["sha256"],"frozen Phase-B input changed: "+name)
    validate_cache(config)
    runtime=str(repo/"src"/"lgfr_runtime")
    if runtime not in sys.path: sys.path.insert(0,runtime)
    import task041_phase_d_fullval_oracle as phase_d
    model,identity,specs=phase_d.load_model(Path(config["project_root"]),
                                             Path(config["checkpoint"]),torch.device("cpu"))
    del model
    require(identity.get("checkpoint_sha256")==EXPECTED_SHA and
            identity.get("classifier_head",{}).get("status")=="loaded" and
            not identity.get("missing_keys") and not identity.get("unexpected_keys"),
            "CPU checkpoint/classifier identity gate failed")
    by_spec=spec_index(specs)
    resolved=[]
    for row in config["unit_rows"].values():
        kind="head" if canonical_type(row["unit_type"])=="attention_head" else "neuron"
        key=(str(row["layer"]),kind)
        require(key in by_spec,"frozen unit layer missing from current model")
        spec=by_spec[key]; index=norm_int(row["unit_index"])
        require(0<=index<int(spec.num_units),"frozen unit index exceeds layer width")
        resolved.append(kind)
    require(len(resolved)==51,"CPU preflight did not resolve all frozen units")
    result={"status":"PREFLIGHT_OK","task042_head":config["prepared_head"],
        "checkpoint_sha256":identity["checkpoint_sha256"],
        "baseline_sha256":config["baseline_cache"]["sha256"],"validation_clips":EXPECTED_CLIPS,
        "unit_identity_count":len(resolved),"attention_unit_count":resolved.count("head"),
        "ffn_unit_count":resolved.count("neuron"),"evaluation_count":config["evaluation_count"],
        "dtype":"torch.float32","amp":False,"model_identity":identity}
    write_json_new(output/"task042_phase_c_preflight.json",result)
    result.pop("model_identity")
    return result

def _live_baseline_check(phase_d,model,batch,baseline_rows,device,torch):
    videos,targets,indices=batch
    index_values=[int(x) for x in indices.tolist()]
    require(index_values==list(range(len(index_values))) and bool(index_values),
            "live first batch is not the exact validation prefix")
    x=videos.float().to(device,non_blocking=True)
    y=targets.long().to(device,non_blocking=True)
    with torch.no_grad(): logits=phase_d.unwrap_logits(model(x))
    metrics=phase_d.metric_tensors(logits,y)
    for offset,index in enumerate(index_values):
        row=baseline_rows[index]
        require(int(metrics["predicted_class"][offset].item())==norm_int(row["predicted_class"]),
                "cached/live baseline predicted classes differ")
        require(abs(float(metrics["cross_entropy"][offset].item())-float(row["cross_entropy"]))<1e-4,
                "cached/live baseline CE differs")
        require(int(metrics["top1_correct"][offset].item())==int(row["top1_correct"]) and
                int(metrics["top5_correct"][offset].item())==int(row["top5_correct"]),
                "cached/live baseline Top-1/Top-5 differs")
    return x.detach(),logits.detach().clone(),videos.detach().clone()

def _target_records(eval_row,units,by_spec):
    records=[]
    for uid in parse_ids(eval_row["masked_task037_ids"]):
        row=units[str(uid)]
        kind="head" if canonical_type(row["unit_type"])=="attention_head" else "neuron"
        key=(str(row["layer"]),kind)
        require(key in by_spec,"model is missing frozen layer "+repr(key))
        spec=by_spec[key]; index=norm_int(row["unit_index"])
        require(0<=index<int(spec.num_units),"frozen unit index outside model width")
        records.append({"task037_global_index":uid,"unit_index":index,"spec":spec})
    return records

def worker(args):
    import torch
    repo,output=Path(args.repo).resolve(),Path(args.output).resolve()
    config=json.loads((output/"task042_phase_c_prepared.json").read_text(encoding="utf-8"))
    current_head(repo,config)
    require(args.num_shards==2 and args.shard_index in (0,1),"exactly two workers are preregistered")
    require(args.physical_gpu in (0,1) and os.environ.get("CUDA_VISIBLE_DEVICES")==str(args.physical_gpu),
            "worker must expose exactly physical GPU 0 or 1 via CUDA_VISIBLE_DEVICES")
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,
            "worker must see exactly one visible CUDA device")
    require(sha256(config["checkpoint"])==EXPECTED_SHA,"checkpoint hash changed before GPU worker")
    require(sha256(config["manifest_path"])==config["manifest_sha256"],"manifest changed before GPU worker")
    baseline_rows=validate_cache(config)
    torch.set_num_threads(2); torch.manual_seed(3407); torch.cuda.manual_seed_all(3407)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda:0"); runtime=str(repo/"src"/"lgfr_runtime")
    if runtime not in sys.path: sys.path.insert(0,runtime)
    import task041_phase_d_fullval_oracle as phase_d
    import task040_htor_probe as task040
    phase_d.set_seed(3407)
    torch.set_num_threads(2)
    model,identity,specs=phase_d.load_model(Path(config["project_root"]),
                                             Path(config["checkpoint"]),device)
    require(identity.get("checkpoint_sha256")==EXPECTED_SHA and
            identity.get("classifier_head",{}).get("status")=="loaded" and
            not identity.get("missing_keys") and not identity.get("unexpected_keys"),
            "GPU checkpoint/classifier identity failed")
    by_spec=spec_index(specs)
    loader,nclips=phase_d.build_full_loader(Path(config["project_root"]),config["val_list"],
           config["frame_root"],num_workers=2,batch_size=4,seed=3407)
    require(nclips==EXPECTED_CLIPS,"loader is not the full 3783-clip validation split")
    initial=next(iter(loader))
    reference_inputs,reference_logits,reference_cpu=_live_baseline_check(
        phase_d,model,initial,baseline_rows,device,torch)
    assigned=[row for row in config["evaluation_rows"]
              if int(row["evaluation_index"])%2==args.shard_index]
    require(bool(assigned),"worker shard has no assigned evaluations")
    shard_path=output/("task042_phase_c_shard_%d.csv"%args.shard_index)
    require(not shard_path.exists(),"worker shard already exists")
    results=[]
    for eval_row in assigned:
        records=_target_records(eval_row,config["unit_rows"],by_spec)
        masked_ids=parse_ids(eval_row["masked_task037_ids"])
        require([int(r["task037_global_index"]) for r in records]==masked_ids,
                "resolved mask ids differ from frozen manifest")
        ce_mask=[]; top1_mask=[]; top5_mask=[]; pred_mask=[]; seen=[]; calls=None
        with joint_temporary_unit_masks(model,records,task040.temporary_unit_mask,torch) as calls:
            with torch.no_grad():
                for videos,targets,indices in iter(loader):
                    index_values=[int(x) for x in indices.tolist()]
                    require(index_values==list(range(len(seen),len(seen)+len(index_values))),
                            "validation index order is not exact")
                    target_values=[int(x) for x in targets.tolist()]
                    for offset,index in enumerate(index_values):
                        require(target_values[offset]==norm_int(baseline_rows[index]["label"]),
                                "masked validation label differs from cache")
                    if not seen:
                        require(torch.equal(videos.cpu(),reference_cpu),
                                "full-validation transform changed between baseline reuse and masking")
                    logits=phase_d.unwrap_logits(model(videos.float().to(device,non_blocking=True)))
                    require(bool(torch.isfinite(logits).all().item()),"masked logits are non-finite")
                    values=phase_d.metric_tensors(logits,targets.long().to(device,non_blocking=True))
                    for offset,index in enumerate(index_values):
                        ce_mask.append(float(values["cross_entropy"][offset].item()))
                        top1_mask.append(int(values["top1_correct"][offset].item()))
                        top5_mask.append(int(values["top5_correct"][offset].item()))
                        pred_mask.append(int(values["predicted_class"][offset].item()))
                        seen.append(index)
        require(len(seen)==EXPECTED_CLIPS and seen==list(range(EXPECTED_CLIPS)),
                "masked inference did not cover all clips in exact order")
        require(calls is not None and len(calls)==len(records) and all(v>0 for v in calls.values()),
                "intended units were not all verified masked")
        with torch.no_grad(): restored=phase_d.unwrap_logits(model(reference_inputs))
        restored_exact=bool(torch.equal(restored,reference_logits))
        require(restored_exact,"unmasked logits changed after joint-mask restoration")
        base_ce=[float(row["cross_entropy"]) for row in baseline_rows]
        base_top1=[int(row["top1_correct"]) for row in baseline_rows]
        base_top5=[int(row["top5_correct"]) for row in baseline_rows]
        base_pred=[norm_int(row["predicted_class"]) for row in baseline_rows]
        ce_delta=[m-b for m,b in zip(ce_mask,base_ce)]
        top1_delta=[m-b for m,b in zip(top1_mask,base_top1)]
        top5_delta=[m-b for m,b in zip(top5_mask,base_top5)]
        flips=[int(m!=b) for m,b in zip(pred_mask,base_pred)]
        results.append({
          "mask_evaluation_id":eval_row["mask_evaluation_id"],"domain_id":eval_row["domain_id"],
          "domain_group":eval_row["domain_group"],"keep_count":eval_row["keep_count"],
          "unit_count":eval_row["unit_count"],"method":eval_row["method"],
          "shared_methods":eval_row["shared_methods"],
          "retained_task037_ids":eval_row["retained_task037_ids"],
          "masked_task037_ids":eval_row["masked_task037_ids"],
          "masked_attention_count":eval_row["masked_attention_count"],
          "masked_ffn_count":eval_row["masked_ffn_count"],"set_relation":eval_row["set_relation"],
          "temporal_orientation_status":eval_row["temporal_orientation_status"],
          "mean_ce_increase":statistics.mean(ce_delta),"median_ce_increase":statistics.median(ce_delta),
          "top1_change_percentage_points":100.0*statistics.mean(top1_delta),
          "top5_change_percentage_points":100.0*statistics.mean(top5_delta),
          "prediction_flip_rate":statistics.mean(flips),"validation_clips":len(seen),
          "verified_mask_unit_count":len(calls),"mask_restored_exact":True,
          "restored_logits_exact":restored_exact,"worker_physical_gpu":args.physical_gpu,
          "worker_gpu_name":torch.cuda.get_device_name(0),"checkpoint_sha256":EXPECTED_SHA,
          "baseline_sha256":config["baseline_cache"]["sha256"]})
        print("WORKER_SET_DONE gpu=%s id=%s domain=%s k=%s clips=%s"%(
              args.physical_gpu,eval_row["mask_evaluation_id"],eval_row["domain_id"],
              eval_row["keep_count"],len(seen)),flush=True)
    write_csv_new(shard_path,results)
    meta={"status":"WORKER_OK","shard_index":args.shard_index,"physical_gpu":args.physical_gpu,
          "gpu_name":torch.cuda.get_device_name(0),"evaluation_count":len(results),
          "evaluation_ids":[r["mask_evaluation_id"] for r in results],
          "validation_clips_per_evaluation":EXPECTED_CLIPS,"all_masks_verified":True,
          "all_masks_restored_exact":True,"checkpoint_sha256":EXPECTED_SHA,
          "baseline_sha256":config["baseline_cache"]["sha256"]}
    write_json_new(output/("task042_phase_c_shard_%d.json"%args.shard_index),meta)
    return meta

def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:(values[i],i)); ranks=[0.0]*len(values); start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]: end+=1
        rank=((start+1)+end)/2.0
        for pos in range(start,end): ranks[order[pos]]=rank
        start=end
    return ranks

def spearman(x,y):
    if len(x)!=len(y) or len(x)<2: return None
    rx,ry=average_ranks(x),average_ranks(y); mx,my=statistics.mean(rx),statistics.mean(ry)
    dx,dy=[v-mx for v in rx],[v-my for v in ry]
    vx,vy=sum(v*v for v in dx),sum(v*v for v in dy)
    return None if vx==0 or vy==0 else sum(a*b for a,b in zip(dx,dy))/math.sqrt(vx*vy)

def finalize(args):
    repo,output=Path(args.repo).resolve(),Path(args.output).resolve()
    config=json.loads((output/"task042_phase_c_prepared.json").read_text(encoding="utf-8"))
    current_head(repo,config); shard_rows=[]
    for shard in (0,1):
        csv_path=output/("task042_phase_c_shard_%d.csv"%shard)
        meta_path=output/("task042_phase_c_shard_%d.json"%shard)
        require(csv_path.is_file() and meta_path.is_file(),"both GPU shards must complete")
        meta=json.loads(meta_path.read_text(encoding="utf-8"))
        require(meta.get("status")=="WORKER_OK" and meta.get("physical_gpu")==shard and
                meta.get("all_masks_verified") is True and meta.get("all_masks_restored_exact") is True,
                "worker identity or mask restoration gate failed")
        shard_rows.extend(read_csv(csv_path))
    by_eval={}
    for row in shard_rows:
        eid=row["mask_evaluation_id"]
        require(eid not in by_eval,"duplicate evaluation across GPU shards")
        require(int(row["validation_clips"])==EXPECTED_CLIPS and
                row["mask_restored_exact"]=="True" and row["restored_logits_exact"]=="True",
                "full-val or exact restore gate failed")
        by_eval[eid]=row
    expected={row["mask_evaluation_id"] for row in config["evaluation_rows"]}
    require(set(by_eval)==expected,"worker shards do not exactly cover the manifest")
    manifests={row["mask_evaluation_id"]:row for row in config["evaluation_rows"]}
    damage=[]
    for eid,result in by_eval.items():
        row=dict(result); row["informative_directional_pair"]=manifests[eid]["informative_directional_pair"]
        row["existing_selection_baseline_status"]=EXISTING_STATUS; damage.append(row)
    pairs=[]
    for pair in config["pair_rows"]:
        t,d=by_eval[pair["temporal_mask_evaluation_id"]],by_eval[pair["descriptor_mask_evaluation_id"]]
        td,dd=float(t["mean_ce_increase"]),float(d["mean_ce_increase"])
        pairs.append({
          "domain_id":pair["domain_id"],"domain_group":pair["domain_group"],
          "unit_count":pair["unit_count"],"keep_count":pair["keep_count"],
          "temporal_retained_task037_ids":pair["temporal_retained_task037_ids"],
          "descriptor_retained_task037_ids":pair["descriptor_retained_task037_ids"],
          "set_relation":pair["set_relation"],
          "temporal_orientation_status":pair["temporal_orientation_status"],
          "informative_directional_pair":pair["informative_directional_pair"],
          "temporal_mask_evaluation_id":pair["temporal_mask_evaluation_id"],
          "descriptor_mask_evaluation_id":pair["descriptor_mask_evaluation_id"],
          "temporal_mean_ce_increase":td,"descriptor_mean_ce_increase":dd,
          "descriptor_minus_temporal_mean_ce":dd-td,
          "temporal_median_ce_increase":float(t["median_ce_increase"]),
          "descriptor_median_ce_increase":float(d["median_ce_increase"]),
          "temporal_top1_change_percentage_points":float(t["top1_change_percentage_points"]),
          "descriptor_top1_change_percentage_points":float(d["top1_change_percentage_points"]),
          "temporal_top5_change_percentage_points":float(t["top5_change_percentage_points"]),
          "descriptor_top5_change_percentage_points":float(d["top5_change_percentage_points"]),
          "temporal_prediction_flip_rate":float(t["prediction_flip_rate"]),
          "descriptor_prediction_flip_rate":float(d["prediction_flip_rate"]),
          "temporal_lower_mean_ce_damage":td<dd,
          "directional_gate_eligible":pair["informative_directional_pair"],
          "temporal_J_max":pair["phase_b_temporal_J_max"],
          "descriptor_J_max_under_temporal_distance":pair["phase_b_descriptor_J_max_under_temporal_distance"],
          "temporal_coverage_regret_descriptor":pair["phase_b_temporal_coverage_regret_descriptor"]})
    eligible=[r for r in pairs if r["directional_gate_eligible"]]
    nonmixed=[r for r in eligible if r["domain_group"]!="mixed"]
    favored=sum(bool(r["temporal_lower_mean_ce_damage"]) for r in eligible)
    agg=statistics.mean(r["descriptor_minus_temporal_mean_ce"] for r in eligible) if eligible else None
    nmagg=statistics.mean(r["descriptor_minus_temporal_mean_ce"] for r in nonmixed) if nonmixed else None
    majority=bool(eligible and favored>len(eligible)/2)
    agg_favors=bool(agg is not None and agg>0)
    nm_favors=bool(nonmixed and nmagg is not None and nmagg>0)
    if majority and agg_favors and nm_favors:
        decision="A. TEMPORAL_DIVERSITY_PRESERVATION_SUPPORTED"
    else:
        reverse_majority=bool(eligible and len(eligible)-favored>len(eligible)/2)
        reverse_agg=bool(agg is not None and agg<0)
        reverse_nm=bool(nonmixed and nmagg is not None and nmagg<0)
        decision=("C. TEMPORAL_DIVERSITY_PRESERVATION_REJECTED"
                  if reverse_majority and reverse_agg and reverse_nm
                  else "B. TEMPORAL_DIVERSITY_PRESERVATION_WEAK_OR_UNRESOLVED")
    radius=[]
    for row in pairs:
        result=by_eval[row["temporal_mask_evaluation_id"]]
        radius.append({"domain_id":row["domain_id"],"domain_group":row["domain_group"],
          "keep_count":row["keep_count"],"unit_count":row["unit_count"],
          "temporal_J_max":row["temporal_J_max"],
          "temporal_mean_ce_increase":row["temporal_mean_ce_increase"],
          "temporal_orientation_status":row["temporal_orientation_status"],
          "included_in_spearman":row["temporal_orientation_status"]!="TEMPORAL_ORIENTATION_TIED",
          "mask_evaluation_id":row["temporal_mask_evaluation_id"],
          "mask_restored_exact":result["mask_restored_exact"]})
    radius_used=[r for r in radius if r["included_in_spearman"]]
    radius_rho=spearman([float(r["temporal_J_max"]) for r in radius_used],
                         [float(r["temporal_mean_ce_increase"]) for r in radius_used])
    regret=[]
    for row in pairs:
        regret.append({"domain_id":row["domain_id"],"domain_group":row["domain_group"],
          "keep_count":row["keep_count"],"set_relation":row["set_relation"],
          "temporal_orientation_status":row["temporal_orientation_status"],
          "temporal_coverage_regret_descriptor":row["temporal_coverage_regret_descriptor"],
          "descriptor_minus_temporal_mean_ce":row["descriptor_minus_temporal_mean_ce"],
          "informative_directional_pair":row["informative_directional_pair"],
          "included_in_spearman":bool(row["set_relation"]=="A_TEMPORAL_DIFFERS" and
                   row["temporal_orientation_status"]!="TEMPORAL_ORIENTATION_TIED")})
    regret_used=[r for r in regret if r["included_in_spearman"]]
    regret_rho=spearman([float(r["temporal_coverage_regret_descriptor"]) for r in regret_used],
                         [float(r["descriptor_minus_temporal_mean_ce"]) for r in regret_used])
    strong=[]
    for row in pairs:
        if row["domain_group"] not in ("strong_structured","diverse_control"): continue
        for method,ce,t1,t5,flip,eid in (
          ("temporal","temporal_mean_ce_increase","temporal_top1_change_percentage_points",
           "temporal_top5_change_percentage_points","temporal_prediction_flip_rate","temporal_mask_evaluation_id"),
          ("descriptor","descriptor_mean_ce_increase","descriptor_top1_change_percentage_points",
           "descriptor_top5_change_percentage_points","descriptor_prediction_flip_rate","descriptor_mask_evaluation_id")):
            strong.append({"domain_id":row["domain_id"],"domain_group":row["domain_group"],
             "keep_count":row["keep_count"],"unit_count":row["unit_count"],"method":method,
             "mask_evaluation_id":row[eid],"set_relation":row["set_relation"],
             "temporal_orientation_status":row["temporal_orientation_status"],
             "temporal_J_max":row["temporal_J_max"],"mean_ce_increase":row[ce],
             "top1_change_percentage_points":row[t1],"top5_change_percentage_points":row[t5],
             "prediction_flip_rate":row[flip]})
    unit_rows=config["unit_rows"]; mixed=[]
    for row in pairs:
        if row["domain_id"] not in MIXED: continue
        for method,eid_key,ids_key,ce_key,t1_key,t5_key,flip_key in (
          ("temporal","temporal_mask_evaluation_id","temporal_retained_task037_ids",
           "temporal_mean_ce_increase","temporal_top1_change_percentage_points",
           "temporal_top5_change_percentage_points","temporal_prediction_flip_rate"),
          ("descriptor","descriptor_mask_evaluation_id","descriptor_retained_task037_ids",
           "descriptor_mean_ce_increase","descriptor_top1_change_percentage_points",
           "descriptor_top5_change_percentage_points","descriptor_prediction_flip_rate")):
            eid=row[eid_key]; ev=manifests[eid]; retained=parse_ids(row[ids_key])
            heads=sum(canonical_type(unit_rows[str(uid)]["unit_type"])=="attention_head" for uid in retained)
            mixed.append({"domain_id":row["domain_id"],"keep_count":row["keep_count"],
              "method":ev["method"],"shared_methods":ev["shared_methods"],
              "retained_task037_ids":json.dumps(retained,separators=(",",":")),
              "retained_attention_count":heads,"retained_ffn_count":len(retained)-heads,
              "retains_both_types":heads>0 and len(retained)-heads>0,
              "masked_attention_count":ev["masked_attention_count"],
              "masked_ffn_count":ev["masked_ffn_count"],"mean_ce_increase":row[ce_key],
              "top1_change_percentage_points":row[t1_key],"top5_change_percentage_points":row[t5_key],
              "prediction_flip_rate":row[flip_key],"mask_evaluation_id":eid,
              "temporal_orientation_status":row["temporal_orientation_status"]})
    phase_outputs={
      OUTPUT_NAMES[1]:damage,OUTPUT_NAMES[2]:pairs,OUTPUT_NAMES[3]:strong,
      OUTPUT_NAMES[4]:radius,OUTPUT_NAMES[5]:regret,OUTPUT_NAMES[6]:mixed}
    for filename,rows in phase_outputs.items(): write_csv_new(output/filename,rows)
    baseline=config["baseline_cache"]
    summary={"task":"Task042 Phase C temporal representative-set joint masking oracle",
      "decision":decision,"branch":BRANCH,"git_head":config["prepared_head"],
      "server":"jixinye25@192.168.0.12","checkpoint_sha256":EXPECTED_SHA,
      "baseline_cache_sha256":baseline["sha256"],"baseline_cache_reused":True,
      "baseline_mean_ce":baseline["mean_ce"],"baseline_top1":baseline["top1"],
      "baseline_top5":baseline["top5"],"validation_clips":EXPECTED_CLIPS,
      "dtype":"float32","amp":False,"physical_gpus":[0,1],"pair_count":len(pairs),
      "unique_joint_mask_evaluations":len(by_eval),"informative_directional_pair_count":len(eligible),
      "temporal_lower_damage_count":favored,"strict_majority_favors_temporal":majority,
      "aggregate_descriptor_minus_temporal_mean_ce":agg,
      "aggregate_ce_direction_favors_temporal":agg_favors,
      "nonmixed_informative_pair_count":len(nonmixed),
      "nonmixed_aggregate_descriptor_minus_temporal_mean_ce":nmagg,
      "not_supported_only_by_mixed_domains":nm_favors,
      "temporal_radius_spearman_rho_unique_orientation_only":radius_rho,
      "coverage_regret_spearman_rho_differing_non_tied_pairs":regret_rho,
      "temporal_orientation_tied_pair_count":sum(r["temporal_orientation_status"]=="TEMPORAL_ORIENTATION_TIED" for r in pairs),
      "same_set_pair_count":sum(r["set_relation"]=="B_SETS_EQUAL" for r in pairs),
      "different_set_pair_count":sum(r["set_relation"]=="A_TEMPORAL_DIFFERS" for r in pairs),
      "existing_selection_baseline_status":EXISTING_STATUS,
      "existing_selection_rule_evidence":["/home/jixinye25/jxy_work1/swintrans_task037/reproduction_report.md",
        "/home/jixinye25/jxy_work1/swintrans_task037/task014_n09/functional_selection_trace.csv"],
      "all_full_validation_counts":all(int(r["validation_clips"])==EXPECTED_CLIPS for r in by_eval.values()),
      "all_intended_masks_verified":all(int(r["verified_mask_unit_count"])==len(parse_ids(r["masked_task037_ids"]))
                                        for r in by_eval.values()),
      "all_hook_and_parameter_restoration_exact":all(r["mask_restored_exact"]=="True" for r in by_eval.values()),
      "all_unmasked_logits_exact_after_restore":all(r["restored_logits_exact"]=="True" for r in by_eval.values()),
      "primary_gate":{"strict_majority":majority,"aggregate_direction_favors_temporal":agg_favors,
        "nonmixed_aggregate_favors_temporal":nm_favors,"eligible_pairs":len(eligible),
        "temporal_favorable_pairs":favored,"nonmixed_eligible_pairs":len(nonmixed)},
      "interpretation":"Temporary joint-mask oracle only; no physical pruning, shape changes, training, fine-tuning, threshold, lambda, quota, learned score, or global pruning.",
      "outputs":{}}
    for filename in OUTPUT_NAMES[:7]:
        summary["outputs"][filename]={"path":str(output/filename),"sha256":sha256(output/filename)}
    report=make_report(summary,pairs,strong,mixed,radius_rho,regret_rho)
    report_path=output/OUTPUT_NAMES[8]
    require(not report_path.exists(),"refusing to overwrite report")
    report_path.write_text(report,encoding="utf-8")
    summary["outputs"][OUTPUT_NAMES[8]]={"path":str(report_path),"sha256":sha256(report_path)}
    write_json_new(output/OUTPUT_NAMES[7],summary)
    return {"status":"FINALIZE_OK","decision":decision,"informative_pairs":len(eligible),
            "temporal_favorable":favored,"aggregate_delta":agg,"report":str(report_path)}

def make_report(s,pairs,strong,mixed,radius_rho,regret_rho):
    lines=["# Task042 Phase C — Temporal Representative-Set Joint-Masking Oracle","",
      "## Decision","", "**"+s["decision"]+"**","",
      "The preregistered A gate requires a strict majority of informative, non-tied "
      "temporal-versus-descriptor set pairs to favor temporal retention, a positive aggregate "
      "descriptor-minus-temporal CE-damage direction, and a positive aggregate direction among "
      "non-mixed informative domains. The reverse pattern yields C; other outcomes yield B. "
      "Exact temporal ties are excluded from directional claims.","",
      "## Protocol and baseline","",
      "- Full UCF101 validation: %d / %d clips."%(s["validation_clips"],s["validation_clips"]),
      "- Checkpoint SHA-256: %s."%s["checkpoint_sha256"],
      "- Phase-D unmasked baseline reused, SHA-256: %s."%s["baseline_cache_sha256"],
      "- Baseline mean CE %.8f; Top-1 %.4f; Top-5 %.4f."%(s["baseline_mean_ce"],s["baseline_top1"],s["baseline_top5"]),
      "- FP32, AMP disabled; independent worker processes used physical GPUs 0 and 1.",
      "- %d distinct joint-mask evaluations; each used all %d validation clips."%(s["unique_joint_mask_evaluations"],s["validation_clips"]),
      "- All intended masks verified: %s; all hooks/parameter versions restored: %s; "
      "unmasked logits bitwise restored: %s."%(s["all_intended_masks_verified"],
      s["all_hook_and_parameter_restoration_exact"],s["all_unmasked_logits_exact_after_restore"]),
      "- Existing Task037 fixed-domain/fixed-keep selection baseline: unavailable. Task037 records "
      "an adaptive global parameter-budget sequence; an exact set for this audit cannot be recovered "
      "from that rule, so no approximation was substituted.","",
      "## Temporal versus descriptor representatives","","|Domain|Group|k|Set relation|Orientation|Temporal CE damage|Descriptor CE damage|Descriptor - temporal|Top-1 Δ T/D (pp)|Flip rate T/D|",
      "|---:|---|---:|---|---|---:|---:|---:|---:|---:|"]
    for r in pairs:
      lines.append("|%s|%s|%s|%s|%s|%.7f|%.7f|%.7f|%.3f / %.3f|%.4f / %.4f|"%
        (r["domain_id"],r["domain_group"],r["keep_count"],r["set_relation"],
         r["temporal_orientation_status"],r["temporal_mean_ce_increase"],
         r["descriptor_mean_ce_increase"],r["descriptor_minus_temporal_mean_ce"],
         r["temporal_top1_change_percentage_points"],r["descriptor_top1_change_percentage_points"],
         r["temporal_prediction_flip_rate"],r["descriptor_prediction_flip_rate"]))
    lines += ["","- Set-different pairs: %d; equal sets (one shared evaluation): %d."%
        (s["different_set_pair_count"],s["same_set_pair_count"]),
      "- Informative non-tied pairs: %d; temporal lower CE damage: %d; strict majority: %s."%
        (s["informative_directional_pair_count"],s["temporal_lower_damage_count"],
         s["strict_majority_favors_temporal"]),
      "- Aggregate descriptor-minus-temporal CE damage: %s."%s["aggregate_descriptor_minus_temporal_mean_ce"],
      "- Non-mixed aggregate direction: %s; gate condition passes: %s."%
        (s["nonmixed_aggregate_descriptor_minus_temporal_mean_ce"],s["not_supported_only_by_mixed_domains"]),
      "- Phase-B exact temporal-orientation ties: %d; these do not identify a scientifically preferred "
       "individual deletion."%s["temporal_orientation_tied_pair_count"],"",
      "## Strong/structured domains and diverse controls","","|Domain|Group|k|Method|Temporal Jmax|Mean CE damage|Top-1 Δ (pp)|Flip rate|",
      "|---:|---|---:|---|---:|---:|---:|---:|"]
    for r in strong:
      lines.append("|%s|%s|%s|%s|%.6f|%.7f|%.3f|%.4f|"%(r["domain_id"],r["domain_group"],
       r["keep_count"],r["method"],float(r["temporal_J_max"]),float(r["mean_ce_increase"]),
       float(r["top1_change_percentage_points"]),float(r["prediction_flip_rate"])))
    lines += ["","## Temporal radius and coverage-regret diagnostics","",
      "- Spearman of Phase-B temporal Jmax versus temporal joint-mask damage, unique orientations only: %s."%radius_rho,
      "- Spearman of descriptor temporal-coverage regret versus descriptor-minus-temporal damage, "
       "different non-tied sets only: %s."%regret_rho,
      "- Both associations are diagnostic only; Phase-B geometry and objectives stayed frozen.","",
      "## Mixed domains (secondary)","","|Domain|k|Method|Kept Attention|Kept FFN|Mean CE damage|Top-1 Δ (pp)|",
      "|---:|---:|---|---:|---:|---:|---:|"]
    for r in mixed:
      lines.append("|%s|%s|%s|%s|%s|%.7f|%.3f|"%(r["domain_id"],r["keep_count"],r["method"],
       r["retained_attention_count"],r["retained_ffn_count"],float(r["mean_ce_increase"]),
       float(r["top1_change_percentage_points"])))
    lines += ["","The two mixed domains are secondary descriptive cases only; they do not establish "
      "type fairness or justify an Attention/FFN quota.","","## Scope","",
      "This audit applied simultaneous temporary whole-unit masks using exact Phase-B full_10x3 "
      "temporal and frozen-descriptor representative sets at identical within-domain keep counts. "
      "No physical pruning, tensor-shape change, training, fine-tuning, threshold, lambda, quota, "
      "learned score, or global pruning ratio was used.",""]
    return "\n".join(lines)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=("prepare","preflight","worker","finalize"))
    parser.add_argument("--repo",default=str(REPO_DEFAULT))
    parser.add_argument("--output",default=str(OUTPUT_DEFAULT))
    parser.add_argument("--project-root",default=str(PROJECT_DEFAULT))
    parser.add_argument("--shard-index",type=int)
    parser.add_argument("--num-shards",type=int,default=2)
    parser.add_argument("--physical-gpu",type=int)
    args=parser.parse_args()
    if args.phase=="prepare": result=prepare(args)
    elif args.phase=="preflight": result=preflight(args)
    elif args.phase=="worker":
        require(args.shard_index is not None and args.physical_gpu is not None,
                "worker requires --shard-index and --physical-gpu")
        result=worker(args)
    else: result=finalize(args)
    print(json.dumps(result,indent=2,ensure_ascii=False,default=str),flush=True)

if __name__=="__main__": main()
