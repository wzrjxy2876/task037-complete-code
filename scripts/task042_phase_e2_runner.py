#!/usr/bin/env python3
"""Task042 Phase E.2 temporal-redundancy graph refresh audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_DEFAULT = Path('/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy')
BASE_DEFAULT = Path('/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy')
PROJECT_ROOT = Path('/home/jixinye25/jxy_work1/swintrans_task035')
CHECKPOINT = Path('/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt')
FRAME_ROOT = Path('/data/jixinye25/UCF101_Frame/frames')
BRANCH = 'task_042_post_bms_frame_relation_redundancy'
SEED = 3407
GATE_STAGES = (1.0, 0.75, 0.50, 0.25, 0.0)
RECOVERY_STEPS_PER_STAGE = 30
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
RECOVERY_OBJECTIVE = 'cross_entropy_only'
SPANS = (1, 2, 4, 8, 16)
PAIR_IDENTITIES = [(s, p) for s in SPANS for p in range(16)]
E2_OUTPUTS = (
    'task042_phase_e2_initial_relation_graph.csv',
    'task042_phase_e2_progressive_round_manifest.csv',
    'task042_phase_e2_post_relation_distance.csv',
    'task042_phase_e2_post_relation_graph.csv',
    'task042_phase_e2_graph_change.csv',
    'task042_phase_e2_calibration_stability.csv',
    'task042_phase_e2_pre_vs_post_recovery.csv',
    'task042_phase_e2_summary.json',
    'task042_phase_e2_report.md',
)


def require(ok: Any, message: str) -> None:
    if not ok:
        raise RuntimeError('Task042 Phase E.2 gate failed: ' + str(message))


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def runtime_modules(repo: Path):
    scripts = str(repo / 'scripts')
    runtime = str(repo / 'src' / 'lgfr_runtime')
    for path in (runtime, scripts):
        if path not in sys.path:
            sys.path.insert(0, path)
    import task042_frame_relation_redundancy as task042
    import task042_phase_e2_graph_refresh as graph
    import task042_phase_e0_feasibility as e0
    import task042_phase_e0_temporal_preservation as primitives
    import task042_phase_d_progressive_path as phase_d_path
    import task042_phase_e1_progressive_pilot as e1
    e0.PRIMITIVES = primitives
    core, probe, ctfrs = task042._runtime_modules(repo)
    return task042, graph, e0, primitives, phase_d_path, e1, core, probe, ctfrs


def paths(base: Path) -> Dict[str, Path]:
    return {
        'phase_a_dist': base / 'task042_temporal_pair_distance.csv',
        'phase_a_raw': base / 'task042_frame_pair_sensitivity.csv',
        'units': base / 'task042_unit_manifest.csv',
        'videos': base / 'task042_video_manifest.csv',
        'run_config': base / 'task042_run_config.json',
        'phase_d': base / 'phase_d',
        'phase_e1': base / 'phase_e1',
    }


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def parse_distance_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], float]:
    result = {}
    for r in rows:
        i, j = int(r['task037_global_index_i']), int(r['task037_global_index_j'])
        key = (min(i, j), max(i, j))
        require(key not in result, 'duplicate frozen Phase-A pair distance')
        result[key] = float(r['d_temp'])
    return result


def enrich_graph_rows(graph_rows: Sequence[Mapping[str, Any]], units: Sequence[Mapping[str, Any]],
                      phase: str) -> List[Dict[str, Any]]:
    meta = {int(r['task037_global_index']): r for r in units}
    out = []
    for source in graph_rows:
        row = dict(source)
        uid = int(row['task037_global_index'])
        row.update({'phase': phase,
                    'unit_type': meta[uid]['unit_type'], 'layer': meta[uid]['layer'],
                    'unit_index': meta[uid]['unit_index'], 'stage': meta[uid]['stage']})
        out.append(row)
    return out


def derive_directional_pairs(repo: Path, base: Path, task042: Any, graph: Any,
                              phase_d_path: Any, e1: Any):
    eligible, _e1_selected, _path, _phase_d_summary = e1.frozen_inputs(repo)
    p = paths(base)
    require(p['phase_a_dist'].is_file() and p['phase_a_raw'].is_file(), 'Phase-A relation artifacts are missing')
    require(p['phase_e1'].joinpath('task042_phase_e1_training_manifest.csv').is_file(), 'Phase-E.1 training manifest is missing')
    require(p['phase_e1'].joinpath('work/task042_phase_e1_exact_training_list.txt').is_file(), 'exact Phase-E.1 recovery subset is missing')
    phase_a_pairs = read_csv(p['phase_a_dist'])
    dist0 = parse_distance_rows(phase_a_pairs)
    id_rows = graph.directed_nearest_neighbors(eligible, dist0)
    e0 = graph.reciprocal_edges(id_rows)
    provenance_path = p['phase_d'] / 'task042_phase_d_candidate_provenance.csv'
    provenance = {int(r['task037_global_index']): r for r in read_csv(provenance_path)}
    directed = []
    candidate_ids = []
    for i, j in sorted(e0):
        candidate, peer, rule = graph.choose_directional_candidate((i, j), provenance, phase_d_path.f3_order_key)
        candidate_ids.append(candidate)
        ci, pi = provenance[candidate], provenance[peer]
        directed.append({
            'domain_id': str(ci['domain_id']), 'edge_i_task037_global_index': i,
            'edge_j_task037_global_index': j, 'd_temp_phase_a': dist0[(i, j)],
            'removal_candidate_task037_global_index': candidate,
            'protected_peer_task037_global_index': peer,
            'candidate_R_F3': ci['R_F3'], 'candidate_p_total': ci['p_total'],
            'candidate_p_average': ci['p_average'], 'candidate_directional_score_key': ci['directional_score_key'],
            'peer_R_F3': pi['R_F3'], 'peer_p_total': pi['p_total'], 'peer_p_average': pi['p_average'],
            'peer_directional_score_key': pi['directional_score_key'],
            'directional_criterion_source': 'Phase-D authoritative Task037 F3; lower canonical F3 order key proposed first',
            'direction_rule': rule, 'temporal_distance_used_for_direction': False,
        })
    require(len(candidate_ids) == len(set(candidate_ids)), 'same unit selected from multiple reciprocal pairs')
    candidate_set = set(candidate_ids)
    all_ids = {int(r['task037_global_index']) for r in eligible}
    survivor_ids = all_ids - candidate_set
    graph.validate_identity_partition(all_ids, candidate_set, survivor_ids)
    require(len(eligible) == len(all_ids), 'eligible unit identities are duplicated')
    return eligible, sorted(candidate_ids), sorted(survivor_ids), e0, dist0, id_rows, directed


def prepare(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out = base / 'phase_e2'
    require(git_value(repo, 'branch', '--show-current') == BRANCH, 'must remain on the existing Task042 branch')
    require(not git_value(repo, 'status', '--porcelain'), 'E.2 execution requires the pushed Task042 checkout to be clean')
    require(not out.exists(), 'Phase E.2 output directory already exists; refusing overwrite')
    task042, graph, _e0, _primitives, phase_d_path, e1, _core, _probe, _ctfrs = runtime_modules(repo)
    eligible, candidate_ids, survivor_ids, e0, dist0, graph_rows, directed = derive_directional_pairs(
        repo, base, task042, graph, phase_d_path, e1)
    p = paths(base)
    video_rows = read_csv(p['videos'])
    subsets = graph.calibration_subsets(video_rows)
    phase_e1_manifest = e1.training_manifest_values(p['phase_e1'] / 'task042_phase_e1_training_manifest.csv')
    require(phase_e1_manifest['branch'] == BRANCH and phase_e1_manifest['seed'] == SEED, 'Phase-E.1 seed/branch changed')
    require(phase_e1_manifest['gate_stages'] == list(GATE_STAGES) and phase_e1_manifest['recovery_steps_per_noninitial_stage'] == RECOVERY_STEPS_PER_STAGE,
            'Phase-E.1 progressive recovery schedule changed')
    require(phase_e1_manifest['optimizer'] == {'name': 'SGD', 'learning_rate': LEARNING_RATE, 'momentum': MOMENTUM,
            'weight_decay': WEIGHT_DECAY, 'batch_size': 1, 'gradient_clip_norm': GRAD_CLIP_NORM, 'amp': False,
            'model_mode': 'eval mode with gradients enabled; deterministic dropout/DropPath disabled'},
            'Phase-E.1 optimizer configuration changed')
    require(len(phase_e1_manifest['training_videos']) == 30 and len(phase_e1_manifest['training_schedule_indices']) == 120,
            'Phase-E.1 exact recovery cohort/order is incomplete')
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    initial_rows = enrich_graph_rows(graph_rows, eligible, 'Phase-A frozen checkpoint')
    edge_by_unit = {}
    for i, j in e0:
        edge_by_unit[i] = j
        edge_by_unit[j] = i
    for r in initial_rows:
        r['directional_removal_candidate'] = r['task037_global_index'] in candidate_ids
        r['directional_protected_peer'] = edge_by_unit.get(int(r['task037_global_index']), '') if int(r['task037_global_index']) in candidate_ids else ''
    write_csv(out / E2_OUTPUTS[0], initial_rows,
              ('phase', 'domain_id', 'task037_global_index', 'unit_type', 'layer', 'unit_index', 'stage',
               'nn_task037_global_index', 'nn_distance', 'second_nn_distance', 'nn_margin', 'exact_nearest_tie',
               'reciprocal_partner_task037_global_index', 'is_reciprocal_edge', 'directional_removal_candidate',
               'directional_protected_peer'), exclusive=True)
    candidate_rows = []
    unit_by_id = {int(r['task037_global_index']): r for r in eligible}
    provenance = {int(r['task037_global_index']): r for r in read_csv(p['phase_d'] / 'task042_phase_d_candidate_provenance.csv')}
    edge_orientation = {int(r['removal_candidate_task037_global_index']): r for r in directed}
    for order, uid in enumerate(candidate_ids, 1):
        edge = edge_orientation[uid]
        row = dict(unit_by_id[uid])
        row.update({'candidate_order': order, 'protected_peer_task037_global_index': edge['protected_peer_task037_global_index'],
                    'initial_reciprocal_edge_i': edge['edge_i_task037_global_index'], 'initial_reciprocal_edge_j': edge['edge_j_task037_global_index'],
                    'initial_d_temp': edge['d_temp_phase_a'], 'R_F3': provenance[uid]['R_F3'],
                    'p_total': provenance[uid]['p_total'], 'p_average': provenance[uid]['p_average'],
                    'directional_score_key': provenance[uid]['directional_score_key'],
                    'direction_rule': edge['direction_rule'], 'final_gate_value': 0.0,
                    'directional_criterion_source': edge['directional_criterion_source'],
                    'temporal_distance_used_for_direction': False})
        candidate_rows.append(row)
    fields = ('candidate_order', 'task037_global_index', 'domain_id', 'unit_type', 'layer', 'unit_index', 'stage',
              'protected_peer_task037_global_index', 'initial_reciprocal_edge_i', 'initial_reciprocal_edge_j',
              'initial_d_temp', 'R_F3', 'p_total', 'p_average', 'directional_score_key', 'direction_rule',
              'directional_criterion_source', 'temporal_distance_used_for_direction', 'final_gate_value')
    write_csv(work / 'task042_phase_e2_candidate_direction_manifest.csv', candidate_rows, fields, exclusive=True)
    input_paths = {
        'phase_a_temporal_pair_distance.csv': p['phase_a_dist'],
        'phase_a_frame_pair_sensitivity.csv': p['phase_a_raw'],
        'unit_manifest.csv': p['units'], 'video_manifest.csv': p['videos'], 'run_config.json': p['run_config'],
        'phase_d_candidate_provenance.csv': p['phase_d'] / 'task042_phase_d_candidate_provenance.csv',
        'phase_e1_training_manifest.csv': p['phase_e1'] / 'task042_phase_e1_training_manifest.csv',
        'phase_e1_exact_training_list.txt': p['phase_e1'] / 'work/task042_phase_e1_exact_training_list.txt',
        'checkpoint': CHECKPOINT,
    }
    cfg = {
        'task': 'Task042 Phase E.2 temporal redundancy graph refresh feasibility',
        'branch': BRANCH, 'code_head': git_value(repo, 'rev-parse', 'HEAD'), 'phase_d_head': e1.PHASE_D_HEAD,
        'phase_e1_summary_sha256': sha256_file(p['phase_e1'] / 'task042_phase_e1_summary.json'),
        'checkpoint_sha256': sha256_file(CHECKPOINT), 'input_sha256': {k: sha256_file(v) for k, v in input_paths.items()},
        'candidate_ids_in_frozen_order': candidate_ids, 'survivor_ids_in_frozen_order': survivor_ids,
        'candidate_count': len(candidate_ids), 'survivor_count': len(survivor_ids),
        'initial_reciprocal_edges': [list(e) for e in sorted(e0)],
        'candidate_direction_manifest_sha256': sha256_file(work / 'task042_phase_e2_candidate_direction_manifest.csv'),
        'gate_stages': list(GATE_STAGES), 'recovery_steps_per_noninitial_stage': RECOVERY_STEPS_PER_STAGE,
        'total_optimizer_updates': 4 * RECOVERY_STEPS_PER_STAGE,
        'pre_recovery_boundary': 'At optimizer step 90, after .75/.50/.25 CE recovery, set all selected gates to 0 and capture before any step at gate 0.',
        'recovery_objective': RECOVERY_OBJECTIVE, 'L_TR_used': False,
        'optimizer': phase_e1_manifest['optimizer'], 'seed': SEED,
        'training_videos': phase_e1_manifest['training_videos'],
        'training_schedule_indices': phase_e1_manifest['training_schedule_indices'],
        'training_subset_source': 'exact Phase-E.1 training list and manifest; no resampling',
        'video_manifest': video_rows, 'calibration_subsets': subsets,
        'intervention_identities': [{'span': s, 'pair_index': p} for s, p in PAIR_IDENTITIES],
        'unit_normalization': 'Task042 Phase-A per-unit per-video population z-normalization over all 80 fixed interventions',
        'temporal_distance_metric': 'unchanged Task042 Phase-A (1-Pearson)/2 on per-unit/video z-normalized relative sensitivity signatures',
        'directional_candidate_rule': 'Phase-D authoritative Task037 F3 canonical f3_order_key only; lower key proposed first; if only one pair member has recorded F3 scores select that scored member',
        'directional_candidate_pairs': directed,
        'graph_tie_break': 'ascending task037_global_index for exact nearest-distance ties; no distance threshold',
        'physical_gpus_authorized': [0, 1], 'output_dir': str(out),
        'phase_e2_scope': 'one small 10-class x 3-video structural audit; no physical slicing, no full finetune, no L_TR, no 50% pruning, no Task043',
    }
    write_json(work / 'task042_phase_e2_config.json', cfg)
    print('PHASE_E2_PREPARE=PASS')
    print('E0_EDGES=%d CANDIDATES=%d SURVIVORS=%d' % (len(e0), len(candidate_ids), len(survivor_ids)))
    print('CANDIDATE_IDS=' + ','.join(map(str, candidate_ids)))
    print('SURVIVOR_IDS=' + ','.join(map(str, survivor_ids)))


def load_config(out: Path) -> Dict[str, Any]:
    cfg = json.loads((out / 'work/task042_phase_e2_config.json').read_text(encoding='utf-8'))
    require(cfg['branch'] == BRANCH and cfg['gate_stages'] == list(GATE_STAGES) and cfg['recovery_objective'] == 'cross_entropy_only' and cfg['L_TR_used'] is False,
            'frozen E.2 run configuration was altered')
    return cfg


def capture_worker(args: argparse.Namespace) -> None:
    repo, base, phase, gpu = Path(args.repo).resolve(), Path(args.base).resolve(), args.phase, int(args.gpu)
    require(phase in ('pre', 'post') and gpu in (0, 1), 'capture stage/GPU is not authorized')
    out, work = base / 'phase_e2', base / 'phase_e2' / 'work'
    cfg = load_config(out)
    raw_path = work / ('%s_relation_gpu%d.csv' % (phase, gpu))
    done_path = work / ('%s_relation_gpu%d_done.json' % (phase, gpu))
    require(not raw_path.exists() and not done_path.exists(), 'refusing duplicate E.2 capture worker output')
    task042, _graph, e0, primitives, _phase_d_path, e1, core, _probe, ctfrs = runtime_modules(repo)
    torch, _F, device = e1.torch_runtime(gpu)
    eligible, _selected, _path, _summary = e1.frozen_inputs(repo)
    ids = [int(r['task037_global_index']) for r in eligible]
    candidate_ids, survivor_ids = cfg['candidate_ids_in_frozen_order'], cfg['survivor_ids_in_frozen_order']
    require(set(candidate_ids).isdisjoint(survivor_ids) and set(ids) == set(candidate_ids) | set(survivor_ids),
            'candidate/survivor identity partition differs from the exact BMS eligibility set')
    model, identity, _mods = e1.load_frozen_model(repo, device)
    phase_d_model = e1.runtime_modules(repo)[3]
    if phase == 'post':
        state_path = work / 'task042_phase_e2_post_recovery_state.pth'
    else:
        state_path = work / 'task042_phase_e2_pre_recovery_state.pth'
    require(state_path.is_file(), phase + '-recovery checkpoint is missing')
    load = torch.load(str(state_path), map_location=device)
    model.load_state_dict(load, strict=True)
    state = primitives.GateState(candidate_ids)
    for uid in candidate_ids:
        state.values[int(uid)] = 0.0
    require(set(state.values) == set(candidate_ids) and all(float(v) == 0.0 for v in state.values.values()),
            'final candidate gates are not exactly zero')
    model.eval()
    model.requires_grad_(False)
    capture = e0.PhaseE0Capture(model, eligible, state, torch, detach=True)
    cfg_runtime = json.loads((base / 'task042_run_config.json').read_text(encoding='utf-8'))
    loader = task042._get_loader(ctfrs, cfg_runtime, workers=2)
    videos = read_csv(base / 'task042_video_manifest.csv')
    video_by_name = {Path(r['video_id']).name: r for r in videos}
    shard_indices = task042.deterministic_video_shards(videos)[gpu]
    assigned = set(map(int, shard_indices))
    require(len(assigned) == 15, 'each authorized GPU must receive 15 of the exact 30 frozen videos')
    video_dataset = loader.dataset
    while hasattr(video_dataset, 'dataset'):
        video_dataset = video_dataset.dataset
    ops = core.enumerate_fixed_cardinality_temporal_pairs(32)
    op_ids = task042.fixed_cardinality_identities(core)
    require([(r['span'], r['pair_index']) for r in op_ids] == PAIR_IDENTITIES and len(ops) == 80,
            'the exact 80 Phase-A intervention identities changed')
    expected_capture = set(ids)
    expected_survivors = set(survivor_ids)
    before_gates = dict(state.values)
    row_count = 0
    seen = set()
    fields = ('phase', 'physical_gpu', 'video_index', 'dataset_index', 'video_id', 'label',
              'task037_global_index', 'domain_id', 'unit_type', 'layer', 'unit_index', 'stage',
              'span', 'pair_index', 'frame_a', 'frame_b', 'baseline_norm', 'delta_norm', 'relative_sensitivity')
    with raw_path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for batch in loader:
            clips, labels = batch[0], batch[1]
            local_idx = int(batch[2][0].item())
            name = Path(str(video_dataset.clips[local_idx][0])).name
            require(name in video_by_name, 'loader yielded a video outside Phase-A manifest')
            video = video_by_name[name]
            vi = int(video['video_index'])
            if vi not in assigned:
                continue
            require(vi not in seen and int(labels[0].item()) == int(video['label']), 'duplicate video or label mismatch')
            clip = clips[0].to(device=device, dtype=torch.float32, non_blocking=True)
            require(clip.ndim == 4 and int(clip.shape[1]) == 32 and str(clip.dtype) == 'torch.float32',
                    'capture must use Phase-A FP32 32-frame clips')
            model.eval()
            with torch.inference_mode():
                _logits, base_acts = e0.model_forward(model, capture, clip, phase_d_model)
                require(set(base_acts) == expected_capture, 'baseline did not capture the exact frozen unit identities')
                h0 = dict(base_acts)
                rows = []
                for op, identity_row in zip(ops, op_ids):
                    swapped = core.apply_temporal_interventions(clip, [op], time_dim=1)[0]
                    _logits, h1 = e0.model_forward(model, capture, swapped, phase_d_model)
                    require(set(h1) == expected_capture, 'intervention did not capture exact frozen unit identities')
                    for unit in eligible:
                        uid = int(unit['task037_global_index'])
                        if uid not in expected_survivors:
                            continue
                        h_base, h_intervened = h0[uid], h1[uid]
                        base_norm = float(torch.norm(h_base.reshape(-1), p=2).item())
                        delta_norm = float(torch.norm((h_intervened - h_base).reshape(-1), p=2).item())
                        value = float(primitives.relative_sensitivity(h_base, h_intervened, primitives.EPS_SENSITIVITY).item())
                        require(math.isfinite(value), 'nonfinite Task042 Phase-A relative sensitivity')
                        rows.append({'phase': phase, 'physical_gpu': gpu, 'video_index': vi,
                                     'dataset_index': int(video['dataset_index']), 'video_id': video['video_id'],
                                     'label': int(video['label']), 'task037_global_index': uid,
                                     'domain_id': unit['domain_id'], 'unit_type': unit['unit_type'],
                                     'layer': unit['layer'], 'unit_index': unit['unit_index'], 'stage': unit['stage'],
                                     'span': identity_row['span'], 'pair_index': identity_row['pair_index'],
                                     'frame_a': identity_row['frame_a'], 'frame_b': identity_row['frame_b'],
                                     'baseline_norm': base_norm, 'delta_norm': delta_norm,
                                     'relative_sensitivity': value})
                require(len(rows) == 80 * len(expected_survivors), 'video row count differs from exact survivor x intervention product')
                rows.sort(key=lambda r: (r['task037_global_index'], r['span'], r['pair_index']))
                writer.writerows(rows)
                f.flush()
                row_count += len(rows)
            seen.add(vi)
    capture.close()
    require(seen == assigned and dict(state.values) == before_gates, 'capture identities or gate values changed')
    require(row_count == 15 * 80 * len(survivor_ids), 'GPU capture shard row count mismatch')
    write_json(done_path, {'phase': phase, 'physical_gpu': gpu, 'video_indices': sorted(seen), 'row_count': row_count,
                           'unit_count': len(survivor_ids), 'forward_count': 15 * 81, 'dtype': 'float32', 'amp': False,
                           'checkpoint_sha256': identity['checkpoint_sha256'],
                           'loaded_state_sha256': sha256_file(state_path),
                           'candidate_gates_exactly_zero': True, 'nonselected_units_active': True})
    print('CAPTURE_OK phase=%s gpu=%d videos=%d rows=%d' % (phase, gpu, len(seen), row_count))


def train_round(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e2', base / 'phase_e2' / 'work'
    cfg = load_config(out)
    manifest_path = out / 'task042_phase_e2_progressive_round_manifest.csv'
    pre_state = work / 'task042_phase_e2_pre_recovery_state.pth'
    post_state = work / 'task042_phase_e2_post_recovery_state.pth'
    require(not manifest_path.exists() and not pre_state.exists() and not post_state.exists(), 'refusing duplicate progressive-round training')
    task042, _graph, e0, primitives, _phase_d_path, e1, _core, _probe, ctfrs = runtime_modules(repo)
    torch, F, device = e1.torch_runtime(0)
    eligible, _selected, _path, _summary = e1.frozen_inputs(repo)
    candidate_ids = list(map(int, cfg['candidate_ids_in_frozen_order']))
    survivor_ids = list(map(int, cfg['survivor_ids_in_frozen_order']))
    all_ids = [int(r['task037_global_index']) for r in eligible]
    require(set(candidate_ids).isdisjoint(survivor_ids) and set(candidate_ids) | set(survivor_ids) == set(all_ids),
            'candidate/survivor unit identities changed before training')
    model, identity, _modules = e1.load_frozen_model(repo, device)
    phase_d_model = e1.runtime_modules(repo)[3]
    model.requires_grad_(True)
    model.eval()
    state = primitives.GateState(candidate_ids)
    capture = e0.PhaseE0Capture(model, eligible, state, torch, detach=False)
    e1_manifest = e1.training_manifest_values(base / 'phase_e1/task042_phase_e1_training_manifest.csv')
    train_list = base / 'phase_e1/work/task042_phase_e1_exact_training_list.txt'
    clips, labels, names = e1.load_training_clips(ctfrs, torch, train_list, device)
    expected_videos = e1_manifest['training_videos']
    require(names == [r['video_id'] for r in expected_videos] and labels == [int(r['label']) for r in expected_videos],
            'exact Phase-E.1 recovery video identity/order changed')
    schedule = list(map(int, e1_manifest['training_schedule_indices']))
    require(len(schedule) == 120 and schedule == [i for _stage in range(4) for i in range(30)],
            'E.1 recovery optimizer-step schedule changed')
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    rows = []
    update = 0
    for stage_idx, gate in enumerate(GATE_STAGES[1:], 1):
        e1.set_all_gates(state, candidate_ids, gate)
        for _ in range(RECOVERY_STEPS_PER_STAGE):
            training_index = schedule[update]
            clip, label = clips[training_index], labels[training_index]
            optimizer.zero_grad(set_to_none=True)
            logits, _acts = e0.model_forward(model, capture, clip, phase_d_model)
            target = torch.tensor([label], dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target)
            require(torch.isfinite(ce).item(), 'nonfinite CE during E.2 recovery')
            ce.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            require(torch.isfinite(norm).item(), 'nonfinite CE gradient norm during E.2 recovery')
            optimizer.step()
            update += 1
            rows.append({'record_type': 'optimizer_step', 'optimizer_step': update, 'stage_index': stage_idx,
                         'gate_value': gate, 'training_index': training_index, 'video_id': names[training_index],
                         'label': label, 'objective': RECOVERY_OBJECTIVE, 'cross_entropy': float(ce.detach().item()),
                         'L_TR_used': False, 'gradient_norm_preclip': float(norm.detach().item()),
                         'gradient_clip_norm': GRAD_CLIP_NORM, 'seed': SEED,
                         'candidate_ids_json': json.dumps(candidate_ids, separators=(',', ':')),
                         'survivor_ids_json': json.dumps(survivor_ids, separators=(',', ':')),
                         'checkpoint_sha256': cfg['checkpoint_sha256'], 'optimizer': 'SGD',
                         'learning_rate': LEARNING_RATE, 'momentum': MOMENTUM, 'weight_decay': WEIGHT_DECAY})
        require(update == stage_idx * RECOVERY_STEPS_PER_STAGE, 'progressive recovery stage step count changed')
        if gate == 0.25:
            e1.set_all_gates(state, candidate_ids, 0.0)
            torch.save(model.state_dict(), str(pre_state))
            rows.append({'record_type': 'pre_recovery_boundary', 'optimizer_step': update, 'stage_index': stage_idx,
                         'gate_value': 0.0, 'objective': 'no_update_gate_zero_control', 'cross_entropy': '',
                         'L_TR_used': False, 'candidate_ids_json': json.dumps(candidate_ids, separators=(',', ':')),
                         'survivor_ids_json': json.dumps(survivor_ids, separators=(',', ':')),
                         'note': 'Saved immediately after attenuation to gate 0, before any gate-0 recovery update.'})
    require(update == 120 and all(float(v) == 0.0 for v in state.values.values()) and list(state.values) == candidate_ids,
            'final candidate gates/optimizer budget differ from E.1')
    torch.save(model.state_dict(), str(post_state))
    capture.close()
    fields = ('record_type', 'optimizer_step', 'stage_index', 'gate_value', 'training_index', 'video_id', 'label',
              'objective', 'cross_entropy', 'L_TR_used', 'gradient_norm_preclip', 'gradient_clip_norm', 'seed',
              'candidate_ids_json', 'survivor_ids_json', 'checkpoint_sha256', 'optimizer', 'learning_rate',
              'momentum', 'weight_decay', 'note')
    write_csv(manifest_path, rows, fields, exclusive=True)
    write_json(work / 'task042_phase_e2_training_checkpoints.json', {
        'physical_gpu': 0, 'start_checkpoint_sha256': identity['checkpoint_sha256'], 'seed': SEED,
        'candidate_ids': candidate_ids, 'survivor_ids': survivor_ids, 'gate_stages': list(GATE_STAGES),
        'updates': update, 'recovery_steps_per_stage': RECOVERY_STEPS_PER_STAGE, 'objective': RECOVERY_OBJECTIVE,
        'L_TR_used': False, 'pre_recovery_step': 90, 'pre_state_sha256': sha256_file(pre_state),
        'post_state_sha256': sha256_file(post_state), 'final_candidate_gates_zero': True,
        'nonselected_units_active': True, 'physical_tensor_slicing': False,
    })
    print('TRAIN_OK updates=%d pre_step=90 post_step=120 objective=CE_ONLY candidates=%d' % (update, len(candidate_ids)))


def load_capture_rows(work: Path, phase: str) -> List[Dict[str, str]]:
    rows = []
    for gpu in (0, 1):
        done = json.loads((work / ('%s_relation_gpu%d_done.json' % (phase, gpu))).read_text(encoding='utf-8'))
        require(int(done['physical_gpu']) == gpu and done['phase'] == phase and int(done['forward_count']) == 15 * 81,
                '%s GPU%d capture done manifest identity mismatch' % (phase, gpu))
        rows.extend(read_csv(work / ('%s_relation_gpu%d.csv' % (phase, gpu))))
    return rows


def graph_rows_with_domains(rows: Sequence[Mapping[str, Any]], unit_by_id: Mapping[int, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for source in rows:
        row = dict(source)
        meta = unit_by_id[int(row['task037_global_index'])]
        row.update({'unit_type': meta['unit_type'], 'layer': meta['layer'], 'unit_index': meta['unit_index'], 'stage': meta['stage']})
        out.append(row)
    return out


def set_decision(summary_path: Path, decision: str, rationale: str, report_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(decision in ('A. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_PROMISING',
                         'B. TEMPORAL_GRAPH_MOSTLY_STATIC_OR_UNRESOLVED',
                         'C. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_REJECTED'), 'decision must be exactly A, B, or C')
    require(bool(rationale.strip()), 'a brief evidence-grounded decision rationale is required')
    if decision.startswith('A.'):
        edge_evidence = summary['evidence']['new_post_edges_after_recovery']
        require(edge_evidence > 0 and summary['evidence']['new_post_edges_without_exact_tie'] > 0,
                'A requires genuinely new reciprocal survivor edges that are not exact tie artifacts')
    summary['decision'] = decision
    summary['decision_rationale'] = rationale.strip()
    report_path.write_text(make_report(summary), encoding='utf-8')
    summary.setdefault('outputs', {})['task042_phase_e2_report.md'] = {'sha256': sha256_file(report_path), 'bytes': report_path.stat().st_size}
    write_json(summary_path, summary)


def finalize(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e2', base / 'phase_e2' / 'work'
    cfg = load_config(out)
    task042, graph, _e0, _primitives, _phase_d_path, e1, _core, _probe, _ctfrs = runtime_modules(repo)
    eligible, _selected, _path, _phase_d_summary = e1.frozen_inputs(repo)
    survivor_ids = list(map(int, cfg['survivor_ids_in_frozen_order']))
    survivor_rows = [r for r in eligible if int(r['task037_global_index']) in set(survivor_ids)]
    require(len(survivor_rows) == len(survivor_ids), 'survivor identities do not join exactly to the frozen unit manifest')
    video_rows = read_csv(base / 'task042_video_manifest.csv')
    expected_indices = sorted(int(r['video_index']) for r in video_rows)
    pre_raw, post_raw = load_capture_rows(work, 'pre'), load_capture_rows(work, 'post')
    expected_rows = 30 * 80 * len(survivor_ids)
    require(len(pre_raw) == expected_rows and len(post_raw) == expected_rows, 'pre/post relation capture row counts do not match 30x80xsurvivors')
    for phase_name, rows in (('pre', pre_raw), ('post', post_raw)):
        require(sorted({int(r['video_index']) for r in rows}) == expected_indices, phase_name + ' capture video identities differ')
        require({int(r['task037_global_index']) for r in rows} == set(survivor_ids), phase_name + ' capture survivor identities differ')
        require(all(int(r['physical_gpu']) in (0, 1) and r['phase'] == phase_name for r in rows), 'capture used an unauthorized GPU or phase label')
    dpre, npre, pre_video = graph.aggregate_relation_distances(pre_raw, survivor_rows, expected_indices, task042)
    dpost, npost, post_video = graph.aggregate_relation_distances(post_raw, survivor_rows, expected_indices, task042)
    phase_a_rows = read_csv(base / 'task042_temporal_pair_distance.csv')
    d0_all = parse_distance_rows(phase_a_rows)
    survivor_edges = sorted(dpost)
    require(set(survivor_edges) == set(dpre) and set(survivor_edges) <= set(d0_all), 'survivor same-domain pair set differs from frozen Phase-A domain identities')
    require(all(npre[e] == 30 and npost[e] == 30 for e in survivor_edges), 'pre/post d_temp must use all exact 30 videos')
    unit_by_id = {int(r['task037_global_index']): r for r in eligible}
    distance_rows = []
    delta_values: Dict[str, List[float]] = defaultdict(list)
    for i, j in survivor_edges:
        d0, dp, d1 = d0_all[(i, j)], dpre[(i, j)], dpost[(i, j)]
        di, dj = unit_by_id[i], unit_by_id[j]
        type_pair = ('head-head' if di['unit_type'] == dj['unit_type'] == 'attention_head' else
                     'FFN-FFN' if di['unit_type'] == dj['unit_type'] == 'ffn_neuron' else 'head-FFN')
        delta = float(d1) - float(d0)
        delta_values['ALL'].append(delta)
        delta_values[type_pair].append(delta)
        distance_rows.append({'domain_id': str(di['domain_id']), 'task037_global_index_i': i,
                              'task037_global_index_j': j, 'unit_type_i': di['unit_type'], 'unit_type_j': dj['unit_type'],
                              'type_pair': type_pair, 'd_temp_phase_a': d0, 'd_temp_pre_recovery_gate_zero': dp,
                              'd_temp_post_ce_recovery': d1, 'delta_d_post_minus_phase_a': delta,
                              'delta_d_post_minus_pre_recovery': float(d1) - float(dp),
                              'valid_videos_phase_a': 30, 'valid_videos_pre': npre[(i, j)], 'valid_videos_post': npost[(i, j)]})
    distance_fields = ('domain_id', 'task037_global_index_i', 'task037_global_index_j', 'unit_type_i', 'unit_type_j', 'type_pair',
                       'd_temp_phase_a', 'd_temp_pre_recovery_gate_zero', 'd_temp_post_ce_recovery',
                       'delta_d_post_minus_phase_a', 'delta_d_post_minus_pre_recovery',
                       'valid_videos_phase_a', 'valid_videos_pre', 'valid_videos_post')
    write_csv(out / E2_OUTPUTS[2], distance_rows, distance_fields)
    g0_rows = graph_rows_with_domains(graph.directed_nearest_neighbors(eligible, d0_all), unit_by_id)
    gpre_rows = graph_rows_with_domains(graph.directed_nearest_neighbors(survivor_rows, dpre), unit_by_id)
    gpost_rows = graph_rows_with_domains(graph.directed_nearest_neighbors(survivor_rows, dpost), unit_by_id)
    initial_edges = graph.reciprocal_edges(g0_rows)
    pre_edges, post_edges = graph.reciprocal_edges(gpre_rows), graph.reciprocal_edges(gpost_rows)
    require(initial_edges == {tuple(x) for x in cfg['initial_reciprocal_edges']}, 'initial graph reconstruction differs from preparation')
    post_graph_rows = graph_rows_with_domains(gpost_rows, unit_by_id)
    write_csv(out / E2_OUTPUTS[3], post_graph_rows,
              ('domain_id', 'task037_global_index', 'unit_type', 'layer', 'unit_index', 'stage',
               'nn_task037_global_index', 'nn_distance', 'second_nn_distance', 'nn_margin', 'exact_nearest_tie',
               'reciprocal_partner_task037_global_index', 'is_reciprocal_edge'))
    unit_domain = {int(r['task037_global_index']): str(r['domain_id']) for r in eligible}
    change_initial, _r, _l, new_post = graph.graph_change_rows(initial_edges, post_edges, survivor_ids, unit_domain)
    change_control, _r2, _l2, new_control = graph.graph_change_rows(pre_edges, post_edges, survivor_ids, unit_domain)
    change_rows = []
    for comp, values in (('Phase_A_initial_vs_post_recovery', change_initial), ('pre_recovery_gate_zero_vs_post_CE_recovery', change_control)):
        for item in values:
            row = {'comparison': comp, **item}
            for field in ('initial_edge_ids', 'possible_initial_edge_ids', 'post_edge_ids', 'retained_edge_ids',
                          'lost_edge_ids', 'new_edge_ids', 'impossible_edge_ids'):
                row[field] = json.dumps(row[field], separators=(',', ':'))
            change_rows.append(row)
    write_csv(out / E2_OUTPUTS[4], change_rows,
              ('comparison', 'domain_id', 'initial_edges', 'initial_edges_possible_among_survivors',
               'initial_edges_impossible_removed_endpoint', 'post_edges', 'retained_edges', 'lost_possible_edges',
               'new_edges', 'jaccard_on_possible_edges', 'initial_edge_ids', 'possible_initial_edge_ids',
               'post_edge_ids', 'retained_edge_ids', 'lost_edge_ids', 'new_edge_ids', 'impossible_edge_ids'))
    subsets = graph.calibration_subsets(video_rows)
    stability_rows = []
    subset_graphs = {}
    for name, vids in subsets.items():
        ds, _counts, _byvid = graph.aggregate_relation_distances(post_raw, survivor_rows, vids, task042)
        sub_g = graph.directed_nearest_neighbors(survivor_rows, ds)
        subset_graphs[name] = graph.reciprocal_edges(sub_g)
    post_by_uid = {int(r['task037_global_index']): r for r in gpost_rows}
    for edge in sorted(new_post):
        i, j = edge
        ri, rj = post_by_uid[i], post_by_uid[j]
        exact_tie = bool(ri['exact_nearest_tie'] or rj['exact_nearest_tie'])
        recovered = [name for name in subsets if edge in subset_graphs[name]]
        stability_rows.append({'domain_id': unit_domain[i], 'edge_i_task037_global_index': i,
                               'edge_j_task037_global_index': j, 'd_temp_full_10x3': dpost[edge],
                               'nearest_margin_i': ri['nn_margin'], 'nearest_margin_j': rj['nn_margin'],
                               'exact_nearest_tie_at_full_10x3': exact_tie,
                               'calibration_subsets_recovering_edge_count': len(recovered),
                               'calibration_subset_count': len(subsets),
                               'recovering_subsets_json': json.dumps(recovered, separators=(',', ':')),
                               'subset_edge_identity_results_json': json.dumps({n: edge in subset_graphs[n] for n in subsets}, sort_keys=True)})
    write_csv(out / E2_OUTPUTS[5], stability_rows,
              ('domain_id', 'edge_i_task037_global_index', 'edge_j_task037_global_index', 'd_temp_full_10x3',
               'nearest_margin_i', 'nearest_margin_j', 'exact_nearest_tie_at_full_10x3',
               'calibration_subsets_recovering_edge_count', 'calibration_subset_count',
               'recovering_subsets_json', 'subset_edge_identity_results_json'))
    pre_by_uid = {int(r['task037_global_index']): r for r in gpre_rows}
    pre_post_rows = []
    for i, j in survivor_edges:
        pre_post_rows.append({'domain_id': unit_domain[i], 'task037_global_index_i': i, 'task037_global_index_j': j,
                              'd_temp_pre_recovery_gate_zero': dpre[(i, j)], 'd_temp_post_ce_recovery': dpost[(i, j)],
                              'delta_d_post_minus_pre': dpost[(i, j)] - dpre[(i, j)],
                              'is_pre_recovery_reciprocal_edge': (i, j) in pre_edges,
                              'is_post_recovery_reciprocal_edge': (i, j) in post_edges,
                              'pre_nn_i': pre_by_uid[i]['nn_task037_global_index'], 'pre_nn_j': pre_by_uid[j]['nn_task037_global_index'],
                              'post_nn_i': post_by_uid[i]['nn_task037_global_index'], 'post_nn_j': post_by_uid[j]['nn_task037_global_index']})
    write_csv(out / E2_OUTPUTS[6], pre_post_rows,
              ('domain_id', 'task037_global_index_i', 'task037_global_index_j', 'd_temp_pre_recovery_gate_zero',
               'd_temp_post_ce_recovery', 'delta_d_post_minus_pre', 'is_pre_recovery_reciprocal_edge',
               'is_post_recovery_reciprocal_edge', 'pre_nn_i', 'pre_nn_j', 'post_nn_i', 'post_nn_j'))
    typed_stats = {k: graph.summarize(v) for k, v in delta_values.items()}
    control_stats = graph.summarize([r['delta_d_post_minus_pre_recovery'] for r in distance_rows])
    impossible = [e for e in initial_edges if not set(e) <= set(survivor_ids)]
    tie_new = [e for e in new_post if post_by_uid[e[0]]['exact_nearest_tie'] or post_by_uid[e[1]]['exact_nearest_tie']]
    pre_new = new_post - pre_edges
    evidence = {
        'initial_reciprocal_edge_count': len(initial_edges), 'survivor_count': len(survivor_ids),
        'surviving_same_domain_pair_count': len(survivor_edges), 'pre_recovery_edge_count': len(pre_edges),
        'post_recovery_edge_count': len(post_edges), 'new_post_edges_after_recovery': len(new_post),
        'new_post_edges_already_present_pre_recovery': len(new_post & pre_edges),
        'new_post_edges_absent_pre_recovery': len(pre_new),
        'new_post_edges_without_exact_tie': len(new_post - set(tie_new)),
        'new_post_edges_with_exact_tie': len(tie_new),
        'initial_edges_impossible_removed_endpoint': len(impossible),
        'initial_edges_possible_among_survivors': len(initial_edges) - len(impossible),
        'new_post_edge_subset_recovery_counts': {f'{i}-{j}': next(r['calibration_subsets_recovering_edge_count'] for r in stability_rows if int(r['edge_i_task037_global_index']) == i and int(r['edge_j_task037_global_index']) == j) for i, j in sorted(new_post)},
        'distance_delta_post_minus_phase_a': typed_stats,
        'distance_delta_post_minus_pre_recovery': control_stats,
        'type_pair_counts': {k: len(v) for k, v in delta_values.items()},
        'graph_change_initial_vs_post': next(r for r in change_initial if r['domain_id'] == 'ALL_DOMAINS'),
        'graph_change_pre_vs_post': next(r for r in change_control if r['domain_id'] == 'ALL_DOMAINS'),
        'calibration_subsets': {k: v for k, v in subsets.items()},
    }
    # Remove edge ID arrays from the compact summary; full identities remain in graph_change.csv.
    for k in ('graph_change_initial_vs_post', 'graph_change_pre_vs_post'):
        evidence[k] = {kk: vv for kk, vv in evidence[k].items() if not kk.endswith('_ids')}
    summary = {'task': cfg['task'], 'branch': BRANCH, 'code_head': cfg['code_head'], 'phase_d_head': cfg['phase_d_head'],
               'checkpoint_sha256': cfg['checkpoint_sha256'], 'input_sha256': cfg['input_sha256'],
               'candidate_ids': cfg['candidate_ids_in_frozen_order'], 'survivor_ids': survivor_ids,
               'gate_stages': list(GATE_STAGES), 'recovery_steps_per_stage': RECOVERY_STEPS_PER_STAGE,
               'total_optimizer_updates': 120, 'objective': 'cross_entropy_only', 'L_TR_used': False,
               'physical_gpus_used': [0, 1], 'metric_unchanged': True, 'physical_tensor_slicing': False,
               'full_finetuning': False, 'evidence': evidence, 'decision': 'PENDING_EVIDENCE_REVIEW',
               'decision_rationale': '', 'outputs': {}}
    for name in E2_OUTPUTS[:7]:
        fp = out / name
        summary['outputs'][name] = {'sha256': sha256_file(fp), 'bytes': fp.stat().st_size, 'rows': len(read_csv(fp))}
    summary_path = out / E2_OUTPUTS[7]
    report_path = out / E2_OUTPUTS[8]
    write_json(summary_path, summary)
    report_path.write_text(make_report(summary), encoding='utf-8')
    summary['outputs'][E2_OUTPUTS[8]] = {'sha256': sha256_file(report_path), 'bytes': report_path.stat().st_size}
    write_json(summary_path, summary)
    if args.decision:
        set_decision(summary_path, args.decision, args.rationale or '', report_path)
    print('PHASE_E2_FINALIZE=PASS')
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


def make_report(summary: Mapping[str, Any]) -> str:
    ev = summary['evidence']
    lines = [
        '# Task042 Phase E.2 — Temporal Redundancy Graph Refresh Feasibility', '',
        f"**Decision:** {summary['decision']}", '',
        f"**Rationale:** {summary.get('decision_rationale') or 'Evidence review pending.'}", '',
        '## Scope and frozen protocol', '',
        'This audit tests whether one deterministic progressive attenuation and CE recovery round changes the temporal-redundancy graph among the surviving BMS units. The Task042 Phase-A distance, 10 classes × 3 videos, 80 frame-pair swaps, FP32 capture semantics, and per-unit/per-video normalization were reused unchanged.', '',
        f"The run used {len(summary['candidate_ids'])} directionally selected candidate gates and retained {len(summary['survivor_ids'])} units. Candidate direction came only from the frozen Task037 F3 criterion. The schedule was {summary['gate_stages']} with {summary['recovery_steps_per_stage']} CE-only updates at each noninitial gate (120 total); L_TR was not used, no tensor slicing or full fine-tuning occurred, and only physical GPUs {summary['physical_gpus_used']} were used.", '',
        '## Reciprocal graph result', '',
        f"The frozen Phase-A graph had {ev['initial_reciprocal_edge_count']} reciprocal edges. {ev['initial_edges_impossible_removed_endpoint']} became impossible because a selected endpoint was gated to zero; {ev['initial_edges_possible_among_survivors']} initial edges were still comparable among survivors. Before gate-0 recovery the graph had {ev['pre_recovery_edge_count']} reciprocal edges; after CE recovery it had {ev['post_recovery_edge_count']}. The post graph contains {ev['new_post_edges_after_recovery']} edges absent from the possible Phase-A survivor graph, including {ev['new_post_edges_absent_pre_recovery']} absent from the pre-recovery gate-zero control.", '',
        f"Among post edges new relative to Phase A, {ev['new_post_edges_without_exact_tie']} have no exact nearest-distance tie at either endpoint and {ev['new_post_edges_with_exact_tie']} involve an exact tie. Per-edge recovery counts across the seven frozen calibration subsets are recorded in `task042_phase_e2_calibration_stability.csv`: `{json.dumps(ev['new_post_edge_subset_recovery_counts'], sort_keys=True)}`. These counts are descriptive; no threshold was introduced.", '',
        '## Pairwise distance changes', '',
        'All surviving same-domain pairs are listed in `task042_phase_e2_post_relation_distance.csv`. The reported `delta_d_post_minus_phase_a` is d_temp after CE recovery minus the exact Phase-A d_temp. Summary values by head-head, FFN-FFN, and head-FFN pairs:', '',
        '| Pair type | n | Mean | Median | Q25 | Q75 | Max absolute change |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    stats = ev['distance_delta_post_minus_phase_a']
    for label in ('ALL', 'head-head', 'FFN-FFN', 'head-FFN'):
        s = stats.get(label, {'count': 0, 'mean': None, 'median': None, 'q25': None, 'q75': None, 'max_absolute_change': None})
        fmt = lambda x: '—' if x is None else f'{x:.8f}'
        lines.append(f"| {label} | {s['count']} | {fmt(s['mean'])} | {fmt(s['median'])} | {fmt(s['q25'])} | {fmt(s['q75'])} | {fmt(s['max_absolute_change'])} |")
    lines += ['', f"The paired pre-recovery versus post-recovery d_temp change summary across all survivors is `{json.dumps(ev['distance_delta_post_minus_pre_recovery'], sort_keys=True)}`. Its graph accounting is in `task042_phase_e2_graph_change.csv` under `pre_recovery_gate_zero_vs_post_CE_recovery`.", '',
              '## Interpretation boundary', '',
              'This is a structural feasibility audit only. It does not test final accuracy, choose pruning units from temporal distances, tune thresholds, or authorize a 50% pruning experiment. The prescribed stop point is after E.2.', '',
              f"Code head: `{summary['code_head']}`  ", f"Checkpoint SHA-256: `{summary['checkpoint_sha256']}`  ",
              f"Phase-D base: `{summary['phase_d_head']}`", '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=str(REPO_DEFAULT))
    parser.add_argument('--base', default=str(BASE_DEFAULT))
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    cap = sub.add_parser('capture')
    cap.add_argument('--phase', choices=('pre', 'post'), required=True)
    cap.add_argument('--gpu', type=int, required=True)
    sub.add_parser('train')
    fin = sub.add_parser('finalize')
    fin.add_argument('--decision', choices=('A. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_PROMISING',
                                            'B. TEMPORAL_GRAPH_MOSTLY_STATIC_OR_UNRESOLVED',
                                            'C. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_REJECTED'))
    fin.add_argument('--rationale')
    sd = sub.add_parser('set-decision')
    sd.add_argument('--decision', required=True, choices=('A. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_PROMISING',
                                                           'B. TEMPORAL_GRAPH_MOSTLY_STATIC_OR_UNRESOLVED',
                                                           'C. DYNAMIC_TEMPORAL_REDUNDANCY_REFRESH_REJECTED'))
    sd.add_argument('--rationale', required=True)
    args = parser.parse_args()
    if args.cmd == 'prepare':
        prepare(args)
    elif args.cmd == 'capture':
        capture_worker(args)
    elif args.cmd == 'train':
        train_round(args)
    elif args.cmd == 'finalize':
        finalize(args)
    elif args.cmd == 'set-decision':
        out = Path(args.base).resolve() / 'phase_e2'
        set_decision(out / E2_OUTPUTS[7], args.decision, args.rationale, out / E2_OUTPUTS[8])
        print('DECISION_SET=' + args.decision)


if __name__ == '__main__':
    main()
