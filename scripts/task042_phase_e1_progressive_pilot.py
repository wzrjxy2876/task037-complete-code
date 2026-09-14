#!/usr/bin/env python3
"""Task042 Phase E.1 controlled progressive-pruning pilot."""
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
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_DEFAULT = Path('/home/jixinye25/jxy_work1/task042_post_bms_frame_relation_redundancy')
BASE_DEFAULT = Path('/data/jixinye25/work1/output/task042_post_bms_frame_relation_redundancy')
PROJECT_ROOT = Path('/home/jixinye25/jxy_work1/swintrans_task035')
CHECKPOINT = Path('/home/jixinye25/jxy_work1/pretrained/checkpoint-68.ckpt')
VAL_LIST = Path('/data/jixinye25/UCF101_Frame/val_rgb_split1.txt')
TRAIN_LIST = Path('/data/jixinye25/UCF101_Frame/train_rgb_split1.txt')
FRAME_ROOT = Path('/data/jixinye25/UCF101_Frame/frames')
BRANCH = 'task_042_post_bms_frame_relation_redundancy'
PHASE_D_HEAD = 'a8135a59c7b81ba978c158a56baae2b4f0b6a651'
CHECKPOINT_SHA = '4ce0dad71e51f6af65b07ec2c46a10a3e792b694d6427dedc2626d22c0744c63'
PHASE_D = BASE_DEFAULT / 'phase_d'
UNIT_MANIFEST = BASE_DEFAULT / 'task042_unit_manifest.csv'
VIDEO_MANIFEST = BASE_DEFAULT / 'task042_video_manifest.csv'
RUN_CONFIG = BASE_DEFAULT / 'task042_run_config.json'
SEED = 3407
PILOT_COUNT = 12
GATE_STAGES = (1.00, 0.75, 0.50, 0.25, 0.00)
SPANS = (1, 2, 4, 8, 16)
CALIBRATION_PER_CLASS = 1
TRAIN_VIDEOS_PER_CLASS = 3
RECOVERY_STEPS_PER_STAGE = 30
VALIDATION_BATCH_SIZE = 2
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
MOMENTUM = 0.9
REQUIRED_OUTPUTS = (
    'task042_phase_e1_candidate_manifest.csv',
    'task042_phase_e1_gradient_scale_calibration.csv',
    'task042_phase_e1_training_manifest.csv',
    'task042_phase_e1_validation_trajectory.csv',
    'task042_phase_e1_relation_drift.csv',
    'task042_phase_e1_recovery_auc.csv',
    'task042_phase_e1_arm_comparison.csv',
    'task042_phase_e1_summary.json',
    'task042_phase_e1_report.md',
)


def require(ok: Any, message: str) -> None:
    if not ok:
        raise RuntimeError('Task042 Phase E.1 gate failed: ' + str(message))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv_new(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    require(not path.exists(), 'refusing to overwrite ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def write_csv_replace(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    os.replace(str(tmp), str(path))


def write_json(path: Path, value: Mapping[str, Any], exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = 'x' if exclusive else 'w'
    with path.open(mode, encoding='utf-8') as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def median_gradient_calibration(ce_norms: Sequence[float], tr_norms: Sequence[float]) -> Tuple[float, float, float]:
    require(len(ce_norms) == len(tr_norms) and len(ce_norms) >= 2,
            'calibration requires matching multiple-batch CE/TR gradients')
    require(all(math.isfinite(float(x)) and float(x) >= 0 for x in list(ce_norms) + list(tr_norms)),
            'gradient norms must be finite and nonnegative')
    gce, gtr = statistics.median(map(float, ce_norms)), statistics.median(map(float, tr_norms))
    require(gce > 0.0 and gtr > 0.0, 'median CE/TR gradient norm must be nonzero')
    lam = gce / (gtr + 1e-12)
    require(math.isfinite(lam) and lam > 0.0, 'calibrated lambda is not finite and positive')
    return gce, gtr, lam


class FrozenLossScale:
    """Immutable Python scalar so the calibrated coefficient cannot drift."""
    __slots__ = ('_value',)

    def __init__(self, value: float):
        value = float(value)
        require(math.isfinite(value) and value > 0.0, 'invalid frozen lambda_TR')
        object.__setattr__(self, '_value', value)

    @property
    def value(self) -> float:
        return self._value

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError('lambda_TR is frozen for the entire Phase E.1 run')


def validate_gate_schedule(schedule: Sequence[float]) -> None:
    require(tuple(map(float, schedule)) == GATE_STAGES, 'gate schedule differs from frozen 1,.75,.50,.25,0')


def validate_identical_candidate_sets(candidate_ids: Sequence[int], arm_ids: Mapping[str, Sequence[int]]) -> None:
    frozen = tuple(map(int, candidate_ids))
    require(set(arm_ids) == {'A_ONE_SHOT', 'B_PROGRESSIVE_CE', 'C_PROGRESSIVE_CE_TR'},
            'exact A/B/C arms are required')
    for arm, ids in arm_ids.items():
        require(tuple(map(int, ids)) == frozen, 'candidate identity/order mismatch for ' + arm)


def normalized_trapezoid_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    require(len(xs) == len(ys) and len(xs) >= 2, 'AUC requires at least two matched checkpoints')
    require(all(math.isfinite(float(v)) for v in list(xs) + list(ys)), 'AUC input is nonfinite')
    require(all(float(xs[i]) < float(xs[i + 1]) for i in range(len(xs) - 1)), 'AUC steps must be strictly increasing')
    area = sum((float(xs[i + 1]) - float(xs[i])) *
               (float(ys[i + 1]) + float(ys[i])) * 0.5 for i in range(len(xs) - 1))
    return area / (float(xs[-1]) - float(xs[0]))


def runtime_modules(repo: Path):
    runtime, scripts = str(repo / 'src' / 'lgfr_runtime'), str(repo / 'scripts')
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import task042_frame_relation_redundancy as task042
    import task042_phase_e0_feasibility as e0
    import task042_phase_e0_temporal_preservation as primitives
    import task041_phase_d_fullval_oracle as phase_d
    e0.PRIMITIVES = primitives
    core, probe, ctfrs = task042._runtime_modules(repo)
    return task042, e0, primitives, phase_d, core, probe, ctfrs


def frozen_inputs(repo: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    require(subprocess.check_output(['git', '-C', str(repo), 'branch', '--show-current'], text=True).strip() == BRANCH,
            'must stay on the existing Task042 branch')
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA, 'checkpoint SHA changed')
    summary_path = PHASE_D / 'task042_phase_d_summary.json'
    candidate_path = PHASE_D / 'task042_phase_d_candidate_provenance.csv'
    baseline_path = PHASE_D / 'task042_phase_d_baseline_path.csv'
    identity_path = PHASE_D / 'task042_phase_d_input_identity.json'
    for p in (summary_path, candidate_path, baseline_path, identity_path, UNIT_MANIFEST, VIDEO_MANIFEST, RUN_CONFIG, VAL_LIST, TRAIN_LIST):
        require(p.is_file(), 'frozen Phase-D/data input missing: ' + str(p))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    require(summary.get('git_head') == PHASE_D_HEAD and
            summary.get('decision') == 'C. TEMPORAL_COVERAGE_PROGRESSIVE_PATH_REJECTED',
            'Phase-D scientific state or frozen base differs')
    require(sha256_file(candidate_path) == summary['outputs']['task042_phase_d_candidate_provenance.csv']['sha256'],
            'Phase-D candidate provenance hash mismatch')
    require(sha256_file(baseline_path) == summary['outputs']['task042_phase_d_baseline_path.csv']['sha256'],
            'Phase-D baseline F3 path hash mismatch')
    unit_rows = read_csv(UNIT_MANIFEST)
    units_by_id = {int(r['task037_global_index']): dict(r) for r in unit_rows}
    require(len(units_by_id) == len(unit_rows), 'duplicate Task042 unit identity')
    provenance = {int(r['task037_global_index']): dict(r) for r in read_csv(candidate_path)}
    ordered_path = sorted(read_csv(baseline_path), key=lambda r: int(r['step']))
    require(len(ordered_path) == 21 and [int(r['step']) for r in ordered_path] == list(range(1, 22)),
            'authoritative F3 baseline path is incomplete')
    f3_units: List[Dict[str, Any]] = []
    seen = set()
    for order, p in enumerate(ordered_path, 1):
        uid = int(p['candidate_task037_global_index'])
        require(uid not in seen and uid in units_by_id and uid in provenance, 'F3 candidate identity cannot be joined exactly')
        seen.add(uid)
        src = provenance[uid]
        require(src['selected_for_removal'].lower() == 'true' and
                int(src['f3_global_step']) == int(p['f3_global_step']) and
                src['unit_type'] == p['unit_type'] and src['layer'] == p['layer'] and
                src['domain_id'] == p['domain_id'], 'F3 candidate provenance mismatch')
        unit = dict(units_by_id[uid])
        require(unit['unit_type'] == p['unit_type'] and unit['layer'] == p['layer'] and
                int(unit['unit_index']) == int(p['unit_index']) and unit['domain_id'] == p['domain_id'],
                'frozen unit identity mismatch at F3 order %d' % order)
        unit.update({'f3_order': order, 'f3_global_step': int(p['f3_global_step']),
                     'f3_domain_removal_rank': int(src['within_domain_removal_rank']),
                     'f3_R': float(src['R_F3']), 'f3_path_step': int(p['step']),
                     'cumulative_removed_parameters': int(p['cumulative_removed_parameters']),
                     'cumulative_removed_cohort_ratio': float(p['cumulative_removed_cohort_ratio']),
                     'domain_group': src['domain_group']})
        f3_units.append(unit)
    all_eligible = [dict(units_by_id[int(r['task037_global_index'])]) for r in provenance.values()]
    require(len(all_eligible) == 31, 'Phase-D eligible BMS cohort must remain 31 units')
    all_eligible.sort(key=lambda r: (int(r['domain_id']), int(r['task037_global_index'])))
    domain_types: Dict[str, set] = {}
    for unit in all_eligible:
        domain_types.setdefault(str(unit['domain_id']), set()).add(unit['unit_type'])
    selected = f3_units[:PILOT_COUNT]
    require(len(selected) == PILOT_COUNT, 'pilot candidate prefix not available')
    counts = {t: sum(1 for r in selected if r['unit_type'] == t) for t in ('attention_head', 'ffn_neuron')}
    pilot_domains = {str(r['domain_id']) for r in selected}
    require(counts['attention_head'] >= 2 and counts['ffn_neuron'] >= 2 and len(pilot_domains) > 1,
            'F3 prefix fails mixed structural pilot composition')
    require(domain_types.get('269') == {'attention_head'} and '774' in {str(r['task037_global_index']) for r in selected},
            'pilot prefix does not include the same-type domain 269')
    require(domain_types.get('271') == {'attention_head', 'ffn_neuron'} and '271' in pilot_domains,
            'pilot prefix does not include the mixed domain 271')
    return all_eligible, selected, ordered_path, summary


def project_dataset(ctfrs: Any, list_path: Path, frame_root: Path):
    ctfrs.ensure_project_importable(PROJECT_ROOT)
    old_utils = sys.modules.get('utils')
    installed = old_utils is None
    if installed:
        shim = types.ModuleType('utils')
        shim.UCF_DATA_ROOT = str(frame_root)
        sys.modules['utils'] = shim
    try:
        from dataset import ucf101
    finally:
        if installed:
            sys.modules.pop('utils', None)
    ucf101.UCF_DATA_ROOT = str(frame_root)
    spatial, temporal = ucf101.test_transform()
    dataset = ucf101.attack_ucf101(str(list_path), spatial_transform=spatial,
                                   temporal_transform=temporal)
    return ucf101, dataset, temporal


def validate_dataset_rows(dataset: Any, temporal: Any, list_path: Path, frame_root: Path,
                          expected_count: int, verify_frames: bool) -> List[Dict[str, Any]]:
    lines = list_path.read_text(encoding='utf-8').splitlines()
    require(len(lines) == expected_count == len(dataset.clips),
            '%s must contain exactly %d clips' % (list_path, expected_count))
    rows = []
    for index, (line, clip) in enumerate(zip(lines, dataset.clips)):
        fields = line.split()
        require(len(fields) >= 3, 'malformed split row %d' % index)
        directory, duration, label = clip
        require(Path(fields[0]).name == Path(directory).name and
                int(fields[1]) == int(duration) and int(fields[-1]) == int(label),
                'dataset/list identity mismatch at index %d' % index)
        frame_indices = list(range(1, int(duration) + 1)); sampled = temporal(frame_indices) if temporal else frame_indices
        require(len(sampled) == 32, 'temporal transform did not emit exactly 32 frames')
        if verify_frames:
            require(Path(directory).is_dir(), 'clip directory missing: ' + str(directory))
            missing = [i for i in sampled if not (Path(directory) / ('image_%05d.jpg' % int(i))).is_file()]
            require(not missing, 'missing sampled frame in %s: %s' % (directory, missing[:4]))
        rows.append({'index': index, 'video_id': Path(directory).name,
                     'label': int(label), 'duration': int(duration), 'line': line})
    require(len({r['video_id'] for r in rows}) == expected_count, 'duplicate video identity in split')
    return rows


def choose_training_rows(classes: Sequence[int], train_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_class: Dict[int, List[Mapping[str, Any]]] = {}
    for row in train_rows:
        by_class.setdefault(int(row['label']), []).append(row)
    selected = []
    for label in sorted(set(map(int, classes))):
        candidates = by_class.get(label, [])
        require(len(candidates) >= TRAIN_VIDEOS_PER_CLASS,
                'official training split lacks the frozen validation class %d' % label)
        selected.extend(dict(r) for r in candidates[:TRAIN_VIDEOS_PER_CLASS])
    require(len(selected) == len(set(int(r['label']) for r in selected)) * TRAIN_VIDEOS_PER_CLASS,
            'training subset is not balanced')
    return selected


def candidate_manifest_rows(selected: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    frozen_ids = [int(r['task037_global_index']) for r in selected]
    for row in selected:
        rows.append({
            'candidate_order': row['f3_order'], 'task037_global_index': row['task037_global_index'],
            'unit_type': row['unit_type'], 'domain_id': row['domain_id'],
            'domain_group': row['domain_group'], 'layer': row['layer'], 'unit_index': row['unit_index'],
            'f3_global_step': row['f3_global_step'], 'f3_domain_removal_rank': row['f3_domain_removal_rank'],
            'source_path': 'task042_phase_d_baseline_path.csv:first_12_F3_order',
            'pilot_gate_identity_set': json.dumps(frozen_ids, separators=(',', ':')),
            'final_gate_value': 0.0, 'cumulative_removed_cohort_ratio_at_prefix': row['cumulative_removed_cohort_ratio'],
        })
    return rows


def prepare(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    base = Path(args.base).resolve()
    out = base / 'phase_e1'
    require(not out.exists(), 'Phase E.1 output directory already exists; refusing overwrite')
    eligible, selected, ordered_path, dsummary = frozen_inputs(repo)
    # Create an isolated run directory only after authoritative candidate identity checks pass.
    out.mkdir(parents=True)
    work = out / 'work'
    work.mkdir()
    config = json.loads(RUN_CONFIG.read_text(encoding='utf-8'))
    video_rows = read_csv(VIDEO_MANIFEST)
    classes = sorted({int(r['label']) for r in video_rows})
    require(len(classes) == 10, 'frozen Task042 validation cohort must define exactly 10 training classes')
    train_ucf, train_dataset, train_temporal = project_dataset(
        runtime_modules(repo)[-1], TRAIN_LIST, FRAME_ROOT)
    # Read the authoritative official training list in order and select its first three
    # training clips for each frozen Phase-D validation class.
    raw_train_lines = TRAIN_LIST.read_text(encoding='utf-8').splitlines()
    raw_train = []
    for index, line in enumerate(raw_train_lines):
        fields = line.split()
        require(len(fields) >= 3, 'malformed UCF101 train split row %d' % index)
        raw_train.append({'index': index, 'video_id': Path(fields[0]).name,
                          'label': int(fields[-1]), 'duration': int(fields[1]), 'line': line})
    selected_train = choose_training_rows(classes, raw_train)
    exact_train = work / 'task042_phase_e1_exact_training_list.txt'
    exact_train.write_text('\n'.join(r['line'] for r in selected_train) + '\n', encoding='utf-8')
    train_rows = validate_dataset_rows(train_dataset, train_temporal, TRAIN_LIST, FRAME_ROOT,
                                       len(raw_train_lines), False)
    # Full frame validation is restricted to the 30 frozen recovery clips, while all
    # 3783 validation identities and all sampled frames are checked before evaluation.
    _, pilot_train_dataset, pilot_train_temporal = project_dataset(runtime_modules(repo)[-1], exact_train, FRAME_ROOT)
    pilot_train_rows = validate_dataset_rows(pilot_train_dataset, pilot_train_temporal,
                                            exact_train, FRAME_ROOT, 30, True)
    val_ucf, val_dataset, val_temporal = project_dataset(runtime_modules(repo)[-1], VAL_LIST, FRAME_ROOT)
    val_rows = validate_dataset_rows(val_dataset, val_temporal, VAL_LIST, FRAME_ROOT, 3783, True)
    pair_rows = []
    task042, e0, primitives, phase_d, core, probe, ctfrs = runtime_modules(repo)
    _, pair_rows = e0.build_pair_subset(core)
    require([(int(r['span']), int(r['pair_index'])) for r in pair_rows] == [(s, 0) for s in SPANS],
            'fixed five-span Task042 pair identities changed')
    write_csv_new(out / REQUIRED_OUTPUTS[0], candidate_manifest_rows(selected),
                  ('candidate_order', 'task037_global_index', 'unit_type', 'domain_id',
                   'domain_group', 'layer', 'unit_index', 'f3_global_step',
                   'f3_domain_removal_rank', 'source_path', 'pilot_gate_identity_set',
                   'final_gate_value', 'cumulative_removed_cohort_ratio_at_prefix'))
    preflight = {
        'task': 'Task042 Phase E.1 controlled progressive-pruning pilot',
        'branch': BRANCH, 'current_code_head_before_run': subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'phase_d_head': PHASE_D_HEAD, 'phase_d_decision': dsummary['decision'],
        'checkpoint_sha256': sha256_file(CHECKPOINT),
        'candidate_ids_in_frozen_f3_order': [int(r['task037_global_index']) for r in selected],
        'candidate_manifest_sha256': sha256_file(out / REQUIRED_OUTPUTS[0]),
        'phase_d_candidate_provenance_sha256': sha256_file(PHASE_D / 'task042_phase_d_candidate_provenance.csv'),
        'phase_d_baseline_path_sha256': sha256_file(PHASE_D / 'task042_phase_d_baseline_path.csv'),
        'unit_manifest_sha256': sha256_file(UNIT_MANIFEST), 'video_manifest_sha256': sha256_file(VIDEO_MANIFEST),
        'validation_list': str(VAL_LIST), 'validation_list_sha256': sha256_file(VAL_LIST),
        'validation_clip_count': len(val_rows), 'validation_identity_exact': True,
        'training_list': str(TRAIN_LIST), 'training_list_sha256': sha256_file(TRAIN_LIST),
        'training_video_count': len(pilot_train_rows),
        'training_videos': [{'training_index': i, 'video_id': r['video_id'], 'label': r['label'],
                             'official_train_index': selected_train[i]['index']} for i, r in enumerate(pilot_train_rows)],
        'calibration_training_indices': [i * TRAIN_VIDEOS_PER_CLASS for i in range(10)],
        'training_classes': classes, 'pair_interventions': pair_rows,
        'gate_stages': list(GATE_STAGES), 'recovery_steps_per_stage': RECOVERY_STEPS_PER_STAGE,
        'total_updates_per_arm': 4 * RECOVERY_STEPS_PER_STAGE,
        'pilot_parameter_ratio_from_phase_d': float(ordered_path[PILOT_COUNT - 1]['cumulative_removed_cohort_ratio']),
        'phase_d_eligible_unit_count': len(eligible), 'pilot_unit_count': len(selected),
        'domain_types': {str(k): sorted(v) for k, v in _domain_type_map(eligible).items()},
        'validation_reference_video_indices': [0, 3], 'preflight_training_dataset_rows': len(train_rows),
        'preflight_training_video_frames_validated': len(pilot_train_rows),
        'output_dir': str(out), 'gpu_policy': 'physical GPUs 0 and 1 only',
    }
    write_json(work / 'preflight.json', preflight, exclusive=True)
    print('PHASE_E1_PREPARE=PASS')
    print('CANDIDATES=' + ','.join(map(str, preflight['candidate_ids_in_frozen_f3_order'])))
    print('TRAINING_VIDEOS=%d VALIDATION_VIDEOS=%d' % (len(pilot_train_rows), len(val_rows)))


def _domain_type_map(units: Sequence[Mapping[str, Any]]) -> Dict[str, set]:
    result: Dict[str, set] = {}
    for row in units:
        result.setdefault(str(row['domain_id']), set()).add(str(row['unit_type']))
    return result


def torch_runtime(physical_gpu: int):
    require(int(physical_gpu) in (0, 1), 'only physical GPU 0 or 1 is permitted')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(physical_gpu),
            'CUDA_VISIBLE_DEVICES must isolate the requested physical GPU')
    import torch
    import torch.nn.functional as F
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'worker must see exactly one isolated CUDA device')
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch, F, torch.device('cuda:0')


def load_frozen_model(repo: Path, device: Any):
    task042, e0, primitives, phase_d, core, probe, ctfrs = runtime_modules(repo)
    model, adapter, identity = task042._model_and_identity(PROJECT_ROOT, CHECKPOINT,
                                                           device, probe, ctfrs)
    require(identity.get('checkpoint_sha256') == CHECKPOINT_SHA and
            not identity.get('missing_keys') and not identity.get('unexpected_keys') and
            not identity.get('shape_mismatches') and
            identity.get('classifier_head', {}).get('status') == 'loaded',
            'authoritative Video-Swin checkpoint/classifier identity mismatch')
    model.eval()
    return model, identity, (task042, e0, primitives, phase_d, core, probe, ctfrs)


def build_ucf_loader(ctfrs: Any, torch: Any, list_path: Path, frame_root: Path,
                     batch_size: int, indices: Optional[Sequence[int]] = None,
                     workers: int = 4):
    _, dataset, _ = project_dataset(ctfrs, list_path, frame_root)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, list(map(int, indices)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(batch_size),
                                         shuffle=False, num_workers=int(workers),
                                         pin_memory=True, drop_last=False,
                                         persistent_workers=workers > 0)
    return loader, dataset


def load_training_clips(ctfrs: Any, torch: Any, exact_train_list: Path,
                        device: Any = None) -> Tuple[List[Any], List[int], List[str]]:
    _, dataset, _ = project_dataset(ctfrs, exact_train_list, FRAME_ROOT)
    require(len(dataset) == 30, 'frozen training subset must contain 30 clips')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=2, pin_memory=True, drop_last=False)
    clips: List[Any] = [None] * 30
    labels: List[Optional[int]] = [None] * 30
    names: List[Optional[str]] = [None] * 30
    for videos, targets, indices in loader:
        idx = int(indices[0].item())
        require(0 <= idx < 30 and clips[idx] is None, 'training loader identity duplicated')
        clips[idx] = videos[0].contiguous()
        labels[idx] = int(targets[0].item())
        names[idx] = Path(dataset.clips[idx][0]).name
    require(all(x is not None for x in clips + labels + names), 'training loader did not return all 30 frozen clips')
    if device is not None:
        clips = [c.to(device=device, dtype=torch.float32, non_blocking=True) for c in clips]
    return clips, [int(x) for x in labels], [str(x) for x in names]


def set_all_gates(gate_state: Any, candidate_ids: Sequence[int], value: float) -> None:
    val = float(value)
    require(0.0 <= val <= 1.0, 'gate value outside [0,1]')
    require(set(gate_state.values) == set(map(int, candidate_ids)), 'gate identity set changed')
    for uid in candidate_ids:
        gate_state.values[int(uid)] = val


def pair_identities(core: Any, e0: Any) -> Tuple[List[Any], List[Dict[str, int]]]:
    ops, rows = e0.build_pair_subset(core)
    require([(int(r['span']), int(r['pair_index'])) for r in rows] == [(x, 0) for x in SPANS],
            'five fixed E.0 span/pair identities changed')
    return ops, rows


def reference_worker(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e1', base / 'phase_e1' / 'work'
    gpu, shard = int(args.physical_gpu), int(args.shard)
    require(gpu in (0, 1) and shard == gpu, 'reference shard must match isolated physical GPU')
    torch, F, device = torch_runtime(gpu)
    preflight = json.loads((work / 'preflight.json').read_text(encoding='utf-8'))
    model, identity, modules = load_frozen_model(repo, device)
    task042, e0, primitives, phase_d, core, probe, ctfrs = modules
    val_list = VAL_LIST
    loader, subset = build_ucf_loader(ctfrs, torch, val_list, FRAME_ROOT,
                                      VALIDATION_BATCH_SIZE,
                                      list(range(shard, 3783, 2)), workers=4)
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    rows = []
    model.eval()
    with torch.inference_mode():
        for videos, labels, indices in loader:
            videos = videos.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            logits = phase_d.unwrap_logits(model(videos))
            require(logits.ndim == 2 and logits.shape[0] == labels.shape[0] and logits.shape[1] >= 5,
                    'unexpected full-validation logits')
            losses = F.cross_entropy(logits, labels, reduction='none')
            top5 = logits.topk(5, dim=1).indices
            pred = logits.argmax(dim=1)
            for j, raw_idx in enumerate(indices.tolist()):
                idx = int(raw_idx)
                require(int(labels[j].item()) == int(dataset.clips[idx][2]),
                        'validation label changed at index %d' % idx)
                rows.append({'validation_index': idx, 'video_id': Path(dataset.clips[idx][0]).name,
                             'label': int(labels[j].item()), 'prediction': int(pred[j].item()),
                             'cross_entropy': float(losses[j].item()),
                             'top5_hit': int(int(labels[j].item()) in set(map(int, top5[j].tolist()))),
                             'physical_gpu': gpu})
    expected = list(range(shard, 3783, 2))
    require(sorted(r['validation_index'] for r in rows) == expected,
            'reference shard did not cover its exact validation identities')
    write_csv_new(work / ('reference_gpu%d.csv' % gpu), rows,
                  ('validation_index', 'video_id', 'label', 'prediction',
                   'cross_entropy', 'top5_hit', 'physical_gpu'))

    # Cache frozen teacher relation signatures for the exact recovery clips split
    # between devices. All cached values are Python floats detached from autograd.
    units, selected, _, _ = frozen_inputs(repo)
    capture_state = primitives.GateState([])
    capture = e0.PhaseE0Capture(model, units, capture_state, torch, detach=True)
    pair_ops, pair_rows = pair_identities(core, e0)
    train_clip_tensors, train_labels, train_names = load_training_clips(
        ctfrs, torch, work / 'task042_phase_e1_exact_training_list.txt', device)
    train_targets = []
    model.requires_grad_(False)
    for idx in range(30):
        if idx % 2 != shard:
            continue
        sens, _, _, _ = e0.collect_no_grad(model, capture, capture_state,
                                            [train_clip_tensors[idx]], [train_labels[idx]],
                                            pair_ops, units, core, phase_d, torch, F)
        require(not any(p.grad is not None for p in model.parameters()), 'teacher accumulated gradients')
        train_targets.append({'training_index': idx, 'video_id': train_names[idx],
                              'label': train_labels[idx], 'sensitivities': sens[0]})
    val_targets = []
    if gpu == 0:
        config = json.loads(RUN_CONFIG.read_text(encoding='utf-8'))
        video_manifest = read_csv(VIDEO_MANIFEST)
        val_clips, val_labels, val_rows = e0.load_clips(ctfrs, config, video_manifest,
                                                         torch, device)
        require([int(r['video_index']) for r in val_rows] == [0, 3],
                'E.0 relation-drift validation identities changed')
        for idx, (clip, label, row) in enumerate(zip(val_clips, val_labels, val_rows)):
            sens, _, _, _ = e0.collect_no_grad(model, capture, capture_state,
                                                [clip], [label], pair_ops, units,
                                                core, phase_d, torch, F)
            val_targets.append({'video_index': int(row['video_index']),
                                'video_id': Path(str(row['video_id'])).name,
                                'label': int(label), 'sensitivities': sens[0]})
    capture.close()
    write_json(work / ('teacher_targets_gpu%d.json' % gpu),
               {'physical_gpu': gpu, 'checkpoint_sha256': CHECKPOINT_SHA,
                'teacher_requires_grad': False, 'pair_interventions': pair_rows,
                'training_sensitivities': train_targets,
                'validation_sensitivities': val_targets}, exclusive=True)
    print('REFERENCE_GPU%d_DONE clips=%d teacher_train=%d' % (gpu, len(rows), len(train_targets)))


def finalize_reference(args: argparse.Namespace) -> None:
    base, out, work = Path(args.base).resolve(), Path(args.base).resolve() / 'phase_e1', Path(args.base).resolve() / 'phase_e1' / 'work'
    shards = read_csv(work / 'reference_gpu0.csv') + read_csv(work / 'reference_gpu1.csv')
    shards.sort(key=lambda r: int(r['validation_index']))
    require(len(shards) == 3783 and [int(r['validation_index']) for r in shards] == list(range(3783)),
            'baseline reference does not contain exact 3783/3783 identities')
    expected_lines = VAL_LIST.read_text(encoding='utf-8').splitlines()
    for i, row in enumerate(shards):
        fields = expected_lines[i].split()
        require(Path(fields[0]).name == row['video_id'] and int(fields[-1]) == int(row['label']),
                'baseline reference video/label differs from authoritative validation list')
    write_csv_new(out / 'task042_phase_e1_reference_predictions.csv', shards,
                  ('validation_index', 'video_id', 'label', 'prediction',
                   'cross_entropy', 'top5_hit', 'physical_gpu'))
    n = len(shards)
    ref = {'validation_clips': n,
           'cross_entropy': statistics.mean(float(r['cross_entropy']) for r in shards),
           'top1_percent': 100.0 * sum(int(r['prediction']) == int(r['label']) for r in shards) / n,
           'top5_percent': 100.0 * sum(int(r['top5_hit']) for r in shards) / n,
           'prediction_count': n, 'validation_list_sha256': sha256_file(VAL_LIST),
           'prediction_file_sha256': sha256_file(out / 'task042_phase_e1_reference_predictions.csv')}
    write_json(work / 'reference_metrics.json', ref, exclusive=True)
    t0 = json.loads((work / 'teacher_targets_gpu0.json').read_text(encoding='utf-8'))
    t1 = json.loads((work / 'teacher_targets_gpu1.json').read_text(encoding='utf-8'))
    require(t0['teacher_requires_grad'] is False and t1['teacher_requires_grad'] is False,
            'teacher relation targets were not frozen')
    training = t0['training_sensitivities'] + t1['training_sensitivities']
    training.sort(key=lambda r: int(r['training_index']))
    require([int(r['training_index']) for r in training] == list(range(30)),
            'teacher relation cache does not cover all 30 fixed training videos')
    validation = t0['validation_sensitivities']
    require([int(r['video_index']) for r in validation] == [0, 3],
            'teacher relation cache misses exact E.0 diagnostic clips')
    require(t0['pair_interventions'] == t1['pair_interventions'], 'GPU teacher workers used different frame pairs')
    cache = {'checkpoint_sha256': CHECKPOINT_SHA, 'teacher_detached': True,
             'pair_interventions': t0['pair_interventions'],
             'training_sensitivities': training,
             'validation_sensitivities': validation}
    write_json(work / 'teacher_relation_targets.json', cache, exclusive=True)
    print('REFERENCE_FINALIZED=PASS videos=3783 prediction_hash=' + ref['prediction_file_sha256'])


def calibration_worker(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e1', base / 'phase_e1' / 'work'
    gpu = int(args.physical_gpu)
    torch, F, device = torch_runtime(gpu)
    preflight = json.loads((work / 'preflight.json').read_text(encoding='utf-8'))
    cache = json.loads((work / 'teacher_relation_targets.json').read_text(encoding='utf-8'))
    units, selected, _, _ = frozen_inputs(repo)
    ids = [int(r['task037_global_index']) for r in selected]
    target_by_index = {int(r['training_index']): r['sensitivities'] for r in cache['training_sensitivities']}
    calib_indices = list(map(int, preflight['calibration_training_indices']))
    my_indices = [idx for idx in calib_indices if idx % 2 == gpu]
    task042, e0, primitives, phase_d, core, probe, ctfrs = runtime_modules(repo)
    model, identity, _modules = load_frozen_model(repo, device)
    model.requires_grad_(True)
    model.eval()
    gates = primitives.GateState(ids)
    set_all_gates(gates, ids, 0.0)
    capture = e0.PhaseE0Capture(model, units, gates, torch, detach=False)
    pair_ops, pair_rows = pair_identities(core, e0)
    train_path = work / 'task042_phase_e1_exact_training_list.txt'
    _, dataset, _ = project_dataset(ctfrs, train_path, FRAME_ROOT)
    before_sha = e0.model_parameter_sha(model)
    rows = []
    for idx in my_indices:
        clip_cpu, label, _ = dataset[idx]
        clip = clip_cpu.to(device=device, dtype=torch.float32, non_blocking=True)
        require(label == int(cache['training_sensitivities'][idx]['label']),
                'calibration label differs from frozen teacher cache')
        model.zero_grad(set_to_none=True)
        ce_value = e0.ce_backward(model, capture, [clip], [label], phase_d, torch, F)
        ce_grad = e0.finite_gradient_metrics(model, torch)
        model.zero_grad(set_to_none=True)
        sens, _, _, _ = e0.collect_no_grad(model, capture, gates, [clip], [label],
                                            pair_ops, units, core, phase_d, torch, F)
        alive = e0.alive_mask(units, gates)
        teacher_sens = [target_by_index[idx]]
        ltr_value, _ = e0.ltr_backward(model, capture, [clip], pair_ops, sens,
                                       teacher_sens, units, alive, core, phase_d,
                                       torch, primitives.relation_loss)
        tr_grad = e0.finite_gradient_metrics(model, torch)
        require(ce_grad['finite'] and tr_grad['finite'] and math.isfinite(ce_value) and math.isfinite(ltr_value),
                'nonfinite CE/TR gradient-scale calibration')
        rows.append({'batch_index': idx, 'training_index': idx,
                     'video_id': preflight['training_videos'][idx]['video_id'],
                     'label': label, 'gate_value': 0.0, 'cross_entropy': ce_value,
                     'L_TR': ltr_value, 'g_CE': ce_grad['total_grad_norm'],
                     'g_TR': tr_grad['total_grad_norm'],
                     'candidate_ids_excluded_from_gradients': json.dumps(ids, separators=(',', ':')),
                     'surviving_parameter_gradient_scope': 'all named student parameters at final candidate gates; zero-gated unit slices have zero activation-path gradients'})
        model.zero_grad(set_to_none=True)
    require(e0.model_parameter_sha(model) == before_sha, 'gradient calibration changed student weights')
    write_csv_new(work / ('calibration_gpu%d.csv' % gpu), rows,
                  ('batch_index', 'training_index', 'video_id', 'label', 'gate_value',
                   'cross_entropy', 'L_TR', 'g_CE', 'g_TR',
                   'candidate_ids_excluded_from_gradients',
                   'surviving_parameter_gradient_scope'))
    print('CALIBRATION_GPU%d_DONE batches=%d' % (gpu, len(rows)))


def write_training_manifest(out: Path, manifest: Mapping[str, Any]) -> None:
    rows = [{'field': k, 'value_json': json.dumps(v, ensure_ascii=False, sort_keys=True)}
            for k, v in manifest.items()]
    write_csv_new(out / 'task042_phase_e1_training_manifest.csv', rows, ('field', 'value_json'))


def finalize_calibration(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e1', base / 'phase_e1' / 'work'
    preflight = json.loads((work / 'preflight.json').read_text(encoding='utf-8'))
    rows = read_csv(work / 'calibration_gpu0.csv') + read_csv(work / 'calibration_gpu1.csv')
    rows.sort(key=lambda r: int(r['training_index']))
    expected = list(map(int, preflight['calibration_training_indices']))
    require([int(r['training_index']) for r in rows] == expected,
            'gradient calibration did not use the exact frozen multi-batch subset')
    gce, gtr, lam = median_gradient_calibration([float(r['g_CE']) for r in rows],
                                                 [float(r['g_TR']) for r in rows])
    output_rows = [dict(r, record_type='batch', lambda_batch=float(r['g_CE']) / (float(r['g_TR']) + 1e-12)) for r in rows]
    output_rows.append({'record_type': 'frozen_median', 'batch_index': '',
                        'training_index': '', 'video_id': 'MULTI_BATCH_MEDIAN', 'label': '',
                        'gate_value': 0.0, 'cross_entropy': '', 'L_TR': '', 'g_CE': gce,
                        'g_TR': gtr, 'lambda_batch': lam,
                        'candidate_ids_excluded_from_gradients': rows[0]['candidate_ids_excluded_from_gradients'],
                        'surviving_parameter_gradient_scope': rows[0]['surviving_parameter_gradient_scope']})
    fields = ('record_type', 'batch_index', 'training_index', 'video_id', 'label',
              'gate_value', 'cross_entropy', 'L_TR', 'g_CE', 'g_TR', 'lambda_batch',
              'candidate_ids_excluded_from_gradients', 'surviving_parameter_gradient_scope')
    write_csv_new(out / 'task042_phase_e1_gradient_scale_calibration.csv', output_rows, fields)
    units, selected, path_rows, dsummary = frozen_inputs(repo)
    cache = json.loads((work / 'teacher_relation_targets.json').read_text(encoding='utf-8'))
    pair_ops, pair_rows = runtime_modules(repo)[1].build_pair_subset(runtime_modules(repo)[4])
    require(pair_rows == cache['pair_interventions'], 'calibration/training pair identities differ')
    ids = [int(r['task037_global_index']) for r in selected]
    all_arm_ids = {name: ids[:] for name in ('A_ONE_SHOT', 'B_PROGRESSIVE_CE', 'C_PROGRESSIVE_CE_TR')}
    validate_identical_candidate_sets(ids, all_arm_ids)
    manifest = {
        'task': 'Task042 Phase E.1 controlled progressive-pruning pilot',
        'branch': BRANCH, 'code_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'phase_d_head': PHASE_D_HEAD, 'phase_d_decision': dsummary['decision'],
        'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': CHECKPOINT_SHA,
        'candidate_source': 'first 12 candidates of frozen Phase-D baseline F3 global path',
        'candidate_ids_in_frozen_order': ids,
        'candidate_manifest_sha256': sha256_file(out / 'task042_phase_e1_candidate_manifest.csv'),
        'candidate_sets_by_arm': all_arm_ids, 'same_final_gate_identity': True,
        'gate_stages': list(GATE_STAGES), 'recovery_steps_per_noninitial_stage': RECOVERY_STEPS_PER_STAGE,
        'total_optimizer_updates_per_arm': 4 * RECOVERY_STEPS_PER_STAGE,
        'optimizer': {'name': 'SGD', 'learning_rate': LEARNING_RATE, 'momentum': MOMENTUM,
                      'weight_decay': WEIGHT_DECAY, 'batch_size': 1,
                      'gradient_clip_norm': GRAD_CLIP_NORM, 'amp': False,
                      'model_mode': 'eval mode with gradients enabled; deterministic dropout/DropPath disabled'},
        'seed': SEED, 'same_training_video_order_all_arms': True,
        'training_videos': preflight['training_videos'],
        'training_schedule_indices': [i for _stage in range(4) for i in range(30)],
        'calibration_indices': expected, 'calibration_video_ids': [r['video_id'] for r in rows],
        'calibration_gate_value': 0.0,
        'G_CE': gce, 'G_TR': gtr, 'lambda_TR': lam,
        'lambda_rule': 'median_batch(g_CE) / (median_batch(g_TR) + 1e-12); immutable Python float for all of E.1; no validation tuning',
        'frame_pair_interventions': pair_rows,
        'teacher_relation_cache_sha256': sha256_file(work / 'teacher_relation_targets.json'),
        'teacher_parameters_frozen_and_targets_detached': True,
        'validation': {'list': str(VAL_LIST), 'count': 3783,
                       'list_sha256': sha256_file(VAL_LIST),
                       'reference_prediction_sha256': sha256_file(out / 'task042_phase_e1_reference_predictions.csv'),
                       'batch_size': VALIDATION_BATCH_SIZE, 'drop_last': False,
                       'prediction_flip_reference': 'original unpruned checkpoint'},
        'relation_drift_subset': [{'video_index': int(r['video_index']), 'video_id': r['video_id'], 'label': r['label']}
                                  for r in cache['validation_sensitivities']],
        'relation_scope': 'same-domain pairs only over the 31 Phase-D eligible Task042 units; no cross-domain pairs',
        'physical_gpus': [0, 1], 'started_training': False,
        'scope': 'small deterministic pilot only; no physical pruning, no final 50% experiment, no 100-epoch fine-tune, no lambda search, no Task043',
    }
    write_training_manifest(out, manifest)
    print('CALIBRATION_FINALIZED=PASS G_CE=%.9g G_TR=%.9g lambda_TR=%.9g' % (gce, gtr, lam))
    print('TRAINING_MANIFEST_WRITTEN_BEFORE_TRAINING=True')


def training_manifest_values(path: Path) -> Dict[str, Any]:
    return {r['field']: json.loads(r['value_json']) for r in read_csv(path)}


def save_latest_checkpoint(torch: Any, path: Path, model: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(model.state_dict(), str(tmp))
    os.replace(str(tmp), str(path))


def full_validation(model: Any, capture: Any, loader: Any, dataset: Any,
                    phase_d: Any, torch: Any, F: Any, device: Any,
                    reference_predictions: Mapping[int, int],
                    gates: Any, candidate_ids: Sequence[int]) -> Dict[str, Any]:
    before = dict(gates.values)
    model.eval()
    n = top1 = top5 = flips = 0
    ce_sum = 0.0
    seen = set()
    with torch.inference_mode():
        for videos, labels, indices in loader:
            videos = videos.to(device=device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            if capture is not None:
                capture.clear()
                capture.qkv_v.clear()
            logits = phase_d.unwrap_logits(model(videos))
            require(logits.ndim == 2 and logits.shape[0] == labels.shape[0] and logits.shape[1] >= 5,
                    'validation logits have unexpected shape')
            losses = F.cross_entropy(logits, labels, reduction='none')
            preds = logits.argmax(dim=1)
            top5_preds = logits.topk(5, dim=1).indices
            for j, raw_idx in enumerate(indices.tolist()):
                idx = int(raw_idx)
                require(idx not in seen and int(labels[j].item()) == int(dataset.clips[idx][2]),
                        'duplicate validation identity or label mismatch')
                require(idx in reference_predictions, 'baseline prediction missing for validation index')
                seen.add(idx)
                n += 1
                ce_sum += float(losses[j].item())
                top1 += int(int(preds[j].item()) == int(labels[j].item()))
                top5 += int(int(labels[j].item()) in set(map(int, top5_preds[j].tolist())))
                flips += int(int(preds[j].item()) != int(reference_predictions[idx]))
            if capture is not None:
                capture.clear()
                capture.qkv_v.clear()
    require(seen == set(range(3783)) and n == 3783,
            'full validation identity is not exactly 3783/3783')
    require(gates.values == before, 'read-only validation changed gate state')
    metrics = {'validation_clips': n, 'cross_entropy': ce_sum / n,
               'top1_percent': 100.0 * top1 / n, 'top5_percent': 100.0 * top5 / n,
               'prediction_flip_rate': flips / n, 'validation_identity_exact': True}
    require(all(math.isfinite(float(metrics[k])) for k in
                ('cross_entropy', 'top1_percent', 'top5_percent', 'prediction_flip_rate')),
            'nonfinite full-validation metrics')
    return metrics


def relation_drift_at_checkpoint(model: Any, capture: Any, gates: Any,
                                 clips: Sequence[Any], labels: Sequence[int],
                                 teacher_sens: Sequence[Any], unit_rows: Sequence[Mapping[str, Any]],
                                 pair_ops: Sequence[Any], core: Any, phase_d: Any,
                                 primitives: Any, e0: Any, torch: Any, F: Any,
                                 arm: str, stage: int, gate_value: float,
                                 optimizer_steps: int) -> List[Dict[str, Any]]:
    before = dict(gates.values)
    sens, _, _, _ = e0.collect_no_grad(model, capture, gates, clips, labels,
                                        pair_ops, unit_rows, core, phase_d, torch, F)
    alive = e0.alive_mask(unit_rows, gates)
    _, details = e0.no_grad_relation(sens, teacher_sens, unit_rows, alive,
                                     torch, primitives.relation_loss)
    require(bool(details), 'relation drift has no surviving same-domain pairs')
    values: Dict[str, List[float]] = {}
    for item in details:
        values.setdefault(str(item['domain_id']), []).append(float(item['absolute_drift']))
    values['ALL_DOMAINS'] = [float(r['absolute_drift']) for r in details]
    rows = []
    for domain_id, domain_values in values.items():
        require(domain_values and all(math.isfinite(x) for x in domain_values),
                'relation drift contains no data or nonfinite values')
        rows.append({'arm': arm, 'stage_index': stage, 'gate_value': gate_value,
                     'optimizer_steps': optimizer_steps, 'domain_id': domain_id,
                     'mean_relation_drift': statistics.mean(domain_values),
                     'median_relation_drift': statistics.median(domain_values),
                     'max_relation_drift': max(domain_values), 'pair_video_observation_count': len(domain_values),
                     'teacher_detached': True})
    require(gates.values == before, 'relation drift changed gate state')
    return rows


def arm_worker(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e1', base / 'phase_e1' / 'work'
    arm, gpu = str(args.arm), int(args.physical_gpu)
    arm_names = {'A': 'A_ONE_SHOT', 'B': 'B_PROGRESSIVE_CE', 'C': 'C_PROGRESSIVE_CE_TR'}
    require(arm in arm_names and gpu in (0, 1), 'unknown arm or unauthorized GPU')
    require((out / 'task042_phase_e1_training_manifest.csv').is_file(),
            'frozen candidate/lambda training manifest must exist before training')
    manifest = training_manifest_values(out / 'task042_phase_e1_training_manifest.csv')
    scale = FrozenLossScale(float(manifest['lambda_TR']))
    candidate_ids = list(map(int, manifest['candidate_ids_in_frozen_order']))
    validate_gate_schedule(manifest['gate_stages'])
    validate_identical_candidate_sets(candidate_ids, manifest['candidate_sets_by_arm'])
    torch, F, device = torch_runtime(gpu)
    units, selected, _, _ = frozen_inputs(repo)
    require([int(r['task037_global_index']) for r in selected] == candidate_ids,
            'frozen candidate identities changed after calibration')
    task042, e0, primitives, phase_d, core, probe, ctfrs = runtime_modules(repo)
    model, identity, _modules = load_frozen_model(repo, device)
    model.requires_grad_(True)
    model.eval()
    state = primitives.GateState(candidate_ids)
    capture = e0.PhaseE0Capture(model, units, state, torch, detach=False)
    pair_ops, pair_rows = pair_identities(core, e0)
    cache = json.loads((work / 'teacher_relation_targets.json').read_text(encoding='utf-8'))
    require(pair_rows == cache['pair_interventions'], 'arm frame-pair identities differ from frozen cache')
    teacher_train = {int(r['training_index']): r['sensitivities'] for r in cache['training_sensitivities']}
    teacher_val_rows = cache['validation_sensitivities']
    teacher_val = [r['sensitivities'] for r in teacher_val_rows]
    val_clips, val_labels, val_rows = e0.load_clips(ctfrs,
        json.loads(RUN_CONFIG.read_text(encoding='utf-8')), read_csv(VIDEO_MANIFEST), torch, device)
    require([int(r['video_index']) for r in val_rows] == [0, 3], 'drift video identities differ from E.0')
    require([Path(str(r['video_id'])).name for r in val_rows] == [r['video_id'] for r in teacher_val_rows],
            'teacher/student drift video identities differ')
    train_clips, train_labels, train_names = load_training_clips(
        ctfrs, torch, work / 'task042_phase_e1_exact_training_list.txt', device)
    require(train_names == [r['video_id'] for r in manifest['training_videos']] and
            train_labels == [int(r['label']) for r in manifest['training_videos']],
            'arm training video identity/order differs from frozen manifest')
    ref_rows = read_csv(out / 'task042_phase_e1_reference_predictions.csv')
    ref_preds = {int(r['validation_index']): int(r['prediction']) for r in ref_rows}
    require(len(ref_preds) == 3783, 'baseline reference prediction count differs')
    val_loader, val_dataset = build_ucf_loader(ctfrs, torch, VAL_LIST, FRAME_ROOT,
                                               VALIDATION_BATCH_SIZE, workers=4)
    while hasattr(val_dataset, 'dataset'):
        val_dataset = val_dataset.dataset

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    train_schedule = list(map(int, manifest['training_schedule_indices']))
    require(len(train_schedule) == 4 * RECOVERY_STEPS_PER_STAGE and
            all(0 <= i < len(train_clips) for i in train_schedule),
            'frozen optimizer-step data order is invalid')
    trajectory: List[Dict[str, Any]] = []
    drift_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    arm_name = arm_names[arm]
    updates_done = 0

    def evaluate(stage_index: int, gate: float, steps: int, phase: str) -> None:
        nonlocal trajectory, drift_rows
        metrics = full_validation(model, capture, val_loader, val_dataset, phase_d,
                                  torch, F, device, ref_preds, state, candidate_ids)
        trajectory.append({'arm': arm_name, 'stage_index': stage_index,
                           'gate_value': gate, 'optimizer_steps': steps,
                           'checkpoint_phase': phase, **metrics})
        drift_rows.extend(relation_drift_at_checkpoint(
            model, capture, state, val_clips, val_labels, teacher_val, units,
            pair_ops, core, phase_d, primitives, e0, torch, F,
            arm_name, stage_index, gate, steps))
        write_csv_replace(work / ('arm_%s_validation.csv' % arm), trajectory,
                          ('arm', 'stage_index', 'gate_value', 'optimizer_steps',
                           'checkpoint_phase', 'validation_clips', 'cross_entropy',
                           'top1_percent', 'top5_percent', 'prediction_flip_rate',
                           'validation_identity_exact'))
        write_csv_replace(work / ('arm_%s_relation_drift.csv' % arm), drift_rows,
                          ('arm', 'stage_index', 'gate_value', 'optimizer_steps',
                           'domain_id', 'mean_relation_drift', 'median_relation_drift',
                           'max_relation_drift', 'pair_video_observation_count', 'teacher_detached'))
        print('ARM_%s_CHECKPOINT step=%d gate=%.2f CE=%.8f top1=%.4f top5=%.4f flip=%.6f drift=%.8f' %
              (arm, steps, gate, metrics['cross_entropy'], metrics['top1_percent'],
               metrics['top5_percent'], metrics['prediction_flip_rate'],
               next(r['mean_relation_drift'] for r in drift_rows
                    if r['domain_id'] == 'ALL_DOMAINS' and r['optimizer_steps'] == steps)))

    if arm == 'A':
        set_all_gates(state, candidate_ids, 0.0)
        evaluate(0, 0.0, 0, 'immediate_one_shot_prune_before_recovery')

    for stage_index, gate in enumerate(GATE_STAGES[1:]):
        if arm == 'A':
            set_all_gates(state, candidate_ids, 0.0)
            effective_gate = 0.0
        else:
            set_all_gates(state, candidate_ids, float(gate))
            effective_gate = float(gate)
        for _ in range(RECOVERY_STEPS_PER_STAGE):
            idx = train_schedule[updates_done]
            clip, label = train_clips[idx], train_labels[idx]
            optimizer.zero_grad(set_to_none=True)
            logits, _ = e0.model_forward(model, capture, clip, phase_d)
            target = torch.tensor([label], dtype=torch.long, device=device)
            ce = F.cross_entropy(logits, target)
            require(torch.isfinite(ce).item(), 'nonfinite training CE')
            ce.backward()
            ce_value = float(ce.detach().item())
            ltr_value, lambda_used = 0.0, 0.0
            if arm == 'C':
                ce_grads = [None if p.grad is None else p.grad.detach().clone()
                            for p in model.parameters()]
                optimizer.zero_grad(set_to_none=True)
                sens, _, _, _ = e0.collect_no_grad(model, capture, state, [clip], [label],
                                                    pair_ops, units, core, phase_d, torch, F)
                alive = e0.alive_mask(units, state)
                ltr_value, _ = e0.ltr_backward(model, capture, [clip], pair_ops, sens,
                    [teacher_train[idx]], units, alive, core, phase_d, torch,
                    primitives.relation_loss)
                require(math.isfinite(float(ltr_value)), 'nonfinite training L_TR')
                for parameter, grad_ce in zip(model.parameters(), ce_grads):
                    grad_tr = parameter.grad
                    if grad_ce is None and grad_tr is None:
                        parameter.grad = None
                    else:
                        left = torch.zeros_like(parameter) if grad_ce is None else grad_ce
                        right = torch.zeros_like(parameter) if grad_tr is None else grad_tr
                        parameter.grad = left + scale.value * right
                lambda_used = scale.value
                require(lambda_used == float(manifest['lambda_TR']), 'lambda_TR changed after calibration')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            require(torch.isfinite(grad_norm).item(), 'nonfinite combined gradient norm')
            optimizer.step()
            updates_done += 1
            train_rows.append({'arm': arm_name, 'stage_index': stage_index + 1,
                               'gate_value': effective_gate, 'optimizer_step': updates_done,
                               'training_index': idx, 'video_id': train_names[idx],
                               'label': label, 'cross_entropy': ce_value,
                               'L_TR': ltr_value, 'lambda_calibrated': scale.value,
                               'lambda_used': lambda_used, 'combined_loss_reported': ce_value + lambda_used * ltr_value,
                               'preclip_grad_norm': float(grad_norm.detach().item()),
                               'gradient_clip_norm': GRAD_CLIP_NORM})
        require(updates_done == (stage_index + 1) * RECOVERY_STEPS_PER_STAGE,
                'optimizer-step allocation changed across gate stages')
        if arm in ('B', 'C'):
            require(effective_gate == float(GATE_STAGES[stage_index + 1]),
                    'progressive gate did not match frozen stage')
        evaluate(stage_index + 1, effective_gate, updates_done, 'after_equal_recovery_steps_at_gate_stage')
        save_latest_checkpoint(torch, work / 'checkpoints' / ('arm_%s_latest.pth' % arm), model)

    require(updates_done == 4 * RECOVERY_STEPS_PER_STAGE,
            'arm did not receive the exact frozen update budget')
    require(all(float(v) == 0.0 for v in state.values.values()) and
            list(state.values) == candidate_ids,
            'final gate identities/values differ across arms')
    # Exact restoration audit: temporarily set gates to one, then restore the
    # final all-zero mask and verify identical logits under that restored state.
    clip = val_clips[0]
    capture.clear()
    with torch.inference_mode():
        logits_before = phase_d.unwrap_logits(model(clip.unsqueeze(0))).detach().clone()
    snapshot = dict(state.values)
    set_all_gates(state, candidate_ids, 1.0)
    with torch.inference_mode():
        _ = phase_d.unwrap_logits(model(clip.unsqueeze(0)))
    state.values.clear()
    state.values.update(snapshot)
    capture.clear()
    with torch.inference_mode():
        logits_after = phase_d.unwrap_logits(model(clip.unsqueeze(0))).detach().clone()
    restoration_exact = bool(torch.equal(logits_before, logits_after))
    require(restoration_exact, 'final structural gate restoration was not bit-exact')
    require(not any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                    for p in model.parameters()), 'arm left nonfinite student gradients')
    write_csv_new(work / ('arm_%s_training_steps.csv' % arm), train_rows,
                  ('arm', 'stage_index', 'gate_value', 'optimizer_step',
                   'training_index', 'video_id', 'label', 'cross_entropy', 'L_TR',
                   'lambda_calibrated', 'lambda_used', 'combined_loss_reported',
                   'preclip_grad_norm', 'gradient_clip_norm'))
    summary = {'arm': arm_name, 'physical_gpu': gpu, 'optimizer_updates': updates_done,
               'candidate_ids': candidate_ids, 'final_zero_gate_ids': [int(k) for k, v in state.values.items() if float(v) == 0.0],
               'final_gate_values': list(state.values.values()),
               'pair_interventions': pair_rows, 'teacher_detached': True,
               'gate_restoration_exact': restoration_exact,
               'validation_checkpoints': len(trajectory),
               'validation_clip_counts': [int(r['validation_clips']) for r in trajectory],
               'lambda_calibrated': scale.value,
               'lambda_values_used': sorted(set(float(r['lambda_used']) for r in train_rows)),
               'all_training_losses_finite': all(math.isfinite(float(r['combined_loss_reported'])) for r in train_rows),
               'model_identity_before_training': identity}
    write_json(work / ('arm_%s_summary.json' % arm), summary, exclusive=True)
    print('ARM_%s_DONE updates=%d restoration=%s' % (arm, updates_done, restoration_exact))


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-30 or vy <= 1e-30:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _metric_improvements(left: Mapping[str, float], right: Mapping[str, float]) -> Dict[str, float]:
    """Positive means right is better than left for each named measure."""
    return {
        'final_CE_improvement': float(left['cross_entropy']) - float(right['cross_entropy']),
        'final_top1_improvement': float(right['top1_percent']) - float(left['top1_percent']),
        'final_top5_improvement': float(right['top5_percent']) - float(left['top5_percent']),
        'final_flip_improvement': float(left['prediction_flip_rate']) - float(right['prediction_flip_rate']),
        'CE_AUC_improvement': float(left['CE_AUC']) - float(right['CE_AUC']),
        'top1_AUC_improvement': float(right['top1_AUC']) - float(left['top1_AUC']),
    }


def finalize(args: argparse.Namespace) -> None:
    repo, base = Path(args.repo).resolve(), Path(args.base).resolve()
    out, work = base / 'phase_e1', base / 'phase_e1' / 'work'
    manifest_path = out / 'task042_phase_e1_training_manifest.csv'
    require(manifest_path.is_file(), 'frozen training manifest absent')
    manifest = training_manifest_values(manifest_path)
    candidate_ids = list(map(int, manifest['candidate_ids_in_frozen_order']))
    validate_gate_schedule(manifest['gate_stages'])
    ref = json.loads((work / 'reference_metrics.json').read_text(encoding='utf-8'))
    require(int(ref['validation_clips']) == 3783 and ref['validation_list_sha256'] == sha256_file(VAL_LIST),
            'baseline validation identity does not match full official split')
    arm_names = {'A': 'A_ONE_SHOT', 'B': 'B_PROGRESSIVE_CE', 'C': 'C_PROGRESSIVE_CE_TR'}
    summaries, arm_traj, arm_drift, arm_train = {}, {}, {}, {}
    for key, name in arm_names.items():
        summaries[key] = json.loads((work / ('arm_%s_summary.json' % key)).read_text(encoding='utf-8'))
        arm_traj[key] = read_csv(work / ('arm_%s_validation.csv' % key))
        arm_drift[key] = read_csv(work / ('arm_%s_relation_drift.csv' % key))
        arm_train[key] = read_csv(work / ('arm_%s_training_steps.csv' % key))
        summary = summaries[key]
        require(summary['arm'] == name and summary['candidate_ids'] == candidate_ids and
                summary['optimizer_updates'] == manifest['total_optimizer_updates_per_arm'],
                'arm candidate identities or optimizer-step budget differs: ' + name)
        require(summary['final_zero_gate_ids'] == candidate_ids and
                all(float(v) == 0.0 for v in summary['final_gate_values']),
                'final structural mask differs across arms: ' + name)
        require(summary['pair_interventions'] == manifest['frame_pair_interventions'] and
                summary['teacher_detached'] is True and summary['gate_restoration_exact'] is True,
                'pair, teacher, or gate-restoration audit failed: ' + name)
        require(all(int(r['validation_clips']) == 3783 and r['validation_identity_exact'] == 'True'
                    for r in arm_traj[key]), 'an arm checkpoint missed full validation')
        expected_checkpoints = 5 if key == 'A' else 4
        require(len(arm_traj[key]) == expected_checkpoints, 'validation checkpoint count differs: ' + name)
        require(all(math.isfinite(float(r['cross_entropy'])) and
                    math.isfinite(float(r['top1_percent'])) and
                    math.isfinite(float(r['top5_percent'])) and
                    math.isfinite(float(r['prediction_flip_rate'])) for r in arm_traj[key]),
                'nonfinite validation metric in ' + name)
        require(len(arm_train[key]) == int(manifest['total_optimizer_updates_per_arm']),
                'training step row count differs: ' + name)
        require(all(float(r['lambda_calibrated']) == float(manifest['lambda_TR']) for r in arm_train[key]),
                'calibrated lambda drifted during ' + name)
        if key == 'C':
            require(set(float(r['lambda_used']) for r in arm_train[key]) == {float(manifest['lambda_TR'])},
                    'C did not use one frozen lambda for every update')
        else:
            require(set(float(r['lambda_used']) for r in arm_train[key]) == {0.0},
                    'CE-only arm unexpectedly used the temporal loss')
        require(all(math.isfinite(float(r['combined_loss_reported'])) for r in arm_train[key]),
                'training loss is nonfinite in ' + name)
    validate_identical_candidate_sets(candidate_ids,
        {summaries[k]['arm']: summaries[k]['candidate_ids'] for k in summaries})
    b_gates = [(int(r['optimizer_steps']), float(r['gate_value'])) for r in arm_traj['B']]
    c_gates = [(int(r['optimizer_steps']), float(r['gate_value'])) for r in arm_traj['C']]
    expected_stages = [(RECOVERY_STEPS_PER_STAGE * i, float(GATE_STAGES[i])) for i in range(1, 5)]
    require(b_gates == expected_stages and c_gates == expected_stages,
            'B/C gate schedule or stage-step allocation changed')
    require(all(float(r['gate_value']) == 0.0 for r in arm_traj['A']),
            'A did not apply its one-shot final mask immediately')

    reference_row = {'arm': 'ORIGINAL_REFERENCE', 'stage_index': 0, 'gate_value': 1.0,
                     'optimizer_steps': 0, 'checkpoint_phase': 'before_pruning',
                     'validation_clips': 3783, 'cross_entropy': ref['cross_entropy'],
                     'top1_percent': ref['top1_percent'], 'top5_percent': ref['top5_percent'],
                     'prediction_flip_rate': 0.0, 'validation_identity_exact': True}
    trajectory = [reference_row]
    for key in ('A', 'B', 'C'):
        trajectory.extend(arm_traj[key])
    trajectory_fields = ('arm', 'stage_index', 'gate_value', 'optimizer_steps',
                         'checkpoint_phase', 'validation_clips', 'cross_entropy',
                         'top1_percent', 'top5_percent', 'prediction_flip_rate',
                         'validation_identity_exact')
    write_csv_new(out / 'task042_phase_e1_validation_trajectory.csv', trajectory, trajectory_fields)

    drift_rows: List[Dict[str, Any]] = []
    for key in ('A', 'B', 'C'):
        drift_rows.extend(arm_drift[key])
    # The original checkpoint is identical to its cached teacher relation target.
    eligible_units, _selected_units, _path_rows, _phase_d_summary = frozen_inputs(repo)
    domain_counts: Dict[str, int] = {}
    for unit in eligible_units:
        domain_counts[str(unit['domain_id'])] = domain_counts.get(str(unit['domain_id']), 0) + 1
    baseline_pair_count = 0
    for domain_id, unit_count in sorted(domain_counts.items(), key=lambda x: int(x[0])):
        count = 2 * (unit_count * (unit_count - 1) // 2)
        baseline_pair_count += count
        drift_rows.append({'arm': 'ORIGINAL_REFERENCE', 'stage_index': 0, 'gate_value': 1.0,
                           'optimizer_steps': 0, 'domain_id': domain_id,
                           'mean_relation_drift': 0.0, 'median_relation_drift': 0.0,
                           'max_relation_drift': 0.0, 'pair_video_observation_count': count,
                           'teacher_detached': True})
    drift_rows.append({'arm': 'ORIGINAL_REFERENCE', 'stage_index': 0, 'gate_value': 1.0,
                       'optimizer_steps': 0, 'domain_id': 'ALL_DOMAINS',
                       'mean_relation_drift': 0.0, 'median_relation_drift': 0.0,
                       'max_relation_drift': 0.0, 'pair_video_observation_count': baseline_pair_count,
                       'teacher_detached': True})
    write_csv_new(out / 'task042_phase_e1_relation_drift.csv', drift_rows,
                  ('arm', 'stage_index', 'gate_value', 'optimizer_steps', 'domain_id',
                   'mean_relation_drift', 'median_relation_drift', 'max_relation_drift',
                   'pair_video_observation_count', 'teacher_detached'))
    all_drifts = [float(r['mean_relation_drift']) for r in drift_rows]
    require(all(math.isfinite(x) for x in all_drifts), 'relation drift contains NaN/Inf')

    by_arm_step = {key: {int(r['optimizer_steps']): r for r in arm_traj[key]} for key in ('A', 'B', 'C')}
    ref_as_metrics = {'cross_entropy': float(ref['cross_entropy']),
                      'top1_percent': float(ref['top1_percent']),
                      'top5_percent': float(ref['top5_percent']),
                      'prediction_flip_rate': 0.0}
    series: Dict[str, Dict[str, List[float]]] = {}
    for key in ('A', 'B', 'C'):
        if key == 'A':
            zero = by_arm_step[key][0]
        else:
            zero = ref_as_metrics
        ordered = [zero] + [by_arm_step[key][i * RECOVERY_STEPS_PER_STAGE] for i in range(1, 5)]
        xs = [i * RECOVERY_STEPS_PER_STAGE for i in range(5)]
        series[key] = {}
        for metric in ('cross_entropy', 'top1_percent', 'top5_percent', 'prediction_flip_rate'):
            ys = [float(r[metric]) for r in ordered]
            value = normalized_trapezoid_auc(xs, ys)
            series[key][metric] = ys
            auc_metric = {'cross_entropy': 'CE_AUC', 'top1_percent': 'top1_AUC',
                          'top5_percent': 'top5_AUC', 'prediction_flip_rate': 'flip_AUC'}[metric]
            series[key][auc_metric] = value
    auc_rows = []
    orientations = {'cross_entropy': 'lower', 'top1_percent': 'higher',
                    'top5_percent': 'higher', 'prediction_flip_rate': 'lower'}
    for key in ('A', 'B', 'C'):
        for metric, direction in orientations.items():
            auc_metric = {'cross_entropy': 'CE_AUC', 'top1_percent': 'top1_AUC',
                          'top5_percent': 'top5_AUC', 'prediction_flip_rate': 'flip_AUC'}[metric]
            auc_rows.append({'arm': arm_names[key], 'metric': metric, 'normalized_trapezoid_auc': series[key][auc_metric],
                             'orientation': direction, 'optimizer_step_coordinates': json.dumps([0,30,60,90,120]),
                             'point_values': json.dumps(series[key][metric]), 'point_count': 5})
    write_csv_new(out / 'task042_phase_e1_recovery_auc.csv', auc_rows,
                  ('arm', 'metric', 'normalized_trapezoid_auc', 'orientation',
                   'optimizer_step_coordinates', 'point_values', 'point_count'))

    compare_rows: List[Dict[str, Any]] = []
    for label, left_key, right_key in (('B_vs_A', 'A', 'B'), ('C_vs_B', 'B', 'C')):
        for step in (30, 60, 90, 120):
            left = by_arm_step[left_key][step]
            right = by_arm_step[right_key][step]
            compare_rows.append({'record_type': 'matched_checkpoint', 'comparison': label,
                'optimizer_steps': step, 'left_arm': arm_names[left_key], 'right_arm': arm_names[right_key],
                'gate_left': float(left['gate_value']), 'gate_right': float(right['gate_value']),
                'CE_improvement_right_minus_left': float(left['cross_entropy']) - float(right['cross_entropy']),
                'top1_improvement_right_minus_left': float(right['top1_percent']) - float(left['top1_percent']),
                'top5_improvement_right_minus_left': float(right['top5_percent']) - float(left['top5_percent']),
                'flip_improvement_right_minus_left': float(left['prediction_flip_rate']) - float(right['prediction_flip_rate']),
                'drift_reduction_left_minus_right': '', 'note': 'positive delta favors right arm'})
    drift_global = {(str(r['arm']), int(r['optimizer_steps'])): float(r['mean_relation_drift'])
                    for r in drift_rows if r['domain_id'] == 'ALL_DOMAINS'}
    association_drift, association_ce = [], []
    for row in compare_rows:
        if row['comparison'] != 'C_vs_B':
            continue
        step = int(row['optimizer_steps'])
        db, dc = drift_global[(arm_names['B'], step)], drift_global[(arm_names['C'], step)]
        row['drift_reduction_left_minus_right'] = db - dc
        association_drift.append(db - dc)
        association_ce.append(float(row['CE_improvement_right_minus_left']))
    pearson = _pearson(association_drift, association_ce)
    compare_rows.append({'record_type': 'relation_performance_association',
                         'comparison': 'C_vs_B', 'optimizer_steps': '',
                         'left_arm': arm_names['B'], 'right_arm': arm_names['C'],
                         'gate_left': '', 'gate_right': '',
                         'CE_improvement_right_minus_left': '',
                         'top1_improvement_right_minus_left': '',
                         'top5_improvement_right_minus_left': '',
                         'flip_improvement_right_minus_left': '',
                         'drift_reduction_left_minus_right': '',
                         'note': json.dumps({'matched_steps': [30,60,90,120],
                           'Drift_B_minus_C': association_drift,
                           'CE_B_minus_C': association_ce,
                           'descriptive_pearson_r': pearson}, allow_nan=False)})
    write_csv_new(out / 'task042_phase_e1_arm_comparison.csv', compare_rows,
                  ('record_type', 'comparison', 'optimizer_steps', 'left_arm', 'right_arm',
                   'gate_left', 'gate_right', 'CE_improvement_right_minus_left',
                   'top1_improvement_right_minus_left', 'top5_improvement_right_minus_left',
                   'flip_improvement_right_minus_left', 'drift_reduction_left_minus_right', 'note'))

    drift_improved = sum(1 for s in (30, 60, 90, 120)
                         if drift_global[(arm_names['C'], s)] < drift_global[(arm_names['B'], s)])
    auc_as = {key: {r['metric']: float(r['normalized_trapezoid_auc'])
                    for r in auc_rows if r['arm'] == arm_names[key]} for key in ('A','B','C')}
    final = {key: by_arm_step[key][120] for key in ('A','B','C')}
    left_b = {**{k: float(v) for k, v in final['A'].items()
                 if k in ('cross_entropy','top1_percent','top5_percent','prediction_flip_rate')},
              'CE_AUC': auc_as['A']['cross_entropy'], 'top1_AUC': auc_as['A']['top1_percent']}
    right_b = {**{k: float(v) for k, v in final['B'].items()
                  if k in ('cross_entropy','top1_percent','top5_percent','prediction_flip_rate')},
               'CE_AUC': auc_as['B']['cross_entropy'], 'top1_AUC': auc_as['B']['top1_percent']}
    delta_ba = _metric_improvements(left_b, right_b)
    b_helpful = any(v > 1e-12 for v in delta_ba.values())
    left_c = {**{k: float(v) for k, v in final['B'].items()
                 if k in ('cross_entropy','top1_percent','top5_percent','prediction_flip_rate')},
              'CE_AUC': auc_as['B']['cross_entropy'], 'top1_AUC': auc_as['B']['top1_percent']}
    right_c = {**{k: float(v) for k, v in final['C'].items()
                  if k in ('cross_entropy','top1_percent','top5_percent','prediction_flip_rate')},
               'CE_AUC': auc_as['C']['cross_entropy'], 'top1_AUC': auc_as['C']['top1_percent']}
    delta_cb = _metric_improvements(left_c, right_c)
    c_extra = any(v > 1e-12 for v in delta_cb.values()) and all(v >= -1e-12 for v in delta_cb.values())
    decision_a = drift_improved >= 3 and c_extra
    if decision_a:
        decision = 'A. TEMPORAL_RELATION_PRESERVING_PROGRESSIVE_PRUNING_PROMISING'
    elif b_helpful:
        decision = 'B. PROGRESSIVE_PRUNING_HELPFUL_BUT_TR_VALUE_UNRESOLVED'
    else:
        decision = 'C. PROGRESSIVE_AND_TR_PILOT_REJECTED'

    summary = {
        'task': 'Task042 Phase E.1 controlled progressive-pruning pilot',
        'decision': decision,
        'predeclared_decision_gate': {'C_lower_relation_drift_at_strict_majority': drift_improved >= 3,
          'strict_majority_lower_drift_checkpoints': drift_improved,
          'C_improves_at_least_one_and_regresses_none': c_extra,
          'B_improves_over_A_on_any_endpoint_or_recovery_auc': b_helpful,
          'delta_metrics_C_vs_B_positive_favors_C': delta_cb,
          'delta_metrics_B_vs_A_positive_favors_B': delta_ba},
        'final_checkpoint_metrics': {arm_names[k]: {x: float(final[k][x])
          for x in ('cross_entropy','top1_percent','top5_percent','prediction_flip_rate')}
          for k in ('A','B','C')},
        'auc_metrics': auc_as,
        'relation_drift_reduction_checkpoints': {'steps': [30,60,90,120],
          'C_lower_than_B_count': drift_improved,
          'mean_relation_drift_B': [drift_global[(arm_names['B'],s)] for s in (30,60,90,120)],
          'mean_relation_drift_C': [drift_global[(arm_names['C'],s)] for s in (30,60,90,120)]},
        'relation_performance_association': {'Drift_B_minus_C': association_drift,
          'CE_B_minus_C': association_ce, 'descriptive_pearson_r': pearson,
          'checkpoint_count': 4, 'interpretation': 'diagnostic only; not used to tune the method'},
        'candidate_ids_in_frozen_f3_order': candidate_ids,
        'candidate_manifest_sha256': sha256_file(out / 'task042_phase_e1_candidate_manifest.csv'),
        'candidate_sets_identical_A_B_C': True,
        'final_gate_identities_identical': True,
        'training_updates_per_arm': {summaries[k]['arm']: summaries[k]['optimizer_updates'] for k in summaries},
        'gate_schedule': list(GATE_STAGES), 'recovery_steps_per_stage': RECOVERY_STEPS_PER_STAGE,
        'validation_clip_count_each_checkpoint': 3783,
        'validation_list_sha256': sha256_file(VAL_LIST),
        'validation_identity_exact': True,
        'G_CE': float(manifest['G_CE']), 'G_TR': float(manifest['G_TR']),
        'lambda_TR_frozen': float(manifest['lambda_TR']),
        'lambda_update_count': 0,
        'teacher_detached': True,
        'same_five_frame_pair_identities_all_arms': True,
        'exact_gate_restoration_all_arms': all(summaries[k]['gate_restoration_exact'] for k in summaries),
        'all_training_and_validation_metrics_finite': True,
        'training_manifest_sha256': sha256_file(manifest_path),
        'phase_d_head': PHASE_D_HEAD,
        'code_head': subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'physical_gpus': [0,1],
        'scope': 'controlled pilot only; no final 50% pruning, physical pruning, 100-epoch fine-tuning, lambda search, gate redesign, or Task043',
    }
    report = make_report(summary, manifest, ref)
    report_path = out / 'task042_phase_e1_report.md'
    require(not report_path.exists(), 'refusing to overwrite report')
    report_path.write_text(report, encoding='utf-8')
    summary['outputs'] = {name: {'path': str(out / name), 'sha256': sha256_file(out / name)}
                          for name in REQUIRED_OUTPUTS if name != 'task042_phase_e1_summary.json'}
    write_json(out / 'task042_phase_e1_summary.json', summary, exclusive=True)
    print('PHASE_E1_DECISION=' + decision)
    print('C_LOWER_DRIFT=%d/4 lambda_TR=%.9g' % (drift_improved, float(manifest['lambda_TR'])))
    print('PHASE_E1_FINALIZED=PASS')


def make_report(summary: Mapping[str, Any], manifest: Mapping[str, Any], ref: Mapping[str, Any]) -> str:
    final = summary['final_checkpoint_metrics']
    auc = summary['auc_metrics']
    dr = summary['relation_drift_reduction_checkpoints']
    delta = summary['predeclared_decision_gate']['delta_metrics_C_vs_B_positive_favors_C']
    return '\n'.join([
        '# Task042 Phase E.1：时序关系保持的渐进剪枝受控试验', '',
        '## 预注册边界与设计', '',
        '- 决策只按附件预先规定的三臂对照规则给出；使用现有 Task042 分支和 Phase-D 冻结 F3 顺序，没有创建 Task043。',
        '- 试验臂：A 一次性置零后 CE 恢复；B 按 1→0.75→0.50→0.25→0 渐进门控并用 CE；C 与 B 同门控/步数并用 CE + 固定 Lambda×L_TR。',
        '- 12 个候选是 Phase-D baseline F3 顺序的最短前缀，含多个 Attention head、FFN neuron、same-type 域 269 和 mixed 域 271。三臂最终候选身份与门控完全一致。',
        '- 每个非初始 gate 阶段恢复 30 步；每臂总计 120 步。三臂使用同一初始 checkpoint、30 个固定训练视频、相同顺序、SGD、学习率、权重衰减、梯度裁剪和 FP32。',
        '- 完整 UCF101 validation 每个 checkpoint 均评估 3783/3783 clips；关系漂移固定在 E.0 的两个诊断视频和 span {1,2,4,8,16} 的五个固定帧对上计算。', '',
        '## 一次性梯度尺度校准', '',
        '- 10 个类别平衡的训练校准 batch 上，`G_CE=%.9g`，`G_TR=%.9g`，冻结 `lambda_TR=%.9g`。没有按 validation accuracy 调权重，也没有尝试其他 lambda。' %
        (manifest['G_CE'], manifest['G_TR'], manifest['lambda_TR']),
        '- A/B 只使用 CE；C 的 120 次更新都使用同一个校准系数。teacher 参数冻结，缓存目标脱离 autograd。', '',
        '## 全验证集结果', '',
        '| arm | 最终 CE ↓ | Top-1 ↑ | Top-5 ↑ | prediction flip ↓ | CE AUC ↓ | Top-1 AUC ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|',
        '| A ONE_SHOT | %.6f | %.3f%% | %.3f%% | %.4f | %.6f | %.3f |' %
        (final['A_ONE_SHOT']['cross_entropy'], final['A_ONE_SHOT']['top1_percent'], final['A_ONE_SHOT']['top5_percent'], final['A_ONE_SHOT']['prediction_flip_rate'], auc['A']['cross_entropy'], auc['A']['top1_percent']),
        '| B PROGRESSIVE_CE | %.6f | %.3f%% | %.3f%% | %.4f | %.6f | %.3f |' %
        (final['B_PROGRESSIVE_CE']['cross_entropy'], final['B_PROGRESSIVE_CE']['top1_percent'], final['B_PROGRESSIVE_CE']['top5_percent'], final['B_PROGRESSIVE_CE']['prediction_flip_rate'], auc['B']['cross_entropy'], auc['B']['top1_percent']),
        '| C PROGRESSIVE_CE_TR | %.6f | %.3f%% | %.3f%% | %.4f | %.6f | %.3f |' %
        (final['C_PROGRESSIVE_CE_TR']['cross_entropy'], final['C_PROGRESSIVE_CE_TR']['top1_percent'], final['C_PROGRESSIVE_CE_TR']['top5_percent'], final['C_PROGRESSIVE_CE_TR']['prediction_flip_rate'], auc['C']['cross_entropy'], auc['C']['top1_percent']), '',
        '- 原始 checkpoint（剪枝前）：CE=%.6f，Top-1=%.3f%%，Top-5=%.3f%%。' %
        (ref['cross_entropy'], ref['top1_percent'], ref['top5_percent']),
        '- C 在 %d/4 个匹配 checkpoint 的 mean relation drift 低于 B（严格多数门槛为至少 3/4）。' % dr['C_lower_than_B_count'],
        '- C 相对 B 的终点/AUC 有向改变量（正值表示 C 更好）：`%s`。' % json.dumps(delta, ensure_ascii=False, sort_keys=True),
        '- Drift reduction 与 validation CE 改善的 Pearson 仅作为 4 个匹配 checkpoint 上的描述性关联，不用于调整方法；相关数值在 summary/arm comparison 中。', '',
        '## 预注册结论', '',
        '**%s**' % summary['decision'], '',
        '这一结论只适用于 12-unit、24.3% 候选参数量、30 个训练视频、每阶段 30 步的受控 pilot。它不等价于最终 50% 剪枝或完整微调结论。',
        '本阶段未进行物理剪枝、50% 主实验、100-epoch 微调、Lambda 搜索或 selector/gate schedule 重设计。', '',
        '## 产物', '',
        *['- `%s`' % name for name in REQUIRED_OUTPUTS],
        '',
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=('prepare', 'reference-worker',
        'finalize-reference', 'calibration-worker', 'finalize-calibration', 'arm-worker', 'finalize'))
    parser.add_argument('--repo', default=str(REPO_DEFAULT))
    parser.add_argument('--base', default=str(BASE_DEFAULT))
    parser.add_argument('--physical-gpu', type=int, default=0)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--arm', choices=('A','B','C'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args)
    elif args.phase == 'reference-worker':
        reference_worker(args)
    elif args.phase == 'finalize-reference':
        finalize_reference(args)
    elif args.phase == 'calibration-worker':
        calibration_worker(args)
    elif args.phase == 'finalize-calibration':
        finalize_calibration(args)
    elif args.phase == 'arm-worker':
        require(args.arm is not None, '--arm is required for arm-worker')
        arm_worker(args)
    elif args.phase == 'finalize':
        finalize(args)


if __name__ == '__main__':
    main()
