import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'lgfr_runtime'))
sys.path.insert(0, str(ROOT / 'scripts'))

import task042_frame_relation_redundancy as task042
import task042_phase_e2_graph_refresh as graph
import task042_phase_e2_runner as phase_script


def test_reciprocal_nearest_neighbor_and_exact_tie_break():
    units = [{'task037_global_index': i, 'domain_id': '7'} for i in (10, 20, 30)]
    distances = {(10, 20): 0.1, (10, 30): 0.1, (20, 30): 0.2}
    rows = graph.directed_nearest_neighbors(units, distances)
    by_id = {r['task037_global_index']: r for r in rows}
    assert by_id[10]['nn_task037_global_index'] == 20
    assert by_id[10]['exact_nearest_tie'] is True
    assert graph.reciprocal_edges(rows) == {(10, 20)}


def test_direction_uses_only_authoritative_f3_scores():
    provenance = {
        10: {'R_F3': '0.2', 'p_total': '0.3', 'p_average': '0.4'},
        20: {'R_F3': '0.1', 'p_total': '0.9', 'p_average': '0.8'},
    }
    key = lambda r: (float(r['R_F3']), float(r['p_total']), float(r['p_average']), int(r['global_index']))
    assert graph.choose_directional_candidate((10, 20), provenance, key) == (
        20, 10, 'canonical_Task037_F3_order_key_lower_first')
    # A member without recorded F3 scores cannot override the scored peer.
    provenance[10] = {'R_F3': '', 'p_total': '', 'p_average': ''}
    assert graph.choose_directional_candidate((10, 20), provenance, key) == (
        20, 10, 'only_pair_member_with_recorded_authoritative_F3_score')


def test_temporal_metric_is_the_frozen_task042_metric():
    units = [{'task037_global_index': 1, 'domain_id': '3'},
             {'task037_global_index': 2, 'domain_id': '3'}]
    x = [float(i) for i in range(80)]
    y = [float((i * 7) % 31) for i in range(80)]
    raw = []
    idx = 0
    for span in (1, 2, 4, 8, 16):
        for pair in range(16):
            raw.extend([
                {'video_index': 0, 'task037_global_index': 1, 'span': span, 'pair_index': pair,
                 'relative_sensitivity': x[idx]},
                {'video_index': 0, 'task037_global_index': 2, 'span': span, 'pair_index': pair,
                 'relative_sensitivity': y[idx]},
            ])
            idx += 1
    actual, counts, _per_video = graph.aggregate_relation_distances(raw, units, [0], task042)
    zx = task042.normalize_signature(x)[0]
    zy = task042.normalize_signature(y)[0]
    expected = task042.relation_distance(zx, zy)
    assert actual[(1, 2)] == pytest.approx(expected, abs=0.0)
    assert counts[(1, 2)] == 1


def test_candidate_and_survivor_identity_partition_is_exact():
    assert graph.validate_identity_partition([1, 2, 3, 4], [2, 4], [1, 3])
    with pytest.raises(ValueError):
        graph.validate_identity_partition([1, 2, 3], [2], [1])


def test_graph_accounting_excludes_impossible_removed_endpoints_from_new_edges():
    unit_domain = {1: '7', 2: '7', 3: '7', 4: '7'}
    rows, retained, lost, new = graph.graph_change_rows(
        {(1, 2), (3, 4)}, {(1, 3)}, {1, 3, 4}, unit_domain)
    total = next(r for r in rows if r['domain_id'] == 'ALL_DOMAINS')
    assert retained == set()
    assert lost == {(3, 4)}
    assert new == {(1, 3)}
    assert total['initial_edges_impossible_removed_endpoint'] == 1
    assert total['new_edges'] == 1
    assert total['new_edge_ids'] == [(1, 3)]


def test_calibration_subset_identities_reproduce_three_positions_per_class():
    rows = []
    for label in range(10):
        for position in range(3):
            video_index = label * 3 + position
            rows.append({'label': label, 'video_index': video_index, 'dataset_index': video_index + 100})
    subsets = graph.calibration_subsets(rows)
    assert subsets['A_position1'] == list(range(0, 30, 3))
    assert subsets['B_position2'] == list(range(1, 30, 3))
    assert subsets['C_position3'] == list(range(2, 30, 3))
    assert subsets['AB_positions12'] == sorted(subsets['A_position1'] + subsets['B_position2'])
    assert subsets['full_10x3'] == list(range(30))


def test_recovery_path_is_ce_only_and_does_not_call_temporal_loss():
    source = inspect.getsource(phase_script.train_round)
    assert 'F.cross_entropy' in source
    assert 'ltr_backward' not in source
    assert 'relation_loss' not in source
    assert "'L_TR_used': False" in source
    assert phase_script.RECOVERY_OBJECTIVE == 'cross_entropy_only'
