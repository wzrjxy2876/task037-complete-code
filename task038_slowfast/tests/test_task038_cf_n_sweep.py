from __future__ import annotations

import inspect

from task038_slowfast import task038_cf_n_sweep as sweep
from task038_slowfast import slowfast_f3_selector as selector


def test_exact_n_grid_and_fixed_sigma():
    assert sweep.N_VALUES == (9, 18, 27, 36, 45)
    assert sweep.SIGMA == 0.050
    assert all(value == 0.0 or value == sweep.SIGMA for value in (sweep.SIGMA,))


def test_frozen_budget_and_method_constants():
    assert sweep.TARGET_REMAINING_RATIO == 0.50
    assert sweep.MIN_KEEP_RATIO == 0.10
    source = inspect.getsource(sweep)
    for token in ("max_iters", "sink_merge_tol", "one L2", "no_per_video_normalization"):
        # The first two are recorded in shared identity; the latter two guard
        # the frozen Contribution Field normalization contract.
        assert token in source or token.replace("_", " ") in source


def test_nested_manifest_gates_are_explicit():
    source = inspect.getsource(sweep._build_nested_manifests)
    assert "zip((9, 18, 27, 36), (18, 27, 36, 45))" in source
    assert "master_identity[:left] != master_identity[:right][:left]" in source
    assert "master_n45_sample_manifest.csv" in source
    assert "f\"n{n:03d}_manifest.csv\"" in source


def test_nine_sample_identity_is_preserved_before_extension():
    source = inspect.getsource(sweep._build_nested_manifests)
    assert "existing_identity != expected" in source
    assert "refusing to create a replacement baseline" in source
    assert "master_identity = existing_identity + extension" in source


def test_three_videos_per_selected_class_and_deterministic_seed():
    assert sweep.VIDEOS_PER_CLASS == 3
    assert sweep.SEED == 3407
    source = inspect.getsource(sweep._build_nested_manifests)
    assert "rng.sample(remaining_classes, 12)" in source
    assert "rng.sample(by_class[label], VIDEOS_PER_CLASS)" in source


def test_shared_descriptor_and_bms_artifacts_are_read_only():
    source = inspect.getsource(sweep._prepare_shared)
    assert "REFERENCE_OUTPUT / \"descriptors\"" in source
    assert "REFERENCE_SIGMA_OUTPUT / \"bms\"" in source
    assert "1676" in source
    assert "symlink_to" in inspect.getsource(sweep._link)


def test_master_archive_keeps_raw_float32_and_prefixes_without_averaging():
    source = inspect.getsource(sweep._build_master_archive)
    assert "dtype=np.float32" in source
    assert "target[:9]" in source
    assert "target[9:]" in source
    assert "mean" not in source.lower()


def test_worker_uses_one_logical_cuda_device_per_isolated_process():
    source = inspect.getsource(sweep._run_cli)
    assert '"CUDA_VISIBLE_DEVICES": physical_gpu' in source
    assert '"--device",' in source
    assert '"cuda:1"' not in source


def test_selector_avoids_duplicate_full_archive_gpu_allocation():
    source = inspect.getsource(selector.select_f3)
    assert "all_vectors.div_(" in source
    assert "all_vectors.masked_fill_(" in source
    assert "del all_norms" in source


def test_worker_has_no_finetuning_call_and_has_preft_validation():
    source = inspect.getsource(sweep._run_cli) + inspect.getsource(sweep._run_one)
    assert '"preft"' in source
    assert "fine_tune(" not in source
    assert "100" not in source or "N=100" not in source
    assert "fine_tuning_launched" in source


def test_independent_output_directories_and_status_gates():
    source = inspect.getsource(sweep._run_one)
    assert 'f"n{n:03d}"' in source
    assert '"RUNNING"' in source
    assert '"DONE"' in source
    assert '"FAILED"' in source


def test_result_summary_contains_required_identity_and_metrics():
    source = inspect.getsource(sweep._make_result_summary)
    for token in (
        "sample_manifest_sha256", "descriptor_sha256", "bms_domain_sha256",
        "contribution_field_sha256", "P_original", "P_structural_remaining",
        "remaining_parameter_ratio", "preft_sample_count", "**sim",
    ):
        assert token in source


def test_similarity_is_computed_domain_locally():
    source = inspect.getsource(sweep._similarity_stats)
    assert "for members in domains" in source
    assert "build_functional_similarity" in source
    assert "torch.triu" in source
    assert "55328" not in source


def test_jaccard_correctness():
    assert sweep._jaccard({1, 2}, {2, 3}) == 1 / 3
    assert sweep._jaccard(set(), set()) == 1.0
    assert sweep._jaccard({1}, set()) == 0.0


def test_stability_has_consecutive_and_n45_pairs():
    source = inspect.getsource(sweep._aggregate)
    for pair in ("(9, 18)", "(18, 27)", "(27, 36)", "(36, 45)", "(9, 45)", "(18, 45)", "(27, 45)", "(36, 45)"):
        assert pair in source
    assert "cf_n_selection_stability.csv" in source


def test_aggregation_is_deferred_until_all_workers_done():
    source = inspect.getsource(sweep._aggregate)
    assert 'status["status"] != "DONE"' in source
    assert "cf_n_sweep_summary.csv" in source
    assert "cf_n_sweep_by_preft_top1.csv" in source


def test_descriptive_correlations_are_present():
    source = inspect.getsource(sweep._aggregate)
    for token in ("preft_top1", "conv3_removed_ratio", "fast_removed_ratio", "fast_res4_removed_ratio", "mean_positive_similarity", "jaccard_to_n45"):
        assert token in source
    assert "descriptive_only" in source


def test_n09_reuse_does_not_recompute_selection():
    source = inspect.getsource(sweep._run_one)
    assert "_install_n09_reuse" in source
    assert "if n == 9" in source
    assert source.index("if n == 9") < source.index('"selection"')


def test_prepare_creates_shared_identity_before_workers():
    source = inspect.getsource(sweep._prepare_shared)
    assert "shared_identity.json" in source
    assert "READY.json" in source
    assert "fine_tuning_launched" in source
