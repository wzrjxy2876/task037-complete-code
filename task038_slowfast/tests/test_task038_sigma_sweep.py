
from __future__ import annotations

import inspect

from task038_slowfast import task038_cli
from task038_slowfast import task038_sigma_sweep as sweep


def test_exact_task038_sigma_grid():
    assert sweep.SIGMAS == (
        0.100,
        0.095,
        0.090,
        0.085,
        0.080,
        0.075,
        0.070,
        0.065,
        0.060,
        0.055,
        0.050,
    )


def test_sigma_directory_names_are_stable():
    assert sweep._sigma_name(0.100) == "sigma_0p100"
    assert sweep._sigma_name(0.050) == "sigma_0p050"


def test_cli_exposes_only_sigma_as_bms_override():
    source = inspect.getsource(task038_cli.main)
    run_source = inspect.getsource(task038_cli.run)
    assert "--bms-sigma" in source
    assert "sigma=args.bms_sigma" in run_source
    assert '"tol": 1e-4' in run_source
    assert '"max_iters": 100' in run_source
    assert '"sink_merge_tol": 0.01' in run_source


def test_wrapper_preserves_frozen_target_and_min_keep():
    assert sweep.TARGET_REMAINING_RATIO == 0.50
    assert sweep.MIN_KEEP_RATIO == 0.10
    source = inspect.getsource(sweep._run_cli)
    assert "--target-remaining-ratio" in source
    assert "TARGET_REMAINING_RATIO" in source


def test_wrapper_does_not_launch_finetuning():
    source = inspect.getsource(sweep.main)
    runner = inspect.getsource(sweep._run_cli)
    assert '"finetune"' not in source
    assert '"finetune"' not in runner
    assert '"preft"' in source
    assert "fine_tune(" not in source


def test_wrapper_reuses_official_artifacts_read_only():
    source = inspect.getsource(sweep._prepare_sigma_dir)
    assert "REFERENCE_OUTPUT" in inspect.getsource(sweep.main)
    assert 'symlink_to' in source
    assert "descriptors" in source
    assert "contribution" in source


def test_global_outputs_include_required_diagnostics():
    source = inspect.getsource(sweep._write_global_outputs)
    for field in (
        "sigma_sweep_summary.csv",
        "sigma_sweep_analysis.json",
        "sigma_sweep_report.md",
        "conv3_removed_ratio",
        "fast_res4_removed_ratio",
        "preft_top1",
        "preft_top5",
    ):
        assert field in source


def test_rank_and_correlation_helpers():
    assert sweep._ranks([3.0, 1.0, 2.0]) == [3.0, 1.0, 2.0]
    assert sweep._ranks([1.0, 1.0, 2.0]) == [1.5, 1.5, 3.0]
    assert sweep._pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0

def test_subset_worker_support_keeps_shared_upstream_read_only():
    source = inspect.getsource(sweep.main)
    verify_source = inspect.getsource(sweep._verify_upstream)
    assert "--sigmas" in source
    assert "--read-only-upstream" in source
    assert "write_manifest=not args.read_only_upstream" in source
    assert "selected_sigmas" in source
    assert "write_manifest" in verify_source


def test_subset_worker_defers_global_aggregation():
    source = inspect.getsource(sweep.main)
    assert "global aggregation deferred" in source
    assert "len(selected_sigmas) == len(SIGMAS)" in source
