# Task037 standalone validation report

## Result

Overall: **PASS**

The standalone package satisfies the source-closure gate: unresolved first-party source dependencies are zero and historical source paths required at runtime are zero.

## Scope

Package: /home/jixinye25/jxy_work1/swintrans_task037_complete
Branch: task_037_complete_code_release
Previous commit preserved: cf7be1ee42a4be1afc258e215d68ebdd69f541fe
Bundled Python files: **97**
Bundled shell scripts: **4**
First-party dependency records: **359** (348 Python static/dynamic plus 11 shell)
Copied Task028 source records: **65**
Copied LGFR source records: **33**
Source identity records: **98 exact / 98 total**
Unresolved first-party source dependencies: **0**
Historical source paths required at runtime: **0**

## Isolated validation

- Python compilation: **PASS** (all 97 bundled Python files compiled in memory; no bytecode dependency was created).
- Import smoke: **PASS** (9 principal modules resolved to package-internal paths).
- construct_f3_registry: **PASS**
- domain_total_losses: **PASS**
- BMS implementation resolution: **PASS**
- Focused CPU tests: **PASS**, 76 passed, 1 third-party deprecation warning.
- Shell syntax: **PASS**, all 4 run_task037_*.sh files passed bash -n.

## Exact bundled source tree

Executable/source files copied or provided by the package are listed below. The identity manifest is the authoritative complete path list for copied Task028/LGFR files.

### contribution/

- contribution/probe_contribution_pattern_affinity.py
- contribution/probe_contribution_pattern_coverage.py
- contribution/probe_cstc_contribution_tube.py

### pruning/

- pruning/MC.py
- pruning/functional_competition_pruning.py
- pruning/task034_mid_veto_50_logical_finetune.py

### models/

- models/ucf101_videoswin_my.py

### src/task028_runtime/

- src/task028_runtime/analyze_bms_competition_domains.py
- src/task028_runtime/analyze_descriptor_ablation_consistency.py
- src/task028_runtime/analyze_task011_pruning_regression.py
- src/task028_runtime/analyze_task012_pruning_severity.py
- src/task028_runtime/analyze_task013_cost_decoupling.py
- src/task028_runtime/analyze_task014_functional_pruning.py
- src/task028_runtime/analyze_task016_domain_size_calibration.py
- src/task028_runtime/analyze_task017_high_sparsity_collapse.py
- src/task028_runtime/analyze_task018_high_sparsity_transition.py
- src/task028_runtime/analyze_task019_causal_ablation.py
- src/task028_runtime/analyze_task020_functional_demand.py
- src/task028_runtime/analyze_task022_dual_risk.py
- src/task028_runtime/analyze_task023_average_rescue.py
- src/task028_runtime/analyze_task024_threshold_free_adaptive_safety.py
- src/task028_runtime/analyze_task025_total_domain_state_factorial_ablation.py
- src/task028_runtime/analyze_task027_senior_style_logical_pruning_finetune.py
- src/task028_runtime/analyze_task028_main50_tad_logical_finetune.py
- src/task028_runtime/analyze_task029_task028_type_degradation_diagnosis.py
- src/task028_runtime/analyze_tdd_pruning_validation.py
- src/task028_runtime/analyze_tdd_task009_task010_consistency.py
- src/task028_runtime/analyze_video_specific_third_descriptor.py
- src/task028_runtime/compare_pruning_run_configs.py
- src/task028_runtime/config/swintrans.yaml
- src/task028_runtime/contribution_field_loader.py
- src/task028_runtime/cost_decoupled_selection.py
- src/task028_runtime/coverage_selector.py
- src/task028_runtime/dataset/ssv2.py
- src/task028_runtime/dataset/transforms.py
- src/task028_runtime/dataset/ucf101.py
- src/task028_runtime/probe_descriptor_ablation_consistency.py
- src/task028_runtime/probe_task011_real_activation.py
- src/task028_runtime/probe_tdd_video_sanity.py
- src/task028_runtime/probe_video_specific_third_descriptor.py
- src/task028_runtime/task011_runtime_audit.py
- src/task028_runtime/task012_diagnostics.py
- src/task028_runtime/task012_runtime_audit.py
- src/task028_runtime/task013_diagnostics.py
- src/task028_runtime/task013_runtime_audit.py
- src/task028_runtime/task014_diagnostics.py
- src/task028_runtime/task014_runtime_audit.py
- src/task028_runtime/task015_attention_ffn_diagnosis.py
- src/task028_runtime/task016_preflight.py
- src/task028_runtime/task016_runtime_audit.py
- src/task028_runtime/task017_high_sparsity_diagnosis.py
- src/task028_runtime/task018_high_sparsity_transition.py
- src/task028_runtime/task019_dynamic_ranking_causal_ablation.py
- src/task028_runtime/task020_functional_demand_task_importance.py
- src/task028_runtime/task021_importance_information_retention.py
- src/task028_runtime/task022_average_total_dual_risk.py
- src/task028_runtime/task023_average_rescue_causal_ablation.py
- src/task028_runtime/task024_threshold_free_adaptive_safety.py
- src/task028_runtime/task025_total_domain_state_factorial_ablation.py
- src/task028_runtime/task027_senior_style_logical_pruning_finetune.py
- src/task028_runtime/task028_main50_tad_logical_finetune.py
- src/task028_runtime/task029_task028_type_degradation_diagnosis.py
- src/task028_runtime/task030_average_granularity_causal_validation.py
- src/task028_runtime/task031_domain_conditioned_average_diagnosis.py
- src/task028_runtime/task032_average_causal_granularity_decomposition.py
- src/task028_runtime/task033_average_veto_mechanism_diagnosis.py
- src/task028_runtime/temporal_dynamicity.py
- src/task028_runtime/utils.py

### src/lgfr_runtime/

- src/lgfr_runtime/MC.py
- src/lgfr_runtime/config/swintrans.yaml
- src/lgfr_runtime/dataset/ssv2.py
- src/lgfr_runtime/dataset/transforms.py
- src/lgfr_runtime/dataset/ucf101.py
- src/lgfr_runtime/function_bms_utils.py
- src/lgfr_runtime/functional_redundancy_smoke.py
- src/lgfr_runtime/offline_cfsp_selection.py
- src/lgfr_runtime/offline_function_coverage_probe.py
- src/lgfr_runtime/offline_function_coverage_probe_fixed.py
- src/lgfr_runtime/paired_cluster_prototype_cam_smoke_adapter.py
- src/lgfr_runtime/paired_cluster_prototype_cam_utils.py
- src/lgfr_runtime/probe_class_trajectory_weighted_coverage.py
- src/lgfr_runtime/probe_contribution_pattern_coverage_fast.py
- src/lgfr_runtime/probe_ctfrs_dynamic_function.py
- src/lgfr_runtime/probe_curve_shape_discovery.py
- src/lgfr_runtime/probe_descriptor_distinctiveness.py
- src/lgfr_runtime/probe_descriptor_intervention.py
- src/lgfr_runtime/probe_descriptor_stability.py
- src/lgfr_runtime/probe_feature_intervention.py
- src/lgfr_runtime/probe_functional_redundancy_validation.py
- src/lgfr_runtime/probe_pair_joint_masking_redundancy.py
- src/lgfr_runtime/probe_tdcc_dynamic_function.py
- src/lgfr_runtime/representative_redundancy_cam_smoke.py
- src/lgfr_runtime/representative_redundancy_cam_utils.py
- src/lgfr_runtime/ucf101_videoswin_my.py
- src/lgfr_runtime/ucf101_videoswin_probe_adapter_v2.py
- src/lgfr_runtime/utils.py
- src/lgfr_runtime/validate_paired_cluster_prototype_cam.py
- src/lgfr_runtime/validate_representative_redundancy_cam.py

### scripts/

- scripts/run_task037_all.sh
- scripts/run_task037_contribution.sh
- scripts/run_task037_finetune.sh
- scripts/run_task037_pruning.sh

### verification/

- verification/first_party_dependency_graph.json
- verification/first_party_dependency_graph.md
- verification/runtime_file_dependencies.json
- verification/source_identity_manifest.tsv
- verification/no_external_first_party_source.json
- verification/isolated_source_test.json
- verification/large_artifact_manifest.tsv
- verification/standalone_validation_report.md
- verification/archived_source_snapshots/task021_importance_information_retention.py

Task021's module-name-preserving loader is a new package-local wrapper; the exact Task028 implementation is the archived source snapshot above and is the identity-checked implementation. Its optional raw-field marker is not used by the Task037 reproduction.

## Runtime file classification

- Bundled: all listed Python source, model/pruning/contribution source, dataset loader source, utilities, YAML config, and verification source.
- External dataset: UCF101 frames and split files.
- External checkpoint: checkpoint-68.ckpt and other original weights.
- External generated data: fresh Contribution Field NPZ and read-only Task014–Task033 result artifacts.
- Third-party packages: installed Python dependencies in requirements.txt.

Historical Task028/LGFR path literals appear only in provenance/archived-authority records; executable directories (contribution/, pruning/, models/, src/, scripts/) contain zero such paths.

## Artifact/Git safety

- Server large-file audit: 7 files over 50 MiB; hashes and sizes are recorded in verification/large_artifact_manifest.tsv.
- Their server copies were retained; matching binary patterns are excluded by .gitignore and removed from the current Git index.
- Existing previous Git history contains old binary blobs; this commit does not rewrite history.
