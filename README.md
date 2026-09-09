# Task037 N09 Task034 F3 Reproduction

## Purpose

This package is the consolidated implementation source and recorded artifact
package for the fresh Task037 n=9 reproduction of Task034 F3_MID_VETO 50%
logical pruning. Authoritative implementations are preserved byte-for-byte from
the authoritative Task028 and LGFR source files. The optional Task021 offline
implementation is retained byte-for-byte under verification/archived_source_snapshots/
and exposed through a package-local loader; this changes only source loading,
not algorithms or numerical behavior. This repair only closes the first-party
source dependency graph.

## Bundled source closure

The package contains the complete first-party Python source closure needed by
the reproduced pipeline:

- contribution/ contains the three Task037 Contribution Field entry points.
- pruning/ contains the Task037 MC, functional-pruning, and Task034 F3 entry
  points.
- models/ contains the Video Swin model entry point.
- src/task028_runtime/ contains the recursively required Task028 first-party
  modules, dataset loader source, utility source, and YAML config.
- src/lgfr_runtime/ contains the recursively required LGFR probe, adapter,
  model, dataset, utility, and support modules.
- verification/ contains dependency, identity, runtime-file, and validation
  records.
- configs/, training/, and verification/ are part of the standalone layout.

Only source/config files were added to the two src closure trees. Historical
experiment result directories were not copied as source code.

## Pipeline

Contribution Field

        |

        v

MC.py

        |

        v

functional_competition_pruning.py

        |

        v

F3_MID_VETO

        |

        v

logical pruning

        |

        v

SGD fine-tuning

## Source identity

verification/source_identity_manifest.tsv records every authoritative
Task028/LGFR source file copied into this package. Every authoritative source
row has exact_match=true; the destination SHA256 is calculated from the
standalone copy.

verification/first_party_dependency_graph.json and
verification/first_party_dependency_graph.md record static imports, dynamic
imports, and shell/subprocess discovery. The graph was resolved recursively
until no first-party source dependency remained unresolved.

The original source paths in the identity and provenance records are not
runtime imports. The execution scripts use only source paths inside this
package. The package-local Task021 loader reads the exact archived source
snapshot inside verification/; it does not import a historical checkout.

## Runtime assets

The following may remain external runtime assets:

- the UCF101 dataset and split files;
- the original checkpoint, including checkpoint-68.ckpt;
- fresh generated Contribution Field NPZ/data;
- historical Task014-Task034 result artifacts used as read-only scientific
  reference data;
- normal third-party Python packages listed in requirements.txt.

Large generated NPZ, NPY, PT, PTH, and checkpoint files are kept as server-side
artifacts when present, but are excluded from new Git tracking by .gitignore.
Their paths, sizes, and SHA256 values are recorded in
verification/large_artifact_manifest.tsv.

## Standalone Guarantee

The Task037 implementation source can be imported and executed without using
historical Task028/LGFR source directories. The only external inputs are
runtime assets and installed third-party packages. The isolated-source test
constructs PYTHONPATH exclusively from paths inside this package and verifies
the principal modules and frozen F3 symbols.

## Frozen protocol

The reproduced settings remain unchanged:

- Contribution sample count: 9; seed: 3407.
- Descriptor variant: dynamic3d.
- BMS sigma: 0.1.
- F3 definitions:
  B = max(p_total, domain_damage),
  V = max(p_average - B, 0),
  R_F3 = B + V / 2.
- Ordering: R_F3, p_total, p_average, global_index.
- Fine-tuning: FP32, AMP=False, SGD, lr=5e-4, momentum=0.9,
  weight_decay=1e-5, batch_size=4, GPUs 0 and 1, scheduler NONE,
  seed 3407, 100 epochs.

## Reproduction commands

The four scripts are command sheets. They preserve the original scientific
arguments and use only standalone source paths:

1. run_task037_contribution.sh: fresh n=9 Contribution Field.
2. run_task037_pruning.sh: functional domain_average and domain_total pruning.
3. run_task037_finetune.sh: F3 identity, selection, validation gates, and
   fine-tuning.
4. run_task037_all.sh: execution order.

The scripts require the operator to provide external dataset, checkpoint,
fresh-data, and read-only reference-artifact environment variables. Reference
artifact roots are data inputs only and must never be placed on PYTHONPATH.
