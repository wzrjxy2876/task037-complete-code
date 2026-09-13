# Task041 Phase I — action-class diversity control

## Decision

**C. CLASS_DIVERSITY_DOES_NOT_RESCUE_TEMPORAL_SELECTION**

The 9x1 temporal cohort does not improve safest-candidate accuracy over the equal-size 3x3 cohort. Increasing action-class diversity does not rescue the current post-BMS temporal-stress selector.

## Frozen protocol and cohorts

The 3x3 cohort reuses Phase-H CE records. The 9x1 and 10x1 cohorts use the exact Task041 N=30 manifest, seed 3407, canonical class IDs in ascending order, and the first manifest video per selected class. The full-validation oracle remains the frozen Phase-D 3,783-video artifact. All damage is signed masked CE minus unmasked CE; temporal stress uses T=32 and 16 exact two-frame swaps at each of spans 1, 2, 4, 8, and 16.

| Cohort | Videos | Classes | New-video inference |
|---|---:|---:|---:|
| 3x3 | 9 | 3 | 0 |
| 9x1 | 9 | 9 | 9 |
| 10x1 | 10 | 10 | 10 |

## Same-type oracle

Domain-balanced values are unweighted means across the seven frozen same-type domains.

| Cohort | Method | Spearman | Kendall | Safest accuracy | Low/high correct | Reverse | Tie |
|---|---|---:|---:|---:|---:|---:|---:|
| 3x3 | original | -0.1429 | -0.0952 | 0.2857 | 2 | 5 | 0 |
| 3x3 | temporal | 0.0714 | 0.0476 | 0.2857 | 4 | 3 | 0 |
| 9x1 | original | -0.3333 | -0.3333 | 0.1429 | 2 | 4 | 1 |
| 9x1 | temporal | -0.4286 | -0.4286 | 0.0000 | 2 | 5 | 0 |
| 10x1 | original | -0.1667 | -0.1667 | 0.1429 | 2 | 4 | 1 |
| 10x1 | temporal | -0.2143 | -0.2381 | 0.1429 | 3 | 4 | 0 |

## Equal-N 3x3 versus 9x1

| Method | Domain | 3x3 safest | 9x1 safest | Fullval safest | Correction | Regression |
|---|---|---:|---:|---:|---:|---:|
| temporal | 269 | 774 | 10071 | 774 | False | True |
| temporal | 400 | 8524 | 28638 | 8524 | False | True |
| temporal | 415 | 1553 | 779 | 1549 | False | False |
| temporal | 102 | 166 | 166 | 390 | False | False |
| temporal | 103 | 254 | 254 | 16627 | False | False |
| temporal | 113 | 6 | 6 | 2321 | False | False |
| temporal | 76 | 133 | 36376 | 33306 | False | False |
| temporal | ALL | 29% | 0% | — | 0 corrections | 2 regressions |
| original | 269 | 774 | 10071 | 774 | False | True |
| original | 400 | 8524 | 28638 | 8524 | False | True |
| original | 415 | 1553 | 779 | 1549 | False | False |
| original | 102 | 166 | 166 | 390 | False | False |
| original | 103 | 254 | 254 | 16627 | False | False |
| original | 113 | 6 | 2321 | 2321 | True | False |
| original | 76 | 133 | 133 | 33306 | False | False |
| original | ALL | 29% | 14% | — | 1 corrections | 2 regressions |

The original-only rows show whether class coverage helps ordinary calibration by a similar amount; temporal benefit is not credited when original-only improves equally.

## Incremental temporal value within 9x1

| Method | Spearman | Kendall | Safest accuracy |
|---|---:|---:|---:|
| Original-only | -0.3333 | -0.3333 | 0.1429 |
| Temporal | -0.4286 | -0.4286 | 0.0000 |

Temporal corrections: 0; regressions: 1.

## Per-span 9x1

| Span | Spearman | Kendall | Safest accuracy |
|---:|---:|---:|---:|
| 1 | -0.4286 | -0.4286 | 0.1429 |
| 2 | -0.4286 | -0.4286 | 0.0000 |
| 4 | -0.5000 | -0.5238 | 0.0000 |
| 8 | -0.4286 | -0.4286 | 0.0000 |
| 16 | -0.4286 | -0.4286 | 0.0000 |

## Leave-one-class-out 10x1 stability

| Dropped class | Stable safest domains | Mean within-domain rank Spearman |
|---|---:|---:|
| BoxingPunchingBag (16) | 5/7 | 0.6429 |
| BoxingSpeedBag (17) | 7/7 | 1.0000 |
| GolfSwing (32) | 7/7 | 0.7857 |
| HighJump (39) | 7/7 | 1.0000 |
| Mixing (53) | 7/7 | 0.7857 |
| MoppingFloor (54) | 4/7 | 0.2857 |
| PlayingGuitar (62) | 7/7 | 1.0000 |
| PlayingPiano (63) | 7/7 | 1.0000 |
| Rafting (72) | 6/7 | 0.8571 |
| Surfing (87) | 6/7 | 0.6429 |

## Frozen conclusions and provenance

Phase-H.1 remains unchanged: corrected original-only and temporal safest accuracy are both 2/7; temporal corrections and regressions are both zero; its scientific conclusion remains `NO_DEMONSTRATED_TOP1_SELECTION_VALUE`.

The report uses the exact Phase-D full-validation oracle and performs no pruning, finetuning, descriptor/BMS change, or Task042 creation.

See `task041_phase_i_summary.json` for input hashes, all domain rows, predeclared gate results, and the complete decision audit.
