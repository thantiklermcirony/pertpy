# Archived submission draft

Submitted as [Pertpy PR #1098](https://github.com/scverse/pertpy/pull/1098).
The live pull request is the current review record; this file preserves the draft.

Title: Add baseline-aware evaluator for perturbation predictions

Relates to #1035. Comparing perturbation predictions currently requires callers
to assemble splits, references and baselines themselves. This adds
`pt.tl.PerturbationEvaluator`, which aligns genes by identifier and returns tidy
per-group metrics with explicit unavailable/undefined statuses.

Baselines use training controls and single interventions only. Explicit label,
combination and context holdouts preserve cell membership; known combination
aliases are grouped through an explicit component mapping. Evaluation rejects
declared train/test cell overlap and mismatched feature sets, while retaining
missing prediction groups. MSE and E-distance reuse the existing `Distance` API.
The guide documents point-baseline, measurement-scale and split limitations.

This is a proposed first API slice: control-mean and additive baselines, MSE,
E-distance, delta Pearson, sign agreement and top-k absolute response overlap.
It does not close the entire issue: nearest-neighbour baselines, logFC scoring
and DEG discovery are not implemented. Input contracts for the first two would
benefit from maintainer agreement.

Validation: 44 contribution tests and 40 independent audit cases passed on Python
3.12.3 and 3.14.7; the existing comparison tests passed. Whole-project mypy checked
101 files; applicable pre-commit hooks and the full Sphinx build passed. An
eight-combination Norman example produced 80 score/status rows, agreed with
independent numerical calculations, and preserved the case where additive
prediction is worse than the control baseline.

Runtime and real-data evidence:
https://github.com/thantiklermcirony/pertpy/actions/runs/34415474373

Project checks:
https://github.com/thantiklermcirony/pertpy/actions/runs/34415474369

The clean diff contains only implementation, tests and documentation. The
audit workflow, data recipe and independent tests remain in the fork's audit
branch. No raw data or claim of a new biological prediction method is included.
