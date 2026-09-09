"""Run the Pertpy evaluator on the reproducibly prepared Norman example."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

import pertpy as pt


def run(subset_path: Path, output: Path) -> None:
    """Save aggregate scores and independent checks, never a data copy."""
    data = ad.read_h5ad(subset_path)
    manifest = json.loads(subset_path.with_suffix(".manifest.json").read_text())
    evaluator = pt.tl.PerturbationEvaluator("perturbation_name", "control", context_key="context")
    train, test = evaluator.split(
        data, holdout=manifest["holdout"], strategy="combination", components=manifest["components"]
    )
    assert set(train.obs_names) == set(manifest["train_obs_names"])
    assert set(test.obs_names) == set(manifest["test_obs_names"])
    scores = evaluator.evaluate(test, {}, train=train, components=manifest["components"], top_k=20)
    assert len(scores) == 8 * 2 * 5
    assert (scores.n_true == 200).all()
    assert (scores.n_predicted == 1).all()
    assert (scores.n_features == 500).all()

    # Compute mean and distribution scores independently of Pertpy's Distance.
    x_train = train.X.toarray().astype(np.float64)
    x_test = test.X.toarray().astype(np.float64)
    train_labels = train.obs["perturbation_name"].astype(str).to_numpy()
    test_labels = test.obs["perturbation_name"].astype(str).to_numpy()
    control = x_train[train_labels == "control"].mean(axis=0)
    independent_checks = 0
    for label in manifest["holdout"]:
        real = x_test[test_labels == label]
        additive = control.copy()
        for part in manifest["components"][label]:
            additive += x_train[train_labels == part].mean(axis=0) - control
        for name, predicted in [("control_mean", control), ("additive", additive)]:
            selected = scores[(scores.perturbation == label) & (scores.model == f"baseline:{name}")]
            expected_mse = np.mean((real.mean(axis=0) - predicted) ** 2)
            expected_energy = 2 * cdist(real, predicted[None, :]).mean() - pdist(real).mean()
            for metric, expected in [("mse", expected_mse), ("edistance", expected_energy)]:
                row = selected[selected.metric == metric].iloc[0]
                assert row.status == "ok"
                np.testing.assert_allclose(row.value, expected, rtol=1e-10, atol=1e-10)
                independent_checks += 1

    # Reordering genes in training changes neither alignment nor any score.
    reordered = evaluator.evaluate(
        test, {}, train=train[:, ::-1].copy(), components=manifest["components"], top_k=20
    )
    pd.testing.assert_frame_equal(scores, reordered, check_exact=False, rtol=1e-10, atol=1e-10)
    assert scores.loc[scores.status != "ok", "value"].isna().all()
    assert np.isfinite(scores.loc[scores.status == "ok", "value"]).all()

    mse = scores[scores.metric == "mse"].pivot(index="perturbation", columns="model", values="value")
    mse["additive_relative_mse_reduction"] = 1 - mse["baseline:additive"] / mse["baseline:control_mean"]
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "norman-scores.csv", index=False)
    mse.to_csv(output / "norman-mse-comparison.csv")
    (output / "norman-evaluation-manifest.json").write_text(
        json.dumps(scores.attrs["pertpy_evaluation"], indent=2) + "\n"
    )
    report = {
        "source_url": manifest["source_url"],
        "source_sha256": manifest["source_sha256"],
        "shape": list(data.shape),
        "train_cells": train.n_obs,
        "test_cells": test.n_obs,
        "heldout_combinations": manifest["holdout"],
        "component_mapping": manifest["components"],
        "measurement": manifest["measurement"],
        "selection": manifest["metadata_selection"],
        "limitations": manifest["limitations"],
        "score_rows": len(scores),
        "status_counts": scores.status.value_counts().to_dict(),
        "independent_numerical_checks": independent_checks,
        "gene_order_invariance_check": "passed",
        "additive_lower_mse_groups": int((mse["baseline:additive"] < mse["baseline:control_mean"]).sum()),
        "total_groups": len(mse),
        "macro_mean_mse": {name: float(mse[name].mean()) for name in ["baseline:control_mean", "baseline:additive"]},
        "macro_mean_delta_pearson_additive": float(scores[(scores.model == "baseline:additive") & (scores.metric == "delta_pearson")].value.mean()),
        "data_redistributed": False,
        "package_file": str(Path(pt.__file__).resolve()),
        "versions": {name: importlib.metadata.version(name) for name in ["pertpy", "anndata", "numpy", "pandas", "scipy", "h5py"]},
        "scripts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(__file__).with_name("prepare_norman_subset.py")]},
    }
    (output / "norman-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(mse.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.subset, args.output)
