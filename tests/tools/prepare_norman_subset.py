"""Create a bounded, deterministic Norman combination evaluation example.

This script reads an already downloaded public processed H5AD. It does not fetch
or publish data. Its feature universe inherits the source archive's filtering.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import anndata as ad
from anndata.io import read_elem, sparse_dataset
import h5py
import numpy as np
import pandas as pd
import scipy
from scipy import sparse


SOURCE_SHA256 = "b679c157fea550ef5be4dad91da9dff1f5d8287313ad3f3537f6cf14d4bcd434"
SOURCE_URL = "https://ndownloader.figshare.com/files/34027562"
BAD_GUIDE = "NegCtrl1_NegCtrl0__NegCtrl1_NegCtrl0"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(value: str) -> tuple[str, str]:
    return hashlib.sha256(value.encode("utf-8")).hexdigest(), value


def canonical_guide(guide: str) -> str:
    # The source construct grammar is verified against every perturbation label.
    tokens = guide.split("__", 1)[0].split("_")
    if len(tokens) != 2:
        raise ValueError(f"Unexpected source guide grammar: {guide}")
    genes = sorted(t for t in tokens if re.fullmatch(r"NegCtrl\d+", t) is None)
    if len(set(genes)) != len(genes):
        raise ValueError(f"Repeated non-control component: {guide}")
    return "+".join(genes) if genes else "control"


def prepare(source: Path, output: Path) -> dict:
    if source.suffix != ".h5ad" or not source.is_file():
        raise ValueError("Use a completed .h5ad source file, never a partial download.")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Subset or manifest already exists; no overwrite attempted.")
    source_digest = file_hash(source)
    if source_digest != SOURCE_SHA256:
        raise ValueError("Source SHA-256 differs from the audited public artifact.")

    with h5py.File(source, "r") as handle:
        obs = read_elem(handle["obs"])
        var = read_elem(handle["var"])
        if not obs.index.is_unique or not var.index.is_unique:
            raise ValueError("Source cell and gene identifiers must be unique.")
        mapping = {str(g): canonical_guide(str(g)) for g in obs["guide_identity"].unique()}
        canonical = obs["guide_identity"].astype(str).map(mapping)
        labels = obs["perturbation_name"].astype(str)
        if not np.array_equal(canonical.to_numpy(), labels.to_numpy()):
            raise ValueError("Guide-derived labels disagree with archived perturbation_name.")
        good = obs["good_coverage"]
        if good.dtype != bool:
            raise ValueError("Expected a Boolean good_coverage flag.")
        bad_guide_mask = obs["guide_identity"].astype(str).eq(BAD_GUIDE)
        eligible_mask = good & ~bad_guide_mask
        condition_counts = labels[eligible_mask].value_counts()
        minimum_cells = 50
        eligible_combinations = [
            label for label, count in condition_counts.items()
            if len(label.split("+")) == 2 and count >= minimum_cells
            and all(condition_counts.get(part, 0) >= minimum_cells for part in label.split("+"))
        ]
        combinations = sorted(eligible_combinations, key=stable_order)[:8]
        if len(combinations) != 8 or condition_counts.get("control", 0) < minimum_cells:
            raise ValueError("Insufficient eligible combinations or training controls.")
        components = {label: label.split("+") for label in combinations}
        singles = sorted({part for parts in components.values() for part in parts})
        train_conditions = ["control", *singles]
        conditions = [*train_conditions, *combinations]
        selected_positions: list[int] = []
        selection_counts = {}
        for label in conditions:
            positions = np.flatnonzero((eligible_mask & labels.eq(label)).to_numpy())
            selected = sorted(positions, key=lambda pos: stable_order(str(obs.index[pos])))[:200]
            selected_positions.extend(selected)
            selection_counts[label] = {"eligible_cells": int(len(positions)), "selected_cells": len(selected)}

        # Read only chosen cells across the archived measured feature universe.
        desired = np.asarray(selected_positions)
        sorted_positions = np.sort(desired)
        row_restore = np.searchsorted(sorted_positions, desired)
        counts = sparse_dataset(handle["layers"]["counts"])[sorted_positions, :].tocsr()[row_restore]
        if not np.isfinite(counts.data).all() or (counts.data < 0).any():
            raise ValueError("Selected count values must be finite and nonnegative.")
        if not np.equal(counts.data, np.floor(counts.data)).all():
            raise ValueError("Selected count values must be integer-valued.")
        library_totals = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
        if (library_totals <= 0).any():
            raise ValueError("Selected cells must have nonzero libraries.")
        normalized = counts.astype(np.float64).multiply((10000.0 / library_totals)[:, None]).tocsr()
        normalized.data = np.log1p(normalized.data)
        # Verify the archived X scale separately; never use it to fit the baseline.
        original_x = sparse_dataset(handle["X"])[sorted_positions, :].tocsr()[row_restore]
        difference = (original_x - normalized).tocsr()
        max_original_x_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        if max_original_x_difference > 1e-5:
            raise ValueError("Archived X does not match the documented log1p 10000-count scale.")
        source_uns = {key: str(read_elem(handle["uns"][key])) for key in ("doi", "preprocessing_nb_link")}

    selected_obs = obs.iloc[desired][["perturbation_name", "guide_identity", "gemgroup", "good_coverage"]].copy()
    selected_obs["perturbation_name"] = labels.iloc[desired].to_numpy()
    selected_obs["split"] = np.where(selected_obs["perturbation_name"].isin(combinations), "test", "train")
    selected_obs["context"] = "K562"
    selected_obs["source_row"] = desired
    selected_obs["library_total_19018_genes"] = library_totals
    train_mask = selected_obs["split"].eq("train").to_numpy()
    train = normalized[train_mask]
    means = np.asarray(train.mean(axis=0)).ravel()
    variances = np.asarray(train.power(2).mean(axis=0)).ravel() - means**2
    if (variances < -1e-10).any() or not np.isfinite(variances).all():
        raise ValueError("Invalid training feature variance.")
    variances = np.maximum(variances, 0)
    gene_order = np.asarray(sorted(range(len(var)), key=lambda j: (-variances[j], str(var.index[j])))[:500])
    selected_var = pd.DataFrame(index=var.index[gene_order].copy())
    selected_var["ensembl_id"] = var["index"].iloc[gene_order].astype(str).to_numpy()
    selected_var["training_variance_log1p_10k"] = variances[gene_order]
    selected_var["source_column"] = gene_order
    subset = ad.AnnData(normalized[:, gene_order].astype(np.float32), obs=selected_obs, var=selected_var)
    subset.layers["counts"] = counts[:, gene_order].astype(np.float32)
    subset.uns["measurement_scale"] = "log1p(counts / per-cell library total across 19018 archived genes * 10000)"
    subset.uns["source_url"] = SOURCE_URL
    subset.uns["source_sha256"] = source_digest
    subset.uns["heldout_combinations"] = combinations
    subset.uns["components_json"] = json.dumps(components, sort_keys=True)
    subset.uns["split_protocol"] = "All selected controls and constituent singles train; every selected combination cell test."
    subset.uns["redistribution_license"] = "Unresolved for this exact file; local research subset only."

    train_ids = selected_obs.index[train_mask].astype(str).tolist()
    test_ids = selected_obs.index[~train_mask].astype(str).tolist()
    if set(train_ids) & set(test_ids):
        raise ValueError("Cell overlap across split.")
    if set(selected_obs.loc[train_mask, "perturbation_name"]) & set(combinations):
        raise ValueError("Combination leaked into training.")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL,
        "source_bytes": source.stat().st_size,
        "source_sha256": source_digest,
        "source_shape": [len(obs), len(var)],
        "source_uns": source_uns,
        "source_provenance": "https://raw.githubusercontent.com/theislab/sc-pert/main/data_table.csv",
        "processing_notebook": "https://github.com/theislab/sc-pert/blob/main/datasets/Norman_2019.ipynb",
        "published_processing": [
            "Filter cells with fewer than 200 detected genes; filter genes found in fewer than 20 cells.",
            "Make gene symbols unique; calculate QC metadata.",
            "Retain cells with fewer than 6000 detected genes and mitochondrial count percentage below 15.",
            "Store counts; normalize each cell to 10000 total counts; natural log1p.",
            "Compute 5000 HVGs, PCA, neighbors, Leiden and UMAP; these stored results are ignored here.",
            "Convert the empty guide_ids label to control and commas between targeted genes to '+'."
        ],
        "original_paper": "https://doi.org/10.1126/science.aax4438",
        "exact_file_license_verified": False,
        "redistribution": "Do not publish this subset unless the exact data-file license is independently verified.",
        "shape": list(subset.shape),
        "perturbation_key": "perturbation_name",
        "control": "control",
        "context_key": "context",
        "split_key": "split",
        "train_cells": len(train_ids),
        "test_cells": len(test_ids),
        "holdout": combinations,
        "components": components,
        "training_conditions": train_conditions,
        "condition_counts": selection_counts,
        "metadata_selection": {
            "minimum_cells_per_combination_and_constituent": minimum_cells,
            "maximum_combinations": 8,
            "maximum_cells_per_condition": 200,
            "condition_order": "SHA256(canonical UTF-8 perturbation label), tie by label",
            "cell_order": "SHA256(original UTF-8 cell ID), tie by cell ID",
            "quality_filter": "good_coverage == True; exclude problematic NegCtrl1 construct",
            "good_coverage_false_source_cells": int((~good).sum()),
            "problematic_guide_source_cells": int(bad_guide_mask.sum()),
            "excluded_union_cells": int((~eligible_mask).sum()),
            "eligible_source_cells": int(eligible_mask.sum()),
            "eligible_combinations": len(eligible_combinations),
            "guide_to_label_validation": "All source rows match the two-position guide grammar after removing NegCtrl tokens and sorting genes.",
            "guide_pooling": "Pool constructs with the same intended unordered gene intervention; retain original guide_identity and gemgroup.",
            "problematic_guide": BAD_GUIDE,
            "exclusion_source": "https://raw.githubusercontent.com/theislab/sc-pert/main/datasets/Norman_2019_curation.ipynb"
        },
        "measurement": {
            "input": "layers/counts",
            "selected_count_values": "All finite, nonnegative and integer-valued.",
            "normalization": "Per selected cell: divide by count sum over all 19018 archived genes, multiply 10000, natural log1p.",
            "normalization_dtype": "float64",
            "saved_X_dtype": "float32",
            "source_X_max_absolute_difference_from_recomputed_log1p_10k": max_original_x_difference,
            "feature_selection": "Top 500 population variances on normalized training controls and singles only; gene-symbol tie-break.",
            "inherited_processing_limit": "Source archive already filtered cells/genes globally. The 19018-gene universe is inherited; no claim of a raw-data benchmark free of all preprocessing leakage.",
            "ignored_source_metadata": ["highly_variable", "leiden", "PCA", "UMAP", "neighbors", "all-data means and dispersions"],
            "counts_layer_note": "Saved counts are only the selected genes; library_total_19018_genes preserves the full archived denominator. Do not renormalize using the 500-gene count sum."
        },
        "train_obs_names": train_ids,
        "test_obs_names": test_ids,
        "var_names": subset.var_names.astype(str).tolist(),
        "limitations": [
            "Software demonstration only; no model superiority or biological discovery claim.",
            "Single K562 context; 10X gemgroup lanes are not independent biological replicates.",
            "Point predictions describe means, not a generative distribution.",
            "No source precomputed DEG lists or outcome-dependent condition selection used."
        ],
        "runtime": {"anndata": version("anndata"), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "h5py": h5py.__version__}
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(output, compression="gzip")
    reread = ad.read_h5ad(output)
    if (not reread.obs_names.equals(subset.obs_names) or not reread.var_names.equals(subset.var_names)
            or (reread.X != subset.X).nnz or (reread.layers["counts"] != subset.layers["counts"]).nnz):
        raise ValueError("Written subset differs from the prepared matrix or identifiers.")
    manifest["write_readback_verified"] = True
    manifest["semantic_digests"] = {
        "ordered_cell_ids_sha256": hashlib.sha256("\n".join(subset.obs_names).encode("utf-8")).hexdigest(),
        "ordered_gene_ids_sha256": hashlib.sha256("\n".join(subset.var_names).encode("utf-8")).hexdigest(),
        "dense_X_little_endian_float32_sha256": hashlib.sha256(np.asarray(subset.X.toarray(), dtype="<f4").tobytes(order="C")).hexdigest(),
        "dense_counts_little_endian_float32_sha256": hashlib.sha256(np.asarray(subset.layers["counts"].toarray(), dtype="<f4").tobytes(order="C")).hexdigest()
    }
    manifest["subset_bytes"] = output.stat().st_size
    manifest["subset_sha256"] = file_hash(output)
    manifest["script_sha256"] = file_hash(Path(__file__))
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    selected_obs.to_csv(output.with_suffix(".cells.csv"))
    selected_var.to_csv(output.with_suffix(".genes.csv"))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "subset.h5ad")
    args = parser.parse_args()
    result = prepare(args.source.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in ("shape", "train_cells", "test_cells", "holdout", "subset_bytes", "subset_sha256", "measurement")}, indent=2))
