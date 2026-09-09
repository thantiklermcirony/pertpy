"""Independent public-API acceptance tests for PerturbationEvaluator.

Run against the real installed/source-selected Pertpy package, with no package
stubs. Small expected values were specified before reading the implementation.
These are synthetic correctness tests, not a biological prediction benchmark.
"""

import copy
import json
from math import sqrt

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
import pytest
from scipy import sparse


def dataset(values, labels, prefix, *, features=None, contexts=None, layer=None):
    values = np.asarray(values)
    features = features if features is not None else [f"g{i + 1}" for i in range(values.shape[1])]
    obs = pd.DataFrame({"perturbation": labels}, index=[f"{prefix}_{i}" for i in range(len(labels))])
    if contexts is not None:
        obs["context"] = contexts
    result = ad.AnnData(values.copy(), obs=obs, var=pd.DataFrame(index=features))
    if layer is not None:
        result.layers[layer] = values.copy()
        result.X = np.full(values.shape, 9000.0)
        result.obsm["X_pca"] = np.full(values.shape, -1000.0)
    return result


def row(frame, metric, *, model="model", perturbation="A", scope="all"):
    selected = frame[
        (frame.model == model)
        & (frame.metric == metric)
        & (frame.perturbation == perturbation)
        & (frame.scope == scope)
    ]
    assert len(selected) == 1
    return selected.iloc[0]


@pytest.fixture
def oracle():
    return (
        dataset([[10, 100, 1000, 10000]], ["ctrl"], "train"),
        dataset([[12, 99, 1000, 10003]], ["A"], "test"),
        dataset([[11, 102, 1000, 10003]], ["A"], "pred"),
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("mse", 2.5),
        ("delta_pearson", 3 / sqrt(50)),
        ("direction_accuracy", 2 / 3),
        ("top_k_overlap", 0.5),
    ],
)
def test_hand_declared_delta_oracle(oracle, metric, expected):
    train, truth, prediction = oracle
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    result = evaluator.evaluate(truth, {"model": prediction}, train=train, baselines=(), metrics=(metric,), top_k=2)
    actual = row(result, metric)
    assert actual.status == "ok"
    assert actual.value == pytest.approx(expected)
    assert actual.reference_source == "train_control"


def test_feature_permutation_preserves_values_and_inputs(oracle):
    train, truth, prediction = oracle
    before = [x.copy() for x in oracle]
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    arguments = {"baselines": ("control_mean",), "metrics": ("mse", "delta_pearson", "top_k_overlap"), "top_k": 2}
    expected = evaluator.evaluate(truth, {"model": prediction}, train=train, **arguments)
    result = evaluator.evaluate(
        truth,
        {"model": prediction[:, [2, 0, 3, 1]].copy()},
        train=train[:, [3, 1, 2, 0]].copy(),
        **arguments,
    )
    pd.testing.assert_frame_equal(result, expected)
    for original, snapshot in zip(oracle, before, strict=True):
        np.testing.assert_array_equal(original.X, snapshot.X)
        pd.testing.assert_frame_equal(original.obs, snapshot.obs)
        pd.testing.assert_frame_equal(original.var, snapshot.var)
        assert original.uns == snapshot.uns


@pytest.mark.parametrize("defect", ["missing", "extra", "duplicate"])
def test_feature_id_errors_are_not_silent(oracle, defect):
    train, truth, prediction = oracle
    if defect == "missing":
        prediction = prediction[:, :-1].copy()
    elif defect == "extra":
        prediction = dataset([[11, 102, 1000, 10003, 7]], ["A"], "extra")
    else:
        prediction.var_names = ["g1", "g1", "g3", "g4"]
    with pytest.raises(ValueError):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {"model": prediction}, train=train, baselines=(), metrics=("mse",)
        )


def test_test_control_cannot_change_reference_or_baselines(oracle):
    train, truth, prediction = oracle
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    first = evaluator.evaluate(truth, {"model": prediction}, train=train, metrics=("mse", "delta_pearson"))
    truth_with_control = dataset(
        [[12, 99, 1000, 10003], [10000, -2000, 55, 4]], ["A", "ctrl"], "different_test"
    )
    second = evaluator.evaluate(
        truth_with_control, {"model": prediction}, train=train, metrics=("mse", "delta_pearson")
    )
    pd.testing.assert_frame_equal(first, second)
    assert set(second.reference_source) == {"train_control"}


def test_additive_uses_each_training_component_mean_equally():
    train = dataset(
        [[10, 20], [12, 19], [12, 19], [12, 19], [9, 23]],
        ["ctrl", "A", "A", "A", "B"],
        "train",
    )
    truth = dataset([[11, 22]], ["A+B"], "test")
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    result = evaluator.evaluate(truth, {}, train=train, components={"A+B": ["A", "B"]}, metrics=("mse",))
    assert row(result, "mse", model="baseline:additive", perturbation="A+B").value == pytest.approx(0)
    assert row(result, "mse", model="baseline:control_mean", perturbation="A+B").value == pytest.approx(2.5)
    assert set(result.n_predicted) == {1}


def test_missing_component_remains_unavailable_despite_test_single():
    train = dataset([[10, 20], [12, 19]], ["ctrl", "A"], "train")
    truth = dataset([[11, 22], [9, 23]], ["A+B", "B"], "test")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {}, train=train, components={"A+B": ["A", "B"]}, metrics=("mse",)
    )
    actual = row(result, "mse", model="baseline:additive", perturbation="A+B")
    assert actual.status != "ok"
    assert np.isnan(actual.value)
    assert actual.n_predicted == 0
    assert len(result) == 4


def test_heldout_context_does_not_use_its_test_controls():
    train = dataset([[10, 20], [12, 19]], ["ctrl", "A"], "train", contexts=["C1", "C1"])
    truth = dataset([[30, 40], [31, 42]], ["ctrl", "A"], "test", contexts=["C2", "C2"])
    pred = dataset([[31, 42]], ["A"], "pred", contexts=["C2"])
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl", context_key="context")
    result = evaluator.evaluate(truth, {"model": pred}, train=train, metrics=("mse", "delta_pearson"))
    assert row(result, "mse").value == pytest.approx(0)
    assert row(result, "delta_pearson").status != "ok"
    assert np.isnan(row(result, "delta_pearson").value)
    assert set(result.reference_source) == {"unavailable"}
    unavailable = result[result.model.str.startswith("baseline:")]
    assert unavailable.value.isna().all()
    assert (unavailable.status != "ok").all()


def test_combination_aliases_are_one_holdout_unit():
    data = dataset([[0], [1], [2], [3], [4]], ["ctrl", "A", "B", "A+B", "B+A"], "cell")
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    kwargs = {"strategy": "combination", "holdout": ["A+B"], "components": {"A+B": ["A", "B"], "B+A": ["B", "A"]}}
    train, test = evaluator.split(data, **kwargs)
    assert set(train.obs.perturbation) == {"ctrl", "A", "B"}
    assert set(test.obs.perturbation) == {"A+B", "B+A"}
    assert not set(train.obs_names) & set(test.obs_names)
    again_train, again_test = evaluator.split(data[::-1].copy(), **kwargs)
    assert set(train.obs_names) == set(again_train.obs_names)
    assert set(test.obs_names) == set(again_test.obs_names)
    manifest = test.uns["pertpy_evaluation_split"]
    assert set(manifest["test_obs_names"]) == set(test.obs_names)
    assert set(manifest["train_obs_names"]) == set(train.obs_names)
    assert "pertpy_evaluation_split" not in data.uns


def test_perturbation_split_spans_contexts_and_context_split_includes_control():
    data = dataset(
        [[0], [1], [2], [3], [4], [5]],
        ["ctrl", "A", "B", "ctrl", "A", "B"],
        "cell",
        contexts=["C1", "C1", "C1", "C2", "C2", "C2"],
    )
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl", context_key="context")
    train, test = evaluator.split(data, holdout=["A"])
    assert set(test.obs.perturbation) == {"A"}
    assert set(test.obs.context) == {"C1", "C2"}
    assert "A" not in set(train.obs.perturbation)
    train, test = evaluator.split(data, holdout=["C2"], strategy="context")
    assert set(test.obs.context) == {"C2"}
    assert "ctrl" in set(test.obs.perturbation)
    assert set(train.obs.context) == {"C1"}


@pytest.mark.parametrize("container", [np.asarray, sparse.csr_matrix, sparse.csc_matrix])
def test_explicit_layer_and_sparse_equivalence(container):
    train = dataset([[10, 100, 1000, 10000]], ["ctrl"], "train", layer="measured")
    truth = dataset([[12, 99, 1000, 10003]], ["A"], "test", layer="measured")
    pred = dataset([[11, 102, 1000, 10003]], ["A"], "pred", layer="measured")
    for data in (train, truth, pred):
        data.layers["measured"] = container(data.layers["measured"])
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl", layer_key="measured").evaluate(
        truth, {"model": pred}, train=train, baselines=(), metrics=("mse", "delta_pearson")
    )
    assert row(result, "mse").value == pytest.approx(2.5)
    assert row(result, "delta_pearson").value == pytest.approx(3 / sqrt(50))
    assert result.attrs["pertpy_evaluation"]["layer_key"] == "measured"
    np.testing.assert_array_equal(truth.X, np.full((1, 4), 9000.0))


def test_missing_selected_layer_does_not_fall_back(oracle):
    train, truth, pred = oracle
    train.layers["measured"] = train.X.copy()
    truth.layers["measured"] = truth.X.copy()
    with pytest.raises(ValueError, match="layer"):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl", layer_key="measured").evaluate(
            truth, {"model": pred}, train=train, baselines=(), metrics=("mse",)
        )


def test_missing_prediction_group_preserves_coverage():
    train = dataset([[0, 0]], ["ctrl"], "train")
    truth = dataset([[1, 0], [20, 0]], ["A", "B"], "test")
    partial = dataset([[1, 0]], ["A"], "partial")
    complete = dataset([[1, 0], [20, 0]], ["A", "B"], "complete")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {"partial": partial, "complete": complete}, train=train, baselines=(), metrics=("mse",)
    )
    assert len(result) == 4
    missing = row(result, "mse", model="partial", perturbation="B")
    assert missing.status != "ok"
    assert np.isnan(missing.value)
    assert missing.n_predicted == 0


def test_existing_direct_edistance_is_not_clamped():
    train = dataset([[0]], ["ctrl"], "train")
    truth = dataset([[0], [2]], ["A", "A"], "test")
    pred = dataset([[0], [2]], ["A", "A"], "pred")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {"model": pred}, train=train, baselines=(), metrics=("edistance", "mse")
    )
    assert row(result, "edistance").value == pytest.approx(-2.0)
    assert row(result, "mse").value == pytest.approx(0)


def test_mean_prediction_preserves_singleton_count():
    train = dataset([[0]], ["ctrl"], "train")
    truth = dataset([[-1], [1]], ["A", "A"], "test")
    pred = dataset([[0]], ["A"], "pred")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {"model": pred}, train=train, baselines=("control_mean",), metrics=("mse", "edistance")
    )
    assert set(result.n_predicted) == {1}
    assert set(result.n_true) == {2}
    assert set(result.value) == {0.0}


def test_zero_and_constant_effects_are_undefined():
    train = dataset([[10, 20, 30]], ["ctrl"], "train")
    truth = dataset([[10, 20, 30]], ["A"], "test")
    pred = dataset([[10, 20, 30]], ["A"], "pred")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth,
        {"model": pred},
        train=train,
        baselines=(),
        metrics=("delta_pearson", "direction_accuracy", "top_k_overlap"),
    )
    assert result.value.isna().all()
    assert (result.status != "ok").all()


def test_top_k_ties_use_ids_and_capping_is_explicit():
    train = dataset([[0, 0, 0]], ["ctrl"], "train", features=["z", "a", "b"])
    truth = dataset([[1, 1, 0]], ["A"], "test", features=["z", "a", "b"])
    pred = dataset([[0, 1, 1]], ["A"], "pred", features=["z", "a", "b"])
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    result = evaluator.evaluate(truth, {"model": pred}, train=train, baselines=(), metrics=("top_k_overlap",), top_k=1)
    reordered = evaluator.evaluate(
        truth[:, ::-1].copy(), {"model": pred[:, ::-1].copy()}, train=train[:, ::-1].copy(),
        baselines=(), metrics=("top_k_overlap",), top_k=1,
    )
    assert row(result, "top_k_overlap").value == 1
    assert row(reordered, "top_k_overlap").value == 1
    capped = evaluator.evaluate(truth, {"model": pred}, train=train, baselines=(), metrics=("top_k_overlap",), top_k=20)
    assert row(capped, "top_k_overlap").value == 1
    assert row(capped, "top_k_overlap").n_features == 3
    assert capped.attrs["pertpy_evaluation"]["top_k"] == 20


def test_selected_scope_single_feature_pearson_and_provenance(oracle):
    train, truth, pred = oracle
    supplied = {("A",): ["g2"]}
    original = copy.deepcopy(supplied)
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {"model": pred}, train=train, baselines=(), metrics=("delta_pearson", "mse"), feature_sets=supplied
    )
    assert row(result, "mse", scope="selected").value == pytest.approx(9)
    assert row(result, "delta_pearson", scope="selected").status != "ok"
    assert np.isnan(row(result, "delta_pearson", scope="selected").value)
    assert supplied == original
    # Selected IDs must be recoverable, not merely a count or the full universe.
    metadata = result.attrs["pertpy_evaluation"]
    assert "feature_sets" in metadata
    serializable = json.dumps(metadata["feature_sets"])
    assert "g2" in serializable
    assert "g1" not in serializable


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("as_sparse", [False, True])
def test_nonfinite_measurement_rejected(oracle, bad, as_sparse):
    train, truth, pred = oracle
    values = np.asarray(pred.X, dtype=float)
    values[0, 0] = bad
    pred.X = sparse.csr_matrix(values) if as_sparse else values
    with pytest.raises(ValueError, match="finite"):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {"model": pred}, train=train, baselines=(), metrics=("mse",)
        )


def test_finite_float32_training_means_do_not_overflow():
    # Both arithmetic means are representable in float64; float32 reduction
    # overflows before division if the training matrix is not promoted first.
    train = dataset(np.array([[3e38, 0], [3e38, 0]], dtype=np.float32), ["ctrl", "ctrl"], "train")
    truth = dataset(np.array([[3e38, 0]], dtype=np.float32), ["A"], "test")
    result = pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
        truth, {}, train=train, baselines=("control_mean",), metrics=("mse",)
    )
    actual = row(result, "mse", model="baseline:control_mean")
    assert actual.status == "ok"
    assert actual.value == pytest.approx(0)


def test_arithmetic_overflow_never_becomes_successful_nonfinite_score():
    train = dataset([[0.0, 0.0]], ["ctrl"], "train")
    truth = dataset([[1e308, -1e308]], ["A"], "test")
    pred = dataset([[-1e308, 1e308]], ["A"], "pred")
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    try:
        result = evaluator.evaluate(
            truth, {"model": pred}, train=train, baselines=(), metrics=("mse", "delta_pearson")
        )
    except (ValueError, FloatingPointError):
        return  # Explicit documented rejection is an acceptable fail-closed policy.
    assert np.isfinite(result.loc[result.status == "ok", "value"]).all()
    assert not np.isinf(result.value.to_numpy()).any()
    assert result.loc[result.status != "ok", "value"].isna().all()


def test_overflowed_reference_cannot_produce_successful_direction_or_ranking():
    train = dataset([[1e308, 0.0], [1e308, 0.0]], ["ctrl", "ctrl"], "train")
    truth = dataset([[1e308, 1.0]], ["A"], "test")
    pred = dataset([[1e308, 2.0]], ["A"], "pred")
    evaluator = pt.tl.PerturbationEvaluator("perturbation", "ctrl")
    try:
        result = evaluator.evaluate(
            truth, {"model": pred}, train=train, baselines=(),
            metrics=("delta_pearson", "direction_accuracy", "top_k_overlap"), top_k=1,
        )
    except (ValueError, FloatingPointError):
        return
    # An implementation may calculate the representable mean stably or decline
    # the overflow. It must not rank an artificial -inf first-gene change.
    successful = result[result.status == "ok"]
    if len(successful):
        assert row(result, "delta_pearson").status == "ok"
        assert row(result, "delta_pearson").value == pytest.approx(1.0)
        assert row(result, "direction_accuracy").value == pytest.approx(1.0)
        assert row(result, "top_k_overlap").value == pytest.approx(1.0)
    else:
        assert result.value.isna().all()


def test_self_referential_component_is_invalid():
    train = dataset([[0, 0], [1, 0], [2, 0]], ["ctrl", "A", "A+B"], "train")
    truth = dataset([[3, 0]], ["A+B"], "test")
    with pytest.raises(ValueError):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {}, train=train, components={"A+B": ["A+B", "A"]}, metrics=("mse",)
        )


def test_declared_train_test_id_overlap_is_rejected(oracle):
    train, truth, pred = oracle
    truth.obs_names = train.obs_names.copy()
    with pytest.raises(ValueError, match="overlap"):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {"model": pred}, train=train, baselines=(), metrics=("mse",)
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k_rejected(oracle, top_k):
    train, truth, pred = oracle
    with pytest.raises(ValueError):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {"model": pred}, train=train, top_k=top_k, metrics=("mse",)
        )


def test_reserved_model_prefix_is_rejected(oracle):
    train, truth, pred = oracle
    with pytest.raises(ValueError):
        pt.tl.PerturbationEvaluator("perturbation", "ctrl").evaluate(
            truth, {"baseline:additive": pred}, train=train, metrics=("mse",)
        )
