import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from IPython.display import Image, Markdown


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "lopes2025_reproduction_standalone.ipynb"
REFERENCE = NOTEBOOK.parents[1] / "reference"


def cell_source(marker):
    notebook = json.loads(NOTEBOOK.read_text())
    return next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )


@pytest.fixture
def heatmap_namespace(tmp_path):
    manifest = pd.read_csv(REFERENCE / "acquisition" / "GSE64951" / "manifest.csv")
    included = manifest.loc[manifest["included"].astype(str).str.lower() == "true"]
    labels = included.set_index("gsm")["label"].astype(int)
    degs = pd.read_csv(REFERENCE / "computed" / "GSE64951" / "degs.csv", index_col="gene")
    outputs = []
    return {
        "REFERENCE": REFERENCE,
        "RESULTS": tmp_path,
        "OUTPUT": tmp_path,
        "FIGURES": tmp_path,
        "expression_path": tmp_path / "GSE64951" / "expression.csv",
        "saliva_labels": labels,
        "saliva_degs": degs,
        "PUBLISHED_SALIVA": set(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "json": json,
        "hashlib": hashlib,
        "sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
        "Markdown": Markdown,
        "Image": Image,
        "display": outputs.append,
        "outputs": outputs,
    }


def test_offline_heatmap_uses_verified_individual_sample_snapshot(heatmap_namespace):
    ns = heatmap_namespace
    exec(cell_source("heatmap_path ="), ns)
    saved = REFERENCE / "computed" / "GSE64951" / "heatmap.png"
    assert ns["heatmap_path"].read_bytes() == saved.read_bytes()
    assert ns["outputs"][1].data == saved.read_bytes()
    assert ns["heatmap_provenance"]["samples"] == 94
    assert ns["heatmap_provenance"]["controls"] == 31
    assert ns["heatmap_provenance"]["cancer"] == 63
    assert ns["heatmap_provenance"]["genes"] == 37
    assert "not recomputed" in ns["heatmap_provenance"]["source"]
    assert "z_values" not in ns


def test_local_heatmap_has_all_samples_continuous_colors_and_group_labels(heatmap_namespace, monkeypatch):
    ns = heatmap_namespace
    ns["expression_path"].parent.mkdir()
    ns["expression_path"].touch()
    rng = np.random.default_rng(2025)
    expression = pd.DataFrame(
        rng.normal(size=(37, 94)),
        index=ns["saliva_degs"].index,
        columns=ns["saliva_labels"].index[::-1],
    )
    expression.iloc[0] = 5.0
    expression["excluded"] = 100.0
    ns["saliva_filtered"] = expression
    monkeypatch.setattr(plt, "show", lambda: None)
    try:
        exec(cell_source("heatmap_path ="), ns)
        image_values = np.asarray(ns["image"].get_array())
        assert image_values.shape == (37, 94)
        order = ns["saliva_labels"].sort_values(kind="stable").index
        expected = expression.loc[:, order].to_numpy()
        scale = expected.std(axis=1, keepdims=True)
        expected = (expected - expected.mean(axis=1, keepdims=True)) / np.where(scale > 0, scale, 1)
        np.testing.assert_allclose(image_values, expected)
        np.testing.assert_allclose(image_values[0], 0)
        assert len(np.unique(image_values)) > 2
        tick_labels = [tick.get_text() for tick in ns["ax"].get_xticklabels()]
        assert len(tick_labels) == 94
        assert tick_labels == ns["column_labels"]
        assert all("(Control)" in label for label in tick_labels[:31])
        assert all("(Cancer)" in label for label in tick_labels[31:])
        assert ns["image"].get_clim() == (-3, 3)
        assert ns["heatmap_provenance"]["samples"] == 94
        assert "computed from local" in ns["heatmap_provenance"]["source"]
    finally:
        plt.close("all")


@pytest.mark.parametrize("mutation", ["image", "labels", "genes"])
def test_offline_heatmap_rejects_inconsistent_sources(heatmap_namespace, tmp_path, mutation):
    ns = heatmap_namespace
    if mutation == "image":
        ns["REFERENCE"] = tmp_path / "reference"
        snapshot = ns["REFERENCE"] / "computed" / "GSE64951"
        snapshot.mkdir(parents=True)
        for name in ("summary.json", "degs.csv", "manifest.csv", "heatmap.png"):
            (snapshot / name).write_bytes((REFERENCE / "computed" / "GSE64951" / name).read_bytes())
        (snapshot / "heatmap.png").write_bytes(b"changed image")
        message = "SHA256 changed"
    elif mutation == "labels":
        ns["saliva_labels"].iloc[0] = 1 - ns["saliva_labels"].iloc[0]
        message = "sample IDs or group labels differ"
    else:
        ns["saliva_degs"] = ns["saliva_degs"].iloc[:-1]
        message = "dimensions differ"
    with pytest.raises(ValueError, match=message):
        exec(cell_source("heatmap_path ="), ns)
    assert not (tmp_path / "figure_3c_saliva_heatmap.png").exists()


@pytest.mark.parametrize("mutation", ["empty", "nonfinite"])
def test_heatmap_rejects_unplottable_expression(heatmap_namespace, mutation):
    ns = heatmap_namespace
    if mutation == "empty":
        ns["saliva_degs"] = ns["saliva_degs"].iloc[:0]
        message = "at least one DEG"
    else:
        ns["expression_path"].parent.mkdir()
        ns["expression_path"].touch()
        ns["saliva_filtered"] = pd.DataFrame(
            np.nan, index=ns["saliva_degs"].index, columns=ns["saliva_labels"].index
        )
        message = "finite sample expression"
    with pytest.raises(ValueError, match=message):
        exec(cell_source("heatmap_path ="), ns)
    assert not (ns["FIGURES"] / "figure_3c_saliva_heatmap.png").exists()


@pytest.mark.parametrize("local_expression", [False, True])
def test_original_volcano_preserves_full_universe_snapshot(heatmap_namespace, local_expression):
    ns = heatmap_namespace
    if local_expression:
        ns["expression_path"].parent.mkdir()
        ns["expression_path"].touch()
    exec(cell_source("original_volcano_path ="), ns)
    saved = REFERENCE / "computed" / "GSE64951" / "volcano.png"
    assert ns["original_volcano_path"].read_bytes() == saved.read_bytes()
    assert ns["outputs"][1].data == saved.read_bytes()
    assert "20,527 genes; not recomputed" in ns["outputs"][0].data
    assert ns["original_volcano_path"].name == "figure_3d_saliva_volcano_original.png"


def test_original_volcano_rejects_changed_snapshot(heatmap_namespace, tmp_path):
    ns = heatmap_namespace
    ns["REFERENCE"] = tmp_path / "reference"
    snapshot = ns["REFERENCE"] / "computed" / "GSE64951"
    snapshot.mkdir(parents=True)
    (snapshot / "summary.json").write_bytes(
        (REFERENCE / "computed" / "GSE64951" / "summary.json").read_bytes()
    )
    (snapshot / "volcano.png").write_bytes(b"changed image")
    with pytest.raises(ValueError, match="original saliva volcano SHA256 changed"):
        exec(cell_source("original_volcano_path ="), ns)
    assert not ns["original_volcano_path"].exists()
    assert not ns["outputs"]


@pytest.mark.parametrize("missing", [False, True])
def test_saliva_de_requires_and_selects_every_included_sample(heatmap_namespace, missing):
    ns = heatmap_namespace
    expression = pd.DataFrame(
        [np.arange(95)], index=["GENE"], columns=[*ns["saliva_labels"].index[::-1], "excluded"]
    )
    if missing:
        expression = expression.drop(columns=ns["saliva_labels"].index[0])
    ns["expression_path"].parent.mkdir()
    expression.to_csv(ns["expression_path"])
    ns["filter_expression"] = lambda frame: frame
    calls = []

    def differential_expression(frame, labels):
        calls.append((frame.copy(), labels.copy()))
        return ns["saliva_degs"]

    ns["differential_expression"] = differential_expression
    if missing:
        with pytest.raises(ValueError, match="missing included samples"):
            exec(cell_source("saliva_manifest ="), ns)
        assert not calls
    else:
        exec(cell_source("saliva_manifest ="), ns)
        frame, labels = calls[0]
        assert frame.shape[1] == 94
        pd.testing.assert_index_equal(frame.columns, labels.index, check_names=False)
        assert "excluded" not in frame.columns


def test_standalone_notebook_is_documented_and_self_contained():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["metadata"]["language_info"]["version"].startswith("3.14")
    assert sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) >= 6
    source = "\n".join(
        line
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for line in cell["source"]
    )
    assert "gastric_repro" not in source
    assert "subprocess" not in source
    assert "figure_3c_saliva_heatmap.png" in source
    assert "figure_3d_saliva_volcano.png" in source
    assert "differential_expression" in source
    assert "read_gmt" in source
    assert "bootstrap_auc" in source
    assert "LogisticRegression" in source
