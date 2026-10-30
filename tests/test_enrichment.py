import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gastric_repro.enrichment import GO_ONTOLOGIES, ranked_gene_list, run_go_prerank


@pytest.fixture
def results():
    return pd.DataFrame(
        {"t": np.linspace(5, -5, 30), "is_deg": [True] * 2 + [False] * 28},
        index=[f"G{i:02}" for i in range(30)],
    )


@pytest.fixture
def gmt_files(tmp_path):
    paths = {}
    for ontology in GO_ONTOLOGIES:
        path = tmp_path / f"{ontology}.gmt"
        path.write_text(
            "\n".join(
                f"{ontology}_{name}\tsnapshot\t" + "\t".join(f"G{i:02}" for i in indices)
                for name, indices in (
                    ("up", range(0, 8)),
                    ("down", range(22, 30)),
                    ("mixed", range(0, 30, 3)),
                )
            ) + "\n", encoding="utf-8"
        )
        paths[ontology] = path
    return paths


def test_ranking_uses_entire_universe_and_signed_t(results):
    ranks = ranked_gene_list(results.iloc[::-1])
    assert len(ranks) == 30
    assert ranks.gene.tolist() == results.index.tolist()
    assert ranks.score.iloc[0] == 5
    assert ranks.score.iloc[-1] == -5
    tied = ranked_gene_list(pd.DataFrame({"t": [1, 1, -1]}, index=["b", "a", "c"]))
    assert tied.gene.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame(),
        pd.DataFrame({"t": [np.nan]}, index=["a"]),
        pd.DataFrame({"t": [np.inf]}, index=["a"]),
        pd.DataFrame({"t": [1, 2]}, index=["a", "a"]),
        pd.DataFrame({"t": [1, 2]}, index=[1, "1"]),
        pd.DataFrame({"t": [1]}, index=[""]),
    ],
)
def test_ranking_validation(frame):
    with pytest.raises(ValueError):
        ranked_gene_list(frame)


def test_wrapper_signature_provenance_and_all_ontologies(results, gmt_files, tmp_path, monkeypatch):
    import gseapy

    calls = []

    def fake_prerank(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(res2d=pd.DataFrame({"Term": ["mock"], "NES": [1.2], "FDR q-val": [0.04]}))

    monkeypatch.setattr(gseapy, "prerank", fake_prerank)
    output = tmp_path / "output"
    actual = run_go_prerank(
        results, gmt_files, output, seed=7, permutations=32, min_size=3,
        source_versions=dict.fromkeys(GO_ONTOLOGIES, "test-2026"),
    )
    assert actual.ontology.tolist() == list(GO_ONTOLOGIES)
    assert len(calls) == 3
    for ontology, call in zip(GO_ONTOLOGIES, calls, strict=True):
        assert len(call["rnk"]) == len(results)
        assert call["gene_sets"] == str(gmt_files[ontology].resolve())
        assert call["seed"] == 7
        assert call["permutation_num"] == 32
        assert call["threads"] == 1
        assert call["ascending"] is False
        assert call["no_plot"] is True
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["universe_size"] == 30
    assert provenance["sources"]["BP"]["version"] == "test-2026"
    assert provenance["sources"]["BP"]["gmt_gene_count"] == 21
    assert provenance["sources"]["BP"]["ranked_gene_overlap_count"] == 21
    assert provenance["sources"]["BP"]["ranked_gene_overlap_fraction"] == pytest.approx(21 / 30)
    assert provenance["sources"]["BP"]["sha256"] == hashlib.sha256(gmt_files["BP"].read_bytes()).hexdigest()
    assert provenance["rank_sha256"] == hashlib.sha256((output / "ranked_genes.tsv").read_bytes()).hexdigest()
    assert (output / "go_prerank.tsv").exists()
    assert actual.attrs["provenance"] == provenance


@pytest.mark.parametrize(
    "gmt_symbols,expected_count",
    [(["G00", "G00", "external"], 1), (["g00", "external"], 0),
     ([f"G{i:02}" for i in range(30)], 30)],
)
def test_overlap_is_exact_unique_and_before_size_filtering(
    results, gmt_files, tmp_path, monkeypatch, gmt_symbols, expected_count
):
    import gseapy

    monkeypatch.setattr(
        gseapy, "prerank",
        lambda **kwargs: SimpleNamespace(res2d=pd.DataFrame({"Term": ["mock"]})),
    )
    for path in gmt_files.values():
        path.write_text("test\tsnapshot\t" + "\t".join(gmt_symbols) + "\n")
    output = tmp_path / "overlap"
    run_go_prerank(results, gmt_files, output, min_size=100, max_size=200)
    provenance = json.loads((output / "provenance.json").read_text())
    for source in provenance["sources"].values():
        assert source["gmt_gene_count"] == len(set(gmt_symbols))
        assert source["ranked_gene_overlap_count"] == expected_count
        assert source["ranked_gene_overlap_fraction"] == pytest.approx(expected_count / 30)


def test_real_local_gseapy_is_reproducible(results, gmt_files, tmp_path):
    first = run_go_prerank(results, gmt_files, tmp_path / "first", seed=17, permutations=32, min_size=3)
    second = run_go_prerank(results, gmt_files, tmp_path / "second", seed=17, permutations=32, min_size=3)
    assert set(first.ontology) == set(GO_ONTOLOGIES)
    assert len(first) == 9
    pd.testing.assert_frame_equal(first, second)
    for ontology in GO_ONTOLOGIES:
        top = first[(first.ontology == ontology) & (first.Term == f"{ontology}_up")].iloc[0]
        assert float(top.NES) > 0
        bottom = first[(first.ontology == ontology) & (first.Term == f"{ontology}_down")].iloc[0]
        assert float(bottom.NES) < 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": 2**32},
        {"permutations": 0},
        {"permutations": True},
        {"min_size": 10, "max_size": 5},
        {"threads": 0},
        {"source_versions": {"BP": "release"}},
    ],
)
def test_parameter_validation(results, gmt_files, tmp_path, kwargs):
    with pytest.raises(ValueError):
        run_go_prerank(results, gmt_files, tmp_path / "out", **kwargs)


def test_requires_local_gmt_for_all_ontologies(results, gmt_files, tmp_path):
    with pytest.raises(ValueError, match="exactly"):
        run_go_prerank(results, {"BP": gmt_files["BP"]}, tmp_path / "out")
    with pytest.raises(ValueError, match="does not exist"):
        run_go_prerank(results, dict.fromkeys(GO_ONTOLOGIES, "GO_Biological_Process_2025"), tmp_path / "out")
    gmt_files["BP"].write_text("badrecord\n")
    with pytest.raises(ValueError, match="malformed"):
        run_go_prerank(results, gmt_files, tmp_path / "out")


def test_reject_duplicate_gmt_terms(results, gmt_files, tmp_path):
    gmt_files["BP"].write_text("term\tx\tG01\nterm\tx\tG02\n")
    with pytest.raises(ValueError, match="duplicate GMT"):
        run_go_prerank(results, gmt_files, tmp_path / "out")
