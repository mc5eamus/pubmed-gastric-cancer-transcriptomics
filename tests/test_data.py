"""Offline tests for strict labels, atomic caches and auditable preparation."""

import gzip
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd
import pytest
import requests

from gastric_repro import data


@pytest.fixture
def workspace():
    path = Path(__file__).resolve().parents[1] / ".pytest_cache" / f"data-{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def sample(characteristics, *, title="sample", source="gastric tissue", processing="RMA"):
    return {
        "geo_accession": ["GSM1"], "title": [title], "source_name_ch1": [source],
        "characteristics_ch1": characteristics, "platform_id": ["GPL570"],
        "organism_ch1": ["Homo sapiens"], "data_processing": [processing],
        "supplementary_file": ["ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/GSM1/suppl/GSM1.CEL.gz"],
    }


def soft(samples):
    return "".join(
        f"^SAMPLE = {gsm}\n"
        + "".join(f"!Sample_{key} = {value}\n" for key, values in fields.items() for value in values)
        for gsm, fields in samples.items()
    )


@pytest.mark.parametrize(
    "accession,characteristics,expected",
    [
        ("GSE66229", ["tissue: Gastric tumor", "patient: 107"], 1),
        ("GSE66229", ["tissue: normal gastric tissue"], 0),
        ("GSE54129", ["tissue: gastric cancer tumor"], 1),
        ("GSE54129", ["tissue: normal"], 0),
        ("GSE13911", ["MSI", "TUMOR"], 1),
        ("GSE13911", ["--", "NORMAL"], 0),
        ("GSE64951", ["disease state: gastric cancer", "rna source: cell-free saliva supernatant"], 1),
        ("GSE64951", ["disease state: normal"], 0),
    ],
)
def test_explicit_real_geo_labels(accession, characteristics, expected):
    label, included, reason, kind = data.classify_sample(accession, sample(characteristics))
    assert label == expected and included
    assert "explicit metadata" in reason


def test_normal_tissue_from_cancer_patient_is_not_tumor():
    fields = sample(
        ["tissue: normal"], title="normal_1",
        source="surgically resected from patient of gastric cancer",
    )
    assert data.classify_sample("GSE54129", fields)[0] == 0


@pytest.mark.parametrize(
    "disease", ["oral cancer", "lung cancer", "gastritis", "pancreatic cancer", "non-gastric cancer"]
)
def test_non_gc_disease_excluded_not_control(disease):
    label, included, reason, kind = data.classify_sample("GSE64951", sample([f"disease state: {disease}"]))
    assert label is None and not included and kind == "other_disease"
    assert "excluded non-GC disease" in reason


@pytest.mark.parametrize(
    "characteristics",
    [
        ["tissue: normal", "tissue: gastric tumor"],
        ["disease state: normal and gastric cancer"],
        ["patient: 1", "MSI"],
        ["disease state: unknown"],
        ["disease state: cancer"],
    ],
)
def test_ambiguous_labels_fail(characteristics):
    with pytest.raises(data.MetadataError):
        data.classify_sample("GSE64951", sample(characteristics, source="Saliva"))


def test_manifest_records_all_samples_and_explicit_count_discrepancy():
    samples = {
        "GSM1": sample(["disease state: gastric cancer"]),
        "GSM2": sample(["disease state: normal"]),
        "GSM3": sample(["disease state: lung cancer"]),
        "GSM4": sample(["patient: unknown"]),
    }
    manifest, counts = data.build_manifest("GSE64951", samples)
    assert list(manifest["gsm"]) == list(samples)
    assert manifest["included"].tolist() == [True, True, False, False]
    assert manifest["label"].tolist()[:2] == [1, 0]
    assert manifest["label"].isna().tolist() == [False, False, True, True]
    assert counts["count_status"] == "explicit_discrepancy"
    assert counts["observed"] == {
        "tumor": 1, "normal": 1, "total": 2, "available_total": 4, "excluded": 2,
    }
    assert counts["ambiguous_gsms"] == ["GSM4"]
    assert "https://" in manifest.iloc[0]["cel_urls"]
    assert manifest["biospecimen"].unique().tolist() == ["cell-free saliva"]


def test_excluded_disease_does_not_inflate_included_count_discrepancy(monkeypatch):
    monkeypatch.setitem(data.DATASETS, "GSE64951", {
        "tumor": 1, "normal": 1, "total": 2, "biospecimen": "cell-free saliva",
    })
    samples = {
        "GSM1": sample(["disease state: gastric cancer"]),
        "GSM2": sample(["disease state: normal"]),
        "GSM3": sample(["disease state: oral cancer"]),
    }
    manifest, counts = data.build_manifest("GSE64951", samples)
    assert len(manifest) == 3
    assert counts["observed"]["total"] == 2
    assert counts["observed"]["available_total"] == 3
    assert counts["observed"]["excluded"] == 1
    assert counts["count_status"] == "matches_expected"


def test_soft_preserves_repeated_values_and_checks_membership():
    fields = sample(["tissue: normal", "patient: 11"])
    parsed = data.parse_soft_samples(soft({"GSM1": fields}))
    assert parsed["GSM1"] == fields
    with pytest.raises(data.MetadataError, match="duplicate"):
        data.parse_soft_samples(soft({"GSM1": fields}) * 2)
    with pytest.raises(data.MetadataError, match="geo_accession"):
        data.parse_soft_samples(soft({"GSM2": fields}))
    with pytest.raises(data.MetadataError):
        data.parse_soft_samples("<html>error</html>")


class Response:
    def __init__(self, content=b"payload", url="https://example.org/input", declared=None):
        self.content = content
        self.url = url
        self.headers = {"Content-Length": str(len(content) if declared is None else declared)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content[:3]
        yield b""
        yield self.content[3:]


def test_atomic_download_and_verified_cached_hash(workspace, monkeypatch):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(data.requests, "get", get)
    path = workspace / "input"
    expected = hashlib.sha256(b"payload").hexdigest()
    assert data.download("https://example.org/input", path, expected_sha256=expected) == path
    assert path.read_bytes() == b"payload"
    record = json.loads(path.with_name("input.sha256.json").read_text())
    assert record["hash_policy"] == "independent_expected_sha256"
    assert record["sha256"] == expected
    data.download("https://example.org/input", path)
    assert len(calls) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(data.CacheIntegrityError):
        data.download("https://example.org/input", path)
    assert not list(workspace.glob("*.part"))


def test_cache_requires_sidecar_and_same_source(workspace, monkeypatch):
    path = workspace / "input"
    path.write_text("unverified")
    with pytest.raises(data.CacheIntegrityError):
        data.download("https://example.org/input", path)
    path.unlink()
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response())
    data.download("https://example.org/input", path)
    with pytest.raises(data.CacheIntegrityError):
        data.download("https://example.org/different", path)


def test_failed_downloads_do_not_promote_partial_files(workspace, monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response(declared=100))
    path = workspace / "input"
    with pytest.raises(data.AcquisitionError, match="Incomplete"):
        data.download("https://example.org/input", path, retries=2)
    assert list(workspace.iterdir()) == []
    monkeypatch.setattr(data.requests, "get", lambda *args, **kwargs: Response())
    with pytest.raises(data.CacheIntegrityError, match="expected hash"):
        data.download("https://example.org/input", path, expected_sha256="0" * 64)
    assert list(workspace.iterdir()) == []


def test_download_retries_network_errors_and_forbids_http(workspace, monkeypatch):
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("transient")
        return Response()
    monkeypatch.setattr(data.requests, "get", get)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    path = workspace / "input"
    data.download("https://example.org/input", path)
    assert len(calls) == 2
    with pytest.raises(data.AcquisitionError, match="HTTPS"):
        data.download("http://example.org/input", workspace / "unsafe")


def test_probe_annotation_and_hgnc_aliases_highest_mean():
    annotation = data.parse_probe_annotation(
        "^Annotation\n!platform_table_begin\nID\tGene symbol\n"
        "p1\tMETTL7A\np2\tTMT1A\np3\tSEPT10\np4\tFIGF\np5\tA///B\n"
        "AFFX-control\tCTRL\np6\t---\n!platform_table_end\n"
    )
    assert set(annotation) == {"p1", "p2", "p3", "p4"}
    hgnc = (
        "symbol\tstatus\talias_symbol\tprev_symbol\n"
        "TMT1A\tApproved\tMYSTERY\tMETTL7A\n"
        "SEPTIN10\tApproved\tMYSTERY\tSEPT10\n"
        "VEGFD\tApproved\tFIGF\t\n"
    )
    aliases = data.hgnc_aliases(hgnc)
    assert "MYSTERY" not in aliases
    assert aliases["METTL7A"] == "TMT1A"
    assert aliases["TMT1A"] == "TMT1A"
    expression = pd.DataFrame(
        [[1., 3.], [5., 7.], [4., 6.], [2., 4.]],
        index=["p1", "p2", "p3", "p4"], columns=["GSM1", "GSM2"],
    )
    genes, mapping = data.collapse_probes(expression, annotation, aliases)
    assert set(genes.index) == {"TMT1A", "SEPTIN10", "VEGFD"}
    np.testing.assert_array_equal(genes.loc["TMT1A"], [5., 7.])
    assert mapping.loc[mapping["probe"] == "p1", "selected"].tolist() == [False]
    assert data.canonical_gene("METTL7A") == "TMT1A"
    assert data.canonical_gene("SEPT10") == "SEPTIN10"
    assert data.canonical_gene("FIGF") == "VEGFD"


def test_probe_tie_break_is_deterministic():
    expression = pd.DataFrame([[2., 4.], [3., 3.]], index=["z", "a"])
    genes, mapping = data.collapse_probes(expression, {"z": "A", "a": "A"})
    assert genes.iloc[0].tolist() == [3., 3.]
    assert mapping.loc[mapping["selected"], "probe"].tolist() == ["a"]


@pytest.mark.parametrize(
    "accession,processing,expected",
    [
        ("GSE66229", "RMA using APT", np.array([[1., 2.]]) * np.log2(10)),
        ("GSE54129", "Robust Multichip Analysis (RMA)", np.array([[1., 2.]])),
        ("GSE64951", "Affymetrix Expression Console", np.array([[0., 1.]])),
        ("GSE13911", "MAS 5.0 global normalization", np.array([[0., 1.]])),
    ],
)
def test_processed_fallback_explicit_scale(accession, processing, expected):
    expression = pd.DataFrame([[1., 2.]], index=["p"], columns=["GSM1", "GSM2"])
    actual, description = data.series_matrix_to_log2(accession, expression, {"GSM1": sample([], processing=processing)})
    np.testing.assert_allclose(actual.to_numpy(), expected)
    assert description
    with pytest.raises(data.MetadataError):
        data.series_matrix_to_log2(accession, expression, {"GSM1": sample([], processing="unknown")})


def test_read_series_matrix_gzip_and_strict_values(workspace):
    text = (
        '!Sample_geo_accession\t"GSM1"\t"GSM2"\n'
        '!Sample_data_processing\t"RMA"\t"RMA"\n'
        '!series_matrix_table_begin\n"ID_REF"\t"GSM1"\t"GSM2"\n"p1"\t1.5\t2.5\n'
        "!series_matrix_table_end\n"
    )
    path = workspace / "matrix.txt.gz"
    path.write_bytes(gzip.compress(text.encode()))
    expression, metadata = data.read_series_matrix(path)
    assert expression.loc["p1", "GSM2"] == 2.5
    assert metadata["geo_accession"] == [["GSM1", "GSM2"]]
    path.write_bytes(gzip.compress(text.replace("1.5", "NaN").encode()))
    with pytest.raises(data.FormatError):
        data.read_series_matrix(path)


def configure_prepare(monkeypatch, *, raw_error=None, ambiguous=False):
    samples = {
        "GSM1": sample(["tissue: gastric tumor"]),
        "GSM2": sample(["tissue: normal"] if not ambiguous else ["patient: 2"]),
    }
    samples["GSM2"]["geo_accession"] = ["GSM2"]
    monkeypatch.setattr(data, "acquire_metadata", lambda *args: (samples, []))
    monkeypatch.setattr(data, "download", lambda url, destination, **kwargs: Path(destination))
    monkeypatch.setattr(data, "_artifact", lambda path, role: {"role": role, "sha256": "fixture"})
    monkeypatch.setattr(data, "read_cdf", lambda path: SimpleNamespace(shape=(1164, 1164), probesets=range(54675)))
    monkeypatch.setattr(data, "_read_text", lambda path: "")
    monkeypatch.setattr(data, "parse_probe_annotation", lambda text, aliases=None: {"p1": "METTL7A", "p2": "SEPT10"})
    monkeypatch.setattr(data, "hgnc_aliases", lambda text: data.GENE_ALIASES)
    monkeypatch.setattr(data, "acquire_cels", lambda manifest, cache, artifacts: {"GSM1": Path("a"), "GSM2": Path("b")})
    probes = pd.DataFrame([[3., 4.], [5., 6.]], index=["p1", "p2"], columns=["GSM1", "GSM2"])
    def normalize(*args):
        if raw_error is not None:
            raise raw_error
        return probes
    monkeypatch.setattr(data, "rma", normalize)
    monkeypatch.setattr(data, "_acquire_series_matrix", lambda *args: probes)
    return probes


def test_prepare_raw_writes_auditable_outputs(workspace, monkeypatch):
    configure_prepare(monkeypatch)
    output = workspace / "output"
    provenance = data.prepare_dataset("GSE54129", workspace / "cache", output)
    assert provenance["status"] == "complete"
    assert provenance["normalization"] == "raw_PM_RMA_python"
    assert provenance["raw_rma_parity_validated"] is False
    assert provenance["sample_counts"]["count_status"] == "explicit_discrepancy"
    expression = pd.read_csv(output / "expression.csv", index_col=0)
    assert expression.columns.tolist() == ["GSM1", "GSM2"]
    assert set(expression.index) == {"TMT1A", "SEPTIN10"}
    assert pd.read_csv(output / "manifest.csv")["included"].all()
    saved = json.loads((output / "provenance.json").read_text())
    assert saved == provenance
    for name, record in provenance["outputs"].items():
        assert record["sha256"] == data.sha256_file(output / name)


def test_prepare_requires_opt_in_and_records_fallback(workspace, monkeypatch):
    configure_prepare(monkeypatch, raw_error=data.FormatError("Calvin CEL not supported"))
    output = workspace / "output"
    with pytest.raises(data.AcquisitionError, match="allow_series_matrix=True"):
        data.prepare_dataset("GSE54129", workspace / "cache", output)
    assert json.loads((output / "provenance.json").read_text())["status"] == "failed"
    result = data.prepare_dataset("GSE54129", workspace / "cache", output, allow_series_matrix=True)
    assert result["normalization"] == "submitted_series_matrix_fallback"
    assert "Calvin" in result["fallback_reason"]
    assert "NOT raw RMA" in result["fallback_warning"]


def test_prepare_never_falls_back_on_bad_labels_or_cache_integrity(workspace, monkeypatch):
    configure_prepare(monkeypatch, ambiguous=True)
    output = workspace / "labels"
    with pytest.raises(data.MetadataError, match="Ambiguous"):
        data.prepare_dataset("GSE54129", workspace / "cache", output, allow_series_matrix=True)
    manifest = pd.read_csv(output / "manifest.csv")
    assert manifest.loc[manifest["gsm"] == "GSM2", "class"].tolist() == ["ambiguous"]
    configure_prepare(monkeypatch, raw_error=data.CacheIntegrityError("tampered CEL"))
    with pytest.raises(data.CacheIntegrityError):
        data.prepare_dataset("GSE54129", workspace / "cache", workspace / "integrity", allow_series_matrix=True)


def test_prepare_rejects_matrix_missing_included_sample(workspace, monkeypatch):
    probes = configure_prepare(monkeypatch, raw_error=data.AcquisitionError("no raw"))
    monkeypatch.setattr(data, "_acquire_series_matrix", lambda *args: probes.loc[:, ["GSM1"]])
    with pytest.raises(data.MetadataError, match="GSM mismatch"):
        data.prepare_dataset("GSE54129", workspace / "cache", workspace / "out", allow_series_matrix=True)


def test_download_metadata_only_never_downloads_raw_or_normalizes(workspace, monkeypatch):
    configure_prepare(monkeypatch)
    def forbidden(*args, **kwargs):
        raise AssertionError("Metadata-only stage must not acquire raw data or normalize")
    monkeypatch.setattr(data, "download", forbidden)
    monkeypatch.setattr(data, "acquire_cels", forbidden)
    monkeypatch.setattr(data, "rma", forbidden)
    output = workspace / "metadata"
    result = data.download_dataset("GSE54129", workspace / "cache", output, metadata_only=True)
    assert result["status"] == "complete"
    assert result["stage"] == "metadata_only"
    assert result["normalization"] == "not_run"
    assert result["expression_available"] is False
    assert result["raw_cel_files_downloaded"] == 0
    assert result["n_included"] == 2
    assert not (output / "expression.csv").exists()
    assert result["outputs"]["manifest.csv"]["sha256"] == data.sha256_file(output / "manifest.csv")


def test_download_raw_stage_records_all_input_resources_without_rma(workspace, monkeypatch):
    configure_prepare(monkeypatch)
    roles = []
    def acquire_cels(manifest, cache, artifacts):
        roles.extend(artifact["role"] for artifact in artifacts)
        artifacts.append({"role": "raw_CEL", "sha256": "fixture"})
        return {"GSM1": Path("a"), "GSM2": Path("b")}
    monkeypatch.setattr(data, "acquire_cels", acquire_cels)
    monkeypatch.setattr(data, "rma", lambda *args: pytest.fail("download must not normalize"))
    result = data.download_dataset("GSE54129", workspace / "cache", workspace / "raw")
    assert result["stage"] == "raw_download" and result["normalization"] == "not_run"
    assert result["raw_cel_files_downloaded"] == 2
    assert set(roles) == {"PM_CDF", "probe_gene_annotation", "HGNC_approved_alias_snapshot"}


def test_download_metadata_retains_ambiguous_manifest_on_failure(workspace, monkeypatch):
    configure_prepare(monkeypatch, ambiguous=True)
    output = workspace / "metadata"
    with pytest.raises(data.MetadataError):
        data.download_dataset("GSE54129", workspace / "cache", output, metadata_only=True)
    assert json.loads((output / "provenance.json").read_text())["status"] == "failed"
    assert "ambiguous" in pd.read_csv(output / "manifest.csv")["class"].tolist()


def test_prepare_explicit_local_cdf_is_hashed_and_not_downloaded(workspace, monkeypatch):
    configure_prepare(monkeypatch)
    cdf_path = workspace / "local.cdf"
    cdf_path.write_bytes(b"fixture local CDF")
    urls = []
    def download(url, destination, **kwargs):
        urls.append(url)
        return Path(destination)
    monkeypatch.setattr(data, "download", download)
    result = data.prepare_dataset("GSE54129", workspace / "cache", workspace / "out", cdf=cdf_path)
    assert data.CDF_URL not in urls
    record = next(artifact for artifact in result["artifacts"] if artifact["role"] == "PM_CDF")
    assert record["path"] == str(cdf_path)
    assert record["sha256"] == data.sha256_file(cdf_path)
    assert record["source"] == "explicit user-supplied local CDF"


def test_hgnc_resolves_withdrawn_readthrough_but_not_real_multigene_probe():
    annotation = (
        "!platform_table_begin\nID\tGene symbol\n"
        "p1\tPIR-FIGF///FIGF\np2\tMETTL7A///TMT1A\n"
        "p3\tFIGF///SEPT10\np4\tUNKNOWN1///UNKNOWN2\n!platform_table_end\n"
    )
    aliases = {"FIGF": "VEGFD", "VEGFD": "VEGFD", "METTL7A": "TMT1A", "TMT1A": "TMT1A", "SEPT10": "SEPTIN10"}
    assert data.parse_probe_annotation(annotation, aliases) == {"p1": "VEGFD", "p2": "TMT1A"}
    with pytest.raises(data.FormatError, match="No unambiguous"):
        data.parse_probe_annotation(annotation)
