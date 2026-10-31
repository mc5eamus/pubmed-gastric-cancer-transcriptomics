from pathlib import Path

import pandas as pd
import pytest

from gastric_repro.pipeline import Settings, load_cohort, read_de, report_stage, sha256, write_json


def test_invalid_resampling_settings():
    with pytest.raises(ValueError, match="candidates"):
        Settings(Path("/tmp/data"), Path("/tmp/results"), candidates=11)
    with pytest.raises(ValueError, match="positive"):
        Settings(Path("/tmp/data"), Path("/tmp/results"), bootstrap=0)


def test_manifest_alignment_and_exclusions(tmp_path):
    directory = tmp_path / "GSE64951"
    directory.mkdir()
    pd.DataFrame({"GSM2": [5.], "GSM1": [6.]}, index=["GENE"]).to_csv(
        directory / "expression.csv")
    pd.DataFrame([
        ["GSM1", 1, True], ["GSM2", 0, True], ["GSM3", None, False],
    ], columns=["gsm", "label", "included"]).to_csv(directory / "manifest.csv", index=False)
    expression, labels = load_cohort(tmp_path, "GSE64951")
    assert labels.index.tolist() == expression.columns.tolist()
    assert labels.tolist() == [0, 1]


def test_manifest_invalid_flag_rejected(tmp_path):
    directory = tmp_path / "GSE64951"
    directory.mkdir()
    pd.DataFrame({"GSM1": [6.]}, index=["GENE"]).to_csv(directory / "expression.csv")
    pd.DataFrame([["GSM1", 1, "maybe"]], columns=["gsm", "label", "included"]).to_csv(
        directory / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="inclusion"):
        load_cohort(tmp_path, "GSE64951")


def test_failed_normalization_cannot_reuse_old_expression(tmp_path):
    directory = tmp_path / "GSE64951"
    directory.mkdir()
    write_json(directory / "provenance.json", {"status": "failed"})
    with pytest.raises(ValueError, match="preprocessing did not complete"):
        load_cohort(tmp_path, "GSE64951")


def test_uncomputed_report_does_not_claim_reproduction(tmp_path):
    report_stage(Settings(tmp_path, tmp_path))
    report = (tmp_path / "REPORT.md").read_text()
    assert report.count("Not assessed") == 3
    assert not (tmp_path / "computed_metrics.csv").exists()


def test_stale_de_inputs_and_filter_settings_rejected(tmp_path):
    directory = tmp_path / "GSE64951"
    de_directory = directory / "de"
    de_directory.mkdir(parents=True)
    (directory / "expression.csv").write_text("gene,GSM1\nGENE,5\n")
    (directory / "manifest.csv").write_text("gsm,label,included\nGSM1,1,True\n")
    pd.DataFrame({"is_deg": [True]}, index=["GENE"]).to_csv(
        de_directory / "differential_expression.csv")
    write_json(de_directory / "provenance.json", {
        "filter": "any_above_median",
        "expression_sha256": sha256(directory / "expression.csv"),
        "manifest_sha256": sha256(directory / "manifest.csv"),
    })
    settings = Settings(tmp_path, tmp_path)
    assert read_de(settings, "GSE64951").index.tolist() == ["GENE"]
    with pytest.raises(ValueError, match="filter settings"):
        read_de(Settings(tmp_path, tmp_path, filter_rule="mean_above_median"), "GSE64951")
    (directory / "expression.csv").write_text("gene,GSM1\nGENE,6\n")
    with pytest.raises(ValueError, match="Stale DE"):
        read_de(settings, "GSE64951")
