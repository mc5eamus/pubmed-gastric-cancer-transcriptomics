import json
import sys
from types import SimpleNamespace

import pytest

from gastric_repro import cli


def test_help_without_network_or_models(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "normalise" in capsys.readouterr().out


def test_partial_gmt_rejected(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli.main(["enrichment", "--go-bp", str(tmp_path / "bp.gmt")])
    assert error.value.code == 2


def test_run_rejects_subset():
    with pytest.raises(SystemExit) as error:
        cli.main(["run", "--dataset", "GSE64951"])
    assert error.value.code == 2


def test_report_command_records_actual_stage_only(tmp_path):
    assert cli.main(["report", "--output", str(tmp_path)]) == 0
    record = json.loads((tmp_path / "report_run.json").read_text())
    assert record["status"] == "completed"
    assert record["completed_stages"] == ["report"]
    assert (tmp_path / "REPORT.md").exists()


def test_failure_retains_run_record_and_exits_nonzero(monkeypatch, tmp_path):
    def fail(*args):
        raise ValueError("fixture failure")
    monkeypatch.setattr(cli, "dispatch", fail)
    assert cli.main(["de", "--output", str(tmp_path)]) == 1
    record = json.loads((tmp_path / "de_run.json").read_text())
    assert record["status"] == "failed"
    assert record["completed_stages"] == []
    assert record["error"] == "ValueError: fixture failure"


def test_run_passes_source_models_to_robustness(monkeypatch, tmp_path):
    stages = []
    source = object()
    monkeypatch.setattr(cli, "dispatch", lambda stage, *args: stages.append(stage))
    monkeypatch.setattr(cli, "ml_stage", lambda settings: source)
    def robust(settings, fitted):
        assert fitted is source
        stages.append("robustness")
    monkeypatch.setattr(cli, "robustness_stage", robust)
    monkeypatch.setattr(cli, "report_stage", lambda settings: stages.append("report"))
    assert cli.main(["run", "--output", str(tmp_path)]) == 0
    assert stages == ["evidence", "normalise", "de", "enrichment", "robustness", "report"]
    record = json.loads((tmp_path / "run_run.json").read_text())
    assert record["completed_stages"] == [
        "evidence", "normalise", "de", "enrichment", "ml", "robustness", "report",
    ]


def test_normalise_passes_local_cdf_and_explicit_fallback(monkeypatch, tmp_path):
    calls = []
    def prepare(accession, cache, output, *, cdf, allow_series_matrix):
        calls.append((accession, cache, output, cdf, allow_series_matrix))
    monkeypatch.setitem(sys.modules, "gastric_repro.data",
                        SimpleNamespace(prepare_dataset=prepare))
    assert cli.main([
        "normalise", "--dataset", "GSE64951", "--cache", str(tmp_path / "cache"),
        "--output", str(tmp_path / "output"), "--cdf", str(tmp_path / "chip.cdf"),
        "--allow-series-matrix",
    ]) == 0
    assert calls == [("GSE64951", tmp_path / "cache", tmp_path / "output" / "GSE64951",
                      tmp_path / "chip.cdf", True)]


def test_download_metadata_does_not_invoke_normalization(monkeypatch, tmp_path):
    calls = []
    def download(accession, cache, output, *, metadata_only):
        calls.append((accession, metadata_only))
    monkeypatch.setitem(sys.modules, "gastric_repro.data",
                        SimpleNamespace(download_dataset=download))
    assert cli.main([
        "download", "--dataset", "GSE64951", "--metadata-only",
        "--output", str(tmp_path),
    ]) == 0
    assert calls == [("GSE64951", True)]
