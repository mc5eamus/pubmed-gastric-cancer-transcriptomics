"""Small synthetic fixtures; no GEO/raw archives and no external executables."""

import gzip
from pathlib import Path
import shutil
import struct
import uuid

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from gastric_repro import rma as module


@pytest.fixture
def workspace():
    path = (Path(__file__).resolve().parents[1] / ".pytest_cache" / f"rma-{uuid.uuid4().hex}")
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


def text_cel(values):
    rows, cols = values.shape
    return (
        f"[CEL]\nVersion=3\n[HEADER]\nCols={cols}\nRows={rows}\n"
        f"[INTENSITY]\nNumberCells={values.size}\nCellHeader=X Y MEAN STDV NPIXELS\n"
        + "".join(f"{x} {y} {values[y, x]} 1 16\n" for y in range(rows) for x in range(cols))
        + "[MASKS]\nNumberCells=0\nCellHeader=X Y\n"
    )


def binary_cel(values):
    rows, cols = values.shape
    header = f"Cols={cols}\nRows={rows}\n".encode()
    string = lambda value: struct.pack("<i", len(value)) + value
    return (
        struct.pack("<iiiii", 64, 4, rows, cols, values.size)
        + string(header) + string(b"Percentile") + string(b"Percentile:75")
        + struct.pack("<iIIi", 2, 0, 0, 0)
        + b"".join(struct.pack("<ffh", float(value), 1, 16) for value in values.ravel())
    )


def ascii_cdf():
    # Real GC3 CellHeader syntax, including columns often omitted in toy parsers.
    return """[CDF]
Version=GC3.0
[Chip]
Name=HG-U133_Plus_2
Rows=2
Cols=3
NumberOfUnits=2
MaxUnit=2
NumQCUnits=1
[QCUnit0]
Type=1
NumberCells=1
CellHeader=X Y PROBE PLEN ATOM INDEX
Cell1=0 0 N 25 0 0
[Unit0]
Name=p1_at
Direction=1
NumAtoms=1
NumCells=2
UnitNumber=0
UnitType=3
NumberBlocks=1
[Unit0_Block0]
Name=p1_at
BlockNumber=0
NumAtoms=1
NumCells=2
StartPosition=0
StopPosition=0
CellHeader=X Y PROBE FEAT QUAL EXPOS POS CBASE PBASE TBASE ATOM INDEX
Cell1=2 0 N control control 0 13 A T A 0 2
Cell2=2 1 N control control 0 13 A A A 0 5
[Unit1]
Name=p2_at
Direction=1
NumAtoms=1
NumCells=2
UnitNumber=1
UnitType=3
NumberBlocks=1
[Unit1_Block0]
Name=p2_at
CellHeader=X Y PROBE FEAT QUAL EXPOS POS CBASE PBASE TBASE ATOM INDEX
Cell1=0 1 N control control 0 13 G C G 0 3
Cell2=1 1 N control control 0 13 G G G 0 4
"""


def binary_cdf():
    names = [b"p1_at", b"p2_at"]
    prefix = struct.pack("<iiHHiii", 67, 1, 3, 2, 2, 0, 0)
    prefix += b"".join(name.ljust(64, b"\0") for name in names)
    records = []
    for i, (name, pm, mm) in enumerate(zip(names, [(2, 0), (0, 1)], [(2, 1), (1, 1)], strict=True)):
        record = struct.pack("<HBiiiiB", 1, 1, 1, 1, 2, i, 2)
        record += struct.pack("<iiBBii", 1, 2, 2, 1, 0, 0) + name.ljust(64, b"\0")
        for (x, y), pbase in [(pm, b"T"), (mm, b"A")]:
            record += struct.pack("<iHHicc", 0, x, y, 13, pbase, b"A")
        records.append(record)
    offset = len(prefix) + 8
    return prefix + struct.pack("<II", offset, offset + len(records[0])) + b"".join(records)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_cel_geometry_and_compression(workspace, binary, compressed):
    values = np.array([[1.25, 2.5, 3.75], [4.25, 5.5, 6.75]])
    payload = binary_cel(values) if binary else text_cel(values).encode()
    path = workspace / "input.data"
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    np.testing.assert_array_equal(module.read_cel(path), values)


def test_text_cel_missing_duplicate_negative_and_transform(workspace):
    values = np.arange(6).reshape(2, 3)
    text = text_cel(values)
    path = workspace / "test.CEL"
    for malformed in [
        text.replace("2 1 5 1 16\n", ""),
        text.replace("2 1 5 1 16", "1 1 5 1 16"),
        text.replace("2 1 5 1 16", "2 1 -5 1 16"),
        text.replace("Cols=3", "Cols=3\nOffsetX=1"),
        text.replace("2 1 5 1 16", "3 1 5 1 16"),
        text.replace("Version=3", "Version=2"),
    ]:
        path.write_text(malformed)
        with pytest.raises(module.FormatError):
            module.read_cel(path)


def test_binary_cel_truncation_and_unsupported(workspace):
    path = workspace / "input.CEL"
    path.write_bytes(binary_cel(np.ones((2, 3)))[:-2])
    with pytest.raises(module.FormatError, match="Truncated"):
        module.read_cel(path)
    path.write_bytes(b"\x3b\x01calvin")
    with pytest.raises(module.FormatError, match="Calvin"):
        module.read_cel(path)


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_cdf_real_format_pm_selection(workspace, binary, compressed):
    payload = binary_cdf() if binary else ascii_cdf().encode()
    path = workspace / "chip.cdf"
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    cdf = module.read_cdf(path)
    assert cdf.shape == (2, 3)
    assert list(cdf.probesets) == ["p1_at", "p2_at"]
    np.testing.assert_array_equal(cdf.probesets["p1_at"], [2])
    np.testing.assert_array_equal(cdf.probesets["p2_at"], [3])


def test_cdf_fails_closed(workspace):
    path = workspace / "bad.cdf"
    for malformed in [
        ascii_cdf().replace("NumberOfUnits=2", "NumberOfUnits=3"),
        ascii_cdf().replace("PBASE", "NOBASE"),
        ascii_cdf().replace("Cell1=2 0", "Cell1=3 0"),
        ascii_cdf().replace("Name=p2_at", "Name=p1_at"),
        ascii_cdf().replace("UnitType=3", "UnitType=2"),
        ascii_cdf().replace("NumAtoms=1", "NumAtoms=2"),
    ]:
        path.write_text(malformed)
        with pytest.raises(module.FormatError):
            module.read_cdf(path)
    path.write_bytes(binary_cdf()[:50])
    with pytest.raises(module.FormatError):
        module.read_cdf(path)
    path.write_bytes(binary_cdf().replace(struct.pack("<ii", 67, 1), struct.pack("<ii", 67, 4), 1))
    with pytest.raises(module.FormatError, match="version"):
        module.read_cdf(path)


def test_ascii_cdf_uses_block_names_not_placeholder_unit_names(workspace):
    path = workspace / "chip.cdf"
    text = ascii_cdf().replace("[Unit0]\nName=p1_at", "[Unit0]\nName=NONE")
    text = text.replace("[Unit1]\nName=p2_at", "[Unit1]\nName=NONE")
    path.write_text(text)
    assert list(module.read_cdf(path).probesets) == ["p1_at", "p2_at"]


def test_background_matches_conditional_expectation():
    alpha, mu, sigma = 0.3, 4.0, 2.0
    values = np.array([0, 2, 10, 20], dtype=float)
    actual = module.background_correct(values, parameters=(alpha, mu, sigma))
    expected = []
    for value in values:
        posterior = lambda signal: np.exp(-alpha * signal - (value - signal - mu)**2 / (2 * sigma**2))
        expected.append(quad(lambda signal: signal * posterior(signal), 0, np.inf)[0] / quad(posterior, 0, np.inf)[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-10)


def test_background_stable_extreme_tail_and_estimated_parameters():
    result = module.background_correct(np.array([0., 1., 1e6]), parameters=(1., 1000., 10.))
    assert np.isfinite(result).all() and (result > 0).all()
    assert np.all(np.diff(result) > 0)
    rng = np.random.default_rng(42)
    values = np.maximum(0, rng.normal(100, 8, 20000) + rng.exponential(100, 20000))
    result = module.background_correct(values)
    assert result.shape == values.shape
    assert np.isfinite(result).all() and (result > 0).all()
    with pytest.raises(ValueError):
        module.background_correct(np.ones(20))
    with pytest.raises(ValueError):
        module.background_correct(np.array([-1., 3., 4.]))


def test_quantile_normalization_reference_and_ties():
    values = np.array([[5, 4, 3], [2, 1, 4], [3, 4, 6], [4, 2, 8]], dtype=float)
    actual = module.quantile_normalize(values)
    reference = np.sort(values, axis=0).mean(axis=1)
    np.testing.assert_allclose(actual[:, 0], reference[[3, 0, 1, 2]])
    np.testing.assert_allclose(actual[:, 1], [reference[2:].mean(), reference[0], reference[2:].mean(), reference[1]])
    np.testing.assert_allclose(actual.mean(axis=0), np.repeat(reference.mean(), 3))
    assert not np.shares_memory(values, actual)


def test_median_polish_additive_model_and_outlier():
    additive = np.array([1., 2., 4., 7., 8.])[:, None] + np.array([3., 6., 9.])[None, :]
    np.testing.assert_allclose(module.median_polish(additive), np.median(additive, axis=0))
    noisy = additive.copy()
    noisy[0, 0] += 1000
    np.testing.assert_allclose(module.median_polish(noisy), np.median(additive, axis=0))
    with pytest.raises(ValueError):
        module.median_polish(np.array([[np.nan]]))


def test_rma_pipeline_order_and_geometry(workspace, monkeypatch):
    cdf = workspace / "chip.cdf"
    cdf.write_text(ascii_cdf())
    files = {}
    for gsm, values in {"GSM1": np.arange(1, 7).reshape(2, 3), "GSM2": np.arange(11, 17).reshape(2, 3)}.items():
        path = workspace / f"{gsm}.CEL"
        path.write_text(text_cel(values))
        files[gsm] = path
    monkeypatch.setattr(module, "background_correct", lambda x: x + 1)
    actual = module.rma(files, cdf)
    # PMs: [3,4], [13,14], corrected [4,5],[14,15], quantiles [9,10].
    expected = pd.DataFrame(
        np.log2([[9., 9.], [10., 10.]]),
        index=pd.Index(["p1_at", "p2_at"], name="probe"), columns=["GSM1", "GSM2"],
    )
    pd.testing.assert_frame_equal(actual, expected)
    files["GSM2"].write_text(text_cel(np.ones((3, 2))))
    with pytest.raises(module.FormatError, match="differs"):
        module.rma(files, cdf)
    with pytest.raises(ValueError, match="No CEL"):
        module.rma({}, cdf)
