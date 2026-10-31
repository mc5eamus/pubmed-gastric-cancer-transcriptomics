"""Pure-Python PM-only RMA, with explicit file-format and parity limitations.

``rma(cel_files, cdf_path)`` returns log2 probe-set expression (rows are probe
sets, columns are the keys of ``cel_files``). CEL v3 text and XDA v4 binary,
including gzip wrappers, and Calvin v1 intensity CELs are supported.
CDF input may be genuine Affymetrix ASCII GC1--GC6 or XDA v1 expression CDF.
No R executable or processed-expression fallback is invoked here.

The normal/exponential model and median polish follow RMA, but the binned
Gaussian KDE parameter estimator is not bit-identical to preprocessCore.
Numerical parity against affy/oligo has NOT been established.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import gzip
import io
from pathlib import Path
import re
import struct
from typing import BinaryIO
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import log_ndtr

from .calvin import CalvinFormatError, read_calvin_cel


class FormatError(ValueError):
    """Malformed or unsupported CEL/CDF input; never silently reinterpret it."""


@dataclass(frozen=True)
class CDF:
    """PM indices in CEL row-major order, with (rows, columns) chip geometry."""

    shape: tuple[int, int]
    probesets: dict[str, np.ndarray]


def _open_binary(path: Path) -> BinaryIO:
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rb") if compressed else path.open("rb")


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    if size < 0:
        raise FormatError("Negative binary field length")
    value = handle.read(size)
    if len(value) != size:
        raise FormatError("Truncated binary CEL/CDF")
    return value


def _unpack(handle: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))


def _string(handle: BinaryIO) -> bytes:
    size, = _unpack(handle, "i")
    if not 0 <= size <= 10_000_000:
        raise FormatError("Invalid binary string length")
    return _read_exact(handle, size)


def _geometry(rows: int, cols: int) -> None:
    if rows <= 0 or cols <= 0 or rows * cols > 100_000_000:
        raise FormatError(f"Invalid chip geometry {rows} x {cols}")


def _check_header(header: str, rows: int, cols: int) -> None:
    values = dict(line.split("=", 1) for line in header.splitlines() if "=" in line)
    for key, expected in (("Rows", rows), ("Cols", cols)):
        if key in values and int(values[key]) != expected:
            raise FormatError("CEL binary dimensions disagree with text header")
    for key in ("Axis-invertX", "AxisInvertX", "AxisInvertY", "OffsetX", "OffsetY"):
        if key in values and int(values[key]) != 0:
            raise FormatError(f"Unsupported CEL coordinate transform: {key}")


def read_cel(path: Path) -> np.ndarray:
    """Read intensities as ``array[y, x]``; ignore masks/outliers as standard RMA.

    Intensities must be finite and nonnegative. Missing, duplicate, rotated, or
    truncated cells are errors, not candidates for imputation.
    """
    path = Path(path)
    with _open_binary(path) as handle:
        prefix = handle.read(4)
        handle.seek(0)
        if prefix == struct.pack("<i", 64):
            magic, version, rows, cols, count = _unpack(handle, "iiiii")
            if version != 4:
                raise FormatError(f"Unsupported XDA CEL version {version}")
            _geometry(rows, cols)
            if count != rows * cols:
                raise FormatError("CEL cell count differs from dimensions")
            header = _string(handle).decode("ascii", errors="strict")
            _check_header(header, rows, cols)
            _string(handle)  # algorithm name
            _string(handle)  # algorithm parameters
            margin, outliers, masked, subgrids = _unpack(handle, "iIIi")
            if margin < 0 or subgrids != 0 or max(outliers, masked) > count:
                raise FormatError("Unsupported CEL subgrids or invalid flag counts")
            cells = np.frombuffer(
                _read_exact(handle, 10 * count),
                dtype=np.dtype([("mean", "<f4"), ("sd", "<f4"), ("pixels", "<i2")]),
            )
            result = cells["mean"].astype(float).reshape(rows, cols)
            _read_exact(handle, 4 * (outliers + masked))
        elif prefix.startswith(b"[CEL"):
            result = _read_text_cel(io.TextIOWrapper(handle, encoding="ascii"))
        elif prefix[:1] == b"\x3b":
            try:
                result = read_calvin_cel(path)
            except CalvinFormatError as error:
                raise FormatError(str(error)) from error
        else:
            raise FormatError("Unsupported CEL format (expected v3 text, XDA v4 or Calvin v1)")
    if not np.isfinite(result).all() or (result < 0).any():
        raise FormatError("CEL intensities must be finite and nonnegative")
    return result


def _read_text_cel(handle) -> np.ndarray:
    section = ""
    header: dict[str, str] = {}
    version = None
    result = None
    seen = None
    columns = None
    declared = None
    for raw in handle:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            if section == "INTENSITY":
                if version != "3" or not {"Rows", "Cols"} <= header.keys():
                    raise FormatError("Expected CEL version 3 and dimensions before intensities")
                rows, cols = int(header["Rows"]), int(header["Cols"])
                _geometry(rows, cols)
                _check_header("\n".join(f"{k}={v}" for k, v in header.items()), rows, cols)
                result = np.empty((rows, cols), dtype=float)
                seen = np.zeros((rows, cols), dtype=bool)
            continue
        if section == "CEL" and line.startswith("Version="):
            version = line.split("=", 1)[1]
        elif section == "HEADER" and "=" in line:
            key, value = line.split("=", 1)
            header[key] = value
        elif section == "INTENSITY":
            if line.startswith("NumberCells="):
                declared = int(line.split("=", 1)[1])
            elif line.startswith("CellHeader="):
                columns = line.split("=", 1)[1].split()
                if not {"X", "Y", "MEAN"} <= set(columns):
                    raise FormatError("CEL intensity header lacks X/Y/MEAN")
            else:
                if columns is None:
                    raise FormatError("CEL cells precede CellHeader")
                fields = dict(zip(columns, line.split(), strict=True))
                x, y = int(fields["X"]), int(fields["Y"])
                if not (0 <= x < result.shape[1] and 0 <= y < result.shape[0]):
                    raise FormatError("CEL coordinate outside chip")
                if seen[y, x]:
                    raise FormatError("Duplicate CEL coordinate")
                result[y, x] = float(fields["MEAN"])
                seen[y, x] = True
    if result is None or not seen.all() or declared != result.size:
        raise FormatError("Incomplete CEL intensity table")
    return result


def _finalize_cdf(shape, probesets) -> CDF:
    rows, cols = shape
    _geometry(rows, cols)
    if not probesets:
        raise FormatError("CDF contains no PM expression probe sets")
    converted = {}
    for name, indices in probesets.items():
        array = np.asarray(indices, dtype=np.int64)
        if not len(array) or (array < 0).any() or (array >= rows * cols).any():
            raise FormatError(f"Invalid PM coordinates in {name}")
        if len(np.unique(array)) != len(array):
            raise FormatError(f"Duplicate PM coordinates in {name}")
        converted[name] = array
    return CDF((rows, cols), converted)


def read_cdf(path: Path) -> CDF:
    """Parse real ASCII CDF CellHeader columns, or standard XDA v1 binary CDF.

    A PM cell has unequal PBASE/TBASE in Affymetrix expression CDFs; MM cells
    have equal bases. QCUnit records are not expression probes. Coordinates
    are converted with ``y * Cols + x`` (never inferred from row order).
    """
    with _open_binary(Path(path)) as handle:
        prefix = handle.read(4)
        handle.seek(0)
        if prefix == struct.pack("<i", 67):
            return _read_binary_cdf(handle)
        if prefix != b"[CDF":
            raise FormatError("Unsupported CDF format")
        return _read_ascii_cdf(io.TextIOWrapper(handle, encoding="ascii"))


def _read_ascii_cdf(handle) -> CDF:
    section = ""
    chip = {}
    unit_names = {}
    unit_fields = {}
    unit_cell_counts = {}
    block_names = {}
    cells: dict[str, list[tuple[int, int]]] = {}
    columns = None
    version = None
    for raw in handle:
        line = raw.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            columns = None
            continue
        if "=" not in line:
            raise FormatError("Malformed ASCII CDF line")
        key, value = line.split("=", 1)
        if section == "CDF" and key == "Version":
            version = value
        elif section == "Chip":
            chip[key] = value
        elif re.fullmatch(r"Unit\d+", section):
            if key == "Name":
                if section in unit_names:
                    raise FormatError("Duplicate CDF unit")
                unit_names[section] = value
                cells[section] = []
                unit_fields[section] = {}
                unit_cell_counts[section] = 0
                block_names[section] = []
            elif section in unit_fields:
                unit_fields[section][key] = value
        elif re.fullmatch(r"Unit\d+_Block\d+", section):
            unit = section.split("_", 1)[0]
            if key == "Name":
                if unit not in unit_names:
                    raise FormatError("CDF block precedes unit")
                block_names[unit].append(value)
            elif key == "CellHeader":
                columns = value.upper().split()
                if not {"X", "Y", "PBASE", "TBASE"} <= set(columns):
                    raise FormatError("CDF CellHeader lacks X/Y/PBASE/TBASE")
            elif re.fullmatch(r"Cell\d+", key):
                if columns is None or unit not in cells:
                    raise FormatError("CDF cells precede unit or CellHeader")
                fields = dict(zip(columns, value.split(), strict=True))
                pbase, tbase = fields["PBASE"].upper(), fields["TBASE"].upper()
                if pbase not in "ACGT" or tbase not in "ACGT" or len(pbase) != 1 or len(tbase) != 1:
                    raise FormatError("Unsupported CDF probe/target bases")
                x, y = int(fields["X"]), int(fields["Y"])
                if not (0 <= x < int(chip["Cols"]) and 0 <= y < int(chip["Rows"])):
                    raise FormatError("CDF coordinate outside chip")
                if pbase != tbase:
                    cells[unit].append((x, y))
                unit_cell_counts[unit] += 1
    if version not in {f"GC{v}.0" for v in range(1, 7)}:
        raise FormatError(f"Unsupported ASCII CDF version {version}")
    if not {"Rows", "Cols", "NumberOfUnits"} <= chip.keys():
        raise FormatError("Incomplete CDF Chip header")
    if len(unit_names) != int(chip["NumberOfUnits"]):
        raise FormatError("CDF unit count does not match header")
    rows, cols = int(chip["Rows"]), int(chip["Cols"])
    probesets = {}
    for unit, positions in cells.items():
        fields = unit_fields[unit]
        required = {"UnitType", "NumAtoms", "NumCells", "NumberBlocks"}
        if not required <= fields.keys() or fields["UnitType"] != "3":
            raise FormatError("Only complete ASCII expression CDF units (UnitType=3) are supported")
        atoms = int(fields["NumAtoms"].split()[0])
        if (
            atoms <= 0 or int(fields["NumCells"]) != atoms * 2
            or unit_cell_counts[unit] != atoms * 2 or len(positions) != atoms
            or int(fields["NumberBlocks"]) != len(block_names[unit])
            or len(set(block_names[unit])) != 1
        ):
            raise FormatError("CDF PM/cell/block counts do not match header")
        # Vendor GC-format readers use Block Name for expression probe-set IDs.
        name = block_names[unit][0]
        if not name or name in probesets:
            raise FormatError("Duplicate/empty CDF expression probe-set name")
        probesets[name] = [y * cols + x for x, y in positions]
    return _finalize_cdf(
        (rows, cols), probesets,
    )


def _read_binary_cdf(handle: BinaryIO) -> CDF:
    magic, version, cols, rows, count, qc_count = _unpack(handle, "iiHHii")
    if version != 1:
        raise FormatError(f"Unsupported XDA CDF version {version}; only v1 is supported")
    _geometry(rows, cols)
    if not 0 < count <= rows * cols or not 0 <= qc_count <= rows * cols:
        raise FormatError("Invalid CDF unit counts")
    _string(handle)
    names = [_read_exact(handle, 64).split(b"\0", 1)[0].decode("ascii") for _ in range(count)]
    if len(set(names)) != count or not all(names):
        raise FormatError("Duplicate/empty binary CDF names")
    _read_exact(handle, 4 * qc_count)
    offsets = _unpack(handle, "I" * count)
    probesets = {}
    for name, offset in zip(names, offsets, strict=True):
        handle.seek(offset)
        kind, direction, atoms, groups, cell_count, number, per_atom = _unpack(handle, "HBiiiiB")
        if kind != 1 or groups <= 0 or per_atom != 2:
            raise FormatError("Only paired PM/MM expression CDF units are supported")
        positions = []
        total_cells = 0
        for _ in range(groups):
            group_atoms, n_cells, group_per_atom, group_dir, start, stop = _unpack(handle, "iiBBii")
            _read_exact(handle, 64)
            if n_cells < 0 or n_cells > rows * cols:
                raise FormatError("Invalid CDF group size")
            total_cells += n_cells
            for _ in range(n_cells):
                atom, x, y, exposure, pbase, tbase = _unpack(handle, "iHHicc")
                if x >= cols or y >= rows:
                    raise FormatError("CDF coordinate outside chip")
                if pbase.upper() not in (b"A", b"C", b"G", b"T") or tbase.upper() not in (b"A", b"C", b"G", b"T"):
                    raise FormatError("Unsupported CDF probe/target bases")
                if pbase.upper() != tbase.upper():
                    positions.append(y * cols + x)
        if total_cells != cell_count or len(positions) != atoms:
            raise FormatError("CDF PM/cell counts do not match header")
        probesets[name] = positions
    return _finalize_cdf((rows, cols), probesets)


def _matrix(values, *, positive=False) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError("Expected a nonempty finite probes-by-samples matrix")
    if positive and (array <= 0).any():
        raise ValueError("Expected strictly positive values")
    return array


def _density_mode(values: np.ndarray) -> float:
    """Deterministic 16,384-bin Gaussian KDE with Silverman nrd0 bandwidth."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.ptp(values) <= 0:
        raise ValueError("Insufficient nonconstant probes for RMA background estimation")
    spread = min(np.std(values, ddof=1), np.subtract(*np.percentile(values, [75, 25])) / 1.34)
    if spread <= 0:
        spread = np.std(values, ddof=1)
    bandwidth = 0.9 * spread * len(values) ** -0.2
    lo, hi = values.min() - 3 * bandwidth, values.max() + 3 * bandwidth
    hist, edges = np.histogram(values, bins=16384, range=(lo, hi))
    step = edges[1] - edges[0]
    density = gaussian_filter1d(hist.astype(float), bandwidth / step, mode="constant")
    index = int(np.argmax(density))
    return float((edges[index] + edges[index + 1]) / 2)


def background_correct(
    values: np.ndarray, *, parameters: tuple[float, float, float] | None = None
) -> np.ndarray:
    """Normal/exponential conditional signal expectation for one PM array.

    Optional parameters are ``(alpha, mu, sigma)`` with alpha the exponential
    rate. Otherwise estimate two lower-tail KDE modes, lower-tail RMS sigma,
    and reciprocal upper-tail mode, following the RMA parameter structure.
    KDE discretization differs from preprocessCore; parity is unvalidated.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expected finite nonnegative one-dimensional PM intensities")
    if parameters is None:
        mode = _density_mode(x)
        mu = _density_mode(x[x < mode])
        lower = x[x < mu] - mu
        if len(lower) < 2:
            raise ValueError("Insufficient lower-tail PM probes to estimate RMA sigma")
        sigma = float(np.sqrt(2 * np.sum(lower**2) / (len(lower) - 1)))
        upper_mode = _density_mode(x[x > mu] - mu)
        if upper_mode <= 0:
            raise ValueError("Invalid RMA exponential mode")
        alpha = 1 / upper_mode
    else:
        alpha, mu, sigma = parameters
    if not np.isfinite([alpha, mu, sigma]).all() or alpha <= 0 or sigma <= 0:
        raise ValueError("RMA alpha and sigma must be finite and positive")
    z = (x - mu - alpha * sigma**2) / sigma
    result = np.empty_like(z)
    tail = z < -20
    t = -z[tail]
    result[tail] = sigma * (1 / t - 2 / t**3 + 10 / t**5 - 74 / t**7)
    central = z[~tail]
    ratio = np.exp(-central**2 / 2 - np.log(2 * np.pi) / 2 - log_ndtr(central))
    result[~tail] = sigma * (central + ratio)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("RMA background correction produced invalid signals")
    return result


def quantile_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize columns to mean sorted intensities, averaging tied rank blocks."""
    array = _matrix(values)
    # Two column-wise passes avoid full-size sorted-value and rank workspaces.
    reference = np.zeros(array.shape[0])
    for column in range(array.shape[1]):
        reference += np.sort(array[:, column])
    reference /= array.shape[1]
    result = np.empty_like(array)
    for column in range(array.shape[1]):
        order = np.argsort(array[:, column], kind="stable")
        ordered = array[order, column]
        unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
        sums = np.add.reduceat(reference, starts)
        tied = np.repeat(sums / counts, counts)
        result[order, column] = tied
    return result


def median_polish(values: np.ndarray, *, max_iter: int = 50, tolerance: float = 1e-6) -> np.ndarray:
    """Tukey row/column median polish; return overall + sample (column) effects."""
    residual = _matrix(values).copy()
    if max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid median-polish convergence controls")
    rows = np.zeros(residual.shape[0])
    columns = np.zeros(residual.shape[1])
    overall = 0.0
    previous = np.abs(residual).sum()
    for _ in range(max_iter):
        shift = np.median(residual, axis=1)
        residual -= shift[:, None]
        rows += shift
        center = np.median(columns)
        columns -= center
        overall += center
        shift = np.median(residual, axis=0)
        residual -= shift[None, :]
        columns += shift
        center = np.median(rows)
        rows -= center
        overall += center
        current = np.abs(residual).sum()
        if current == 0 or abs(previous - current) <= tolerance * max(current, 1e-15):
            break
        previous = current
    else:
        warnings.warn("Median polish reached max_iter without convergence", RuntimeWarning, stacklevel=2)
    return overall + columns


def rma(cel_files: Mapping[str, Path], cdf_path: Path) -> pd.DataFrame:
    """Joint PM background→quantile→log2→median-polish normalization per dataset.

    Memory scales with PM probes × samples (about 2 GB for 400 GPL570 arrays
    before normalization workspaces). CELs are read one at a time. Every
    array must match the CDF geometry. R/affy numerical parity is not claimed.
    """
    if not cel_files:
        raise ValueError("No CEL files supplied")
    cdf = read_cdf(cdf_path)
    names = list(cdf.probesets)
    lengths = [len(cdf.probesets[name]) for name in names]
    offsets = np.r_[0, np.cumsum(lengths)]
    indices = np.concatenate(list(cdf.probesets.values()))
    pm = np.empty((len(indices), len(cel_files)), dtype=float)
    for column, path in enumerate(cel_files.values()):
        array = read_cel(path)
        if array.shape != cdf.shape:
            raise FormatError(f"CEL {path} shape {array.shape} differs from CDF {cdf.shape}")
        pm[:, column] = background_correct(array.ravel()[indices])
    normalized = quantile_normalize(pm)
    del pm
    np.log2(normalized, out=normalized)
    expression = np.vstack([
        median_polish(normalized[offsets[i]:offsets[i + 1]]) for i in range(len(names))
    ])
    return pd.DataFrame(expression, index=pd.Index(names, name="probe"), columns=list(cel_files))
