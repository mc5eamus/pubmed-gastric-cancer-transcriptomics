"""Affymetrix Calvin Generic Data CEL reader (big-endian version 1).

Reads the Intensity data set in row-major chip order. Ancillary standard
deviation, pixel, mask and outlier data are not used by PM-only RMA.
"""

import gzip
from pathlib import Path
import struct

import numpy as np


class CalvinFormatError(ValueError):
    """Malformed or unsupported Calvin CEL, not a processed-data substitute."""


class _Reader:
    def __init__(self, payload):
        self.payload = payload
        self.position = 0

    def take(self, count):
        end = self.position + count
        if count < 0 or end > len(self.payload):
            raise CalvinFormatError("Truncated Calvin field")
        value = self.payload[self.position:end]
        self.position = end
        return value

    def number(self, kind="I"):
        return struct.unpack(">" + kind, self.take(struct.calcsize(">" + kind)))[0]

    def text(self, wide=False):
        count = self.number("i")
        if not 0 <= count <= 1_000_000:
            raise CalvinFormatError("Invalid Calvin string length")
        try:
            return self.take(count * (2 if wide else 1)).decode(
                "utf-16-be" if wide else "utf-8")
        except UnicodeDecodeError as error:
            raise CalvinFormatError("Invalid Calvin string encoding") from error

    def seek(self, offset):
        if not 10 <= offset < len(self.payload):
            raise CalvinFormatError("Calvin offset outside file")
        self.position = offset

    def parameters(self):
        count = self.number("i")
        if not 0 <= count <= 10000:
            raise CalvinFormatError("Invalid Calvin parameter count")
        result = {}
        for _ in range(count):
            name = self.text(wide=True)
            size = self.number("i")
            value = self.take(size)
            mime = self.text(wide=True)
            if name in result:
                raise CalvinFormatError(f"Duplicate Calvin parameter: {name}")
            result[name] = (value, mime)
        return result

    def generic_header(self, depth=0):
        if depth > 16:
            raise CalvinFormatError("Excessive Calvin parent-header nesting")
        file_type = self.text()
        self.text()  # file identifier
        self.text(wide=True)  # creation time
        self.text(wide=True)  # locale
        parameters = self.parameters()
        count = self.number("i")
        if not 0 <= count <= 100:
            raise CalvinFormatError("Invalid Calvin parent-header count")
        for _ in range(count):
            self.generic_header(depth + 1)
        return file_type, parameters


def _dimension(parameters, name):
    try:
        value, mime = parameters[name]
    except KeyError as error:
        raise CalvinFormatError(f"Missing Calvin dimension: {name}") from error
    # Calvin numeric MIME values may be padded to 16 bytes.
    if mime not in {"text/x-calvin-integer-32", "text/x-calvin-unsigned-integer-32"} or len(value) < 4:
        raise CalvinFormatError(f"Invalid Calvin dimension encoding: {name}")
    return struct.unpack(">i" if "unsigned" not in mime else ">I", value[:4])[0]


def parse_calvin_cel(payload: bytes) -> np.ndarray:
    """Parse a bounded decompressed CEL payload; reject malformed linked headers."""
    reader = _Reader(payload)
    if (reader.number("B"), reader.number("B")) != (59, 1):
        raise CalvinFormatError("Expected Calvin magic 59, version 1")
    group_count, group_offset = reader.number(), reader.number()
    if not 1 <= group_count <= 100:
        raise CalvinFormatError("Invalid Calvin data-group count")
    file_type, parameters = reader.generic_header()
    if file_type != "affymetrix-calvin-intensity":
        raise CalvinFormatError(f"Not a Calvin intensity CEL: {file_type}")
    rows = _dimension(parameters, "affymetrix-cel-rows")
    cols = _dimension(parameters, "affymetrix-cel-cols")
    if rows <= 0 or cols <= 0 or rows * cols > 10_000_000:
        raise CalvinFormatError("Invalid Calvin chip dimensions")
    header_end = reader.position
    visited_groups, visited_sets = set(), set()
    result = None
    for _ in range(group_count):
        if group_offset in visited_groups or group_offset < header_end:
            raise CalvinFormatError("Invalid or cyclic Calvin group offset")
        visited_groups.add(group_offset)
        reader.seek(group_offset)
        next_group, set_offset, set_count = reader.number(), reader.number(), reader.number("i")
        reader.text(wide=True)
        if not 1 <= set_count <= 1000:
            raise CalvinFormatError("Invalid Calvin data-set count")
        group_header_end = reader.position
        for _ in range(set_count):
            if set_offset in visited_sets or set_offset < group_header_end:
                raise CalvinFormatError("Invalid or cyclic Calvin data-set offset")
            visited_sets.add(set_offset)
            reader.seek(set_offset)
            data_offset, next_set = reader.number(), reader.number()
            name = reader.text(wide=True)
            reader.parameters()
            column_count = reader.number()
            if not 1 <= column_count <= 100:
                raise CalvinFormatError("Invalid Calvin column count")
            columns = []
            for _ in range(column_count):
                column_name = reader.text(wide=True)
                kind, size = reader.number("B"), reader.number("i")
                if size <= 0 or size > 1_000_000:
                    raise CalvinFormatError("Invalid Calvin column width")
                columns.append((column_name, kind, size))
            count = reader.number()
            row_width = sum(column[2] for column in columns)
            if data_offset < reader.position or data_offset + count * row_width > len(payload):
                raise CalvinFormatError("Calvin data extent outside file")
            if name == "Intensity":
                if result is not None:
                    raise CalvinFormatError("Multiple Calvin Intensity data sets")
                if count != rows * cols:
                    raise CalvinFormatError("Calvin intensity count differs from dimensions")
                offsets = np.cumsum([0] + [column[2] for column in columns[:-1]])
                intensity_columns = [
                    (offset, kind, size)
                    for offset, (column, kind, size) in zip(offsets, columns, strict=True)
                    if column == "Intensity"
                ]
                if len(intensity_columns) != 1 or intensity_columns[0][1:] != (6, 4):
                    raise CalvinFormatError("Expected one float32 Calvin Intensity column")
                offset = int(intensity_columns[0][0])
                result = np.ndarray(
                    shape=(count,), dtype=">f4", buffer=payload,
                    offset=data_offset + offset, strides=(row_width,),
                ).astype(float).reshape(rows, cols)
            set_offset = next_set
        group_offset = next_group
    if result is None:
        raise CalvinFormatError("Calvin CEL has no Intensity data set")
    if not np.isfinite(result).all() or (result < 0).any():
        raise CalvinFormatError("Calvin intensities must be finite and nonnegative")
    return result


def read_calvin_cel(path: Path) -> np.ndarray:
    path = Path(path)
    with path.open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        payload = handle.read(256 * 1024 * 1024 + 1)
    if len(payload) > 256 * 1024 * 1024:
        raise CalvinFormatError("Calvin CEL exceeds 256 MiB decompressed limit")
    return parse_calvin_cel(payload)
