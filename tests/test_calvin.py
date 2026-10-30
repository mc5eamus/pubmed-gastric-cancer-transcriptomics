import gzip
import struct

import numpy as np
import pytest

from gastric_repro.calvin import CalvinFormatError, parse_calvin_cel, read_calvin_cel


def _number(value, kind="I"):
    return struct.pack(">" + kind, value)


def _text(value, wide=False):
    return _number(len(value), "i") + value.encode("utf-16-be" if wide else "utf-8")


def _parameter(name, value):
    return (_text(name, True) + _number(16, "i") + _number(value, "i")
            + b"\0" * 12 + _text("text/x-calvin-integer-32", True))


def calvin(values=(1., 2., 3., 4.), *, rows=2, cols=2, kind=6):
    generic = (
        _text("affymetrix-calvin-intensity") + _text("fixture")
        + _text("", True) + _text("en-US", True) + _number(2, "i")
        + _parameter("affymetrix-cel-rows", rows)
        + _parameter("affymetrix-cel-cols", cols) + _number(0, "i")
    )
    group_offset = 10 + len(generic)
    group_tail = _number(1, "i") + _text("Default Group", True)
    set_offset = group_offset + 8 + len(group_tail)
    set_tail = (
        _text("Intensity", True) + _number(0, "i") + _number(1)
        + _text("Intensity", True) + _number(kind, "B") + _number(4, "i")
        + _number(len(values))
    )
    data_offset = set_offset + 8 + len(set_tail)
    return (
        b"\x3b\x01" + _number(1) + _number(group_offset) + generic
        + _number(0) + _number(set_offset) + group_tail
        + _number(data_offset) + _number(0) + set_tail
        + np.asarray(values, dtype=">f4").tobytes()
    )


def test_padded_big_endian_dimensions_and_row_major_intensities():
    np.testing.assert_array_equal(parse_calvin_cel(calvin()), [[1, 2], [3, 4]])


def test_gzip_wrapper(tmp_path):
    path = tmp_path / "sample.CEL.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(calvin())
    np.testing.assert_array_equal(read_calvin_cel(path), [[1, 2], [3, 4]])


@pytest.mark.parametrize("cut", [0, 1, 5, 15, 100, -1])
def test_truncated_data_rejected(cut):
    with pytest.raises(CalvinFormatError):
        parse_calvin_cel(calvin()[:cut])


@pytest.mark.parametrize("values", [(1, 2, 3, -1), (1, 2, 3, np.nan), (1, 2, 3, np.inf)])
def test_invalid_intensities_rejected(values):
    with pytest.raises(CalvinFormatError, match="finite and nonnegative"):
        parse_calvin_cel(calvin(values))


def test_geometry_and_column_encoding_rejected():
    with pytest.raises(CalvinFormatError, match="dimensions"):
        parse_calvin_cel(calvin(rows=3))
    with pytest.raises(CalvinFormatError, match="float32"):
        parse_calvin_cel(calvin(kind=5))


def test_bad_offset_rejected():
    payload = bytearray(calvin())
    payload[6:10] = _number(len(payload) + 10)
    with pytest.raises(CalvinFormatError, match="offset"):
        parse_calvin_cel(bytes(payload))
