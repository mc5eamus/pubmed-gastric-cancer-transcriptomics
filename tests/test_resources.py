import hashlib
import io
import json
import zipfile

import pytest

from gastric_repro.resources import article_tables, docx_tables, fetch_resource


def test_cache_requires_matching_hash_and_url(tmp_path):
    path = tmp_path / "source.gmt"
    path.write_bytes(b"term\t\tGENE\n")
    receipt = path.with_suffix(".gmt.source.json")
    record = {"url": "https://example.test/gmt",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt.write_text(json.dumps(record))
    assert fetch_resource(record["url"], path) == record
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="verification"):
        fetch_resource(record["url"], path)


def test_docx_table_transcription():
    payload = io.BytesIO()
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/'
        'wordprocessingml/2006/main"><w:body><w:tbl><w:tr>'
        '<w:tc><w:p><w:r><w:t>limma</w:t></w:r></w:p></w:tc>'
        '<w:tc><w:p><w:r><w:t>3.58.1</w:t></w:r></w:p></w:tc>'
        '</w:tr></w:tbl></w:body></w:document>'
    )
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("word/document.xml", xml)
    table = docx_tables(payload.getvalue())
    assert table.text.tolist() == ["limma", "3.58.1"]
    assert table.column.tolist() == [1, 2]


def test_article_transcription_preserves_intervals():
    payload = b"""<article><table-wrap id="Tab1"><table><tbody><tr>
    <td>SVM</td><td>0.868 (<bold>0.721</bold>-0.974)</td>
    </tr></tbody></table></table-wrap></article>"""
    table = article_tables(payload)
    assert table.text.tolist() == ["SVM", "0.868 (0.721-0.974)"]
    assert table.table.tolist() == ["Tab1", "Tab1"]
