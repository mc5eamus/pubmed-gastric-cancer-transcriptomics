"""Cache provenance for the publication, supplement and versioned GO snapshots."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests


ARTICLE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12116728/fullTextXML"
SUPPLEMENT_URL = (
    "https://static-content.springer-cdn.com/esm/"
    "art%3A10.1038%2Fs41598-025-96864-0/MediaObjects/"
    "41598_2025_96864_MOESM1_ESM.docx"
)
GO_LIBRARIES = {
    "BP": "GO_Biological_Process_2023",
    "CC": "GO_Cellular_Component_2023",
    "MF": "GO_Molecular_Function_2023",
}


def fetch_resource(url: str, destination: Path) -> dict:
    """Cache bounded reference resources, verifying content on every reuse."""
    destination = Path(destination)
    receipt = destination.with_suffix(destination.suffix + ".source.json")
    if destination.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if record.get("url") != url or record.get("sha256") != actual:
            raise ValueError(f"Cached reference failed provenance verification: {destination}")
        return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    temp = None
    try:
        with requests.get(url, timeout=(30, 180), stream=True) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                temp = Path(handle.name)
                for block in response.iter_content(1024 * 1024):
                    size += len(block)
                    if size > 100 * 1024 * 1024:
                        raise ValueError("Reference download exceeds 100 MiB limit")
                    digest.update(block)
                    handle.write(block)
        if size == 0:
            raise ValueError(f"Empty resource: {url}")
        os.replace(temp, destination)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    record = {"url": url, "sha256": digest.hexdigest(), "bytes": size}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    return record


def go_snapshots(cache: Path) -> dict[str, Path]:
    result = {}
    for ontology, library in GO_LIBRARIES.items():
        path = Path(cache) / "references" / f"{library}.gmt"
        fetch_resource(
            "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=" + library,
            path,
        )
        result[ontology] = path
    return result


def docx_tables(payload: bytes) -> pd.DataFrame:
    """Transcribe every supplement table cell without guessing table headings."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entry = archive.getinfo("word/document.xml")
        if entry.file_size > 20 * 1024 * 1024:
            raise ValueError("Supplement document XML exceeds size limit")
        root = ET.fromstring(archive.read(entry))
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    rows = []
    for table_id, table in enumerate(root.findall(".//w:tbl", ns), start=1):
        for row_id, row in enumerate(table.findall("w:tr", ns), start=1):
            for column_id, cell in enumerate(row.findall("w:tc", ns), start=1):
                text = " ".join(
                    "".join(paragraph.itertext())
                    for paragraph in cell.findall("w:p", ns)
                ).strip()
                rows.append(dict(table=table_id, row=row_id, column=column_id, text=text))
    return pd.DataFrame(rows, columns=["table", "row", "column", "text"])


def article_tables(payload: bytes) -> pd.DataFrame:
    root = ET.fromstring(payload)
    rows = []
    for table in root.findall(".//table-wrap"):
        for row_id, row in enumerate(table.findall(".//tr"), start=1):
            for column_id, cell in enumerate(list(row), start=1):
                rows.append(dict(
                    table=table.attrib.get("id"), row=row_id, column=column_id,
                    text="".join(cell.itertext()).strip(),
                ))
    return pd.DataFrame(rows, columns=["table", "row", "column", "text"])


def publication_evidence(cache: Path, output: Path) -> dict:
    """Keep failures visible; an unavailable supplement is never invented."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for key, url, filename in (
        ("article", ARTICLE_URL, "PMC12116728.xml"),
        ("supplement", SUPPLEMENT_URL, "41598_2025_96864_MOESM1_ESM.docx"),
    ):
        path = Path(cache) / "references" / filename
        try:
            sources[key] = {"status": "downloaded", **fetch_resource(url, path)}
            if key == "supplement":
                docx_tables(path.read_bytes()).to_csv(output / "supplement_tables.csv", index=False)
            else:
                article_tables(path.read_bytes()).to_csv(output / "article_tables.csv", index=False)
        except (requests.RequestException, OSError, ValueError, zipfile.BadZipFile,
                KeyError, ET.ParseError) as error:
            sources[key] = {"status": "unavailable", "url": url, "error": str(error)}
    (output / "publication_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    return sources
