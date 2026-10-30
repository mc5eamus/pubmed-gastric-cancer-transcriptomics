"""Auditable GEO acquisition and GPL570 gene-expression preparation.

CLI integration: call ``prepare_dataset(accession, cache, output,
allow_series_matrix=False)``. It writes gene×GSM ``expression.csv``, an
all-samples ``manifest.csv``, ``probe_gene_map.csv`` and ``provenance.json``.
Paths are resolved absolutely. Put ``cache`` under the ignored ``data/``
directory; multi-GB raw archives are NOT needed: individual CEL URLs are
taken directly from each GSM's metadata. No label is inferred from sample
order or from expression. A metadata conflict aborts the dataset.

Downloads use HTTPS, retries, atomic replacement and SHA256 sidecars. GEO
does not generally publish SHA256: absent an independently supplied hash,
the first download is trust-on-first-use, not publisher authentication.
Every cache hit is rehashed. CDF, annotation, HGNC and metadata have the
same checksummed provenance as CELs. Raw RMA is the default and has not
been validated for numerical parity with R/affy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse
import uuid

import numpy as np
import pandas as pd
import requests

from .rma import FormatError, read_cdf, rma


DATASETS = {
    "GSE66229": {"tumor": 300, "normal": 100, "total": 400, "biospecimen": "gastric tissue"},
    "GSE64951": {"tumor": 63, "normal": 31, "total": 94, "biospecimen": "cell-free saliva"},
    "GSE54129": {"tumor": 111, "normal": 21, "total": 132, "biospecimen": "gastric tissue"},
    "GSE13911": {"tumor": 38, "normal": 31, "total": 69, "biospecimen": "gastric tissue"},
}
CDF_URL = (
    "https://aroma-project.org/data/annotationData/chipTypes/"
    "HG-U133_Plus_2/HG-U133_Plus_2.cdf.gz"
)
ANNOTATION_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/"
    "tsv/tsv/hgnc_complete_set.txt"
)
GENE_ALIASES = {"METTL7A": "TMT1A", "SEPT10": "SEPTIN10", "FIGF": "VEGFD"}


class AcquisitionError(RuntimeError):
    """Acquisition failed; an explicit processed-data opt-in may be necessary."""


class CacheIntegrityError(AcquisitionError):
    """A cached object/sidecar is missing or disagrees; do not silently trust it."""


class MetadataError(ValueError):
    """Missing, conflicting, or unsupported sample metadata."""


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    try:
        with partial.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def _https(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "ftp" and parsed.netloc == "ftp.ncbi.nlm.nih.gov":
        parsed = parsed._replace(scheme="https")
    if parsed.scheme != "https":
        raise AcquisitionError(f"Only HTTPS acquisition URLs are supported: {url}")
    return parsed.geturl()


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    retries: int = 3,
    timeout: tuple[float, float] = (20, 180),
) -> Path:
    """Stream an atomic, checksummed HTTPS download; verify every cached read.

    Sidecar ``<filename>.sha256.json`` stores source, size, hash and timestamp.
    A cache without its sidecar, with the wrong source, or with a changed
    hash fails closed. Remove both files deliberately to reacquire. Optional
    ``expected_sha256`` verifies an independently known publisher hash.
    No partial file is promoted on a short read, failed status or bad hash.
    """
    url = _https(url)
    destination = Path(destination).resolve()
    sidecar = destination.with_name(destination.name + ".sha256.json")
    if retries < 1:
        raise ValueError("retries must be positive")
    if expected_sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", expected_sha256):
        raise ValueError("expected_sha256 must be a SHA256 hex digest")
    expected_sha256 = expected_sha256.lower() if expected_sha256 else None
    if destination.exists() or sidecar.exists():
        try:
            record = json.loads(sidecar.read_text())
            actual = sha256_file(destination)
            valid = (
                record["url"] == url
                and record["sha256"] == actual
                and record["bytes"] == destination.stat().st_size
                and (expected_sha256 is None or actual == expected_sha256)
            )
        except (OSError, ValueError, KeyError) as exc:
            raise CacheIntegrityError(f"Unverifiable cache entry: {destination}") from exc
        if not valid:
            raise CacheIntegrityError(f"Cached source, size or SHA256 mismatch: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    error = None
    for attempt in range(retries):
        partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            digest = hashlib.sha256()
            size = 0
            with requests.get(
                url, stream=True, timeout=timeout,
                headers={"User-Agent": "gastric-repro/0.1 (auditable GEO acquisition)", "Accept-Encoding": "identity"},
            ) as response:
                response.raise_for_status()
                _https(response.url)
                with partial.open("xb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                declared = response.headers.get("Content-Length")
                if declared and not response.headers.get("Content-Encoding") and size != int(declared):
                    raise AcquisitionError(f"Incomplete download: expected {declared} bytes, got {size}")
            if size == 0:
                raise AcquisitionError(f"Empty download: {url}")
            actual = digest.hexdigest()
            if expected_sha256 and actual != expected_sha256:
                raise CacheIntegrityError(f"Downloaded SHA256 differs from expected hash: {url}")
            record = {
                "url": url, "resolved_url": response.url, "sha256": actual, "bytes": size,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "hash_policy": "independent_expected_sha256" if expected_sha256 else "trust_on_first_use",
            }
            os.replace(partial, destination)
            _atomic_json(sidecar, record)
            return destination
        except (requests.RequestException, OSError, AcquisitionError, ValueError) as exc:
            error = exc
            if isinstance(exc, CacheIntegrityError):
                raise
            if attempt + 1 < retries:
                time.sleep(2**attempt)
        finally:
            partial.unlink(missing_ok=True)
    raise AcquisitionError(f"Download failed after {retries} attempts: {url}: {error}") from error


def _read_text(path: Path) -> str:
    with Path(path).open("rb") as handle:
        compressed = handle.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        return handle.read()


def _artifact(path: Path, role: str) -> dict:
    """Called after download verification; include source sidecar in provenance."""
    record = json.loads(path.with_name(path.name + ".sha256.json").read_text())
    return {"role": role, "path": str(path.resolve()), **record}


def parse_soft_samples(text: str) -> dict[str, dict[str, list[str]]]:
    """Read metadata-only GEO SOFT, preserving repeated characteristics."""
    samples = {}
    current = None
    in_table = False
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            gsm = line.split(" = ", 1)[1].strip()
            if not re.fullmatch(r"GSM\d+", gsm) or gsm in samples:
                raise MetadataError("Invalid or duplicate GSM in GEO metadata")
            current = defaultdict(list)
            samples[gsm] = current
            in_table = False
        elif line.startswith("^"):
            current = None
        elif line.lower() == "!sample_table_begin":
            in_table = True
        elif line.lower() == "!sample_table_end":
            in_table = False
        elif current is not None and not in_table and line.startswith("!Sample_") and " = " in line:
            key, value = line.split(" = ", 1)
            current[key.removeprefix("!Sample_")].append(value.strip())
    if not samples:
        raise MetadataError("No GEO sample records found (not valid SOFT metadata)")
    for gsm, sample in samples.items():
        if sample.get("geo_accession") != [gsm]:
            raise MetadataError(f"Missing/inconsistent geo_accession for {gsm}")
    return {gsm: dict(fields) for gsm, fields in samples.items()}


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().replace("_", " ")).strip()


def classify_sample(accession: str, sample: Mapping[str, list[str]]) -> tuple[int | None, bool, str, str]:
    """Return (label, included, reason, class) using explicit disease/tissue fields.

    Structured disease and tissue fields take precedence over contextual
    source text (normal tissue may come *from a gastric cancer patient*).
    Conflicting structured fields raise rather than silently choose a label.
    Non-GC disease is excluded, never relabelled as a healthy control.
    """
    if accession not in DATASETS:
        raise ValueError(f"Unsupported dataset: {accession}")
    platform = sample.get("platform_id", [])
    organism = sample.get("organism_ch1", [])
    if platform != ["GPL570"]:
        return None, False, "excluded: not GPL570", "unsupported_platform"
    if organism != ["Homo sapiens"]:
        return None, False, "excluded: not Homo sapiens", "unsupported_organism"
    normal = {
        "normal", "healthy", "healthy control", "control", "normal control",
        "normal gastric tissue", "normal gastric mucosa", "adjacent normal tissue",
        "noncancerous gastric tissue", "non-cancerous gastric tissue",
        "non-tumor", "non tumor", "non-tumour", "non tumour",
    }
    cancer = {
        "gastric cancer", "gastric carcinoma", "gastric adenocarcinoma",
        "gastric cancer tumor", "gastric cancer tumour", "gastric tumor", "gastric tumour",
    }
    generic_cancer = {"tumor", "tumour", "cancer", "adenocarcinoma", "carcinoma"}
    disease_keys = {"disease", "disease state", "disease status", "diagnosis"}
    tissue_keys = {"tissue", "tissue type", "histology", "sample type"}
    evidence: list[tuple[str, str, str]] = []
    for value in sample.get("characteristics_ch1", []):
        if ":" in value:
            key, value = value.split(":", 1)
            key, value = _normalized(key), _normalized(value)
            if key not in disease_keys | tissue_keys:
                continue
        else:
            key, value = "unkeyed tissue", _normalized(value)
        if value in normal:
            evidence.append(("normal", key, value))
        elif value in cancer or (accession != "GSE64951" and value in generic_cancer):
            evidence.append(("gastric_cancer", key, value))
        elif key in disease_keys and value not in {"", "--", "unknown", "not available", "na"}:
            if value in generic_cancer:
                raise MetadataError(f"Disease annotation does not specify gastric versus non-GC: {value}")
            explicitly_non_gc = bool(re.search(r"\bnon[- ]gastric\b", value))
            if not explicitly_non_gc and "gastric" in value and any(
                term in value for term in ("cancer", "carcinoma", "tumor", "tumour")
            ):
                raise MetadataError(f"Unrecognized/ambiguous gastric disease annotation: {value}")
            evidence.append(("other_disease", key, value))
    classes = {item[0] for item in evidence}
    if len(classes) > 1:
        raise MetadataError(f"Conflicting structured tissue/disease labels: {evidence}")
    if not classes:
        # Only exact descriptive labels are admissible, not arbitrary title substrings.
        for key in ("source_name_ch1", "title"):
            for value in sample.get(key, []):
                value = _normalized(value)
                value = re.sub(r"[\s-]*\d+$", "", value).strip()
                if value in normal:
                    evidence.append(("normal", key, value))
                elif value in cancer or (accession != "GSE64951" and value in generic_cancer | {"tumor biopsy"}):
                    evidence.append(("gastric_cancer", key, value))
        classes = {item[0] for item in evidence}
    if len(classes) != 1:
        raise MetadataError(f"Ambiguous or missing explicit sample label: {evidence or dict(sample)}")
    kind = classes.pop()
    proof = "; ".join(f"{key}: {value}" for _, key, value in evidence)
    if kind == "other_disease":
        return None, False, f"excluded non-GC disease: {proof}", kind
    return int(kind == "gastric_cancer"), True, f"included from explicit metadata: {proof}", kind


def build_manifest(accession: str, samples: Mapping[str, Mapping[str, list[str]]]) -> tuple[pd.DataFrame, dict]:
    """Classify every GSM and report expected-versus-observed counts explicitly."""
    rows = []
    errors = []
    for gsm, sample in samples.items():
        try:
            label, included, reason, kind = classify_sample(accession, sample)
        except MetadataError as exc:
            label, included, reason, kind = None, False, f"ERROR: {exc}", "ambiguous"
            errors.append(gsm)
        urls = [
            _https(url) for url in sample.get("supplementary_file", [])
            if re.search(r"\.cel(?:\.gz)?$", url, re.IGNORECASE)
        ]
        rows.append({
            "gsm": gsm, "dataset": accession, "label": label, "included": included,
            "reason": reason, "class": kind, "title": " | ".join(sample.get("title", [])),
            "source": " | ".join(sample.get("source_name_ch1", [])),
            "characteristics": " | ".join(sample.get("characteristics_ch1", [])),
            "biospecimen": DATASETS[accession]["biospecimen"],
            "platform": " | ".join(sample.get("platform_id", [])),
            "source_series": " | ".join(sample.get("series_id", [])),
            "cel_urls": json.dumps(urls), "cel_path": "", "cel_sha256": "",
        })
    manifest = pd.DataFrame(rows)
    manifest["label"] = pd.array(manifest["label"], dtype="Int64")
    observed = {
        "tumor": int(((manifest["label"] == 1) & manifest["included"]).sum()),
        "normal": int(((manifest["label"] == 0) & manifest["included"]).sum()),
        "total": int(manifest["included"].sum()),
        "available_total": len(manifest),
        "excluded": int((~manifest["included"]).sum()),
    }
    expected = {key: DATASETS[accession][key] for key in ("tumor", "normal", "total")}
    discrepancies = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items() if observed[key] != value
    }
    report = {
        "expected": expected, "observed": observed,
        "count_status": "explicit_discrepancy" if discrepancies else "matches_expected",
        "discrepancies": discrepancies, "ambiguous_gsms": errors,
    }
    return manifest, report


def _table(text: str, begin: str, end: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next((i + 1 for i, line in enumerate(lines) if line.lower() == begin.lower()), None)
    stop = next((i for i, line in enumerate(lines) if line.lower() == end.lower()), None)
    if start is None or stop is None or stop <= start:
        raise FormatError(f"Missing or incomplete table: {begin}")
    return pd.read_csv(io.StringIO("\n".join(lines[start:stop])), sep="\t", dtype=str, keep_default_na=False)


def parse_probe_annotation(
    text: str, aliases: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Parse GPL570 annotation/SOFT table; omit controls and ambiguous mappings.

    With a maintained HGNC map, a multi-symbol annotation may resolve to one
    current approved gene after unrecognized/withdrawn names are removed.
    Example: the 2016 ``PIR-FIGF///FIGF`` annotation resolves to VEGFD because
    PIR-FIGF is not a current HGNC symbol/alias, while FIGF is VEGFD's old name.
    Multiple remaining current genes are still excluded; no first-symbol pick.
    """
    for begin, end in (
        ("!platform_table_begin", "!platform_table_end"),
        ("!annotation_table_begin", "!annotation_table_end"),
    ):
        if begin in text.lower():
            table = _table(text, begin, end)
            break
    else:
        raise FormatError("No GPL570 annotation table")
    columns = {column.casefold().replace("_", " "): column for column in table}
    probe_column = columns.get("id") or columns.get("id ref")
    gene_column = columns.get("gene symbol") or columns.get("symbol")
    if probe_column is None or gene_column is None:
        raise FormatError("GPL570 annotation lacks ID / Gene symbol columns")
    mapping = {}
    for probe, gene in zip(table[probe_column], table[gene_column], strict=True):
        probe, gene = probe.strip(), gene.strip()
        if not probe or probe.upper().startswith("AFFX-") or gene in {"", "---", "--"}:
            continue
        if re.search(r"///|//|;|,|\s", gene):
            if aliases is None:
                continue
            symbols = re.split(r"\s*(?:///|//|;|,)\s*", gene)
            resolved = {aliases[symbol] for symbol in symbols if symbol in aliases}
            if len(resolved) != 1:
                continue
            gene = resolved.pop()
        if probe in mapping and mapping[probe] != gene:
            raise FormatError(f"Conflicting GPL570 annotation for {probe}")
        mapping[probe] = gene
    if not mapping:
        raise FormatError("No unambiguous GPL570 probe-to-gene mappings")
    return mapping


def hgnc_aliases(text: str) -> dict[str, str]:
    """Map unique HGNC previous/alias symbols to current approved symbols.

    Current approved names always win. Aliases pointing at multiple genes are
    omitted, not arbitrarily selected. The downloaded HGNC snapshot is hashed.
    """
    table = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    if not {"symbol", "status", "alias_symbol", "prev_symbol"} <= set(table):
        raise FormatError("Invalid HGNC complete-set columns")
    table = table.loc[table["status"] == "Approved"]
    approved = set(table["symbol"])
    if not approved:
        raise FormatError("HGNC table has no approved genes")
    candidates = defaultdict(set)
    for row in table.itertuples(index=False):
        for field in (row.alias_symbol, row.prev_symbol):
            for alias in field.split("|"):
                if alias and alias not in approved:
                    candidates[alias].add(row.symbol)
    aliases = {alias: next(iter(targets)) for alias, targets in candidates.items() if len(targets) == 1}
    aliases.update({symbol: symbol for symbol in approved})
    for alias, canonical in GENE_ALIASES.items():
        if canonical in approved and alias not in approved:
            aliases[alias] = canonical
    return aliases


def canonical_gene(symbol: str, aliases: Mapping[str, str] | None = None) -> str:
    """Resolve current HGNC names; the three documented renamings work offline."""
    if aliases is not None:
        return aliases.get(symbol, symbol)
    return GENE_ALIASES.get(symbol, symbol)


def collapse_probes(
    expression: pd.DataFrame, annotation: Mapping[str, str], aliases: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the highest-mean probe per canonical gene, ties by probe ID.

    Multi-gene probes must have been excluded by ``parse_probe_annotation``.
    Selection uses all included samples, without their outcome labels.
    Return gene×sample values plus the auditable probe/gene/mean/selected map.
    """
    values = expression.to_numpy(dtype=float)
    if expression.empty or expression.index.has_duplicates or expression.columns.has_duplicates or not np.isfinite(values).all():
        raise ValueError("Expression must have unique axes and finite nonempty values")
    rows = [
        {"probe": str(probe), "gene": canonical_gene(annotation[str(probe)], aliases), "mean": float(mean)}
        for probe, mean in zip(expression.index, values.mean(axis=1), strict=True)
        if str(probe) in annotation
    ]
    if not rows:
        raise ValueError("No expression probes map unambiguously to genes")
    mapping = pd.DataFrame(rows).sort_values(["gene", "mean", "probe"], ascending=[True, False, True])
    mapping["selected"] = ~mapping.duplicated("gene")
    selected = mapping.loc[mapping["selected"]]
    result = expression.loc[selected["probe"]].copy()
    result.index = pd.Index(selected["gene"], name="gene")
    return result, mapping.reset_index(drop=True)


def acquire_metadata(accession: str, cache: Path) -> tuple[dict, list[dict]]:
    """Acquire metadata without downloading full SOFT expression or RAW archives.

    GSE66229's superseries metadata contains both GSE62254 tumors and GSE66222
    normals; CEL links are GSM-level and therefore also work for subseries.
    """
    artifacts = []
    base = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
    series = download(
        f"{base}?acc={accession}&targ=self&form=text&view=full",
        Path(cache) / accession / "series.soft",
    )
    sample_path = download(
        f"{base}?acc={accession}&targ=gsm&form=text&view=brief",
        Path(cache) / accession / "samples.soft",
    )
    artifacts.extend([_artifact(series, "series_metadata"), _artifact(sample_path, "sample_metadata")])
    series_text = _read_text(series)
    if f"^SERIES = {accession}" not in series_text:
        raise MetadataError("GEO response is not the requested series metadata")
    expected_gsms = set(re.findall(r"^!Series_sample_id = (GSM\d+)\s*$", series_text, re.MULTILINE))
    samples = parse_soft_samples(_read_text(sample_path))
    if expected_gsms != set(samples):
        raise MetadataError("Series membership differs from acquired GSM metadata")
    return samples, artifacts


def acquire_cels(manifest: pd.DataFrame, cache: Path, artifacts: list[dict]) -> dict[str, Path]:
    """Download only included individual GSM CELs, rejecting ambiguous file links."""
    result = {}
    for index, row in manifest.loc[manifest["included"]].iterrows():
        urls = json.loads(row["cel_urls"])
        if len(urls) != 1:
            raise AcquisitionError(f"{row['gsm']}: expected exactly one CEL link, found {len(urls)}")
        suffix = ".CEL.gz" if urls[0].lower().endswith(".gz") else ".CEL"
        path = download(urls[0], Path(cache) / row["dataset"] / "cel" / (row["gsm"] + suffix))
        artifact = _artifact(path, "raw_CEL")
        artifacts.append({"gsm": row["gsm"], **artifact})
        manifest.at[index, "cel_path"] = str(path)
        manifest.at[index, "cel_sha256"] = artifact["sha256"]
        result[row["gsm"]] = path
    return result


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read processed GEO matrix and per-sample processing metadata, never raw CEL."""
    text = _read_text(path)
    metadata = defaultdict(list)
    for fields in csv.reader(io.StringIO(text), delimiter="\t"):
        if fields and fields[0].startswith("!Sample_"):
            metadata[fields[0].removeprefix("!Sample_")].append(fields[1:])
    table = _table(text, "!series_matrix_table_begin", "!series_matrix_table_end")
    if "ID_REF" not in table:
        raise FormatError("Series matrix lacks ID_REF")
    try:
        table = table.set_index("ID_REF").apply(pd.to_numeric, errors="raise")
    except ValueError as exc:
        raise FormatError("Series matrix contains nonnumeric or missing intensities") from exc
    if table.empty or table.index.has_duplicates or table.columns.has_duplicates or not np.isfinite(table.to_numpy()).all():
        raise FormatError("Invalid processed series matrix")
    return table, dict(metadata)


def series_matrix_to_log2(accession: str, expression: pd.DataFrame, samples: Mapping) -> tuple[pd.DataFrame, str]:
    """Apply a source-documented scale conversion, never a range heuristic.

    GSE66229: submitted log10 RMA → log2; GSE54129: submitted log2 RMA;
    GSE64951/GSE13911: submitted linear MAS5 → log2 (strictly positive).
    GSM processing metadata must corroborate the normalization family.
    """
    processing = " ".join(
        value for sample in samples.values() for value in sample.get("data_processing", [])
    ).casefold()
    if accession == "GSE66229":
        if "rma" not in processing:
            raise MetadataError("GSE66229 fallback requires documented RMA processing")
        return expression * np.log2(10), "submitted log10 RMA multiplied by log2(10); not raw reprocessing"
    if accession == "GSE54129":
        if "rma" not in processing and "robust multichip" not in processing:
            raise MetadataError("GSE54129 fallback requires documented RMA processing")
        return expression.copy(), "submitted RMA log2 values unchanged; not raw reprocessing"
    if accession in {"GSE64951", "GSE13911"}:
        if not any(term in processing for term in ("mas", "expression console", "global normalization")):
            raise MetadataError("Fallback requires documented MAS5/Expression Console processing")
        if (expression.to_numpy() <= 0).any():
            raise ValueError("Cannot log2-transform nonpositive submitted MAS5 intensities")
        return np.log2(expression), "log2 of submitted linear MAS5 signals; NOT RMA"
    raise ValueError(f"Unsupported dataset: {accession}")


def _acquire_series_matrix(accession: str, cache: Path, artifacts: list[dict]) -> pd.DataFrame:
    # Superseries can expose only subseries matrices; verify actual coverage below.
    sources = [accession] if accession != "GSE66229" else ["GSE62254", "GSE66222"]
    tables = []
    for source in sources:
        group = source[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{group}/{source}/matrix/{source}_series_matrix.txt.gz"
        path = download(url, Path(cache) / accession / f"{source}_series_matrix.txt.gz")
        artifacts.append(_artifact(path, "submitted_series_matrix"))
        table, _ = read_series_matrix(path)
        tables.append(table)
    if len(tables) > 1 and not all(set(table.index) == set(tables[0].index) for table in tables[1:]):
        raise FormatError("Subseries matrices have different probe sets")
    result = pd.concat(tables, axis=1)
    if result.columns.has_duplicates:
        raise FormatError("Duplicate GSMs across submitted matrices")
    return result


def _write_frame(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    _atomic_bytes(path, frame.to_csv(index=index, float_format="%.10g").encode())


def download_dataset(
    accession: str, cache: Path, output: Path, *, metadata_only: bool = False,
) -> dict:
    """Acquire one cohort without normalization; return acquisition provenance.

    Always write ``manifest.csv`` and ``provenance.json`` under ``output``.
    With ``metadata_only=True``, no CEL, CDF, annotation or HGNC downloads
    occur. Otherwise acquire all raw inputs needed by ``prepare_dataset``;
    its subsequent normalization reuses and rehashes the same cache.
    There is no implicit processed-data fallback in this acquisition stage.
    A complete acquisition is NOT a complete expression reproduction:
    ``normalization`` is always ``not_run`` and ``expression_available`` false.
    """
    accession = accession.upper()
    if accession not in DATASETS:
        raise ValueError(f"Supported datasets: {', '.join(DATASETS)}")
    cache, output = Path(cache).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "dataset": accession, "status": "in_progress", "platform": "GPL570",
        "stage": "metadata_only" if metadata_only else "raw_download",
        "biospecimen": DATASETS[accession]["biospecimen"],
        "normalization": "not_run", "expression_available": False,
        "raw_rma_parity_validated": False, "artifacts": [],
        "label_policy": "explicit metadata; exclude non-GC disease; fail ambiguity",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "provenance.json", provenance)
    try:
        samples, artifacts = acquire_metadata(accession, cache)
        provenance["artifacts"] = artifacts
        manifest, counts = build_manifest(accession, samples)
        provenance["sample_counts"] = counts
        _write_frame(manifest, output / "manifest.csv", index=False)
        if counts["ambiguous_gsms"]:
            raise MetadataError(f"Ambiguous labels; inspect manifest.csv: {counts['ambiguous_gsms']}")
        if not metadata_only:
            for url, relative, role in (
                (CDF_URL, "GPL570/HG-U133_Plus_2.cdf.gz", "PM_CDF"),
                (ANNOTATION_URL, "GPL570/GPL570.annot.gz", "probe_gene_annotation"),
                (HGNC_URL, "HGNC/hgnc_complete_set.txt", "HGNC_approved_alias_snapshot"),
            ):
                path = download(url, cache / relative)
                artifacts.append(_artifact(path, role))
            acquire_cels(manifest, cache, artifacts)
            _write_frame(manifest, output / "manifest.csv", index=False)
        provenance.update({
            "status": "complete",
            "n_included": int(manifest["included"].sum()),
            "n_excluded": int((~manifest["included"]).sum()),
            "raw_cel_files_downloaded": 0 if metadata_only else int(manifest["included"].sum()),
            "outputs": {
                "manifest.csv": {
                    "path": str(output / "manifest.csv"),
                    "sha256": sha256_file(output / "manifest.csv"),
                },
            },
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        provenance.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _atomic_json(output / "provenance.json", provenance)
        raise
    _atomic_json(output / "provenance.json", provenance)
    return provenance


def prepare_dataset(
    accession: str, cache: Path, output: Path, *, cdf: Path | None = None,
    allow_series_matrix: bool = False,
) -> dict:
    """Prepare a supported dataset and return the JSON-serializable provenance.

    Full downloads and raw normalization can take hours and several GB RAM.
    Run this explicitly, not during unit tests. ``allow_series_matrix`` permits
    fallback ONLY after a recorded acquisition/format/RMA failure; metadata
    ambiguity and cache-integrity failures never trigger fallback. Count
    discrepancies are recorded prominently; unknown labels abort with the
    all-GSM manifest retained. Raw/processed paths are never mixed per sample.
    Output provenance has status ``complete`` only after every artifact exists.
    ``cdf`` optionally supplies a local standard GPL570 ASCII/XDA CDF (also
    gzip); it is SHA256-recorded instead of downloading the public CDF mirror.
    """
    accession = accession.upper()
    if accession not in DATASETS:
        raise ValueError(f"Supported datasets: {', '.join(DATASETS)}")
    cache, output = Path(cache).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Replace any old success marker before a new attempt can fail.
    provenance = {
        "dataset": accession, "status": "in_progress", "platform": "GPL570",
        "biospecimen": DATASETS[accession]["biospecimen"], "artifacts": [],
        "raw_rma_parity_validated": False, "allow_series_matrix": allow_series_matrix,
        "label_policy": "explicit metadata; exclude non-GC disease; fail ambiguity",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output / "provenance.json", provenance)
    try:
        samples, artifacts = acquire_metadata(accession, cache)
        provenance["artifacts"] = artifacts
        manifest, counts = build_manifest(accession, samples)
        provenance["sample_counts"] = counts
        _write_frame(manifest, output / "manifest.csv", index=False)
        if counts["ambiguous_gsms"]:
            raise MetadataError(f"Ambiguous labels; inspect manifest.csv: {counts['ambiguous_gsms']}")
        included = manifest.loc[manifest["included"], "gsm"].tolist()
        if not included or not {0, 1} <= set(manifest.loc[manifest["included"], "label"]):
            raise MetadataError("Both tumor and normal samples are required")
        annotation_path = download(ANNOTATION_URL, cache / "GPL570" / "GPL570.annot.gz")
        artifacts.append(_artifact(annotation_path, "probe_gene_annotation"))
        hgnc_path = download(HGNC_URL, cache / "HGNC" / "hgnc_complete_set.txt")
        artifacts.append(_artifact(hgnc_path, "HGNC_approved_alias_snapshot"))
        aliases = hgnc_aliases(_read_text(hgnc_path))
        annotation = parse_probe_annotation(_read_text(annotation_path), aliases)
        provenance["gene_policy"] = (
            "unique GPL570 symbol or unique current HGNC gene after removing unrecognized/"
            "withdrawn names from multi-symbol annotations; exclude genuinely multi-gene "
            "probes; current HGNC aliases; highest mean probe; probe-ID tie break"
        )
        try:
            if cdf is None:
                cdf_path = download(CDF_URL, cache / "GPL570" / "HG-U133_Plus_2.cdf.gz")
                artifacts.append(_artifact(cdf_path, "PM_CDF"))
            else:
                cdf_path = Path(cdf).resolve()
                artifacts.append({
                    "role": "PM_CDF", "path": str(cdf_path),
                    "source": "explicit user-supplied local CDF",
                    "sha256": sha256_file(cdf_path), "bytes": cdf_path.stat().st_size,
                    "hash_policy": "local_snapshot; not publisher-authenticated",
                })
            chip = read_cdf(cdf_path)
            if chip.shape != (1164, 1164) or len(chip.probesets) != 54675:
                raise FormatError("CDF is not the standard 54,675-probe-set GPL570 definition")
            del chip
            cel_files = acquire_cels(manifest, cache, artifacts)
            _write_frame(manifest, output / "manifest.csv", index=False)
            probes = rma(cel_files, cdf_path)
            provenance.update({
                "normalization": "raw_PM_RMA_python", "expression_scale": "log2",
                "normalization_details": "normal-exponential background; mean-rank quantiles; log2; Tukey median polish",
                "parity_caveat": "Binned KDE differs from preprocessCore; no affy/oligo numerical parity validation",
            })
        except CacheIntegrityError:
            raise
        except (AcquisitionError, FormatError, ValueError, OSError) as exc:
            if not allow_series_matrix:
                raise AcquisitionError(
                    f"Raw CEL RMA failed: {exc}. Processed fallback requires allow_series_matrix=True."
                ) from exc
            submitted = _acquire_series_matrix(accession, cache, artifacts)
            missing = set(included) - set(submitted.columns)
            extra = set(submitted.columns) - set(samples)
            if missing or extra:
                raise MetadataError(f"Series-matrix GSM mismatch: missing={sorted(missing)}, unknown={sorted(extra)}")
            probes, conversion = series_matrix_to_log2(accession, submitted.loc[:, included], samples)
            provenance.update({
                "normalization": "submitted_series_matrix_fallback", "expression_scale": "log2",
                "fallback_reason": f"{type(exc).__name__}: {exc}", "scale_conversion": conversion,
                "fallback_warning": "Processed submitted values are NOT raw RMA reproduction",
            })
        if list(probes.columns) != included:
            raise MetadataError("Expression columns are not exactly the included manifest GSMs")
        expression, mapping = collapse_probes(probes, annotation, aliases)
        _write_frame(expression, output / "expression.csv", index=True)
        _write_frame(mapping, output / "probe_gene_map.csv", index=False)
        _write_frame(manifest, output / "manifest.csv", index=False)
        provenance["outputs"] = {
            name: {"path": str(output / name), "sha256": sha256_file(output / name)}
            for name in ("expression.csv", "manifest.csv", "probe_gene_map.csv")
        }
        provenance.update({
            "status": "complete", "n_genes": len(expression), "n_samples": len(included),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as exc:
        provenance.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _atomic_json(output / "provenance.json", provenance)
        raise
    _atomic_json(output / "provenance.json", provenance)
    return provenance
