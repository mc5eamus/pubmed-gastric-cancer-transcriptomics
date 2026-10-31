"""Reproducible GO preranking against user-supplied, local GMT snapshots."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GO_ONTOLOGIES = ("BP", "CC", "MF")


def ranked_gene_list(results: pd.DataFrame) -> pd.DataFrame:
    """Rank the ENTIRE filtered testing universe by signed moderated t.

    Do not subset to DEGs before calling. Gene identifiers must already be in
    the identifier namespace used by the GMT. Ties are ordered by gene name.
    """
    if not isinstance(results, pd.DataFrame) or results.empty or "t" not in results:
        raise ValueError("results must be a nonempty DataFrame with moderated t")
    if not results.index.is_unique or results.index.hasnans:
        raise ValueError("ranked gene identifiers must be unique and nonmissing")
    names = results.index.astype(str)
    if not names.is_unique or any(not name.strip() for name in names):
        raise ValueError("gene identifiers must be unique nonempty strings")
    scores = results["t"].to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("moderated t scores must be finite")
    ranks = pd.DataFrame({"gene": names, "score": scores})
    return ranks.sort_values(["score", "gene"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _gmt_provenance(path: Path, ranked_genes: set[str]) -> dict:
    if not path.is_file():
        raise ValueError(f"local GMT file does not exist: {path}")
    payload = path.read_bytes()
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"GMT must be UTF-8: {path}") from exc
    terms = set()
    gmt_genes = set()
    for line in lines:
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 3 or not fields[0].strip() or not any(gene.strip() for gene in fields[2:]):
            raise ValueError(f"malformed GMT record in {path}")
        if fields[0] in terms:
            raise ValueError(f"duplicate GMT term {fields[0]!r} in {path}")
        terms.add(fields[0])
        gmt_genes.update(gene for gene in fields[2:] if gene.strip())
    if not terms:
        raise ValueError(f"GMT contains no gene sets: {path}")
    overlap = len(ranked_genes & gmt_genes)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "terms": len(terms),
        "gmt_gene_count": len(gmt_genes),
        "ranked_gene_overlap_count": overlap,
        "ranked_gene_overlap_fraction": overlap / len(ranked_genes),
    }


def run_go_prerank(
    results: pd.DataFrame,
    gene_sets: dict[str, str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    permutations: int = 1000,
    min_size: int = 15,
    max_size: int = 500,
    threads: int = 1,
    source_versions: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Run gseapy prerank for BP, CC, and MF using immutable local snapshots.

    A local GMT for every ontology is mandatory; online library aliases are
    deliberately not accepted. SHA256 identifies each exact snapshot, with
    optional human-readable release labels. FDR is gseapy's per-ontology
    permutation-based GSEA FDR, not a pooled BH adjustment across ontologies.
    Results and machine-readable provenance are written to output_dir. Each
    ontology records exact-symbol overlap with the ranked universe before
    min/max gene-set filtering, without case conversion or alias rewriting.
    """
    ranks = ranked_gene_list(results)
    if not isinstance(gene_sets, dict) or set(gene_sets) != set(GO_ONTOLOGIES):
        raise ValueError("gene_sets must contain exactly BP, CC, and MF local GMT paths")
    for name, value in (("seed", seed), ("permutations", permutations), ("min_size", min_size),
                        ("max_size", max_size), ("threads", threads)):
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a {'nonnegative' if name == 'seed' else 'positive'} integer")
    if seed > 2**32 - 1:
        raise ValueError("seed must fit in an unsigned 32-bit integer")
    if min_size > max_size:
        raise ValueError("min_size must not exceed max_size")
    if source_versions is not None and (
        set(source_versions) != set(GO_ONTOLOGIES)
        or any(not isinstance(value, str) or not value.strip() for value in source_versions.values())
    ):
        raise ValueError("source_versions must provide a nonempty release label for BP, CC, and MF")
    sources = {}
    ranked_genes = set(ranks["gene"])
    for ontology in GO_ONTOLOGIES:
        path = Path(gene_sets[ontology])
        sources[ontology] = _gmt_provenance(path, ranked_genes)
        if source_versions is not None:
            sources[ontology]["version"] = source_versions[ontology]
    # Import only after validating local inputs; this module has no network fallback.
    import gseapy

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rank_bytes = ranks.to_csv(index=False, sep="\t", float_format="%.17g").encode("utf-8")
    (output / "ranked_genes.tsv").write_bytes(rank_bytes)
    provenance = {
        "method": "gseapy.prerank",
        "gseapy_version": gseapy.__version__,
        "ranking": "signed moderated t, all supplied filtered genes, descending; gene-name tie break",
        "universe_size": len(ranks),
        "tied_score_count": int(ranks["score"].duplicated(keep=False).sum()),
        "rank_sha256": hashlib.sha256(rank_bytes).hexdigest(),
        "seed": seed,
        "permutations": permutations,
        "min_size": min_size,
        "max_size": max_size,
        "threads": threads,
        "sources": sources,
        "overlap_definition": "exact-symbol ranked genes in any GMT term, before min/max filtering; fraction of ranked universe",
        "fdr_scope": "per ontology, GSEA permutation FDR",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    tables = []
    for ontology in GO_ONTOLOGIES:
        run = gseapy.prerank(
            rnk=ranks.copy(),
            gene_sets=str(Path(gene_sets[ontology]).resolve()),
            outdir=str(output / ontology),
            seed=seed,
            permutation_num=permutations,
            min_size=min_size,
            max_size=max_size,
            threads=threads,
            ascending=False,
            no_plot=True,
            verbose=False,
        )
        table = run.res2d.copy()
        table.insert(0, "ontology", ontology)
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(output / "go_prerank.tsv", sep="\t", index=False)
    combined.attrs["provenance"] = provenance
    return combined
