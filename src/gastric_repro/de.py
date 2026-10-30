"""Two-group, log2-scale OLS with empirical-Bayes variance moderation.

This is a transparent limma-style implementation, not an R/limma binding.
It assumes independent samples, an intercept and a binary group coefficient,
and does not implement limma's trend or robust empirical-Bayes extensions.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


DEFAULT_FILTER_RULE = "any_above_median"
FILTER_RULES = (
    DEFAULT_FILTER_RULE,
    "mean_above_median",
    "half_above_median",
)


def _validate_expression(expression: pd.DataFrame) -> np.ndarray:
    if not isinstance(expression, pd.DataFrame):
        raise TypeError("expression must be a genes-by-samples DataFrame")
    if expression.empty:
        raise ValueError("expression must contain genes and samples")
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("gene and sample identifiers must be unique")
    if expression.index.hasnans or expression.columns.hasnans:
        raise ValueError("gene and sample identifiers must not be missing")
    try:
        values = expression.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("expression must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("expression must contain only finite values")
    return values


def _aligned_labels(expression: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    if not isinstance(labels, pd.Series):
        raise TypeError("labels must be a Series indexed by sample")
    if not labels.index.is_unique or labels.index.hasnans:
        raise ValueError("label sample identifiers must be unique and nonmissing")
    if len(labels) != expression.shape[1] or set(labels.index) != set(expression.columns):
        raise ValueError("labels must match expression samples exactly")
    aligned = labels.reindex(expression.columns)
    if aligned.isna().any() or not aligned.isin([0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if aligned.nunique() != 2:
        raise ValueError("both groups 0 and 1 are required")
    return aligned.to_numpy(dtype=int)


def filter_expression(
    expression: pd.DataFrame, rule: str = DEFAULT_FILTER_RULE
) -> pd.DataFrame:
    """Filter using a fixed global median, without consulting DE outcomes.

    All comparisons are strict. ``half_above_median`` requires at least
    ceil(number of samples / 2) samples above the global matrix median.
    No rule is selected to attain a target gene or differential-expression count.
    """
    values = _validate_expression(expression)
    if rule not in FILTER_RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {FILTER_RULES}")
    median = float(np.median(values))
    above = values > median
    if rule == "any_above_median":
        keep = above.any(axis=1)
    elif rule == "mean_above_median":
        keep = values.mean(axis=1) > median
    else:
        keep = above.sum(axis=1) >= (values.shape[1] + 1) // 2
    filtered = expression.loc[keep].copy()
    filtered.attrs.update(
        filter_rule=rule,
        global_median=median,
        input_gene_count=len(expression),
        retained_gene_count=int(keep.sum()),
    )
    return filtered


def filter_sensitivity_counts(expression: pd.DataFrame) -> pd.DataFrame:
    """Return retained/removed counts for each predeclared median rule."""
    return pd.DataFrame(
        [
            {
                "rule": rule,
                "input_genes": len(expression),
                "retained_genes": len(filter_expression(expression, rule)),
                "global_median": float(np.median(_validate_expression(expression))),
            }
            for rule in FILTER_RULES
        ]
    ).assign(removed_genes=lambda frame: frame.input_genes - frame.retained_genes)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, preserving the input order."""
    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all():
        raise ValueError("pvalues must be a finite one-dimensional array")
    if ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError("pvalues must lie in [0, 1]")
    if not pvalues.size:
        return pvalues.copy()
    order = np.argsort(pvalues, kind="stable")
    adjusted = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def squeeze_var(variances: np.ndarray, df_residual: float) -> dict:
    """Moment-match a scaled inverse-chi-square prior on residual variances.

    The sampling log-variance mean and variance are corrected using digamma
    and trigamma at df_residual/2. Nonpositive excess log-variance implies an
    infinite-prior-df (common-variance) limit. Zeros are floored only for prior
    estimation; original residual variances enter posterior pooling.
    """
    variances = np.asarray(variances, dtype=float)
    if variances.ndim != 1 or not variances.size:
        raise ValueError("variances must be a nonempty one-dimensional array")
    if not np.isfinite(variances).all() or (variances < 0).any():
        raise ValueError("variances must be finite and nonnegative")
    if not np.isfinite(df_residual) or df_residual <= 0:
        raise ValueError("residual degrees of freedom must be positive and finite")
    positive = variances[variances > 0]
    floor = max(float(np.median(positive)) * 1e-8, np.finfo(float).tiny) if positive.size else 1e-12
    safe = np.maximum(variances, floor)
    corrected = np.log(safe) - special.digamma(df_residual / 2) + np.log(df_residual / 2)
    excess = (
        float(np.var(corrected, ddof=1) - special.polygamma(1, df_residual / 2))
        if variances.size > 1 else 0.0
    )
    if excess <= 0:
        df_prior = np.inf
        log_prior = float(corrected.mean())
    else:
        # Solve on a log scale to retain accuracy for small or large priors.
        root = optimize.brentq(
            lambda log_x: float(special.polygamma(1, np.exp(log_x))) - excess,
            -350,
            350,
        )
        df_prior = float(2 * np.exp(root))
        log_prior = float(
            corrected.mean()
            + special.digamma(df_prior / 2)
            - np.log(df_prior / 2)
        )
    prior = float(np.exp(np.clip(log_prior, np.log(np.finfo(float).tiny), np.log(np.finfo(float).max))))
    if np.isinf(df_prior):
        posterior = np.full_like(variances, prior)
    else:
        weight = df_residual / (df_residual + df_prior)
        posterior = weight * variances + (1 - weight) * prior
    posterior = np.maximum(posterior, np.finfo(float).tiny)
    return {"var_prior": prior, "df_prior": df_prior, "var_post": posterior}


def differential_expression(expression: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Fit group 1 minus group 0 on already normalized log2 expression.

    All supplied genes form the BH testing universe. DEG criteria are strictly
    |log2FC| > 1 and padj < 0.05. Filtering must be performed separately.
    """
    values = _validate_expression(expression)
    groups = _aligned_labels(expression, labels)
    df_residual = values.shape[1] - 2
    if df_residual <= 0:
        raise ValueError("at least three samples are required for residual degrees of freedom")
    zero, one = values[:, groups == 0], values[:, groups == 1]
    with np.errstate(over="ignore", invalid="ignore"):
        mean_zero, mean_one = zero.mean(axis=1), one.mean(axis=1)
        logfc = mean_one - mean_zero
        residual = ((zero - mean_zero[:, None]) ** 2).sum(axis=1)
        residual += ((one - mean_one[:, None]) ** 2).sum(axis=1)
        variances = residual / df_residual
    if not np.isfinite(variances).all() or not np.isfinite(logfc).all():
        raise ValueError("expression magnitude exceeds stable floating-point OLS range")
    moderation = squeeze_var(variances, df_residual)
    unscaled = np.sqrt(1 / zero.shape[1] + 1 / one.shape[1])
    standard_error = np.sqrt(moderation["var_post"]) * unscaled
    with np.errstate(over="ignore"):
        t_values = logfc / standard_error
    t_values = np.clip(t_values, -np.finfo(float).max, np.finfo(float).max)
    df_total = min(df_residual + moderation["df_prior"], len(expression) * df_residual)
    pvalues = 2 * stats.t.sf(np.abs(t_values), df=df_total)
    adjusted = benjamini_hochberg(pvalues)
    is_deg = (np.abs(logfc) > 1) & (adjusted < 0.05)
    result = pd.DataFrame(
        {
            "log2FC": logfc,
            "t": t_values,
            "pvalue": pvalues,
            "padj": adjusted,
            "is_deg": is_deg,
            "direction": np.where(is_deg, np.where(logfc > 0, "up", "down"), "not_deg"),
            "mean_group0": mean_zero,
            "mean_group1": mean_one,
            "var_residual": variances,
            "var_post": moderation["var_post"],
            "standard_error": standard_error,
            "df_residual": df_residual,
            "df_total": df_total,
            "df_prior": moderation["df_prior"],
            "var_prior": moderation["var_prior"],
        },
        index=expression.index.copy(),
    )
    result.attrs.update(
        contrast="group1 - group0",
        method="two-group OLS with log-moment empirical-Bayes variance moderation",
        input_scale="normalized log2 expression",
        deg_rule="abs(log2FC) > 1 and padj < 0.05",
    )
    return result


def plot_volcano(results: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a headless volcano plot using BH-adjusted p-values."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    required = {"log2FC", "padj", "is_deg"}
    if not required.issubset(results.columns):
        raise ValueError(f"results must contain {sorted(required)}")
    x = results["log2FC"].to_numpy(dtype=float)
    p = results["padj"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("volcano values must be finite with adjusted p-values in [0, 1]")
    figure = Figure(figsize=(7, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    colors = np.where(results["is_deg"], np.where(x > 0, "firebrick", "royalblue"), "gray")
    axis.scatter(x, -np.log10(np.maximum(p, np.finfo(float).tiny)), c=colors, s=12, alpha=0.7)
    axis.axvline(-1, color="black", linestyle="--", linewidth=0.7)
    axis.axvline(1, color="black", linestyle="--", linewidth=0.7)
    axis.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=0.7)
    axis.set(xlabel="log2 fold change (group 1 - group 0)", ylabel="-log10 BH-adjusted p")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    return path


def plot_heatmap(
    expression: pd.DataFrame,
    output_path: str | Path,
    *,
    results: pd.DataFrame | None = None,
    genes: list | None = None,
    top_n: int = 50,
) -> Path:
    """Write row-z-scored expression, selecting DEGs by |t| if supplied.

    Columns preserve the input sample order. No clustering is implied.
    Constant rows are rendered as zeros. With no DEG rows, raise rather than
    silently displaying non-DEGs.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    _validate_expression(expression)
    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if genes is None:
        if results is None:
            genes = expression.index[:top_n].tolist()
        else:
            if not {"is_deg", "t"}.issubset(results.columns) or not results.index.is_unique:
                raise ValueError("results must have unique genes and is_deg/t columns")
            selected = results.loc[results["is_deg"].astype(bool), "t"].abs()
            genes = selected.sort_values(ascending=False, kind="stable").index[:top_n].tolist()
    if not len(genes) or not pd.Index(genes).is_unique or not pd.Index(genes).isin(expression.index).all():
        raise ValueError("heatmap requires nonempty, unique genes present in expression")
    values = expression.loc[genes].to_numpy(dtype=float)
    scale = values.std(axis=1, keepdims=True)
    zscore = (values - values.mean(axis=1, keepdims=True)) / np.where(scale > 0, scale, 1)
    figure = Figure(figsize=(max(6, expression.shape[1] * 0.3), max(3, len(genes) * 0.18)))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    image = axis.imshow(zscore, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    axis.set_xticks(np.arange(expression.shape[1]), expression.columns.astype(str), rotation=90)
    axis.set_yticks(np.arange(len(genes)), [str(gene) for gene in genes], fontsize=7)
    figure.colorbar(image, ax=axis, label="row z-score")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=150)
    return path
