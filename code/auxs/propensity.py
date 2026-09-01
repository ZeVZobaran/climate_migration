"""Propensity-score weighting helpers for POLOCENTRO comparisons."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _weighted_mean_variance(
    values: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return np.nan, np.nan
    values = values[keep]
    weights = weights[keep]
    total = weights.sum()
    mean = np.dot(values, weights) / total
    variance = np.dot(weights, (values - mean) ** 2) / total
    return float(mean), float(variance)


def propensity_balance(
    weighted_data: pd.DataFrame,
    treatment: Hashable,
    covariates: Sequence[Hashable],
    weight_column: str = "psw_weight",
) -> pd.DataFrame:
    """Compute covariate standardized mean differences before and after PSW."""
    treatment_values = weighted_data[treatment].to_numpy(dtype=int)
    weights = weighted_data[weight_column].to_numpy(dtype=float)
    rows = []
    for covariate in covariates:
        values = pd.to_numeric(weighted_data[covariate], errors="coerce").to_numpy(float)
        control = treatment_values == 0
        treated = treatment_values == 1
        mean_c, var_c = _weighted_mean_variance(values[control], np.ones(control.sum()))
        mean_t, var_t = _weighted_mean_variance(values[treated], np.ones(treated.sum()))
        pooled = np.sqrt((var_c + var_t) / 2)
        before = (mean_t - mean_c) / pooled if np.isfinite(pooled) and pooled > 0 else np.nan

        mean_c_w, var_c_w = _weighted_mean_variance(values[control], weights[control])
        mean_t_w, var_t_w = _weighted_mean_variance(values[treated], weights[treated])
        pooled_w = np.sqrt((var_c_w + var_t_w) / 2)
        after = (
            (mean_t_w - mean_c_w) / pooled_w
            if np.isfinite(pooled_w) and pooled_w > 0
            else np.nan
        )
        rows.append({
            "covariate": str(covariate),
            "control_mean_before": mean_c,
            "treated_mean_before": mean_t,
            "smd_before": before,
            "control_mean_after": mean_c_w,
            "treated_mean_after": mean_t_w,
            "smd_after": after,
        })
    return pd.DataFrame(rows)


def generate_psw_weights(
    df: pd.DataFrame,
    id_column: Hashable,
    treatment: Hashable,
    covariates: Sequence[Hashable],
    *,
    estimand: str = "ATT",
    trim: float | None = 0.01,
    empirical_common_support: bool = True,
    ridge_alpha: float = 1e-4,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Estimate propensity scores and inverse-probability weights.

    The model is estimated once per unit. Numeric covariates are median-imputed
    and standardized. An ordinary logit is attempted first; a ridge-regularized
    binomial GLM is used if separation or non-convergence prevents a reliable
    fit. Observations outside empirical overlap and the optional absolute trim
    are retained for auditing but receive missing PSW weights.
    """
    covariates = list(dict.fromkeys(covariates))
    required = [id_column, treatment, *covariates]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Columns not found for propensity model: {missing}")
    if df[id_column].duplicated().any():
        raise ValueError("Propensity model input must have one row per unit")
    if estimand.upper() not in {"ATT", "ATE"}:
        raise ValueError("estimand must be 'ATT' or 'ATE'")
    if trim is not None and not 0 <= trim < 0.5:
        raise ValueError("trim must lie in [0, 0.5)")

    out = df[required].copy()
    treatment_numeric = pd.to_numeric(out[treatment], errors="coerce")
    if treatment_numeric.isna().any() or not treatment_numeric.isin([0, 1]).all():
        raise ValueError("Treatment must contain only non-missing binary values")
    out[treatment] = treatment_numeric.astype(int)
    if out[treatment].nunique() != 2:
        raise ValueError("Both treated and control units are required")

    usable_covariates = []
    imputation_counts: dict[str, int] = {}
    design = pd.DataFrame(index=out.index)
    for covariate in covariates:
        numeric = pd.to_numeric(out[covariate], errors="coerce")
        finite = numeric.where(np.isfinite(numeric))
        if finite.notna().sum() < 2 or finite.dropna().nunique() < 2:
            continue
        median = finite.median()
        imputation_counts[str(covariate)] = int(finite.isna().sum())
        filled = finite.fillna(median).astype(float)
        scale = filled.std(ddof=0)
        if not np.isfinite(scale) or scale <= 0:
            continue
        out[covariate] = filled
        design[str(covariate)] = (filled - filled.mean()) / scale
        usable_covariates.append(covariate)
    if not usable_covariates:
        raise ValueError("No non-constant numeric covariates are available")

    design = sm.add_constant(design, has_constant="add")
    target = out[treatment].astype(float)
    model_method = "unpenalized_logit"
    model = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            candidate = sm.Logit(target, design).fit(disp=False, maxiter=500)
            converged = bool(candidate.mle_retvals.get("converged", False))
            predicted = candidate.predict(design)
            reliable = converged and np.isfinite(predicted).all()
            if not reliable:
                raise RuntimeError("Unpenalized logit did not converge reliably")
            model = candidate
        except Exception:
            model_method = "ridge_binomial_glm"
            model = sm.GLM(target, design, family=sm.families.Binomial()).fit_regularized(
                alpha=ridge_alpha,
                L1_wt=0.0,
                maxiter=1_000,
            )

    pscore = np.asarray(model.predict(design), dtype=float)
    pscore = np.clip(pscore, 1e-6, 1 - 1e-6)
    out["pscore"] = pscore

    treated_scores = pscore[out[treatment].to_numpy() == 1]
    control_scores = pscore[out[treatment].to_numpy() == 0]
    absolute_lower = trim if trim is not None else 0.0
    absolute_upper = 1 - trim if trim is not None else 1.0
    if empirical_common_support:
        support_lower = max(absolute_lower, treated_scores.min(), control_scores.min())
        support_upper = min(absolute_upper, treated_scores.max(), control_scores.max())
    else:
        support_lower, support_upper = absolute_lower, absolute_upper
    if support_lower >= support_upper:
        raise RuntimeError("Treated and control propensity scores have no common support")

    out["in_common_support"] = out["pscore"].between(support_lower, support_upper)
    treatment_array = out[treatment].to_numpy(dtype=float)
    if estimand.upper() == "ATT":
        raw_weight = treatment_array + (1 - treatment_array) * pscore / (1 - pscore)
    else:
        raw_weight = treatment_array / pscore + (1 - treatment_array) / (1 - pscore)
    out["psw_weight"] = np.where(out["in_common_support"], raw_weight, np.nan)

    used = out[out["in_common_support"]]
    used_weights = used["psw_weight"].to_numpy(float)
    effective_n = float(used_weights.sum() ** 2 / np.square(used_weights).sum())
    diagnostics: dict[str, object] = {
        "estimand": estimand.upper(),
        "model_method": model_method,
        "ridge_alpha_if_used": ridge_alpha,
        "units_total": int(len(out)),
        "treated_total": int(out[treatment].sum()),
        "control_total": int((1 - out[treatment]).sum()),
        "units_on_support": int(out["in_common_support"].sum()),
        "treated_on_support": int(used[treatment].sum()),
        "control_on_support": int((1 - used[treatment]).sum()),
        "support_lower": float(support_lower),
        "support_upper": float(support_upper),
        "effective_sample_size": effective_n,
        "maximum_weight": float(np.nanmax(out["psw_weight"])),
        "covariates_requested": [str(value) for value in covariates],
        "covariates_used": [str(value) for value in usable_covariates],
        "imputation_counts": imputation_counts,
    }
    balance = propensity_balance(out, treatment, usable_covariates)
    diagnostics["maximum_absolute_smd_before"] = float(balance["smd_before"].abs().max())
    diagnostics["maximum_absolute_smd_after"] = float(balance["smd_after"].abs().max())
    return out, balance, diagnostics


def plot_propensity_diagnostics(
    weighted_data: pd.DataFrame,
    treatment: Hashable,
    *,
    title: str = "Propensity-score overlap",
    pscore: str = "pscore",
    weight_column: str = "psw_weight",
    support_column: str = "in_common_support",
    bins: int = 50,
    zoom_bins: int = 120,
) -> tuple[Figure, np.ndarray]:
    """Plot raw propensity-score overlap on the retained common support.

    The upper panel preserves the full probability scale from zero to one.
    The lower panel zooms to the largest estimated score in the complete
    sample and uses finer bins so overlap near zero remains legible.
    """
    data = weighted_data.copy()
    control = data[treatment].astype(int) == 0
    treated = data[treatment].astype(int) == 1
    support = data[support_column].astype(bool)
    lower = data.loc[support, pscore].min()
    upper = data.loc[support, pscore].max()
    maximum_score = float(data[pscore].max())
    maximum_score = max(maximum_score, upper, 1e-3)
    full_edges = np.linspace(0, 1, bins + 1)
    zoom_edges = np.linspace(0, maximum_score, zoom_bins + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    colors = {"control": "#1f77b4", "treated": "#ff7f0e"}
    for ax in axes:
        ax.axvspan(0, lower, color="0.85", alpha=0.6, linewidth=0)
        ax.axvspan(upper, 1, color="0.85", alpha=0.6, linewidth=0)
        ax.axvline(lower, color="0.4", linestyle="--", linewidth=1)
        ax.axvline(upper, color="0.4", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.2)

    axes[0].hist(
        data.loc[control & support, pscore], bins=full_edges, density=True, alpha=0.45,
        color=colors["control"], label="Control",
    )
    axes[0].hist(
        data.loc[treated & support, pscore], bins=full_edges, density=True, alpha=0.45,
        color=colors["treated"], label="Treated",
    )
    axes[0].set_ylabel("Density")
    axes[0].set_xlim(0, 1)
    axes[0].set_title("Common support · full 0–1 scale", loc="left", fontsize=12)
    axes[0].legend(frameon=False)

    axes[1].hist(
        data.loc[control & support, pscore], bins=zoom_edges, density=True,
        alpha=0.45, color=colors["control"], label="Control",
    )
    axes[1].hist(
        data.loc[treated & support, pscore], bins=zoom_edges, density=True,
        alpha=0.45, color=colors["treated"], label="Treated",
    )
    axes[1].set_xlim(0, maximum_score)
    axes[1].set_xlabel("Estimated propensity score")
    axes[1].set_ylabel("Density")
    axes[1].set_title(
        f"Common support · zoomed to pre-trim sample maximum ({maximum_score:.3f})",
        loc="left",
        fontsize=12,
    )
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontsize=15)
    return fig, axes


def plot_propensity_model_comparison(
    models: Sequence[tuple[str, pd.DataFrame]],
    treatment: Hashable,
    *,
    title: str = "Propensity-score overlap",
    pscore: str = "pscore",
    support_column: str = "in_common_support",
    bins: int = 120,
) -> tuple[Figure, np.ndarray]:
    """Compare zoomed common-support distributions across PS specifications.

    Each panel uses its own sample maximum as the upper probability bound so
    overlap near zero remains legible. Gray regions and dashed lines identify
    observations excluded by empirical common-support restrictions.
    """
    if not models:
        raise ValueError("At least one propensity-score model is required")
    fig, axes = plt.subplots(
        1, len(models), figsize=(7 * len(models), 4.8), constrained_layout=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    colors = {"control": "#1f77b4", "treated": "#ff7f0e"}
    for ax, (model_label, source) in zip(axes_flat, models):
        data = source.copy()
        support = data[support_column].astype(bool)
        control = data[treatment].astype(int).eq(0)
        treated = data[treatment].astype(int).eq(1)
        lower = float(data.loc[support, pscore].min())
        upper = float(data.loc[support, pscore].max())
        maximum_score = max(float(data[pscore].max()), upper, 1e-3)
        edges = np.linspace(0, maximum_score, bins + 1)
        ax.axvspan(0, lower, color="0.85", alpha=0.65, linewidth=0)
        ax.axvspan(upper, maximum_score, color="0.85", alpha=0.65, linewidth=0)
        ax.axvline(lower, color="0.4", linestyle="--", linewidth=1)
        ax.axvline(upper, color="0.4", linestyle="--", linewidth=1)
        ax.hist(
            data.loc[control & support, pscore], bins=edges, density=True,
            alpha=0.45, color=colors["control"], label="Control",
        )
        ax.hist(
            data.loc[treated & support, pscore], bins=edges, density=True,
            alpha=0.45, color=colors["treated"], label="Treated",
        )
        ax.set_xlim(0, maximum_score)
        ax.set_xlabel("Estimated propensity score")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{model_label}\nZoomed to sample maximum ({maximum_score:.3f})",
            loc="left", fontsize=12,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle(title, fontsize=15)
    return fig, axes_flat
