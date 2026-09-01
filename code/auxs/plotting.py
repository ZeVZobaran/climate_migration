"""Reusable plotting helpers for the climate-migration project."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import seaborn as sns


def _ordered_levels(series: pd.Series) -> list[object]:
    """Return observed levels, respecting ordered categoricals when possible."""
    observed = list(pd.unique(series.dropna()))
    if isinstance(series.dtype, pd.CategoricalDtype) and series.dtype.ordered:
        observed_set = set(observed)
        return [level for level in series.cat.categories if level in observed_set]

    try:
        return sorted(observed)
    except TypeError:
        # Mixed, non-comparable types still get a stable order.
        return observed


def boxplot(
    title: str,
    df: pd.DataFrame,
    time_dimension: Hashable,
    treatment_dimension: Hashable,
    outcome: Hashable,
    *,
    treatment_start: float | int | None = None,
) -> tuple[Figure, Axes]:
    """Plot outcome distributions by time and treatment.

    Each box represents one ``(time_dimension, treatment_dimension)`` group.
    The function computes the count, mean, median, sample standard deviation,
    first and third quartiles, and 10th and 90th percentiles for every group.
    Boxes span Q1--Q3, the line is the median, the diamond is the mean, and the
    whiskers span the 10th--90th percentiles.

    The summary used for plotting is also available as
    ``ax.boxplot_summary``. It has one row per time-treatment group.

    Parameters
    ----------
    title:
        Figure title.
    df:
        Source dataframe.
    time_dimension:
        Column shown on the x-axis.
    treatment_dimension:
        Column represented by adjacent colored boxes and the legend.
    outcome:
        Numeric column summarized on the y-axis.
    treatment_start:
        Optional treatment-start value on the time axis. When supplied, the
        first treated-period boundary is marked with a dotted black line and
        the post-treatment region is lightly shaded.

    Returns
    -------
    (fig, ax):
        The Matplotlib figure and axes.
    """
    dimensions = [time_dimension, treatment_dimension, outcome]
    missing = [column for column in dimensions if column not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("time_dimension, treatment_dimension, and outcome must be different columns")

    data = df[dimensions].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=dimensions)
    if data.empty:
        raise ValueError("No valid observations remain after dropping missing/non-numeric values")

    summary = (
        data.groupby(
            [time_dimension, treatment_dimension],
            observed=True,
            sort=False,
            as_index=False,
        )
        .agg(
            count=(outcome, "count"),
            mean=(outcome, "mean"),
            median=(outcome, "median"),
            stdev=(outcome, "std"),
            q1=(outcome, lambda values: values.quantile(0.25)),
            q3=(outcome, lambda values: values.quantile(0.75)),
            p10=(outcome, lambda values: values.quantile(0.10)),
            p90=(outcome, lambda values: values.quantile(0.90)),
        )
    )

    time_levels = _ordered_levels(data[time_dimension])
    treatment_levels = _ordered_levels(data[treatment_dimension])
    time_positions = {level: position for position, level in enumerate(time_levels)}

    n_treatments = len(treatment_levels)
    total_group_width = 0.8
    box_width = total_group_width / n_treatments
    offsets = (
        np.arange(n_treatments, dtype=float) - (n_treatments - 1) / 2
    ) * box_width
    colors = plt.get_cmap("tab10").colors

    figure_width = max(7.0, 1.15 * len(time_levels) + 2.0)
    fig, ax = plt.subplots(figsize=(figure_width, 5.5), constrained_layout=True)

    for treatment_index, treatment in enumerate(treatment_levels):
        treatment_summary = summary[summary[treatment_dimension] == treatment]
        stats = []
        positions = []
        for row in treatment_summary.itertuples(index=False, name=None):
            values = dict(zip(summary.columns, row))
            stats.append(
                {
                    "label": str(values[time_dimension]),
                    "mean": values["mean"],
                    "med": values["median"],
                    "q1": values["q1"],
                    "q3": values["q3"],
                    "whislo": values["p10"],
                    "whishi": values["p90"],
                    "fliers": [],
                }
            )
            positions.append(
                time_positions[values[time_dimension]] + offsets[treatment_index]
            )

        if not stats:
            continue

        color = colors[treatment_index % len(colors)]
        artists = ax.bxp(
            stats,
            positions=positions,
            widths=box_width * 0.88,
            showmeans=True,
            showfliers=False,
            patch_artist=True,
            manage_ticks=False,
            medianprops={"color": "black", "linewidth": 1.4},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": color,
                "markersize": 4.5,
            },
            whiskerprops={"color": color, "linewidth": 1.1},
            capprops={"color": color, "linewidth": 1.1},
        )
        for box in artists["boxes"]:
            box.set_facecolor(color)
            box.set_edgecolor(color)
            box.set_alpha(0.72)

    ax.set_title(title)
    ax.set_xlabel(str(time_dimension))
    ax.set_ylabel(str(outcome))
    ax.set_xticks(range(len(time_levels)), [str(level) for level in time_levels])
    ax.grid(axis="y", alpha=0.25)

    if treatment_start is not None:
        numeric_times = pd.to_numeric(pd.Series(time_levels), errors="coerce")
        post_indices = np.flatnonzero(numeric_times.to_numpy() >= treatment_start)
        if len(post_indices):
            boundary = float(post_indices[0]) - 0.5
            ax.axvspan(
                boundary,
                len(time_levels) - 0.5,
                color="0.92",
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )
            ax.axvline(
                boundary,
                color="black",
                linestyle=(0, (2, 2)),
                linewidth=1.2,
                zorder=1,
            )
            ax.text(
                boundary + 0.05,
                0.98,
                f"Treatment begins ({treatment_start:g})",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
                zorder=4,
            )
    ax.legend(
        handles=[
            Patch(
                facecolor=colors[index % len(colors)],
                edgecolor=colors[index % len(colors)],
                alpha=0.72,
                label=str(treatment),
            )
            for index, treatment in enumerate(treatment_levels)
        ],
        title=str(treatment_dimension),
        frameon=False,
    )

    # Keep the requested two-item return while making every computed statistic
    # available to callers that need the underlying table.
    ax.boxplot_summary = summary
    return fig, ax


def meanplot(
    title: str,
    df: pd.DataFrame,
    time_dimension: Hashable,
    treatment_dimension: Hashable,
    outcome: Hashable,
) -> tuple[Figure, Axes]:
    """Compare group means with one- and two-standard-deviation bands.

    For each treatment group, the solid line connects the outcome mean at each
    time value. The darker band is mean +/- 1 sample standard deviation and the
    lighter band is mean +/- 2 sample standard deviations. These are descriptive
    dispersion bands, not confidence intervals for the mean.

    The grouped statistics and band limits are also available as
    ``ax.meanplot_summary``.
    """
    dimensions = [time_dimension, treatment_dimension, outcome]
    missing = [column for column in dimensions if column not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in dataframe: {missing}")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError(
            "time_dimension, treatment_dimension, and outcome must be different columns"
        )

    data = df[dimensions].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=dimensions)
    if data.empty:
        raise ValueError(
            "No valid observations remain after dropping missing/non-numeric values"
        )

    summary = (
        data.groupby(
            [time_dimension, treatment_dimension],
            observed=True,
            sort=False,
            as_index=False,
        )
        .agg(
            count=(outcome, "count"),
            mean=(outcome, "mean"),
            stdev=(outcome, "std"),
        )
    )
    summary["mean_minus_1sd"] = summary["mean"] - summary["stdev"]
    summary["mean_plus_1sd"] = summary["mean"] + summary["stdev"]
    summary["mean_minus_2sd"] = summary["mean"] - 2 * summary["stdev"]
    summary["mean_plus_2sd"] = summary["mean"] + 2 * summary["stdev"]

    time_levels = _ordered_levels(data[time_dimension])
    treatment_levels = _ordered_levels(data[treatment_dimension])
    x = np.arange(len(time_levels), dtype=float)
    colors = plt.get_cmap("tab10").colors

    figure_width = max(7.0, 1.15 * len(time_levels) + 2.0)
    fig, ax = plt.subplots(figsize=(figure_width, 5.5), constrained_layout=True)
    treatment_handles = []

    for treatment_index, treatment in enumerate(treatment_levels):
        treatment_summary = (
            summary[summary[treatment_dimension] == treatment]
            .set_index(time_dimension)
            .reindex(time_levels)
        )
        means = treatment_summary["mean"].to_numpy(dtype=float)
        stdevs = treatment_summary["stdev"].to_numpy(dtype=float)
        color = colors[treatment_index % len(colors)]

# =============================================================================
#         ax.fill_between(
#             x,
#             means - 2 * stdevs,
#             means + 2 * stdevs,
#             color=color,
#             alpha=0.10,
#             linewidth=0,
#         )
# =============================================================================
        ax.fill_between(
            x,
            means - stdevs,
            means + stdevs,
            color=color,
            alpha=0.22,
            linewidth=0,
        )
        line, = ax.plot(
            x,
            means,
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=str(treatment),
        )
        treatment_handles.append(line)

    ax.set_title(title)
    ax.set_xlabel(str(time_dimension))
    ax.set_ylabel(str(outcome))
    ax.set_xticks(x, [str(level) for level in time_levels])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            *treatment_handles,
            Patch(facecolor="gray", alpha=0.22, label="Mean +/- 1 SD"),
#               Patch(facecolor="gray", alpha=0.10, label="Mean +/- 2 SD"),
        ],
        title=str(treatment_dimension),
        frameon=False,
    )

    ax.meanplot_summary = summary
    return fig, ax


def _latex_escape(value: object) -> str:
    """Escape plain text for use in a LaTeX table."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _latex_number(value: object, significant_digits: int) -> str:
    """Format one scalar compactly, using LaTeX notation for small exponents."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(number):
        return "--"
    if number == 0:
        return "0"

    text = f"{number:.{significant_digits}g}"
    if "e" not in text.lower():
        return text
    coefficient, exponent = text.lower().split("e")
    return rf"${coefficient}\mathbin{{\times}}10^{{{int(exponent)}}}$"


def boxplot_summary_to_latex(
    summary: pd.DataFrame,
    time_dimension: Hashable,
    treatment_dimension: Hashable,
    outcome_label: str,
    *,
    caption: str | None = None,
    label: str | None = None,
    control_value: object | None = None,
    treated_value: object | None = None,
    control_label: str = "Control",
    treated_label: str = "Treated",
    significant_digits: int = 2,
    include_mean_difference: bool = True,
    include_smd: bool = False,
    unit_label: str = "regions",
) -> str:
    """Turn ``ax.boxplot_summary`` into a publication-ready LaTeX table.

    Time is placed down the rows to avoid an excessively wide table. Control
    and treated groups form the column blocks; each reports mean (SD), median
    [Q1, Q3], and the number of valid observations. The optional SMD is mainly
    intended for assessing pre-treatment covariate balance, not as a treatment
    effect estimate.

    The returned table uses ``booktabs`` commands, so the LaTeX document must
    include ``\\usepackage{booktabs}``.
    """
    required = {
        time_dimension,
        treatment_dimension,
        "count",
        "mean",
        "median",
        "stdev",
        "q1",
        "q3",
    }
    missing = [column for column in required if column not in summary.columns]
    if missing:
        raise KeyError(f"Columns not found in boxplot summary: {missing}")
    if significant_digits < 1:
        raise ValueError("significant_digits must be at least 1")

    table_data = summary[
        [
            time_dimension,
            treatment_dimension,
            "count",
            "mean",
            "median",
            "stdev",
            "q1",
            "q3",
        ]
    ].copy()
    if table_data.duplicated([time_dimension, treatment_dimension]).any():
        raise ValueError("Summary must have one row per time-treatment group")

    treatment_levels = _ordered_levels(table_data[treatment_dimension])
    if len(treatment_levels) != 2:
        raise ValueError(
            "The comparison table requires exactly two observed treatment groups"
        )
    if control_value is None and treated_value is None:
        control_value, treated_value = treatment_levels
    elif control_value is None:
        remaining = [level for level in treatment_levels if level != treated_value]
        if len(remaining) != 1:
            raise ValueError("Could not infer control_value")
        control_value = remaining[0]
    elif treated_value is None:
        remaining = [level for level in treatment_levels if level != control_value]
        if len(remaining) != 1:
            raise ValueError("Could not infer treated_value")
        treated_value = remaining[0]
    if control_value == treated_value:
        raise ValueError("control_value and treated_value must be different")
    if control_value not in treatment_levels or treated_value not in treatment_levels:
        raise ValueError("control_value and treated_value must be observed in the summary")

    time_levels = _ordered_levels(table_data[time_dimension])
    indexed = table_data.set_index([time_dimension, treatment_dimension])

    comparison_headers = []
    if include_mean_difference:
        comparison_headers.append("Mean diff.")
    if include_smd:
        comparison_headers.append("SMD")
    comparison_columns = len(comparison_headers)
    total_columns = 7 + comparison_columns

    caption_text = caption or f"Descriptive statistics for {outcome_label}"
    column_spec = "@{}l" + "c" * (6 + comparison_columns) + "@{}"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption_text)}}}",
    ]
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.extend([
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
    ])

    first_header = (
        rf"& \multicolumn{{3}}{{c}}{{{_latex_escape(control_label)}}} "
        rf"& \multicolumn{{3}}{{c}}{{{_latex_escape(treated_label)}}}"
    )
    if comparison_columns:
        first_header += rf" & \multicolumn{{{comparison_columns}}}{{c}}{{Comparison}}"
    lines.append(first_header + r" \\")
    rule = r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
    if comparison_columns:
        rule += rf"\cmidrule(lr){{8-{total_columns}}}"
    lines.append(rule)

    second_header = (
        r"Year & Mean (SD) & Median [Q1, Q3] & $N$ "
        r"& Mean (SD) & Median [Q1, Q3] & $N$"
    )
    if comparison_headers:
        second_header += " & " + " & ".join(comparison_headers)
    lines.extend([second_header + r" \\", r"\midrule"])

    for time_value in time_levels:
        try:
            control = indexed.loc[(time_value, control_value)]
        except KeyError:
            control = pd.Series(dtype=float)
        try:
            treated = indexed.loc[(time_value, treated_value)]
        except KeyError:
            treated = pd.Series(dtype=float)

        def value(row: pd.Series, column: str) -> float:
            return row[column] if column in row.index else np.nan

        def mean_sd(row: pd.Series) -> str:
            return (
                f"{_latex_number(value(row, 'mean'), significant_digits)} "
                f"({_latex_number(value(row, 'stdev'), significant_digits)})"
            )

        def median_iqr(row: pd.Series) -> str:
            return (
                f"{_latex_number(value(row, 'median'), significant_digits)} "
                f"[{_latex_number(value(row, 'q1'), significant_digits)}, "
                f"{_latex_number(value(row, 'q3'), significant_digits)}]"
            )

        def count(row: pd.Series) -> str:
            count_value = value(row, "count")
            return f"{int(count_value):,}" if np.isfinite(count_value) else "--"

        row_values = [
            _latex_escape(time_value),
            mean_sd(control),
            median_iqr(control),
            count(control),
            mean_sd(treated),
            median_iqr(treated),
            count(treated),
        ]
        difference = value(treated, "mean") - value(control, "mean")
        if include_mean_difference:
            row_values.append(_latex_number(difference, significant_digits))
        if include_smd:
            pooled_sd = np.sqrt(
                (value(control, "stdev") ** 2 + value(treated, "stdev") ** 2) / 2
            )
            smd = difference / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
            row_values.append(_latex_number(smd, significant_digits))
        lines.append(" & ".join(row_values) + r" \\")

    escaped_outcome = _latex_escape(outcome_label)
    escaped_units = _latex_escape(unit_label)
    note = (
        rf"\textit{{Notes:}} Outcome: {escaped_outcome}. SD is the sample standard "
        rf"deviation across {escaped_units}; Q1 and Q3 are the 25th and 75th "
        r"percentiles. Mean differences are Treated minus Control and are unadjusted."
    )
    if include_smd:
        note += (
            r" SMD divides the mean difference by the square root of the average "
            r"within-group variance."
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{0.98\linewidth}",
        r"\footnotesize " + note,
        r"\end{minipage}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def plot_propensity_scores(
    df,
    treatment,
    pscore="pscore",
    bins=30,
    title="Propensity Score Overlap"
):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df[df[treatment] == 0],
        x=pscore,
        bins=bins,
        stat="density",
        alpha=0.4,
        color="tab:blue",
        label="Control"
    )

    sns.histplot(
        data=df[df[treatment] == 1],
        x=pscore,
        bins=bins,
        stat="density",
        alpha=0.4,
        color="tab:orange",
        label="Treated"
    )

    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return

def plot_coefficients(
    title: str,
    coefficients: Sequence[float],
    std_vars: Sequence[float],
    x: Sequence | None = None,
    xlabel: str = "",
    ylabel: str = "Coefficient",
    color: str = "tab:blue",
    figsize: tuple[float, float] = (10, 5),
):
    """
    Plot coefficients with shaded ±1 and ±2 standard-error bands.

    Parameters
    ----------
    title
        Plot title.
    coefficients
        Sequence of estimated coefficients.
    std_vars
        Sequence of standard errors or standard deviations.
    x
        Optional x-axis values. Defaults to 0, 1, ..., n-1.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    std_vars = np.asarray(std_vars, dtype=float)

    if coefficients.ndim != 1 or std_vars.ndim != 1:
        raise ValueError("coefficients and std_vars must be one-dimensional.")

    if len(coefficients) != len(std_vars):
        raise ValueError("coefficients and std_vars must have the same length.")

    if np.any(std_vars < 0):
        raise ValueError("std_vars cannot contain negative values.")

    if x is None:
        x = np.arange(len(coefficients))
    else:
        x = np.asarray(x)

    if len(x) != len(coefficients):
        raise ValueError("x must have the same length as coefficients.")

    fig, ax = plt.subplots(figsize=figsize)

    # Lighter outer band: coefficient ± 2 standard errors
    ax.fill_between(
        x,
        coefficients - 2 * std_vars,
        coefficients + 2 * std_vars,
        color=color,
        alpha=0.15,
        label=r"$\pm 2$ standard errors",
    )

    # Darker inner band: coefficient ± 1 standard error
    ax.fill_between(
        x,
        coefficients - std_vars,
        coefficients + std_vars,
        color=color,
        alpha=0.30,
        label=r"$\pm 1$ standard error",
    )

    ax.plot(
        x,
        coefficients,
        color=color,
        linewidth=2,
        marker="o",
        label="Coefficient",
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, ax


def compare_regression_coefficients(
    regressions: Mapping[str, object],
    *,
    coefficients: Sequence[Hashable] | None = None,
    x_labels: Sequence[str] | Mapping[Hashable, str] | None = None,
    confidence_level: float = 0.95,
    title: str = "Coefficient comparison",
    xlabel: str = "Coefficient",
    ylabel: str = "Estimate",
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Compare coefficient estimates from several regression specifications.

    Each value in ``regressions`` must expose callable ``.coef()`` and
    ``.se()`` methods. Their results may be pandas Series, mappings, or
    one-dimensional array-like objects. Named results (Series or mappings) are
    aligned by coefficient name; array-like results are aligned by position.

    By default, the function draws central 90% normal-approximation confidence
    intervals, whose endpoints are the 5th and 95th percentiles. Estimates for
    the different specifications are horizontally offset so their whiskers do
    not overlap. The data used for the plot are available afterward as
    ``ax.coefficient_summary``.

    Parameters
    ----------
    regressions:
        Mapping from the desired legend label to a fitted regression object.
    coefficients:
        Optional coefficient names (or positional indices) to plot, in the
        desired order. By default, all coefficients are shown, starting with
        the order returned by the first specification.
    x_labels:
        Optional display labels for the x-axis. Pass either one label per
        plotted coefficient, in order, or a mapping from coefficient names to
        display labels. Missing entries in a mapping use the coefficient name.
    confidence_level:
        Central confidence level. The default of 0.90 gives 5th--95th
        percentile whiskers.
    title, xlabel, ylabel:
        Axis labels and figure title.
    figsize:
        Figure size used when ``ax`` is not supplied. By default, width grows
        with the number of coefficients.
    ax:
        Optional existing Matplotlib axes on which to draw.

    Returns
    -------
    (fig, ax):
        The Matplotlib figure and axes.
    """
    if not isinstance(regressions, Mapping) or not regressions:
        raise ValueError("regressions must be a non-empty mapping")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")

    def extract(model: object, method_name: str) -> object:
        method = getattr(model, method_name, None)
        if not callable(method):
            raise TypeError(
                f"Regression objects must provide a callable .{method_name}() method"
            )
        return method()

    def as_series(
        values: object,
        value_name: str,
        preferred_index: pd.Index | None = None,
    ) -> pd.Series:
        if isinstance(values, pd.Series):
            series = values.copy()
        elif isinstance(values, Mapping):
            series = pd.Series(values)
        else:
            array = np.asarray(values)
            if array.ndim != 1:
                raise ValueError(f"{value_name} must be one-dimensional")
            if preferred_index is not None and len(array) == len(preferred_index):
                series = pd.Series(array, index=preferred_index)
            else:
                series = pd.Series(array)

        if series.index.has_duplicates:
            raise ValueError(f"{value_name} contains duplicate coefficient names")
        return pd.to_numeric(series, errors="coerce").astype(float)

    extracted: dict[str, tuple[pd.Series, pd.Series]] = {}
    available_coefficients: list[Hashable] = []
    seen: set[Hashable] = set()

    for raw_label, model in regressions.items():
        label = str(raw_label)
        estimates = as_series(extract(model, "coef"), f"{label}.coef()")
        standard_errors = as_series(
            extract(model, "se"),
            f"{label}.se()",
            preferred_index=estimates.index,
        )
        if not standard_errors.index.equals(estimates.index):
            missing_errors = estimates.index.difference(standard_errors.index).tolist()
            if missing_errors:
                raise ValueError(
                    f"{label}.se() has no entries for coefficients: {missing_errors}"
                )
            standard_errors = standard_errors.reindex(estimates.index)
        if (standard_errors.dropna() < 0).any():
            raise ValueError(f"{label}.se() contains negative values")

        extracted[label] = (estimates, standard_errors)
        for coefficient in estimates.index:
            if coefficient not in seen:
                seen.add(coefficient)
                available_coefficients.append(coefficient)

    if coefficients is None:
        coefficient_order = available_coefficients
    else:
        coefficient_order = list(coefficients)
        if len(set(coefficient_order)) != len(coefficient_order):
            raise ValueError("coefficients cannot contain duplicates")
        unknown = [name for name in coefficient_order if name not in seen]
        if unknown:
            raise KeyError(f"Coefficients not found in any specification: {unknown}")
    if not coefficient_order:
        raise ValueError("No coefficients are available to plot")

    if x_labels is None:
        display_labels = [str(coefficient) for coefficient in coefficient_order]
    elif isinstance(x_labels, Mapping):
        display_labels = [
            str(x_labels.get(coefficient, coefficient))
            for coefficient in coefficient_order
        ]
    else:
        if isinstance(x_labels, (str, bytes)):
            raise TypeError("x_labels must be a sequence of labels, not a string")
        display_labels = [str(label) for label in x_labels]
        if len(display_labels) != len(coefficient_order):
            raise ValueError(
                "x_labels must have one entry per plotted coefficient "
                f"({len(coefficient_order)} expected, got {len(display_labels)})"
            )

    alpha = 1 - confidence_level
    critical_value = NormalDist().inv_cdf(1 - alpha / 2)
    rows: list[dict[str, object]] = []
    for label, (estimates, standard_errors) in extracted.items():
        for coefficient in coefficient_order:
            if coefficient not in estimates.index:
                continue
            estimate = estimates.loc[coefficient]
            standard_error = standard_errors.loc[coefficient]
            if not np.isfinite(estimate) or not np.isfinite(standard_error):
                continue
            rows.append(
                {
                    "specification": label,
                    "coefficient": coefficient,
                    "estimate": estimate,
                    "std_error": standard_error,
                    "ci_lower": estimate - critical_value * standard_error,
                    "ci_upper": estimate + critical_value * standard_error,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No finite coefficient estimates and standard errors are available")

    if ax is None:
        if figsize is None:
            figsize = (max(8.0, 0.9 * len(coefficient_order) + 2.0), 5.5)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=False)
    else:
        fig = ax.figure

    labels = list(extracted)
    n_specifications = len(labels)
    spacing = min(0.18, 0.75 / n_specifications)
    offsets = (
        np.arange(n_specifications, dtype=float) - (n_specifications - 1) / 2
    ) * spacing
    base_positions = {
        coefficient: position for position, coefficient in enumerate(coefficient_order)
    }
    colors = plt.get_cmap("tab10").colors

    for specification_index, label in enumerate(labels):
        specification_data = summary[summary["specification"] == label]
        if specification_data.empty:
            continue
        x_positions = np.array(
            [base_positions[name] for name in specification_data["coefficient"]],
            dtype=float,
        ) + offsets[specification_index]
        estimates = specification_data["estimate"].to_numpy(dtype=float)
        lower_errors = estimates - specification_data["ci_lower"].to_numpy(dtype=float)
        upper_errors = specification_data["ci_upper"].to_numpy(dtype=float) - estimates

        ax.errorbar(
            x_positions,
            estimates,
            yerr=np.vstack([lower_errors, upper_errors]),
            fmt="o",
            markersize=5,
            capsize=3,
            elinewidth=1.4,
            color=colors[specification_index % len(colors)],
            label=label,
            zorder=3,
        )

    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7, zorder=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(
        np.arange(len(coefficient_order), dtype=float),
        display_labels,
    )
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.legend(
        title=f"Specification\n({lower_percentile:g}--{upper_percentile:g}% CI)",
        frameon=False
    )
    fig.autofmt_xdate(rotation=35, ha="right")

    ax.coefficient_summary = summary
    return fig, ax
