"""Build the paired POLOCENTRO event-study Beamer presentations.

The source analysis is executed once.  Its in-memory ``results`` dictionary is
then used to save the propensity-score and coefficient figures and to build a
matching nine-column regression-table deck.
"""

from __future__ import annotations

import argparse
import math
import runpy
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "code" / "polocentro_es" / "event_study.py"

TREATMENTS = [
    ("polocentro_operational_core", "Core", "core"),
    ("polocentro_operational_any_overlap", "Any overlap", "lax"),
    ("polocentro_operational_majority_area", "Majority area", "rigid"),
]

SPECS = [
    ("Year FEs", "Year FE"),
    ("Year+AMC FEs", "AMC+Year FE"),
    ("Year+AMC FEs +PS Weights", "AMC+Year FE, PSW"),
]

OUTCOME_LABELS = {
    "log_gdp_total_2010_brl_thousand": "Log GDP",
    "log_va_agriculture_2010_brl_thousand": "Log agricultural VA",
    "gdp_total_annualized_log_growth_pct": "Annualized GDP growth",
    "va_agriculture_annualized_log_growth_pct": "Agricultural VA growth",
    "pasture_share_of_mapped": "Pasture share",
    "agriculture_share_of_mapped": "Agriculture share",
    "soybean_share_of_mapped": "Soybean share",
    "native_vegetation_net_loss_share_of_1985": "Native-vegetation loss",
    "weighted_interstate_migrants": "Interstate migrant count",
    "interstate_migrant_share_of_destination_age5plus": "Interstate migrant share",
    "log_population": "Log population",
    "population_annualized_log_growth_pct": "Population growth",
}

SKIP_COLUMNS = {
    "amc_code",
    "year",
    "gdp_growth_year_gap",
    "va_agriculture_growth_year_gap",
    "population_growth_year_gap",
    "has_gdp_outcome",
    "has_population_outcome",
    "has_migration_outcome",
    "has_mapbiomas_outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / "Apts",
    )
    return parser.parse_args()


def latex_escape(value: object) -> str:
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


def coefficient_years(regressions: dict[str, object]) -> list[int]:
    years: set[int] = set()
    for model in regressions.values():
        for coefficient in model.coef().index:
            text = str(coefficient)
            if text.startswith("treated_"):
                try:
                    years.add(int(text.rsplit("_", 1)[-1]))
                except ValueError:
                    pass
    return sorted(years)


def coefficient_value(model: object, year: int) -> tuple[float, float] | None:
    name = f"treated_{year}"
    estimates = model.coef()
    errors = model.se()
    if name not in estimates.index or name not in errors.index:
        return None
    estimate = float(estimates.loc[name])
    error = float(errors.loc[name])
    if not (math.isfinite(estimate) and math.isfinite(error) and error > 0):
        return None
    return estimate, error


def p_value(estimate: float, error: float) -> float:
    return math.erfc(abs(estimate / error) / math.sqrt(2.0))


def assignment_signal(regressions: dict[str, object]) -> str:
    """Summarize post-treatment evidence, emphasizing the PS-weighted model."""
    preferred = regressions["Year+AMC FEs +PS Weights"]
    corroborating = regressions["Year+AMC FEs"]
    years = [year for year in coefficient_years(regressions) if year > 1975]
    preferred_values = [
        (year, *value)
        for year in years
        if (value := coefficient_value(preferred, year)) is not None
    ]
    if not preferred_values:
        return "no clear effect"

    significant_10 = [
        (year, estimate, error)
        for year, estimate, error in preferred_values
        if p_value(estimate, error) < 0.10
    ]
    positive_10 = sum(estimate > 0 for _, estimate, _ in significant_10)
    negative_10 = sum(estimate < 0 for _, estimate, _ in significant_10)
    if positive_10 and negative_10:
        return "mixed effects over time"

    if significant_10:
        direction = 1 if positive_10 else -1
    else:
        median_estimate = float(np.median([value[1] for value in preferred_values]))
        direction = 1 if median_estimate >= 0 else -1

    preferred_05 = sum(
        p_value(estimate, error) < 0.05 and np.sign(estimate) == direction
        for _, estimate, error in preferred_values
    )
    preferred_10 = sum(
        p_value(estimate, error) < 0.10 and np.sign(estimate) == direction
        for _, estimate, error in preferred_values
    )
    corroborated_05 = 0
    for year in years:
        value = coefficient_value(corroborating, year)
        if value is None:
            continue
        estimate, error = value
        if p_value(estimate, error) < 0.05 and np.sign(estimate) == direction:
            corroborated_05 += 1

    movement = "increase" if direction > 0 else "decline"
    if preferred_05 >= 2:
        return f"clear {movement}"
    if preferred_05 >= 1 and corroborated_05 >= 1:
        return f"likely {movement}"
    if preferred_10 >= 1 or corroborated_05 >= 1:
        return f"suggestive {movement}"
    return "no clear effect"


def takeaway_title(outcome: str, results: dict[str, object]) -> str:
    label = OUTCOME_LABELS.get(outcome, outcome.replace("_", " ").title())
    signals = {
        short_label: assignment_signal(results[treatment][outcome]["regs"])
        for treatment, _, short_label in TREATMENTS
    }
    compact = {
        "clear increase": "clear rise",
        "likely increase": "likely rise",
        "suggestive increase": "suggestive rise",
        "clear decline": "clear fall",
        "likely decline": "likely fall",
        "suggestive decline": "suggestive fall",
        "mixed effects over time": "mixed over time",
        "no clear effect": "inconclusive",
    }

    def direction(signal: str) -> str:
        if "increase" in signal:
            return "rise"
        if "decline" in signal:
            return "fall"
        if signal.startswith("mixed"):
            return "mixed"
        return "none"

    directions = {name: direction(signal) for name, signal in signals.items()}
    detected = list(directions.values())
    if len(set(detected)) == 1:
        shared = detected[0]
        if shared in {"rise", "fall"}:
            if len(set(signals.values())) == 1:
                return f"{label}: {compact[next(iter(signals.values()))]} across all assignments"
            ranks = {"clear": 3, "likely": 2, "suggestive": 1}
            strengths = {
                name: ranks.get(signal.split()[0], 0) for name, signal in signals.items()
            }
            maximum = max(strengths.values())
            strongest = [name for name, strength in strengths.items() if strength == maximum]
            strongest_text = " and ".join(strongest)
            verb = "rises" if shared == "rise" else "falls"
            return f"{label} {verb} across assignments; strongest under {strongest_text}"
        if shared == "mixed":
            return f"{label}: effects are mixed over time across all assignments"
        return f"{label}: no assignment yields a clear effect"

    counts = {kind: detected.count(kind) for kind in set(detected)}
    majority_direction = max(counts, key=counts.get)
    if counts[majority_direction] == 2:
        majority = [name for name, kind in directions.items() if kind == majority_direction]
        other = next(name for name, kind in directions.items() if kind != majority_direction)
        majority_text = " and ".join(majority)
        if majority_direction == "rise":
            first = f"{label} rises under {majority_text}"
        elif majority_direction == "fall":
            first = f"{label} falls under {majority_text}"
        elif majority_direction == "mixed":
            first = f"{label} is mixed under {majority_text}"
        else:
            first = f"{label} is inconclusive under {majority_text}"
        other_direction = directions[other]
        if other_direction == "rise":
            second = f"{other} points to a rise"
        elif other_direction == "fall":
            second = f"{other} points to a fall"
        elif other_direction == "mixed":
            second = f"{other} is mixed over time"
        else:
            second = f"{other} is inconclusive"
        return f"{first}; {second}"

    return f"{label}: " + "; ".join(
        f"{name} {compact[signal]}" for name, signal in signals.items()
    )


def preamble(short_title: str) -> list[str]:
    return [
        r"% !TEX program = pdflatex",
        r"\documentclass[aspectratio=169]{beamer}",
        r"\usetheme{Madrid}",
        r"\useinnertheme{rectangles}",
        r"\useoutertheme{miniframes}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{blocks}[default]",
        r"\setbeamertemplate{items}[circle]",
        r"\setbeamertemplate{sections/subsections in toc}[sections numbered]",
        r"\setbeamertemplate{miniframes}[default]",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usefonttheme{professionalfonts}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage{microtype}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{xcolor}",
        r"\definecolor{ZDarkBlue}{HTML}{1E3A5F}",
        r"\definecolor{ZBlue}{HTML}{2D5B8A}",
        r"\definecolor{ZMidGray}{HTML}{6B7280}",
        r"\definecolor{ZLightGray}{HTML}{E9EDF2}",
        r"\definecolor{ZNearWhite}{HTML}{F7F8FA}",
        r"\setbeamercolor{background canvas}{bg=ZNearWhite}",
        r"\setbeamercolor{normal text}{fg=black,bg=ZNearWhite}",
        r"\setbeamercolor{structure}{fg=ZDarkBlue}",
        r"\setbeamercolor{frametitle}{fg=ZDarkBlue,bg=ZLightGray}",
        r"\setbeamercolor{palette primary}{fg=white,bg=ZDarkBlue}",
        r"\setbeamercolor{palette secondary}{fg=white,bg=ZBlue}",
        r"\setbeamercolor{palette tertiary}{fg=white,bg=ZDarkBlue}",
        r"\setbeamercolor{palette quaternary}{fg=white,bg=ZDarkBlue}",
        r"\setbeamercolor{section in head/foot}{fg=white,bg=ZDarkBlue}",
        r"\setbeamercolor{subsection in head/foot}{fg=white,bg=ZBlue}",
        r"\setbeamercolor{block title}{fg=ZDarkBlue,bg=ZLightGray}",
        r"\setbeamercolor{block body}{fg=black,bg=white}",
        r"\setbeamerfont{frametitle}{size=\normalsize}",
        r"\setbeamertemplate{headline}{%",
        r"  \leavevmode",
        r"  \begin{beamercolorbox}[wd=\paperwidth,ht=2.6ex,dp=1.2ex,leftskip=1em,rightskip=1em]{section in head/foot}",
        r"    \usebeamerfont{section in head/foot}\insertsectionhead\hfill",
        r"    \usebeamerfont{section in head/foot}\insertsubsectionhead",
        r"  \end{beamercolorbox}%",
        r"}",
        r"\setbeamertemplate{footline}{%",
        r"  \leavevmode\hbox{%",
        r"    \begin{beamercolorbox}[wd=.85\paperwidth,ht=2.8ex,dp=1.2ex,leftskip=1em,rightskip=1em]{author in head/foot}%",
        rf"      \scriptsize {latex_escape(short_title)}",
        r"    \end{beamercolorbox}%",
        r"    \begin{beamercolorbox}[wd=.15\paperwidth,ht=2.8ex,dp=1.2ex,center]{date in head/foot}%",
        r"      \scriptsize\insertframenumber/\inserttotalframenumber",
        r"    \end{beamercolorbox}%",
        r"  }%",
        r"}",
        r"\setlength{\parskip}{0.25em}",
        r"\setbeamersize{text margin left=8mm,text margin right=8mm}",
        r"\begin{document}",
    ]


def save_assets(results: dict[str, object], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    for treatment, display, short_label in TREATMENTS:
        ps_figure = results[treatment]["ps_density"]
        ps_figure.set_size_inches(9.0, 6.2)
        ps_figure.suptitle(f"{display} assignment", fontsize=15, fontweight="bold")
        ps_figure.savefig(
            assets_dir / f"ps_{short_label}.pdf",
            bbox_inches="tight",
        )

        for outcome, output in results[treatment].items():
            if outcome in {"ps", "ps_density"}:
                continue
            figure = output["coef_comp"]
            axis = figure.axes[0]
            figure.set_size_inches(5.0, 4.2)
            axis.set_title(display, fontsize=12, fontweight="bold")
            axis.set_xlabel("Event year", fontsize=9)
            axis.set_ylabel("Estimate", fontsize=9)
            axis.tick_params(axis="both", labelsize=7)
            tick_labels = axis.get_xticklabels()
            if len(tick_labels) > 15:
                for index, tick_label in enumerate(tick_labels):
                    tick_label.set_visible(index % 5 == 0 or index == len(tick_labels) - 1)
            legend = axis.get_legend()
            if legend is not None:
                legend.set_title("Specification (95% CI)", prop={"size": 7})
                for text in legend.get_texts():
                    text.set_fontsize(6.4)
            figure.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.22)
            figure.savefig(
                assets_dir / f"coef_{short_label}_{outcome}.pdf",
                bbox_inches="tight",
            )


def ps_frames(lines: list[str], results: dict[str, object], asset_rel: str) -> None:
    lines.append(r"\section{Propensity-score overlap}")
    for treatment, display, short_label in TREATMENTS:
        _, _, analytics = results[treatment]["ps"]
        title = (
            f"{display}: common support retains "
            f"{analytics['treated_on_support']}/{analytics['treated_total']} treated AMCs"
        )
        lines.extend(
            [
                rf"\begin{{frame}}{{{latex_escape(title)}}}",
                r"\centering",
                rf"\includegraphics[width=0.88\textwidth,height=0.68\textheight,keepaspectratio]{{{asset_rel}/ps_{short_label}.pdf}}\par",
                r"\vspace{-0.2em}",
                (
                    r"{\tiny\color{ZMidGray} ATT weighting; empirical common support "
                    rf"[{analytics['support_lower']:.3f}, {analytics['support_upper']:.3f}]; "
                    rf"effective $N={analytics['effective_sample_size']:.1f}$; "
                    rf"maximum post-weighting $|$SMD$|={analytics['maximum_absolute_smd_after']:.3f}$.}}"
                ),
                r"\end{frame}",
            ]
        )


def coefficient_frames(
    lines: list[str],
    results: dict[str, object],
    outcomes: list[str],
    asset_rel: str,
) -> None:
    lines.append(r"\section{Event-study estimates}")
    for outcome in outcomes:
        ref_year = int(results[TREATMENTS[0][0]][outcome]["ref_year"])
        lines.extend(
            [
                rf"\begin{{frame}}{{{latex_escape(takeaway_title(outcome, results))}}}",
                r"\begin{columns}[T,totalwidth=\textwidth]",
            ]
        )
        for _, _, short_label in TREATMENTS:
            lines.extend(
                [
                    r"\column{0.325\textwidth}",
                    rf"\centering\includegraphics[width=\linewidth,height=0.68\textheight,keepaspectratio]{{{asset_rel}/coef_{short_label}_{outcome}.pdf}}",
                ]
            )
        lines.extend(
            [
                r"\end{columns}",
                r"\vspace{-0.35em}",
                (
                    r"{\tiny\color{ZMidGray} AMC-clustered 95\% confidence intervals. "
                    rf"Reference year: {ref_year}. "
                    r"Title confidence emphasizes the PS-weighted two-way-FE model: clear = at least two post-treatment coefficients at 5\%; "
                    r"likely = one at 5\% corroborated by unweighted two-way FE; suggestive = evidence at 10\%; otherwise no clear effect.}"
                ),
                r"\end{frame}",
            ]
        )


def formatted_number(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if abs(value) < 0.0005:
        return "0.000"
    return f"{value:.3f}"


def regression_cell(model: object, year: int) -> str:
    value = coefficient_value(model, year)
    if value is None:
        return "--"
    estimate, error = value
    probability = p_value(estimate, error)
    stars = "***" if probability < 0.01 else "**" if probability < 0.05 else "*" if probability < 0.10 else ""
    star_text = rf"$^{{{stars}}}$" if stars else ""
    return rf"\shortstack{{{formatted_number(estimate)}{star_text}\\({formatted_number(error)})}}"


def observations(model: object) -> str:
    value = getattr(model, "_N", None)
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "--"


def _table_frame_unpaginated(lines: list[str], results: dict[str, object], outcome: str) -> None:
    title = takeaway_title(outcome, results)
    ref_year = int(results[TREATMENTS[0][0]][outcome]["ref_year"])
    year_union: set[int] = {ref_year}
    for treatment, _, _ in TREATMENTS:
        year_union.update(coefficient_years(results[treatment][outcome]["regs"]))
    years = sorted(year_union)

    lines.extend(
        [
            rf"\begin{{frame}}{{{latex_escape(title)}}}",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.1pt}",
            r"\renewcommand{\arraystretch}{0.86}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{@{}l*{9}{c}@{}}",
            r"\toprule",
            r"& \multicolumn{3}{c}{Core} & \multicolumn{3}{c}{Any overlap} & \multicolumn{3}{c}{Majority area} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
            r"Event year & Year FE & AMC+Year & PSW & Year FE & AMC+Year & PSW & Year FE & AMC+Year & PSW \\",
            r"\midrule",
        ]
    )

    for year in years:
        if year == ref_year:
            cells = [rf"{year} (ref.)"] + ["--"] * 9
        else:
            cells = [str(year)]
            for treatment, _, _ in TREATMENTS:
                regressions = results[treatment][outcome]["regs"]
                cells.extend(regression_cell(regressions[key], year) for key, _ in SPECS)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    n_cells = ["Observations"]
    for treatment, _, _ in TREATMENTS:
        regressions = results[treatment][outcome]["regs"]
        n_cells.extend(observations(regressions[key]) for key, _ in SPECS)
    lines.append(" & ".join(n_cells) + r" \\")
    lines.extend(
        [
            r"AMC fixed effects & No & Yes & Yes & No & Yes & Yes & No & Yes & Yes \\",
            r"Year fixed effects & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\",
            r"ATT propensity weights & No & No & Yes & No & No & Yes & No & No & Yes \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.35em}",
            (
                r"{\tiny\color{ZMidGray} Each cell reports the event-time coefficient with AMC-clustered standard error in parentheses. "
                r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
                rf"The omitted event year is {ref_year}. "
                + (
                    r"Models are fixed-effects PPML. "
                    if outcome == "weighted_interstate_migrants"
                    else r"Models are OLS. "
                )
                + r"PSW columns use ATT weights on empirical common support.}"
            ),
            r"\end{frame}",
        ]
    )


def table_frame(lines: list[str], results: dict[str, object], outcome: str) -> None:
    """Emit one AER-style table, paginated by event-year rows when needed."""
    title = takeaway_title(outcome, results)
    ref_year = int(results[TREATMENTS[0][0]][outcome]["ref_year"])
    year_union: set[int] = {ref_year}
    for treatment, _, _ in TREATMENTS:
        year_union.update(coefficient_years(results[treatment][outcome]["regs"]))
    years = sorted(year_union)
    page_count = math.ceil(len(years) / 6)
    base_size, extra = divmod(len(years), page_count)
    chunk_sizes = [base_size + (index < extra) for index in range(page_count)]
    chunks = []
    start = 0
    for size in chunk_sizes:
        chunks.append(years[start:start + size])
        start += size

    for page_index, page_years in enumerate(chunks, start=1):
        continuation = f" [{page_index}/{len(chunks)}]" if len(chunks) > 1 else ""
        lines.extend(
            [
                rf"\begin{{frame}}{{{latex_escape(title + continuation)}}}",
                r"\centering",
                r"\scriptsize",
                r"\setlength{\tabcolsep}{2.1pt}",
                r"\renewcommand{\arraystretch}{0.68}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{@{}l*{9}{c}@{}}",
                r"\toprule",
                r"& \multicolumn{3}{c}{Core} & \multicolumn{3}{c}{Any overlap} & \multicolumn{3}{c}{Majority area} \\",
                r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
                r"Event year & Year FE & AMC+Year & PSW & Year FE & AMC+Year & PSW & Year FE & AMC+Year & PSW \\",
                r"\midrule",
            ]
        )

        for year in page_years:
            if year == ref_year:
                cells = [rf"{year} (ref.)"] + ["--"] * 9
            else:
                cells = [str(year)]
                for treatment, _, _ in TREATMENTS:
                    regressions = results[treatment][outcome]["regs"]
                    cells.extend(regression_cell(regressions[key], year) for key, _ in SPECS)
            lines.append(" & ".join(cells) + r" \\")

        lines.append(r"\midrule")
        n_cells = ["Observations"]
        for treatment, _, _ in TREATMENTS:
            regressions = results[treatment][outcome]["regs"]
            n_cells.extend(observations(regressions[key]) for key, _ in SPECS)
        lines.append(" & ".join(n_cells) + r" \\")
        lines.extend(
            [
                r"AMC fixed effects & No & Yes & Yes & No & Yes & Yes & No & Yes & Yes \\",
                r"Year fixed effects & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes & Yes \\",
                r"ATT propensity weights & No & No & Yes & No & No & Yes & No & No & Yes \\",
                r"\bottomrule",
                r"\end{tabular}%",
                r"}",
                r"\vspace{0.1em}",
                (
                    r"{\tiny\color{ZMidGray} AMC-clustered standard errors in parentheses. "
                    r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$. "
                    rf"Reference year: {ref_year}. "
                    + (r"Fixed-effects PPML. " if outcome == "weighted_interstate_migrants" else r"OLS. ")
                    + r"PSW columns use ATT weights on common support.}"
                ),
                r"\end{frame}",
            ]
        )


def build_decks(results: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "polocentro_event_study_assets"
    save_assets(results, assets_dir)
    asset_rel = assets_dir.name

    first_treatment = TREATMENTS[0][0]
    outcomes = [
        key
        for key in results[first_treatment]
        if key not in {"ps", "ps_density"} and key not in SKIP_COLUMNS
    ]

    coefficient_lines = preamble("POLOCENTRO event study - coefficient comparisons")
    ps_frames(coefficient_lines, results, asset_rel)
    coefficient_frames(coefficient_lines, results, outcomes, asset_rel)
    coefficient_lines.append(r"\end{document}")
    coefficient_path = output_dir / "polocentro_event_study_coefficients.tex"
    coefficient_path.write_text("\n".join(coefficient_lines) + "\n", encoding="utf-8")

    table_lines = preamble("POLOCENTRO event study - regression tables")
    ps_frames(table_lines, results, asset_rel)
    table_lines.append(r"\section{Event-study regression tables}")
    for outcome in outcomes:
        table_frame(table_lines, results, outcome)
    table_lines.append(r"\end{document}")
    table_path = output_dir / "polocentro_event_study_regression_tables.tex"
    table_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    return coefficient_path, table_path


def main() -> None:
    args = parse_args()
    namespace = runpy.run_path(str(ANALYSIS))
    results = namespace["results"]
    coefficient_path, table_path = build_decks(results, args.output_dir.resolve())
    print(coefficient_path)
    print(table_path)
    plt.close("all")


if __name__ == "__main__":
    main()
