"""Build the POLOCENTRO treatment-control presentation as LaTeX Beamer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "figs" / "treatment_control_comparisons" / "analysis_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "figs" / "treatment_control_comparisons"
            / "polocentro_treatment_control_comparisons.tex"
        ),
    )
    return parser.parse_args()


def latex_escape(value: object) -> str:
    text = str(value)
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
        "–": "--",
        "—": "---",
        "·": r"\,$\cdot$\,",
        "×": r"$\times$",
        "±": r"$\pm$",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_path(value: str) -> str:
    return value.replace("\\", "/")


def fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    absolute = abs(number)
    if absolute >= 1e9:
        return f"{number / 1e9:.3g}b"
    if absolute >= 1e6:
        return f"{number / 1e6:.3g}m"
    if absolute >= 1e3:
        return f"{number / 1e3:.3g}k"
    if absolute < 1e-4:
        return f"{number:.2e}"
    return f"{number:.3g}"


def treated(value: object) -> bool:
    return value in (True, 1, "true", "True")


def summary_rows(outcome: dict[str, object]) -> list[list[str]]:
    grouped: dict[object, dict[str, dict[str, object]]] = {}
    time_column = str(outcome["time_column"])
    treatment_column = str(outcome["treatment_column"])
    for row in outcome["summary"]:
        time = row[time_column]
        grouped.setdefault(time, {})["treated" if treated(row[treatment_column]) else "control"] = row
    rows = []
    if outcome.get("crop_order"):
        ordered_times = [value for value in outcome["crop_order"] if value in grouped]
    else:
        try:
            ordered_times = sorted(grouped, key=float)
        except (TypeError, ValueError):
            ordered_times = sorted(grouped, key=str)
    for time in ordered_times:
        pair = grouped[time]
        control = pair.get("control", {})
        treatment = pair.get("treated", {})
        rows.append([
            str(time),
            f"{fmt(control.get('mean'))} ({fmt(control.get('stdev'))})",
            f"{fmt(treatment.get('mean'))} ({fmt(treatment.get('stdev'))})",
        ])
    return rows


def comparison_claim(outcome: dict[str, object]) -> str:
    differences = []
    time_column = str(outcome["time_column"])
    treatment_column = str(outcome["treatment_column"])
    grouped: dict[object, dict[str, dict[str, object]]] = {}
    for row in outcome["summary"]:
        time = row[time_column]
        grouped.setdefault(time, {})["treated" if treated(row[treatment_column]) else "control"] = row
    for pair in grouped.values():
        try:
            differences.append(float(pair["treated"]["mean"]) - float(pair["control"]["mean"]))
        except (KeyError, TypeError, ValueError):
            continue
    higher = sum(value > 0 for value in differences)
    lower = sum(value < 0 for value in differences)
    count = len(differences)
    if count and higher == count:
        return f"Treated means are higher in all {count} observed years"
    if count and lower == count:
        return f"Treated means are lower in all {count} observed years"
    if higher >= lower:
        return f"Treated means are higher in {higher} of {count} observed years"
    return f"Treated means are lower in {lower} of {count} observed years"


def preamble() -> list[str]:
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
        r"\setbeamercolor{title}{fg=ZNearWhite}",
        r"\setbeamercolor{subtitle}{fg=ZLightGray}",
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
        r"      \scriptsize\insertshorttitle",
        r"    \end{beamercolorbox}%",
        r"    \begin{beamercolorbox}[wd=.15\paperwidth,ht=2.8ex,dp=1.2ex,center]{date in head/foot}%",
        r"      \scriptsize\insertframenumber/\inserttotalframenumber",
        r"    \end{beamercolorbox}%",
        r"  }%",
        r"}",
        r"\setlength{\parskip}{0.25em}",
        r"\setbeamersize{text margin left=10mm,text margin right=10mm}",
        r"\title[POLOCENTRO comparisons]{Treatment and Control Comparisons}",
        r"\subtitle{AMC outcomes; municipal GDP; MapBiomas land cover; FAO/GAEZ suitability; propensity-score diagnostics}",
        r"\author{José Vidal Zobaran}",
        r"\institute{FGV-EPGE}",
        r"\date{\today}",
        r"\begin{document}",
    ]


def title_and_methods(lines: list[str]) -> None:
    lines.extend([
        r"\begin{frame}",
        r"  \titlepage",
        r"\end{frame}",
        r"\section{Methods and scope}",
        r"\begin{frame}{The comparisons preserve each dataset's analytical grain}",
        r"\small",
        r"Every chart is descriptive and unadjusted. Propensity scores are estimated separately with and without static FAO/GAEZ controls.",
        r"\begin{columns}[T,totalwidth=\textwidth]",
        r"\column{0.24\textwidth}",
        r"\begin{block}{AMC outcomes}",
        r"One AMC-census-year row. Resident traits, migration totals, and GDP/value added per capita where matching census population is observed.",
        r"\end{block}",
        r"\column{0.24\textwidth}",
        r"\begin{block}{Municipal GDP}",
        r"One municipality-GDP-year row. Real levels, natural logs, annualized log growth, and sector shares.",
        r"\end{block}",
        r"\column{0.24\textwidth}",
        r"\begin{block}{MapBiomas}",
        r"One AMC-year from 1985--2025. Mapped-area shares and cumulative native-cover net loss are entirely post-treatment.",
        r"\end{block}",
        r"\column{0.24\textwidth}",
        r"\begin{block}{GAEZ and PSW}",
        r"Eight static high-input rainfed suitability controls augment the same 1970 census and GDP selection model.",
        r"\end{block}",
        r"\end{columns}",
        r"\end{frame}",
    ])


def section_frame(lines: list[str], treatment: dict[str, object]) -> None:
    title = latex_escape(treatment["title"])
    description = latex_escape(treatment["description"])
    total = (
        len(treatment["amc_outcomes"])
        + len(treatment["municipality_outcomes"])
        + len(treatment["environment_outcomes"])
    )
    lines.extend([
        rf"\section{{{title}}}",
        r"\begin{frame}[plain]",
        r"\vfill",
        rf"{{\usebeamerfont{{title}}\usebeamercolor[fg]{{structure}}\Huge\bfseries {title}\par}}",
        r"\vspace{1em}",
        rf"{{\large {description}\par}}",
        r"\vspace{2em}",
        r"\begin{columns}[T,totalwidth=\textwidth]",
        rf"\column{{0.3\textwidth}}\centering\Huge\bfseries {treatment['amc_treated']}\\\normalsize treated AMCs",
        rf"\column{{0.3\textwidth}}\centering\Huge\bfseries {treatment['municipalities_treated']}\\\normalsize treated municipalities",
        rf"\column{{0.3\textwidth}}\centering\Huge\bfseries {total}\\\normalsize time-varying outcomes",
        r"\end{columns}",
        r"\vfill",
        r"\end{frame}",
    ])


def outcome_frame(lines: list[str], treatment: dict[str, object], outcome: dict[str, object]) -> None:
    title = latex_escape(comparison_claim(outcome))
    descriptor = latex_escape(
        f"{treatment['title']} · {str(outcome['level']).upper()} · {outcome['category']}"
    )
    path = latex_path(str(outcome["plot_path"]))
    rows = summary_rows(outcome)
    lines.extend([
        rf"\begin{{frame}}{{{title}}}",
        rf"{{\scriptsize\color{{ZBlue}}\textbf{{{descriptor}}}}}\par\vspace{{0.2em}}",
        r"\begin{columns}[T,totalwidth=\textwidth]",
        r"\column{0.64\textwidth}",
        rf"\includegraphics[width=\linewidth,height=0.70\textheight,keepaspectratio]{{\detokenize{{{path}}}}}",
        r"\column{0.34\textwidth}",
        r"\centering\scriptsize\textbf{UNADJUSTED MEANS (SD)}\par\vspace{0.25em}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Year & Control & Treated \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{columns}",
        r"\vspace{-0.3em}",
        r"{\tiny Box: Q1--Q3; whiskers: P10--P90; diamond: mean; gray field/dotted line: post-1975. Full table saved separately.}",
        r"\end{frame}",
    ])


def gaez_frame(lines: list[str], treatment: dict[str, object]) -> None:
    gaez = treatment["gaez"]
    rows = summary_rows(gaez)
    title = (
        comparison_claim(gaez)
        .replace("Treated means are ", "Treated suitability is ")
        .replace(" in all ", " for all ")
        .replace("observed years", "crops")
    )
    path = latex_path(str(gaez["plot_path"]))
    lines.extend([
        r"\subsection{FAO/GAEZ crop suitability}",
        rf"\begin{{frame}}{{\hspace{{1.5em}}{latex_escape(title)}}}",
        rf"{{\scriptsize\color{{ZBlue}}\textbf{{{latex_escape(treatment['title'])} $\cdot$ STATIC HIGH-INPUT RAINFED SUITABILITY}}}}\par\vspace{{0.2em}}",
        r"\begin{columns}[T,totalwidth=\textwidth]",
        r"\column{0.64\textwidth}",
        rf"\includegraphics[width=\linewidth,height=0.70\textheight,keepaspectratio]{{\detokenize{{{path}}}}}",
        r"\column{0.34\textwidth}",
        r"\centering\tiny\textbf{SUITABILITY: MEAN (SD)}\par\vspace{0.25em}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Crop & Control & Treated \\",
        r"\midrule",
    ])
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{columns}",
        r"\vspace{-0.2em}",
        r"{\tiny Static 1961--1990 climate normal; rainfed; all land; commercial/modern-input management; index 0--100.}",
        r"\end{frame}",
    ])


def propensity_frame(lines: list[str], treatment: dict[str, object]) -> None:
    propensity = treatment["propensity"]
    base = propensity["without_gaez"]["diagnostics"]
    gaez = propensity["with_gaez"]["diagnostics"]
    path = latex_path(str(propensity["plot_path"]))
    direction = (
        "lowers"
        if gaez["maximum_absolute_smd_after"] < base["maximum_absolute_smd_after"]
        else "raises"
    )
    title = (
        f"GAEZ {direction} max |SMD| after weighting: "
        f"{fmt(base['maximum_absolute_smd_after'])} to "
        f"{fmt(gaez['maximum_absolute_smd_after'])}"
    )
    lines.extend([
        r"\subsection{Propensity-score weighting}",
        rf"\begin{{frame}}{{{latex_escape(title)}}}",
        r"\begin{columns}[T,totalwidth=\textwidth]",
        r"\column{0.70\textwidth}",
        rf"\includegraphics[width=\linewidth,height=0.73\textheight,keepaspectratio]{{\detokenize{{{path}}}}}",
        r"\column{0.28\textwidth}",
        r"\centering\tiny",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Diagnostic & Base & +GAEZ \\",
        r"\midrule",
        rf"Treated retained & {base['treated_on_support']}/{base['treated_total']} & {gaez['treated_on_support']}/{gaez['treated_total']} \\",
        rf"Controls retained & {base['control_on_support']:,} & {gaez['control_on_support']:,} \\",
        rf"Effective N & {fmt(base['effective_sample_size'])} & {fmt(gaez['effective_sample_size'])} \\",
        rf"Support low & {fmt(base['support_lower'])} & {fmt(gaez['support_lower'])} \\",
        rf"Support high & {fmt(base['support_upper'])} & {fmt(gaez['support_upper'])} \\",
        rf"Max $|$SMD$|$ after & {fmt(base['maximum_absolute_smd_after'])} & {fmt(gaez['maximum_absolute_smd_after'])} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{0.7em}\par\raggedright\scriptsize Both panels show the zoomed common-support comparison. The inclusive model adds eight high-input rainfed crop-suitability controls.",
        r"\end{columns}",
        r"\end{frame}",
    ])


def synthesis_frame(lines: list[str], manifest: dict[str, object]) -> None:
    lines.extend([
        r"\section{Synthesis}",
        r"\begin{frame}{\hspace{1.5em}GAEZ changes overlap across assignments}",
        r"\small Across definitions, agronomic controls alter retained treated counts and residual weighted imbalance. Preferred outcome models should report both specifications.",
        r"\vspace{0.7em}",
        r"\centering\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Assignment & Treated & On support base & On support +GAEZ & Max $|$SMD$|$ base & Max $|$SMD$|$ +GAEZ \\",
        r"\midrule",
    ])
    for treatment in manifest["treatments"]:
        base = treatment["propensity"]["without_gaez"]["diagnostics"]
        gaez = treatment["propensity"]["with_gaez"]["diagnostics"]
        row = [
            latex_escape(treatment["title"]),
            str(treatment["amc_treated"]),
            str(base["treated_on_support"]),
            str(gaez["treated_on_support"]),
            fmt(base["maximum_absolute_smd_after"]),
            fmt(gaez["maximum_absolute_smd_after"]),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\vspace{1em}",
        r"\begin{block}{Next analytical step}",
        r"Estimate outcome models with the saved ATT weights and report sensitivity to each assignment definition and overlap restriction.",
        r"\end{block}",
        r"\end{frame}",
    ])


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lines = preamble()
    title_and_methods(lines)
    for treatment in manifest["treatments"]:
        section_frame(lines, treatment)
        lines.append(r"\subsection{Outcome comparisons}")
        for outcome in [
            *treatment["amc_outcomes"],
            *treatment["municipality_outcomes"],
            *treatment["environment_outcomes"],
        ]:
            outcome_frame(lines, treatment, outcome)
        gaez_frame(lines, treatment)
        propensity_frame(lines, treatment)
    synthesis_frame(lines, manifest)
    lines.append(r"\end{document}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "frames="
        + str(
            2
            + sum(
                3
                + len(t["amc_outcomes"])
                + len(t["municipality_outcomes"])
                + len(t["environment_outcomes"])
                for t in manifest["treatments"]
            )
            + 1
        )
    )
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
