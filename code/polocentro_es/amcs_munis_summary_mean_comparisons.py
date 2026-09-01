"""Generate POLOCENTRO treatment-control comparisons and PSW diagnostics.

The script produces one boxplot, LaTeX table, and CSV summary for every
requested outcome under every treatment definition. Propensity scores are
estimated twice per AMC: first using 1970 census and GDP characteristics, and
then with eight static FAO/GAEZ crop-suitability controls added. All outputs
and a machine-readable deck manifest are written under
``figs/treatment_control_comparisons``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from auxs.plotting import boxplot, boxplot_summary_to_latex
from auxs.propensity import generate_psw_weights, plot_propensity_model_comparison


possible_treat_assigns = [
    "polocentro_operational_any_overlap",
    "polocentro_operational_majority_area",
    "polocentro_operational_any_seat_inside",
    "polocentro_operational_core",
]

TREATMENT_SPECS = {
    "polocentro_operational_any_overlap": {
        "title": "Any operational-area overlap",
        "description": "Treated when any non-trivial part of the unit overlaps an operational POLOCENTRO area.",
        "municipality_column": "polocentro_operational_any_overlap",
    },
    "polocentro_operational_majority_area": {
        "title": "Majority operational-area overlap",
        "description": "Treated when at least 50% of unit land area overlaps an operational POLOCENTRO area.",
        "municipality_column": "polocentro_operational_majority_area",
    },
    "polocentro_operational_any_seat_inside": {
        "title": "Municipal seat inside an operational area",
        "description": "AMC treated when any constituent municipal seat is inside; municipality treated when its own seat is inside.",
        "municipality_column": "polocentro_operational_seat_inside",
    },
    "polocentro_operational_core": {
        "title": "Operational core",
        "description": "Treated when a seat lies inside or at least 10% of land area overlaps an operational POLOCENTRO area.",
        "municipality_column": "polocentro_operational_core",
    },
}

TREATMENT_START_YEAR = 1975

ENVIRONMENT_DIR = ROOT / "data" / "processed" / "polocentro_environment"

MAPBIOMAS_OUTCOMES = [
    "native_vegetation_share_of_mapped",
    "anthropic_use_share_of_mapped",
    "farming_share_of_mapped",
    "pasture_share_of_mapped",
    "agriculture_share_of_mapped",
    "soybean_share_of_mapped",
    "urban_area_share_of_mapped",
    "native_vegetation_net_loss_share_of_1985",
]

MAPBIOMAS_LABELS = {
    "native_vegetation_share_of_mapped": "Native vegetation share of mapped area",
    "anthropic_use_share_of_mapped": "Anthropic-use share of mapped area",
    "farming_share_of_mapped": "Farming share of mapped area",
    "pasture_share_of_mapped": "Pasture share of mapped area",
    "agriculture_share_of_mapped": "Agriculture share of mapped area",
    "soybean_share_of_mapped": "Soybean share of mapped area",
    "urban_area_share_of_mapped": "Urban-area share of mapped area",
    "native_vegetation_net_loss_share_of_1985": (
        "Native-vegetation net loss relative to 1985"
    ),
}

GAEZ_CROP_ORDER = [
    "Soybean",
    "Maize",
    "Dryland rice",
    "Phaseolus bean",
    "Wheat",
    "Cotton",
    "Coffee",
    "Sugarcane",
]

GAEZ_HIGH_INPUT_COVARIATES = [
    "gaez_soy_high_mean_0_100",
    "gaez_mze_high_mean_0_100",
    "gaez_rcd_high_mean_0_100",
    "gaez_phb_high_mean_0_100",
    "gaez_whe_high_mean_0_100",
    "gaez_cot_high_mean_0_100",
    "gaez_cof_high_mean_0_100",
    "gaez_suc_high_mean_0_100",
]

GDP_LEVEL_COLUMNS = [
    "gdp_total_2010_brl_thousand",
    "va_industry_2010_brl_thousand",
    "va_services_private_2010_brl_thousand",
    "va_public_administration_2010_brl_thousand",
    "va_agriculture_2010_brl_thousand",
]

GDP_CONCEPT_NAMES = {
    "gdp_total_2010_brl_thousand": "GDP total",
    "va_industry_2010_brl_thousand": "Industry value added",
    "va_services_private_2010_brl_thousand": "Private-services value added",
    "va_public_administration_2010_brl_thousand": "Public-administration value added",
    "va_agriculture_2010_brl_thousand": "Agriculture value added",
}

OUTCOME_LABELS = dict(MAPBIOMAS_LABELS)
for _column, _concept in GDP_CONCEPT_NAMES.items():
    _stem = _column.removesuffix("_2010_brl_thousand")
    OUTCOME_LABELS[_column] = f"{_concept} (constant 2010 R$ thousand)"
    OUTCOME_LABELS[f"log_{_column}"] = f"Log {_concept} (constant 2010 R$ thousand)"
    OUTCOME_LABELS[f"{_stem}_per_capita_2010_brl"] = (
        f"{_concept} per capita (constant 2010 R$)"
    )
    OUTCOME_LABELS[f"{_stem}_annualized_log_growth_pct"] = (
        f"Annualized {_concept} log growth (%)"
    )


amc_outcomes = [
    "weighted_interstate_migrants",
    "interstate_migrant_share_of_destination_age5plus",
    "education_fundamental_share_age25plus",
    "education_less_fundamental_share_age25plus",
    "education_secondary_share_age25plus",
    "education_tertiary_share_age25plus",
    "literacy_share_age15plus",
    "mean_education_years_age25plus",
    "employment_share_age15plus",
    "labor_force_share_age15plus",
    "electricity_share",
    "automobile_share",
    "mean_bathrooms",
    "mean_bedrooms",
    "mean_rooms",
    "refrigerator_share",
    "zero_income_share_age15plus",
    "positive_income_q1_share_age15plus",
    "positive_income_q2_share_age15plus",
    "positive_income_q3_share_age15plus",
    "positive_income_q4_share_age15plus",
    "positive_income_q5_share_age15plus",
    "weighted_population",
    "female_share",
    "race_asian_share",
    "race_black_share",
    "race_indigenous_share",
    "race_pardo_share",
    "race_white_share",
    "mean_household_size",
    "urban_share",
    "mean_age",
    "share_age_0_4",
    "share_age_15_24",
    "share_age_25_64",
    "share_age_5_14",
    "share_age_65plus",
    *[
        f"{column.removesuffix('_2010_brl_thousand')}_per_capita_2010_brl"
        for column in GDP_LEVEL_COLUMNS
    ],
]

muni_outcomes = [
    *GDP_LEVEL_COLUMNS,
    *[f"log_{column}" for column in GDP_LEVEL_COLUMNS],
    *[
        f"{column.removesuffix('_2010_brl_thousand')}_annualized_log_growth_pct"
        for column in GDP_LEVEL_COLUMNS
    ],
    "va_industry_share_of_gdp",
    "va_services_private_share_of_gdp",
    "va_public_administration_share_of_gdp",
    "va_agriculture_share_of_gdp",
]

PSW_COVARIATES = [
    "log_population_1970",
    "mean_age",
    "female_share",
    "literacy_share_age15plus",
    "mean_education_years_age25plus",
    "employment_share_age15plus",
    "urban_share",
    "mean_household_size",
    "electricity_share",
    "automobile_share",
    "race_black_share",
    "race_pardo_share",
    "race_indigenous_share",
    "log_gdp_total_1970",
    "log_gdp_per_resident_1970",
    "va_agriculture_share_of_gdp",
    "va_industry_share_of_gdp",
    "va_services_private_share_of_gdp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "amcs")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "figs" / "treatment_control_comparisons",
    )
    parser.add_argument(
        "--environment-data",
        type=Path,
        default=ENVIRONMENT_DIR,
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def outcome_label(column: str) -> str:
    if column in OUTCOME_LABELS:
        return OUTCOME_LABELS[column]
    replacements = {
        "gdp": "GDP", "va": "value added", "brl": "R$",
        "q1": "Q1", "q2": "Q2", "q3": "Q3", "q4": "Q4", "q5": "Q5",
        "age5plus": "age 5+", "age15plus": "age 15+", "age25plus": "age 25+",
    }
    words = [replacements.get(word, word) for word in column.split("_")]
    label = " ".join(words)
    return label[:1].upper() + label[1:]


def outcome_category(column: str, level: str) -> str:
    if level == "environment":
        if column == "native_vegetation_net_loss_share_of_1985":
            return "MapBiomas cumulative land-cover change"
        return "MapBiomas land-cover shares"
    if level == "municipality":
        if "annualized_log_growth_pct" in column:
            return "Municipal GDP growth"
        if column.startswith("log_"):
            return "Municipal GDP logs"
        return "Municipal GDP levels" if "share_of_gdp" not in column else "Municipal GDP composition"
    if "per_capita_2010_brl" in column:
        return "AMC GDP per capita"
    if "migrant" in column or "destination" in column:
        return "Migration"
    if "education" in column or "literacy" in column:
        return "Education"
    if "employment" in column or "labor_force" in column:
        return "Labor market"
    if any(token in column for token in ("electricity", "automobile", "bathroom", "bedroom", "rooms", "refrigerator")):
        return "Housing and amenities"
    if "income" in column:
        return "Income distribution"
    if "race" in column or "female" in column or "household" in column or "urban" in column:
        return "Demographics"
    if "age" in column:
        return "Age composition"
    if "population" in column:
        return "Population"
    return "Other AMC outcomes"


def build_amc_year_panel(data_dir: Path) -> pd.DataFrame:
    """Return one AMC-year row, combining resident traits and total migration."""
    characteristics = pd.read_parquet(data_dir / "amc_year_characteristics.parquet")
    flows = pd.read_parquet(data_dir / "amc_origin_uf_year_flows.parquet")
    migration = (
        flows.groupby(["census_year", "amc_code"], observed=True, as_index=False)
        .agg(
            weighted_interstate_migrants=("weighted_interstate_migrants", "sum"),
            interstate_migrant_share_of_destination_age5plus=(
                "interstate_migrant_share_of_destination_age5plus", "sum"
            ),
            population_share_of_destination_age5plus=(
                "population_share_of_destination_age5plus", "sum"
            ),
        )
    )
    panel = characteristics.merge(
        migration, on=["census_year", "amc_code"], how="left", validate="one_to_one"
    )
    combined = pd.read_parquet(data_dir / "amc_year_panel.parquet")
    per_capita = combined[["year", "amc_code", "weighted_population", *GDP_LEVEL_COLUMNS]].copy()
    for column in GDP_LEVEL_COLUMNS:
        stem = column.removesuffix("_2010_brl_thousand")
        per_capita[f"{stem}_per_capita_2010_brl"] = (
            per_capita[column] * 1_000
            / per_capita["weighted_population"].where(per_capita["weighted_population"] > 0)
        )
    per_capita_columns = [
        "year", "amc_code",
        *[
            f"{column.removesuffix('_2010_brl_thousand')}_per_capita_2010_brl"
            for column in GDP_LEVEL_COLUMNS
        ],
    ]
    per_capita = per_capita[per_capita_columns].rename(columns={"year": "census_year"})
    panel = panel.merge(
        per_capita,
        on=["census_year", "amc_code"],
        how="left",
        validate="one_to_one",
    )
    if panel.duplicated(["census_year", "amc_code"]).any():
        raise RuntimeError("AMC panel is not unique by AMC-year")
    return panel


def add_municipality_gdp_transformations(frame: pd.DataFrame) -> pd.DataFrame:
    """Add logs and annualized within-municipality growth for GDP concepts."""
    result = frame.copy()
    result = result.sort_values(["municipality_code", "year"]).reset_index(drop=True)
    prior_year = result.groupby("municipality_code", observed=True)["year"].shift(1)
    year_gap = result["year"] - prior_year
    for column in GDP_LEVEL_COLUMNS:
        positive = pd.to_numeric(result[column], errors="coerce").where(result[column] > 0)
        result[f"log_{column}"] = np.log(positive)
        prior_log = result.groupby("municipality_code", observed=True)[f"log_{column}"].shift(1)
        stem = column.removesuffix("_2010_brl_thousand")
        result[f"{stem}_annualized_log_growth_pct"] = (
            100 * (result[f"log_{column}"] - prior_log) / year_gap.where(year_gap > 0)
        )
    return result


def build_psw_baseline(amc_panel: pd.DataFrame, municipality_gdp: pd.DataFrame) -> pd.DataFrame:
    """Combine 1970 AMC census traits with 1970 municipal GDP at AMC level."""
    baseline_columns = [
        "amc_code", "weighted_population", "mean_age", "female_share",
        "literacy_share_age15plus", "mean_education_years_age25plus",
        "employment_share_age15plus", "urban_share", "mean_household_size",
        "electricity_share", "automobile_share", "race_black_share",
        "race_pardo_share", "race_indigenous_share",
    ]
    census = amc_panel.loc[amc_panel["census_year"] == 1970, baseline_columns].copy()
    if census["amc_code"].duplicated().any():
        raise RuntimeError("1970 AMC characteristics contain duplicate AMCs")

    level_columns = [
        "gdp_total_2010_brl_thousand", "va_industry_2010_brl_thousand",
        "va_services_private_2010_brl_thousand",
        "va_public_administration_2010_brl_thousand",
        "va_agriculture_2010_brl_thousand",
    ]
    gdp_1970 = municipality_gdp[municipality_gdp["year"] == 1970]
    gdp = gdp_1970.groupby("amc_code", observed=True, as_index=False)[level_columns].sum(min_count=1)
    denominator = gdp["gdp_total_2010_brl_thousand"].where(
        gdp["gdp_total_2010_brl_thousand"] > 0
    )
    gdp["va_industry_share_of_gdp"] = gdp["va_industry_2010_brl_thousand"] / denominator
    gdp["va_services_private_share_of_gdp"] = gdp["va_services_private_2010_brl_thousand"] / denominator
    gdp["va_agriculture_share_of_gdp"] = gdp["va_agriculture_2010_brl_thousand"] / denominator

    baseline = census.merge(gdp, on="amc_code", how="left", validate="one_to_one")
    baseline["log_population_1970"] = np.log1p(baseline["weighted_population"].clip(lower=0))
    baseline["log_gdp_total_1970"] = np.log1p(
        baseline["gdp_total_2010_brl_thousand"].clip(lower=0)
    )
    gdp_per_resident = (
        baseline["gdp_total_2010_brl_thousand"] * 1_000
        / baseline["weighted_population"].where(baseline["weighted_population"] > 0)
    )
    baseline["log_gdp_per_resident_1970"] = np.log1p(gdp_per_resident.clip(lower=0))
    return baseline


def prettify_boxplot(ax: plt.Axes, label: str) -> None:
    ax.set_xlabel("Year")
    ax.set_ylabel(label)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Assignment")
        texts = legend.get_texts()
        if len(texts) == 2:
            texts[0].set_text("Control")
            texts[1].set_text("Treated")


def json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records"))


def generate_outcome_artifacts(
    frame: pd.DataFrame,
    time_column: str,
    treatment_column: str,
    outcomes: list[str],
    treatment_key: str,
    treatment_title: str,
    level: str,
    output_root: Path,
    dpi: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    plot_dir = output_root / treatment_key / level / "plots"
    table_dir = output_root / treatment_key / level / "tables"
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []

    for outcome in outcomes:
        if outcome not in frame.columns:
            skipped.append({"level": level, "outcome": outcome, "reason": "column missing"})
            continue
        numeric = pd.to_numeric(frame[outcome], errors="coerce")
        if numeric.notna().sum() < 4 or numeric.dropna().nunique() < 2:
            skipped.append({"level": level, "outcome": outcome, "reason": "insufficient variation"})
            continue
        use = frame[[time_column, treatment_column, outcome]].copy()
        valid = use.dropna(subset=[time_column, treatment_column, outcome])
        if valid[treatment_column].nunique() != 2:
            skipped.append({"level": level, "outcome": outcome, "reason": "both groups not observed"})
            continue

        label = outcome_label(outcome)
        title = f"{label}: {treatment_title}"
        fig, ax = boxplot(
            title,
            use,
            time_column,
            treatment_column,
            outcome,
            treatment_start=TREATMENT_START_YEAR,
        )
        prettify_boxplot(ax, label)
        summary = ax.boxplot_summary.copy()
        filename = slugify(outcome)
        plot_path = plot_dir / f"{filename}.png"
        table_path = table_dir / f"{filename}.tex"
        csv_path = table_dir / f"{filename}.csv"
        fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        latex = boxplot_summary_to_latex(
            summary, time_column, treatment_column, label,
            caption=f"{label}: {treatment_title}",
            label=f"tab:{slugify(treatment_key)}_{filename}",
            control_value=False, treated_value=True, significant_digits=3,
            include_mean_difference=True, include_smd=False,
            unit_label="municipalities" if level == "municipality" else "AMCs",
        )
        table_path.write_text(latex, encoding="utf-8")
        summary.to_csv(csv_path, index=False)
        generated.append({
            "outcome": outcome, "label": label,
            "category": outcome_category(outcome, level), "level": level,
            "plot_path": str(plot_path.resolve()),
            "table_path": str(table_path.resolve()),
            "summary_csv_path": str(csv_path.resolve()),
            "time_column": time_column, "treatment_column": treatment_column,
            "summary": json_records(summary),
        })
        print(f"  {level}: {outcome}", flush=True)
    return generated, skipped


def generate_gaez_artifact(
    gaez_long: pd.DataFrame,
    treatment: pd.DataFrame,
    treatment_key: str,
    treatment_title: str,
    output_root: Path,
    dpi: int,
) -> dict[str, object]:
    """Compare high-input rainfed suitability across eight GAEZ crops."""
    use = gaez_long.loc[
        gaez_long["input_level"].eq("high"),
        ["amc_code", "crop_name", "mean_suitability_0_100"],
    ].copy()
    use["crop_name"] = pd.Categorical(
        use["crop_name"], categories=GAEZ_CROP_ORDER, ordered=True
    )
    use = use.merge(
        treatment[["amc_code", treatment_key]],
        on="amc_code", how="inner", validate="many_to_one",
    )
    label = "High-input rainfed crop suitability (0-100)"
    fig, ax = boxplot(
        f"FAO/GAEZ crop suitability: {treatment_title}",
        use, "crop_name", treatment_key, "mean_suitability_0_100",
    )
    ax.set_xlabel("Crop")
    ax.set_ylabel(label)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=18)
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Assignment")
        texts = legend.get_texts()
        if len(texts) == 2:
            texts[0].set_text("Control")
            texts[1].set_text("Treated")
    summary = ax.boxplot_summary.copy()

    directory = output_root / treatment_key / "gaez"
    directory.mkdir(parents=True, exist_ok=True)
    plot_path = directory / "high_input_crop_suitability.png"
    table_path = directory / "high_input_crop_suitability.tex"
    csv_path = directory / "high_input_crop_suitability.csv"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    latex = boxplot_summary_to_latex(
        summary, "crop_name", treatment_key, label,
        caption=f"FAO/GAEZ high-input rainfed crop suitability: {treatment_title}",
        label=f"tab:{slugify(treatment_key)}_gaez_high_input",
        control_value=False, treated_value=True, significant_digits=3,
        include_mean_difference=True, include_smd=True, unit_label="AMCs",
    )
    table_path.write_text(latex, encoding="utf-8")
    summary.to_csv(csv_path, index=False)
    return {
        "title": "FAO/GAEZ high-input rainfed suitability",
        "description": (
            "Static 1961-1990 all-land suitability under commercial, modern-input "
            "management; 0-100 index."
        ),
        "input_level": "high",
        "crop_order": GAEZ_CROP_ORDER,
        "plot_path": str(plot_path.resolve()),
        "table_path": str(table_path.resolve()),
        "summary_csv_path": str(csv_path.resolve()),
        "time_column": "crop_name",
        "treatment_column": treatment_key,
        "summary": json_records(summary),
    }


def generate_propensity_artifacts(
    baseline: pd.DataFrame,
    gaez_wide: pd.DataFrame,
    treatment: pd.DataFrame,
    treatment_key: str,
    treatment_title: str,
    output_root: Path,
    dpi: int,
) -> dict[str, object]:
    base_use = baseline.merge(
        treatment[["amc_code", treatment_key]],
        on="amc_code", how="inner", validate="one_to_one",
    )
    gaez_use = base_use.merge(
        gaez_wide[["amc_code", *GAEZ_HIGH_INPUT_COVARIATES]],
        on="amc_code", how="inner", validate="one_to_one",
    )
    base_weighted, base_balance, base_diagnostics = generate_psw_weights(
        base_use, id_column="amc_code", treatment=treatment_key,
        covariates=PSW_COVARIATES, estimand="ATT", trim=0.01,
        empirical_common_support=True,
    )
    gaez_weighted, gaez_balance, gaez_diagnostics = generate_psw_weights(
        gaez_use, id_column="amc_code", treatment=treatment_key,
        covariates=[*PSW_COVARIATES, *GAEZ_HIGH_INPUT_COVARIATES],
        estimand="ATT", trim=0.01, empirical_common_support=True,
    )
    directory = output_root / treatment_key / "propensity"
    directory.mkdir(parents=True, exist_ok=True)
    plot_path = directory / "propensity_score_overlap_gaez_comparison.png"
    fig, _ = plot_propensity_model_comparison(
        [
            ("1970 census and GDP", base_weighted),
            ("1970 census and GDP + GAEZ", gaez_weighted),
        ],
        treatment_key, title=f"Propensity-score overlap: {treatment_title}",
    )
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    def save_model(
        name: str,
        weighted: pd.DataFrame,
        balance: pd.DataFrame,
        diagnostics: dict[str, object],
    ) -> dict[str, object]:
        weights_path = directory / f"amc_1970_psw_weights_{name}.parquet"
        balance_path = directory / f"covariate_balance_{name}.csv"
        diagnostics_path = directory / f"diagnostics_{name}.json"
        weighted.to_parquet(weights_path, index=False)
        balance.to_csv(balance_path, index=False)
        diagnostics_path.write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        return {
            "weights_path": str(weights_path.resolve()),
            "balance_path": str(balance_path.resolve()),
            "diagnostics_path": str(diagnostics_path.resolve()),
            "diagnostics": diagnostics,
            "balance": json_records(balance),
        }

    return {
        "plot_path": str(plot_path.resolve()),
        "without_gaez": save_model(
            "without_gaez", base_weighted, base_balance, base_diagnostics
        ),
        "with_gaez": save_model(
            "with_gaez", gaez_weighted, gaez_balance, gaez_diagnostics
        ),
        "gaez_covariates": GAEZ_HIGH_INPUT_COVARIATES,
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Reading and harmonizing analysis panels...", flush=True)
    amc_panel = build_amc_year_panel(args.data)
    municipality_gdp = add_municipality_gdp_transformations(
        pd.read_parquet(args.data / "municipality_year_gdp.parquet")
    )
    amc_treatment = pd.read_parquet(args.data / "polocentro_1975_amc_treatment.parquet")
    municipality_treatment = pd.read_parquet(
        args.data / "polocentro_1975_municipality_treatment.parquet"
    )
    mapbiomas = pd.read_parquet(
        args.environment_data / "mapbiomas_collection11_amc_year.parquet"
    )
    # Annual source coverage is preserved in the input panel. Five-year display
    # intervals keep 41 annual boxplot groups and slide tables legible.
    mapbiomas = mapbiomas.loc[
        mapbiomas["year"].eq(1985) | mapbiomas["year"].mod(5).eq(0)
    ].copy()
    gaez_long = pd.read_parquet(
        args.environment_data / "gaez_v4_6190_amc_crop_suitability.parquet"
    )
    gaez_wide = pd.read_parquet(
        args.environment_data / "gaez_v4_6190_amc_crop_suitability_wide.parquet"
    )
    baseline = build_psw_baseline(amc_panel, municipality_gdp)

    manifest: dict[str, object] = {
        "title": "POLOCENTRO treatment-control comparisons",
        "generated_from": str(Path(__file__).resolve()),
        "output_root": str(args.output.resolve()),
        "analysis_grains": {
            "amc": "one AMC-census year observation",
            "municipality": "one municipality-GDP year observation",
            "propensity": "one AMC using 1970 census and aggregated 1970 municipal GDP covariates",
            "gdp_per_capita": "AMC-aggregated GDP or value added divided by harmonized AMC census population",
            "mapbiomas": "one AMC-year observation from MapBiomas Collection 11, 1985-2025",
            "gaez": "one static AMC-crop observation from FAO/GAEZ v4, 1961-1990 climate normal",
        },
        "transformations": {
            "logs": "natural logs of strictly positive constant-2010-R$ municipality values",
            "per_capita": "constant 2010 R$ per AMC resident in years with both GDP and census population",
            "growth": "100 times the within-municipality log change divided by the elapsed years",
            "treatment_start_year": TREATMENT_START_YEAR,
            "mapbiomas_shares": "land-cover hectares divided by mapped territory; hectare outcomes excluded",
            "mapbiomas_display_years": (
                "1985 and five-year intervals through 2025; the source panel remains annual"
            ),
            "gaez": (
                "high-input rainfed all-land suitability on a 0-100 scale; "
                "eight crop-specific controls added to the inclusive PS model"
            ),
        },
        "treatments": [],
    }
    all_skipped: list[dict[str, str]] = []

    for treatment_key in possible_treat_assigns:
        spec = TREATMENT_SPECS[treatment_key]
        print(f"Generating {spec['title']}...", flush=True)
        amc_use = amc_panel.merge(
            amc_treatment[["amc_code", treatment_key]],
            on="amc_code", how="inner", validate="many_to_one",
        )
        municipality_column = str(spec["municipality_column"])
        municipality_assignment = municipality_treatment[
            ["municipality_code", municipality_column]
        ].rename(columns={municipality_column: treatment_key})
        municipality_use = municipality_gdp.merge(
            municipality_assignment,
            on="municipality_code", how="inner", validate="many_to_one",
        )
        environment_use = mapbiomas.merge(
            amc_treatment[["amc_code", treatment_key]],
            on="amc_code", how="inner", validate="many_to_one",
        )

        amc_generated, amc_skipped = generate_outcome_artifacts(
            amc_use, "census_year", treatment_key, amc_outcomes,
            treatment_key, str(spec["title"]), "amc", args.output, args.dpi,
        )
        municipality_generated, municipality_skipped = generate_outcome_artifacts(
            municipality_use, "year", treatment_key, muni_outcomes,
            treatment_key, str(spec["title"]), "municipality", args.output, args.dpi,
        )
        environment_generated, environment_skipped = generate_outcome_artifacts(
            environment_use, "year", treatment_key, MAPBIOMAS_OUTCOMES,
            treatment_key, str(spec["title"]), "environment", args.output, args.dpi,
        )
        gaez = generate_gaez_artifact(
            gaez_long, amc_treatment, treatment_key, str(spec["title"]),
            args.output, args.dpi,
        )
        propensity = generate_propensity_artifacts(
            baseline, gaez_wide, amc_treatment, treatment_key, str(spec["title"]),
            args.output, args.dpi,
        )
        all_skipped.extend(
            [{"treatment": treatment_key, **item} for item in (
                amc_skipped + municipality_skipped + environment_skipped
            )]
        )
        manifest["treatments"].append({
            "key": treatment_key, "title": spec["title"],
            "description": spec["description"],
            "amc_treated": int(amc_treatment[treatment_key].sum()),
            "amc_control": int((~amc_treatment[treatment_key]).sum()),
            "municipality_treatment_column": municipality_column,
            "municipalities_treated": int(municipality_treatment[municipality_column].sum()),
            "municipalities_control": int((~municipality_treatment[municipality_column]).sum()),
            "amc_outcomes": amc_generated,
            "municipality_outcomes": municipality_generated,
            "environment_outcomes": environment_generated,
            "gaez": gaez,
            "propensity": propensity,
        })

    manifest["skipped_outcomes"] = all_skipped
    manifest_path = args.output / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(all_skipped).to_csv(args.output / "skipped_outcomes.csv", index=False)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
