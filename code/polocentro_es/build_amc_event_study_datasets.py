"""Build analysis-ready AMC-year outcome and predetermined-control datasets.

The outputs combine the harmonized 1970--2010 AMC census/GDP panels with
MapBiomas Collection 11 and FAO/IIASA GAEZ v4.  They are deliberately kept
separate from the POLOCENTRO treatment-assignment table and merge cleanly on
``amc_code`` (and, for the two panels, ``year``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
AMC_DIR = ROOT / "data" / "processed" / "amcs"
ENV_DIR = ROOT / "data" / "processed" / "polocentro_environment"
OUTPUT_DIR = ROOT / "data" / "processed" / "polocentro_event_study"

GDP_TOTAL = "gdp_total_2010_brl_thousand"
AGRI_VA = "va_agriculture_2010_brl_thousand"

UF_CODE_TO_ABBR = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}


def positive_log(series: pd.Series) -> pd.Series:
    """Natural log with zero and negative values represented as missing."""
    numeric = pd.to_numeric(series, errors="coerce")
    return np.log(numeric.where(numeric > 0))


def add_annualized_log_growth(
    frame: pd.DataFrame,
    level_column: str,
    log_column: str,
    growth_column: str,
    gap_column: str,
) -> pd.DataFrame:
    """Add 100 * log change / elapsed years within AMC source observations."""
    result = frame.sort_values(["amc_code", "year"]).copy()
    result[log_column] = positive_log(result[level_column])
    grouped = result.groupby("amc_code", observed=True, sort=False)
    prior_year = grouped["year"].shift(1)
    prior_log = grouped[log_column].shift(1)
    result[gap_column] = result["year"] - prior_year
    valid_gap = result[gap_column].where(result[gap_column] > 0)
    result[growth_column] = 100 * (result[log_column] - prior_log) / valid_gap
    return result


def build_migration_panel() -> pd.DataFrame:
    flows = pd.read_parquet(AMC_DIR / "amc_origin_uf_year_flows.parquet")
    migration = (
        flows.groupby(["amc_code", "census_year"], observed=True, as_index=False)
        .agg(
            weighted_interstate_migrants=("weighted_interstate_migrants", "sum"),
            interstate_migrant_share_of_destination_age5plus=(
                "interstate_migrant_share_of_destination_age5plus", "sum"
            ),
        )
        .rename(columns={"census_year": "year"})
    )
    return migration


def build_state_crosswalk(amc_codes: pd.Series) -> pd.DataFrame:
    crosswalk = pd.read_parquet(AMC_DIR / "municipality_to_amc_crosswalk.parquet")
    crosswalk = crosswalk.loc[
        crosswalk["census_year"].eq(2010), ["amc_code", "current_uf"]
    ].copy()
    crosswalk["uf_code"] = (
        crosswalk["current_uf"].astype("string").str.extract(r"(\d+)", expand=False)
        .str.zfill(2)
    )
    crosswalk["state_member"] = crosswalk["uf_code"].map(UF_CODE_TO_ABBR)
    if crosswalk["state_member"].isna().any():
        bad = crosswalk.loc[crosswalk["state_member"].isna(), "current_uf"].unique()
        raise RuntimeError(f"Unmapped UF codes in 2010 AMC crosswalk: {bad.tolist()}")

    counts = (
        crosswalk.groupby(["amc_code", "state_member"], observed=True, as_index=False)
        .size()
        .sort_values(["amc_code", "size", "state_member"], ascending=[True, False, True])
    )
    primary = counts.drop_duplicates("amc_code").rename(
        columns={"state_member": "state_uf"}
    )[["amc_code", "state_uf"]]
    state_sets = (
        crosswalk.groupby("amc_code", observed=True)["state_member"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("state_uf_set")
        .reset_index()
    )
    states = primary.merge(state_sets, on="amc_code", validate="one_to_one")
    states["cross_state_amc"] = states["state_uf_set"].str.contains("|", regex=False)
    states["state_assignment_method"] = np.where(
        states["cross_state_amc"],
        "plurality of 2010 constituent municipalities",
        "unique 2010 constituent UF",
    )
    states = pd.DataFrame({"amc_code": amc_codes.astype(str).unique()}).merge(
        states, on="amc_code", how="left", validate="one_to_one"
    )
    if states["state_uf"].isna().any():
        missing = states.loc[states["state_uf"].isna(), "amc_code"].tolist()
        raise RuntimeError(f"State is missing for AMC codes: {missing[:20]}")
    return states


def build_outcomes(amc_codes: pd.Series, years: list[int]) -> pd.DataFrame:
    grid = pd.MultiIndex.from_product(
        [sorted(amc_codes.astype(str).unique()), years], names=["amc_code", "year"]
    ).to_frame(index=False)

    combined = pd.read_parquet(AMC_DIR / "amc_year_panel.parquet")
    economic = combined[["amc_code", "year", GDP_TOTAL, AGRI_VA]].copy()
    economic = add_annualized_log_growth(
        economic,
        GDP_TOTAL,
        "log_gdp_total_2010_brl_thousand",
        "gdp_total_annualized_log_growth_pct",
        "gdp_growth_year_gap",
    )
    economic = add_annualized_log_growth(
        economic,
        AGRI_VA,
        "log_va_agriculture_2010_brl_thousand",
        "va_agriculture_annualized_log_growth_pct",
        "va_agriculture_growth_year_gap",
    )
    economic = economic[
        [
            "amc_code", "year", "log_gdp_total_2010_brl_thousand",
            "log_va_agriculture_2010_brl_thousand",
            "gdp_total_annualized_log_growth_pct", "gdp_growth_year_gap",
            "va_agriculture_annualized_log_growth_pct",
            "va_agriculture_growth_year_gap",
        ]
    ]

    population = combined.loc[
        combined["weighted_population"].notna(),
        ["amc_code", "year", "weighted_population"],
    ].copy()
    population = add_annualized_log_growth(
        population,
        "weighted_population",
        "log_population",
        "population_annualized_log_growth_pct",
        "population_growth_year_gap",
    )[
        [
            "amc_code", "year", "log_population",
            "population_annualized_log_growth_pct", "population_growth_year_gap",
        ]
    ]

    mapbiomas = pd.read_parquet(ENV_DIR / "mapbiomas_collection11_amc_year.parquet")
    environment_columns = [
        "amc_code", "year", "pasture_share_of_mapped",
        "agriculture_share_of_mapped", "soybean_share_of_mapped",
        "native_vegetation_net_loss_share_of_1985",
    ]
    environment = mapbiomas[environment_columns].copy()
    migration = build_migration_panel()

    outcomes = grid.merge(economic, on=["amc_code", "year"], how="left", validate="one_to_one")
    outcomes = outcomes.merge(population, on=["amc_code", "year"], how="left", validate="one_to_one")
    outcomes = outcomes.merge(environment, on=["amc_code", "year"], how="left", validate="one_to_one")
    outcomes = outcomes.merge(migration, on=["amc_code", "year"], how="left", validate="one_to_one")
    outcomes["has_gdp_outcome"] = outcomes["log_gdp_total_2010_brl_thousand"].notna()
    outcomes["has_population_outcome"] = outcomes["log_population"].notna()
    outcomes["has_migration_outcome"] = outcomes["weighted_interstate_migrants"].notna()
    outcomes["has_mapbiomas_outcome"] = outcomes["pasture_share_of_mapped"].notna()

    ordered = [
        "amc_code", "year",
        "log_gdp_total_2010_brl_thousand",
        "log_va_agriculture_2010_brl_thousand",
        "gdp_total_annualized_log_growth_pct",
        "va_agriculture_annualized_log_growth_pct",
        "pasture_share_of_mapped", "agriculture_share_of_mapped",
        "soybean_share_of_mapped",
        "native_vegetation_net_loss_share_of_1985",
        "weighted_interstate_migrants",
        "interstate_migrant_share_of_destination_age5plus",
        "log_population", "population_annualized_log_growth_pct",
        "gdp_growth_year_gap", "va_agriculture_growth_year_gap",
        "population_growth_year_gap", "has_gdp_outcome",
        "has_population_outcome", "has_migration_outcome",
        "has_mapbiomas_outcome",
    ]
    return outcomes[ordered].sort_values(["amc_code", "year"]).reset_index(drop=True)


def build_controls(outcomes: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    combined = pd.read_parquet(AMC_DIR / "amc_year_panel.parquet")
    baseline_columns = [
        "amc_code", "weighted_population", "positive_income_q1_share_age15plus",
        "positive_income_q3_share_age15plus", "female_share", "urban_share",
        "share_age_65plus", "literacy_share_age15plus", GDP_TOTAL,
        "gdp_per_capita_2010_brl", "va_agriculture_share_of_gdp",
    ]
    baseline = combined.loc[combined["year"].eq(1970), baseline_columns].copy()
    if baseline["amc_code"].duplicated().any():
        raise RuntimeError("1970 baseline is not unique by AMC")
    baseline["log_population_1970"] = positive_log(baseline["weighted_population"])
    baseline["log_gdp_total_1970"] = positive_log(baseline[GDP_TOTAL])
    baseline["log_gdp_per_capita_1970"] = positive_log(
        baseline["gdp_per_capita_2010_brl"]
    )
    baseline = baseline.rename(
        columns={
            "positive_income_q1_share_age15plus": "positive_income_q1_share_age15plus_1970",
            "positive_income_q3_share_age15plus": "positive_income_q3_share_age15plus_1970",
            "female_share": "female_share_1970",
            "urban_share": "urban_share_1970",
            "share_age_65plus": "share_age_65plus_1970",
            "literacy_share_age15plus": "literacy_share_age15plus_1970",
            "va_agriculture_share_of_gdp": "va_agriculture_share_of_gdp_1970",
        }
    )
    baseline = baseline[
        [
            "amc_code", "positive_income_q1_share_age15plus_1970",
            "positive_income_q3_share_age15plus_1970", "female_share_1970",
            "urban_share_1970", "share_age_65plus_1970",
            "log_population_1970", "log_gdp_total_1970",
            "log_gdp_per_capita_1970", "va_agriculture_share_of_gdp_1970",
            "literacy_share_age15plus_1970",
        ]
    ]

    gaez = pd.read_parquet(ENV_DIR / "gaez_v4_6190_amc_crop_suitability_wide.parquet")
    gaez_columns = [column for column in gaez.columns if column.startswith("gaez_")]
    static = states.merge(baseline, on="amc_code", how="left", validate="one_to_one")
    static = static.merge(gaez, on="amc_code", how="left", validate="one_to_one")
    controls = outcomes[["amc_code", "year"]].merge(
        static, on="amc_code", how="left", validate="many_to_one"
    )
    ordered = [
        "amc_code", "year", "state_uf", "state_uf_set", "cross_state_amc",
        "state_assignment_method", "positive_income_q1_share_age15plus_1970",
        "positive_income_q3_share_age15plus_1970", "female_share_1970",
        "urban_share_1970", "share_age_65plus_1970",
        "log_population_1970", "log_gdp_total_1970",
        "log_gdp_per_capita_1970", "va_agriculture_share_of_gdp_1970",
        "literacy_share_age15plus_1970", *gaez_columns,
    ]
    return controls[ordered].sort_values(["amc_code", "year"]).reset_index(drop=True)


def variable_dictionary() -> dict[str, dict[str, str]]:
    dictionary = {
        "amc_code": {"unit": "identifier", "definition": "1970-2010 minimum comparable area code."},
        "year": {"unit": "calendar year", "definition": "Outcome year; controls are repeated over the outcome key grid."},
        "log_gdp_total_2010_brl_thousand": {"unit": "natural log", "definition": "Log AMC GDP in constant 2010 R$ thousand; strictly positive values only."},
        "log_va_agriculture_2010_brl_thousand": {"unit": "natural log", "definition": "Log AMC agriculture value added in constant 2010 R$ thousand; strictly positive values only."},
        "gdp_total_annualized_log_growth_pct": {"unit": "percent per year", "definition": "100 times the log GDP change divided by years since the prior observed GDP year."},
        "va_agriculture_annualized_log_growth_pct": {"unit": "percent per year", "definition": "100 times the log agriculture-VA change divided by years since the prior observed GDP year."},
        "pasture_share_of_mapped": {"unit": "share 0-1", "definition": "MapBiomas pasture hectares divided by mapped AMC territory."},
        "agriculture_share_of_mapped": {"unit": "share 0-1", "definition": "MapBiomas agriculture hectares divided by mapped AMC territory."},
        "soybean_share_of_mapped": {"unit": "share 0-1", "definition": "MapBiomas soybean hectares divided by mapped AMC territory."},
        "native_vegetation_net_loss_share_of_1985": {"unit": "share of 1985 stock", "definition": "Signed cumulative native-vegetation stock loss since 1985 divided by 1985 native-vegetation hectares; recovery can offset loss."},
        "weighted_interstate_migrants": {"unit": "persons", "definition": "Survey-weighted interstate in-migrant stock in the age-5+ migration universe."},
        "interstate_migrant_share_of_destination_age5plus": {"unit": "share 0-1", "definition": "Interstate in-migrants divided by destination AMC residents age 5+."},
        "log_population": {"unit": "natural log", "definition": "Log survey-weighted usual-resident population; strictly positive values only."},
        "population_annualized_log_growth_pct": {"unit": "percent per year", "definition": "100 times the log population change divided by years since the prior census."},
        "positive_income_q1_share_age15plus_1970": {"unit": "share 0-1", "definition": "1970 share of age-15+ positive-income residents in the national lowest positive-income quintile."},
        "positive_income_q3_share_age15plus_1970": {"unit": "share 0-1", "definition": "1970 share of age-15+ positive-income residents in the national third positive-income quintile."},
        "female_share_1970": {"unit": "share 0-1", "definition": "Female share of the AMC population in 1970."},
        "urban_share_1970": {"unit": "share 0-1", "definition": "Urban share of the AMC population in 1970."},
        "share_age_65plus_1970": {"unit": "share 0-1", "definition": "Share of the AMC population age 65 or older in 1970."},
        "log_population_1970": {"unit": "natural log", "definition": "Predetermined 1970 log population; added scale control."},
        "log_gdp_total_1970": {"unit": "natural log", "definition": "Predetermined 1970 log total GDP in constant 2010 R$ thousand; added baseline-outcome/scale control."},
        "log_gdp_per_capita_1970": {"unit": "natural log", "definition": "Predetermined 1970 log GDP per resident in constant 2010 R$; added development control."},
        "va_agriculture_share_of_gdp_1970": {"unit": "share 0-1", "definition": "1970 agriculture value added divided by GDP; added productive-structure control."},
        "literacy_share_age15plus_1970": {"unit": "share 0-1", "definition": "1970 literacy share among residents age 15+; added human-capital control."},
    }
    return dictionary


def validate_and_summarize(
    outcomes: pd.DataFrame, controls: pd.DataFrame, treatment: pd.DataFrame
) -> pd.DataFrame:
    if outcomes.duplicated(["amc_code", "year"]).any():
        raise RuntimeError("Outcome dataset has duplicate AMC-year keys")
    if controls.duplicated(["amc_code", "year"]).any():
        raise RuntimeError("Control dataset has duplicate AMC-year keys")
    if not outcomes[["amc_code", "year"]].equals(controls[["amc_code", "year"]]):
        raise RuntimeError("Outcome and control keys differ")
    treatment_codes = set(treatment["amc_code"].astype(str))
    output_codes = set(outcomes["amc_code"].astype(str))
    if treatment_codes != output_codes:
        raise RuntimeError("AMC universe differs from treatment-assignment universe")

    checks: list[dict[str, object]] = [
        {"check": "outcome_rows", "value": len(outcomes)},
        {"check": "control_rows", "value": len(controls)},
        {"check": "unique_amcs", "value": outcomes["amc_code"].nunique()},
        {"check": "unique_years", "value": outcomes["year"].nunique()},
        {"check": "duplicate_outcome_keys", "value": int(outcomes.duplicated(["amc_code", "year"]).sum())},
        {"check": "duplicate_control_keys", "value": int(controls.duplicated(["amc_code", "year"]).sum())},
        {"check": "treatment_amc_universe_match", "value": True},
        {"check": "cross_state_amcs", "value": int(controls.drop_duplicates("amc_code")["cross_state_amc"].sum())},
    ]
    for column in outcomes.columns[2:]:
        checks.append({"check": f"nonmissing_outcomes::{column}", "value": int(outcomes[column].notna().sum())})
    for column in controls.columns[2:]:
        checks.append({"check": f"nonmissing_controls::{column}", "value": int(controls[column].notna().sum())})
    for column in [
        "pasture_share_of_mapped", "agriculture_share_of_mapped",
        "soybean_share_of_mapped", "interstate_migrant_share_of_destination_age5plus",
    ]:
        values = outcomes[column].dropna()
        if ((values < -1e-10) | (values > 1 + 1e-10)).any():
            raise RuntimeError(f"Share outside [0,1]: {column}")
    return pd.DataFrame(checks)


def build_control_assessment(
    controls: pd.DataFrame, treatment: pd.DataFrame
) -> pd.DataFrame:
    """Document requested/supplemental controls and unweighted baseline SMDs."""
    static = controls.drop_duplicates("amc_code").merge(
        treatment[["amc_code", "polocentro_operational_core"]],
        on="amc_code", how="left", validate="one_to_one",
    )
    specifications = {
        "state_uf": ("requested", "geography", "State category; use mainly for state-by-year fixed effects."),
        "positive_income_q1_share_age15plus_1970": ("requested", "income distribution", "Selected from the 1970 balance comparisons."),
        "positive_income_q3_share_age15plus_1970": ("requested", "income distribution", "Selected from the 1970 balance comparisons."),
        "female_share_1970": ("requested", "demographics", "Selected from the 1970 balance comparisons."),
        "urban_share_1970": ("requested", "settlement pattern", "Selected from the 1970 balance comparisons."),
        "share_age_65plus_1970": ("requested", "age composition", "Selected from the 1970 balance comparisons."),
        "log_population_1970": ("added", "scale/exposure", "Important exposure control for migration counts and baseline unit size."),
        "log_gdp_total_1970": ("added", "baseline outcome/scale", "Conditions on aggregate pre-program economic scale."),
        "log_gdp_per_capita_1970": ("added", "development/productivity", "Captures baseline development beyond income-bin composition."),
        "va_agriculture_share_of_gdp_1970": ("added", "productive structure", "Captures pre-program agricultural specialization and selection."),
        "literacy_share_age15plus_1970": ("added", "human capital", "Large pre-treatment imbalance and a distinct growth/migration determinant."),
    }
    for column in controls.columns:
        if column.startswith("gaez_"):
            specifications[column] = (
                "requested", "agro-climatic endowment",
                "Full static GAEZ crop-suitability set requested for selection adjustment/heterogeneity.",
            )

    rows = []
    treated_flag = static["polocentro_operational_core"].astype(bool)
    for variable, (status, dimension, rationale) in specifications.items():
        smd = np.nan
        treated_mean = np.nan
        control_mean = np.nan
        treated_n = 0
        control_n = 0
        if variable != "state_uf":
            values = pd.to_numeric(static[variable], errors="coerce")
            treated = values[treated_flag].dropna()
            control = values[~treated_flag].dropna()
            treated_n, control_n = len(treated), len(control)
            treated_mean, control_mean = treated.mean(), control.mean()
            pooled_sd = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
            if pd.notna(pooled_sd) and pooled_sd > 0:
                smd = (treated_mean - control_mean) / pooled_sd
        rows.append({
            "variable": variable,
            "selection_status": status,
            "control_dimension": dimension,
            "treated_n_core": treated_n,
            "control_n_core": control_n,
            "treated_mean_core": treated_mean,
            "control_mean_core": control_mean,
            "smd_before_core": smd,
            "absolute_smd_before_core": abs(smd) if pd.notna(smd) else np.nan,
            "rationale": rationale,
        })
    return pd.DataFrame(rows).sort_values(
        ["selection_status", "absolute_smd_before_core"],
        ascending=[False, False], na_position="last",
    ).reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    treatment = pd.read_parquet(AMC_DIR / "polocentro_1975_amc_treatment.parquet")
    amc_codes = treatment["amc_code"].astype(str)
    combined = pd.read_parquet(AMC_DIR / "amc_year_panel.parquet", columns=["year"])
    mapbiomas = pd.read_parquet(
        ENV_DIR / "mapbiomas_collection11_amc_year.parquet", columns=["year"]
    )
    years = sorted(set(combined["year"].astype(int)) | set(mapbiomas["year"].astype(int)))

    outcomes = build_outcomes(amc_codes, years)
    states = build_state_crosswalk(amc_codes)
    controls = build_controls(outcomes, states)
    validation = validate_and_summarize(outcomes, controls, treatment)
    control_assessment = build_control_assessment(controls, treatment)

    outcomes.to_parquet(OUTPUT_DIR / "amc_year_outcomes.parquet", index=False)
    outcomes.to_csv(OUTPUT_DIR / "amc_year_outcomes.csv.gz", index=False, compression="gzip")
    controls.to_parquet(OUTPUT_DIR / "amc_year_controls.parquet", index=False)
    controls.to_csv(OUTPUT_DIR / "amc_year_controls.csv.gz", index=False, compression="gzip")
    validation.to_csv(OUTPUT_DIR / "validation_summary.csv", index=False)
    control_assessment.to_csv(OUTPUT_DIR / "control_selection_assessment.csv", index=False)

    dictionary = variable_dictionary()
    for column in controls.columns:
        if column.startswith("gaez_"):
            dictionary[column] = {
                "unit": "suitability index 0-100",
                "definition": "FAO/IIASA GAEZ v4 rainfed 1961-1990 climate-normal crop-suitability statistic; see source metadata for crop/input/statistic encoded in the name.",
            }
    metadata = {
        "created_by": str(Path(__file__).resolve()),
        "unit_of_observation": "AMC-year",
        "key": ["amc_code", "year"],
        "amc_universe": "3,800 Ehrl/geobr 1970-2010 minimum comparable areas; identical to POLOCENTRO treatment assignment",
        "year_grid": years,
        "row_count_each": len(outcomes),
        "construction": {
            "grid": "Balanced AMC by union-of-outcome-years grid; missing source outcomes remain null and are never interpolated.",
            "logs": "Natural logs of strictly positive levels.",
            "growth": "100 times within-AMC log change divided by elapsed years between consecutive observed source years.",
            "state": "2010 constituent-municipality UF; AMC 11052 uses ES by municipality plurality and retains ES|MG plus a cross-state flag.",
            "controls": "Predetermined 1970 census/GDP variables and static 1961-1990 GAEZ suitability repeated over the outcome year grid.",
        },
        "control_selection_assessment": {
            "requested": [
                "state_uf", "positive_income_q1_share_age15plus_1970",
                "positive_income_q3_share_age15plus_1970", "female_share_1970",
                "urban_share_1970", "share_age_65plus_1970",
                "all GAEZ wide crop-suitability columns",
            ],
            "added": {
                "log_population_1970": "Baseline scale and exposure for count outcomes.",
                "log_gdp_total_1970": "Baseline outcome/aggregate economic scale.",
                "log_gdp_per_capita_1970": "Baseline development/productivity not captured by income-bin composition alone.",
                "va_agriculture_share_of_gdp_1970": "Predetermined productive specialization, central to treatment selection and agricultural outcomes.",
                "literacy_share_age15plus_1970": "Human capital; the operational-core treated-control standardized difference is about 0.54 before adjustment.",
            },
            "econometric_note": "All controls are time-invariant. AMC fixed effects absorb their levels; use them for weighting/matching, interactions with time, state-by-year fixed effects, or heterogeneous effects rather than adding their raw levels to a unit-FE regression.",
        },
        "important_cautions": [
            "MapBiomas starts in 1985, ten years after January 1975 assignment, so land-cover outcomes cannot identify pre-treatment trends.",
            "Population and interstate migration have only the 1970 pre-treatment census observation; they cannot support conventional multi-lead pre-trend tests without another source or design.",
            "The 1980 interstate-migration measure uses recent residence plus birth-UF as a proxy; other years use the definitions documented in the AMC metadata.",
            "Native-vegetation loss is net stock loss, not gross deforestation; within-AMC recovery can offset clearing.",
            "The 1970 income-quintile shares are conditional on positive income among residents age 15+.",
        ],
        "sources": [
            "data/processed/amcs/amc_year_panel.parquet",
            "data/processed/amcs/amc_origin_uf_year_flows.parquet",
            "data/processed/amcs/municipality_to_amc_crosswalk.parquet",
            "data/processed/amcs/polocentro_1975_amc_treatment.parquet",
            "data/processed/polocentro_environment/mapbiomas_collection11_amc_year.parquet",
            "data/processed/polocentro_environment/gaez_v4_6190_amc_crop_suitability_wide.parquet",
        ],
        "variable_dictionary": dictionary,
    }
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {len(outcomes):,} rows to each AMC-year dataset")
    print(f"AMC count: {outcomes['amc_code'].nunique():,}; years: {len(years)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
