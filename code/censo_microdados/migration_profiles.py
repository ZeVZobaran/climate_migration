r"""Compare immigrants, emigrants, and stayers in Brazilian census microdata.

The script scans the harmonized person Parquet dataset in batches, applies the
census-specific migration definition, and writes survey-weighted distributions
and SVG profile plots for every state-year and for a person-weighted pool of all
censuses.

Migration definitions
---------------------
1970: previous state of residence (``last_origin_uf_code``).
1980: state of birth (``birth_uf_code``).
1991/2000/2010: state of residence five years before the census.

The analysis universe is people aged five or older with a positive person
weight. International migrants and records whose internal origin cannot be
identified are outside the three-group comparison. A mover contributes as an
immigrant to the destination state and as an emigrant from the origin state.

Examples
--------
Run the complete analysis::

    .venv\Scripts\python.exe -u code\censo_microdados\migration_profiles.py

Regenerate plots from cached aggregate CSV files::

    .venv\Scripts\python.exe -u code\censo_microdados\migration_profiles.py --plots-only
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


YEARS = (1970, 1980, 1991, 2000, 2010)
STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)
STATE_TO_INDEX = {state: i for i, state in enumerate(STATES)}

IBGE_TO_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}

# State ordinals used by the 1970 public-use file. Guanabara is merged into RJ
# and Fernando de Noronha into PE to make state profiles usable across years.
ORIGIN_1970 = {
    "01": "RO", "02": "AC", "03": "AM", "04": "RR", "05": "PA",
    "06": "AP", "07": "MA", "08": "PI", "09": "CE", "10": "RN",
    "11": "PB", "12": "PE", "13": "AL", "14": "PE", "15": "SE",
    "16": "BA", "17": "MG", "18": "ES", "19": "RJ", "20": "RJ",
    "21": "SP", "22": "PR", "23": "SC", "24": "RS", "25": "MT",
    "26": "GO", "27": "DF",
}

# State ordinals used in the 1980 birthplace field. The historical sequence
# contains Fernando de Noronha (merged here into PE) and does not contain TO.
ORIGIN_1980 = {
    "01": "RO", "02": "AC", "03": "AM", "04": "RR", "05": "PA",
    "06": "AP", "07": "MA", "08": "PI", "09": "CE", "10": "RN",
    "11": "PB", "12": "PE", "13": "AL", "14": "PE", "15": "SE",
    "16": "BA", "17": "MG", "18": "ES", "19": "RJ", "20": "SP",
    "21": "PR", "22": "SC", "23": "RS", "24": "MS", "25": "MT",
    "26": "GO", "27": "DF",
}

# Modern state ordinals used for true movers in the 2000 five-year-origin
# field. This sequence contains TO and no longer contains Fernando de Noronha.
ORIGIN_2000 = {
    f"{i:02d}": state for i, state in enumerate(
        ("RO", "AC", "AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE",
         "RN", "PB", "PE", "AL", "SE", "BA", "MG", "ES", "RJ", "SP",
         "PR", "SC", "RS", "MS", "MT", "GO", "DF"),
        start=1,
    )
}

CURRENT_1970 = {state: state for state in STATES}
CURRENT_1970.update({"GB": "RJ", "FN": "PE"})

GROUPS = ("immigrant", "emigrant", "stayer")
GROUP_INDEX = {group: i for i, group in enumerate(GROUPS)}
GROUP_LABEL = {
    "immigrant": "Immigrants",
    "emigrant": "Emigrants",
    "stayer": "Stayers",
}
GROUP_COLOR = {
    "immigrant": "#0072B2",
    "emigrant": "#D55E00",
    "stayer": "#6B7280",
}

PROXY_LABEL = {
    1970: "Proxy: previous state of residence",
    1980: "Proxy: state of birth",
    1991: "Five-year state of residence",
    2000: "Five-year state of residence",
    2010: "Five-year state of residence",
}

INCOME_COLUMN = {
    1970: "income_total",
    1980: "income_main",
    1991: "income_total",
    2000: "income_total",
    2010: "income_total",
}

# Public-use missing-value sentinels that survived numeric parsing.
INCOME_UPPER_EXCLUSIVE = {
    1970: 9_999,
    1980: 9_999_999,
    1991: 99_999_999,
    2000: math.inf,
    2010: math.inf,
}


@dataclass(frozen=True)
class Dimension:
    key: str
    title: str
    labels: tuple[str, ...]


DIMENSIONS = (
    Dimension("age", "Age", ("5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65+")),
    Dimension("income_rank", "Income rank", ("No\nincome", "Q1", "Q2", "Q3", "Q4", "Q5")),
    Dimension("sex", "Sex", ("Women", "Men")),
    Dimension("literacy", "Literacy (age 15+)", ("Not\nliterate", "Literate")),
    Dimension("schooling", "Years of schooling (age 15+)", ("0", "1-3", "4-7", "8-10", "11-14", "15+")),
    Dimension("household_size", "Household size", ("1", "2", "3", "4", "5", "6+")),
    Dimension("rooms", "Rooms in dwelling", ("1", "2", "3", "4", "5", "6+")),
    Dimension("refrigerator", "Refrigerator in dwelling", ("No", "Yes")),
)
DIMENSION_BY_KEY = {dimension.key: dimension for dimension in DIMENSIONS}

SCAN_COLUMNS = (
    "census_year", "current_uf", "person_weight", "age_years", "sex",
    "literacy_code", "education_years", "income_main", "income_total",
    "household_size", "rooms", "refrigerator_code", "birth_uf_code",
    "born_muni_code", "last_origin_uf_code", "origin_5yr_uf_code",
    "internal_migrant_5yr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persons",
        type=Path,
        default=Path("data/processed/censo_microdados/persons"),
        help="Hive-partitioned harmonized person Parquet dataset.",
    )
    parser.add_argument(
        "--aggregates",
        type=Path,
        default=Path("data/processed/censo_microdados/migration_profiles"),
        help="Directory for reusable aggregate CSV and metadata files.",
    )
    parser.add_argument(
        "--figures",
        type=Path,
        default=Path("figs/censo_microdados/migration_profiles"),
        help="Directory for SVG plots.",
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip the microdata scan and render from existing aggregate CSVs.",
    )
    return parser.parse_args()


def clean_code(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.zfill(2) if text.isdigit() and len(text) < 2 else text


def map_current(values: pd.Series, year: int) -> np.ndarray:
    clean = values.astype("string").str.strip()
    if year == 1970:
        mapped = clean.map(CURRENT_1970)
    else:
        mapped = clean.str[:2].map(IBGE_TO_UF)
    return mapped.map(STATE_TO_INDEX).fillna(-1).to_numpy(dtype=np.int16)


def map_origin(values: pd.Series, year: int) -> np.ndarray:
    clean = values.astype("string").str.strip()
    if year == 1970:
        mapped = clean.str.zfill(2).map(ORIGIN_1970)
    elif year == 1980:
        mapped = clean.str.zfill(2).map(ORIGIN_1980)
    elif year == 2000:
        mapped = clean.str.zfill(2).map(ORIGIN_2000)
    elif year == 1991:
        mapped = clean.str[:2].map(IBGE_TO_UF)
    elif year == 2010:
        mapped = clean.str[:2].map(IBGE_TO_UF)
    else:  # pragma: no cover - guarded by YEARS
        raise ValueError(year)
    return mapped.map(STATE_TO_INDEX).fillna(-1).to_numpy(dtype=np.int16)


def valid_income(values: np.ndarray, year: int) -> np.ndarray:
    return np.isfinite(values) & (values >= 0) & (values < INCOME_UPPER_EXCLUSIVE[year])


def exact_income_quintiles(
    dataset: ds.Dataset, year: int, batch_size: int
) -> np.ndarray:
    """Return exact weighted quintile cutoffs among positive-income adults."""
    column = INCOME_COLUMN[year]
    value_weights: defaultdict[float, float] = defaultdict(float)
    scanner = dataset.scanner(
        columns=["age_years", "person_weight", column],
        filter=ds.field("census_year") == year,
        batch_size=batch_size,
        use_threads=True,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        age = frame["age_years"].to_numpy(dtype=float, na_value=np.nan)
        weight = frame["person_weight"].to_numpy(dtype=float, na_value=np.nan)
        income = frame[column].to_numpy(dtype=float, na_value=np.nan)
        keep = (
            (age >= 15)
            & np.isfinite(weight)
            & (weight > 0)
            & valid_income(income, year)
            & (income > 0)
        )
        if not keep.any():
            continue
        unique, inverse = np.unique(income[keep], return_inverse=True)
        sums = np.bincount(inverse, weights=weight[keep])
        for value, total_weight in zip(unique, sums, strict=True):
            value_weights[float(value)] += float(total_weight)

    if not value_weights:
        raise RuntimeError(f"No positive-income observations found for {year}")
    values = np.array(sorted(value_weights), dtype=float)
    weights = np.array([value_weights[value] for value in values], dtype=float)
    cumulative = np.cumsum(weights)
    targets = np.array((0.2, 0.4, 0.6, 0.8)) * cumulative[-1]
    indices = np.searchsorted(cumulative, targets, side="left")
    return values[np.minimum(indices, len(values) - 1)]


def category_arrays(frame: pd.DataFrame, year: int, cutoffs: np.ndarray) -> dict[str, np.ndarray]:
    n = len(frame)
    age = frame["age_years"].to_numpy(dtype=float, na_value=np.nan)

    age_category = np.full(n, -1, dtype=np.int8)
    valid_age = np.isfinite(age) & (age >= 5) & (age <= 120)
    age_category[valid_age] = np.digitize(age[valid_age], (15, 25, 35, 45, 55, 65)).astype(np.int8)

    income = frame[INCOME_COLUMN[year]].to_numpy(dtype=float, na_value=np.nan)
    income_category = np.full(n, -1, dtype=np.int8)
    valid = valid_age & (age >= 15) & valid_income(income, year)
    income_category[valid & (income == 0)] = 0
    positive = valid & (income > 0)
    # Keep identical incomes together. When multiple weighted-quantile cutoffs
    # coincide (notably the 2010 minimum-wage mass), place the tied value in the
    # middle applicable quintile rather than creating an arbitrary upper-bin
    # jump.
    positive_income = income[positive]
    left = np.searchsorted(cutoffs, positive_income, side="left")
    right = np.searchsorted(cutoffs, positive_income, side="right")
    income_category[positive] = (1 + ((left + right) // 2)).astype(np.int8)

    sex_category = np.full(n, -1, dtype=np.int8)
    sex = frame["sex"].astype("string")
    sex_category[(sex == "female").fillna(False).to_numpy()] = 0
    sex_category[(sex == "male").fillna(False).to_numpy()] = 1

    literacy_category = np.full(n, -1, dtype=np.int8)
    literacy = frame["literacy_code"].astype("string").str.strip()
    adult = valid_age & (age >= 15)
    if year == 1970:
        yes = literacy == "1"
        no = literacy == "2"
    elif year == 1980:
        yes = literacy == "2"
        no = literacy.isin(("4", "6"))
    else:
        yes = literacy == "1"
        no = literacy == "2"
    literacy_category[adult & no.fillna(False).to_numpy()] = 0
    literacy_category[adult & yes.fillna(False).to_numpy()] = 1

    schooling_category = np.full(n, -1, dtype=np.int8)
    schooling = frame["education_years"].to_numpy(dtype=float, na_value=np.nan)
    valid_school = adult & np.isfinite(schooling) & (schooling >= 0) & (schooling <= 30)
    schooling_category[valid_school] = np.digitize(
        schooling[valid_school], (1, 4, 8, 11, 15)
    ).astype(np.int8)

    def size_categories(column: str) -> np.ndarray:
        values = frame[column].to_numpy(dtype=float, na_value=np.nan)
        out = np.full(n, -1, dtype=np.int8)
        valid_size = np.isfinite(values) & (values >= 1) & (values <= 30)
        out[valid_size] = np.minimum(values[valid_size].astype(np.int16), 6).astype(np.int8) - 1
        return out

    refrigerator_category = np.full(n, -1, dtype=np.int8)
    refrigerator = frame["refrigerator_code"].astype("string").str.strip()
    if year == 1970:
        yes = refrigerator == "1"
        no = refrigerator == "2"
    elif year == 1980:
        yes = refrigerator == "1"
        no = refrigerator == "8"
    elif year == 1991:
        yes = refrigerator.isin(("1", "2", "3", "4"))
        no = refrigerator == "0"
    else:
        yes = refrigerator == "1"
        no = refrigerator == "2"
    refrigerator_category[no.fillna(False).to_numpy()] = 0
    refrigerator_category[yes.fillna(False).to_numpy()] = 1

    return {
        "age": age_category,
        "income_rank": income_category,
        "sex": sex_category,
        "literacy": literacy_category,
        "schooling": schooling_category,
        "household_size": size_categories("household_size"),
        "rooms": size_categories("rooms"),
        "refrigerator": refrigerator_category,
    }


def migration_masks_and_states(
    frame: pd.DataFrame, year: int, current_state: np.ndarray, universe: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stayer mask, mover mask, and mover origin-state indices."""
    if year == 1970:
        raw = frame["last_origin_uf_code"].astype("string").str.strip()
        origin = map_origin(raw, year)
        raw_missing = raw.isna().to_numpy() | (raw == "").fillna(False).to_numpy()
        stayer = universe & (raw_missing | ((origin >= 0) & (origin == current_state)))
        mover = universe & (origin >= 0) & (origin != current_state)
    elif year == 1980:
        origin = map_origin(frame["birth_uf_code"], year)
        stayer = universe & (origin >= 0) & (origin == current_state)
        mover = universe & (origin >= 0) & (origin != current_state)
    else:
        internal = frame["internal_migrant_5yr"].astype("boolean")
        internal_true = internal.fillna(False).to_numpy(dtype=bool)
        internal_false = (~internal).fillna(False).to_numpy(dtype=bool)
        if year == 1991:
            # The 1991 five-year-origin fields are structurally blank for
            # people who report being born in the current municipality. They
            # are definite state stayers, not unobserved migration cases.
            born_current_muni = (
                frame["born_muni_code"].astype("string").str.strip() == "1"
            ).fillna(False).to_numpy(dtype=bool)
            internal_false |= internal.isna().to_numpy() & born_current_muni
        origin = map_origin(frame["origin_5yr_uf_code"], year)
        stayer = universe & internal_false
        mover = universe & internal_true & (origin >= 0) & (origin != current_state)
    return stayer, mover, origin


def accumulate_contribution(
    group_sizes: np.ndarray,
    distributions: dict[str, np.ndarray],
    year_index: int,
    group: str,
    mask: np.ndarray,
    state_index: np.ndarray,
    weight: np.ndarray,
    categories: dict[str, np.ndarray],
) -> None:
    group_index = GROUP_INDEX[group]
    good = mask & (state_index >= 0)
    if not good.any():
        return
    state = state_index[good]
    contribution_weight = weight[good]
    group_sizes[year_index, :, group_index] += np.bincount(
        state, weights=contribution_weight, minlength=len(STATES)
    )
    source_rows = np.flatnonzero(good)
    for dimension in DIMENSIONS:
        category = categories[dimension.key][source_rows]
        valid = category >= 0
        if not valid.any():
            continue
        flat_index = state[valid] * len(dimension.labels) + category[valid]
        values = np.bincount(
            flat_index,
            weights=contribution_weight[valid],
            minlength=len(STATES) * len(dimension.labels),
        ).reshape(len(STATES), len(dimension.labels))
        distributions[dimension.key][year_index, :, group_index, :] += values


def aggregate(
    dataset: ds.Dataset, batch_size: int
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, np.ndarray], dict[str, int]]:
    income_cutoffs: dict[int, np.ndarray] = {}
    for year in YEARS:
        print(f"Computing {year} national positive-income quintiles...", flush=True)
        income_cutoffs[year] = exact_income_quintiles(dataset, year, batch_size)
        print(f"  cutoffs: {income_cutoffs[year].tolist()}", flush=True)

    group_sizes = np.zeros((len(YEARS), len(STATES), len(GROUPS)), dtype=np.float64)
    distributions = {
        dimension.key: np.zeros(
            (len(YEARS), len(STATES), len(GROUPS), len(dimension.labels)),
            dtype=np.float64,
        )
        for dimension in DIMENSIONS
    }
    diagnostics = {
        "rows_scanned": 0,
        "universe_rows": 0,
        "stayer_rows": 0,
        "mover_rows": 0,
        "unclassified_rows": 0,
    }

    for year_index, year in enumerate(YEARS):
        print(f"Aggregating {year} migration profiles...", flush=True)
        scanner = dataset.scanner(
            columns=list(SCAN_COLUMNS),
            filter=ds.field("census_year") == year,
            batch_size=batch_size,
            use_threads=True,
        )
        batch_number = 0
        for batch in scanner.to_batches():
            batch_number += 1
            frame = batch.to_pandas()
            n = len(frame)
            diagnostics["rows_scanned"] += n
            age = frame["age_years"].to_numpy(dtype=float, na_value=np.nan)
            weight = frame["person_weight"].to_numpy(dtype=float, na_value=np.nan)
            current_state = map_current(frame["current_uf"], year)
            universe = (
                np.isfinite(age)
                & (age >= 5)
                & (age <= 120)
                & np.isfinite(weight)
                & (weight > 0)
                & (current_state >= 0)
            )
            categories = category_arrays(frame, year, income_cutoffs[year])
            stayer, mover, origin_state = migration_masks_and_states(
                frame, year, current_state, universe
            )

            diagnostics["universe_rows"] += int(universe.sum())
            diagnostics["stayer_rows"] += int(stayer.sum())
            diagnostics["mover_rows"] += int(mover.sum())
            diagnostics["unclassified_rows"] += int((universe & ~stayer & ~mover).sum())

            accumulate_contribution(
                group_sizes, distributions, year_index, "stayer", stayer,
                current_state, weight, categories,
            )
            accumulate_contribution(
                group_sizes, distributions, year_index, "immigrant", mover,
                current_state, weight, categories,
            )
            accumulate_contribution(
                group_sizes, distributions, year_index, "emigrant", mover,
                origin_state, weight, categories,
            )
            if batch_number % 10 == 0:
                print(f"  processed {batch_number} batches", flush=True)

    return group_sizes, distributions, income_cutoffs, diagnostics


def write_aggregates(
    output: Path,
    group_sizes: np.ndarray,
    distributions: dict[str, np.ndarray],
    income_cutoffs: dict[int, np.ndarray],
    diagnostics: dict[str, int],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    size_path = output / "group_sizes.csv"
    with size_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("state", "period", "group", "population", "population_thousands", "proxy"))
        for year_index, year in enumerate(YEARS):
            for state_index, state in enumerate(STATES):
                for group_index, group in enumerate(GROUPS):
                    population = group_sizes[year_index, state_index, group_index]
                    writer.writerow((state, year, group, population, population / 1000, PROXY_LABEL[year]))
        pooled = group_sizes.sum(axis=0)
        for state_index, state in enumerate(STATES):
            for group_index, group in enumerate(GROUPS):
                population = pooled[state_index, group_index]
                writer.writerow((state, "pooled", group, population, population / 1000, "Person-weighted census pool"))

    distribution_path = output / "weighted_distributions.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "state", "period", "group", "dimension", "category",
            "weighted_population", "valid_dimension_population", "percent",
        ))
        for dimension in DIMENSIONS:
            values = distributions[dimension.key]
            for year_index, year in enumerate(YEARS):
                write_distribution_rows(writer, year, dimension, values[year_index])
            write_distribution_rows(writer, "pooled", dimension, values.sum(axis=0))

    with (output / "income_quintile_cutoffs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("census_year", "income_measure", "q20", "q40", "q60", "q80"))
        for year in YEARS:
            writer.writerow((year, INCOME_COLUMN[year], *income_cutoffs[year]))

    metadata = {
        "years": list(YEARS),
        "states": list(STATES),
        "universe": "Age 5+, positive person weight, internally classifiable migration status",
        "migration_proxies": PROXY_LABEL,
        "pooled_definition": "Sum of person weights across censuses; not an equal-year average",
        "income_definition": (
            "No income plus national weighted quintiles among positive-income adults. "
            "1980 uses main-job income; other years use total person income."
        ),
        "geography_note": (
            "1970 Guanabara is merged into RJ and Fernando de Noronha into PE. "
            "1970 MT and GO retain their historical boundaries."
        ),
        "diagnostics": diagnostics,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def write_distribution_rows(
    writer: csv.writer,
    period: int | str,
    dimension: Dimension,
    values: np.ndarray,
) -> None:
    for state_index, state in enumerate(STATES):
        for group_index, group in enumerate(GROUPS):
            valid_population = float(values[state_index, group_index].sum())
            for category_index, category in enumerate(dimension.labels):
                population = float(values[state_index, group_index, category_index])
                percent = 100 * population / valid_population if valid_population > 0 else math.nan
                writer.writerow((
                    state, period, group, dimension.key, category.replace("\n", " "),
                    population, valid_population, percent,
                ))


def read_aggregates(
    output: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[int, np.ndarray], dict[str, int]]:
    sizes = pd.read_csv(output / "group_sizes.csv", dtype={"period": str})
    distributions_csv = pd.read_csv(output / "weighted_distributions.csv", dtype={"period": str})
    cutoffs_csv = pd.read_csv(output / "income_quintile_cutoffs.csv")

    group_sizes = np.zeros((len(YEARS), len(STATES), len(GROUPS)), dtype=float)
    distributions = {
        dimension.key: np.zeros(
            (len(YEARS), len(STATES), len(GROUPS), len(dimension.labels)), dtype=float
        )
        for dimension in DIMENSIONS
    }
    for year_index, year in enumerate(YEARS):
        part = sizes[sizes.period == str(year)]
        for row in part.itertuples(index=False):
            group_sizes[year_index, STATE_TO_INDEX[row.state], GROUP_INDEX[row.group]] = row.population
        for dimension in DIMENSIONS:
            part = distributions_csv[
                (distributions_csv.period == str(year))
                & (distributions_csv.dimension == dimension.key)
            ]
            label_to_index = {label.replace("\n", " "): i for i, label in enumerate(dimension.labels)}
            for row in part.itertuples(index=False):
                distributions[dimension.key][
                    year_index,
                    STATE_TO_INDEX[row.state],
                    GROUP_INDEX[row.group],
                    label_to_index[row.category],
                ] = row.weighted_population
    cutoffs = {
        int(row.census_year): np.array((row.q20, row.q40, row.q60, row.q80), dtype=float)
        for row in cutoffs_csv.itertuples(index=False)
    }
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    return group_sizes, distributions, cutoffs, metadata.get("diagnostics", {})


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 14,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#1F2937",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}"{transform}>{html.escape(str(text))}</text>'
    )


def multiline_text(x: float, y: float, text: str, size: int = 12) -> str:
    lines = str(text).split("\n")
    tspans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else size + 2
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="{size}" fill="#374151">'
        + "".join(tspans)
        + "</text>"
    )


def rounded_axis_max(maximum: float) -> int:
    if not np.isfinite(maximum) or maximum <= 0:
        return 20
    return int(min(100, max(20, math.ceil(maximum / 10) * 10)))


def render_panel(
    x: float,
    y: float,
    width: float,
    height: float,
    dimension: Dimension,
    values: np.ndarray,
) -> str:
    parts = [svg_text(x, y + 18, dimension.title, size=17, weight=600)]
    totals = values.sum(axis=1, keepdims=True)
    shares = np.divide(values, totals, out=np.zeros_like(values), where=totals > 0) * 100
    if not np.any(totals > 0):
        parts.append(svg_text(x + width / 2, y + height / 2, "Not available", size=15, anchor="middle", fill="#6B7280"))
        return "\n".join(parts)

    left = x + 54
    right = x + width - 14
    top = y + 38
    bottom = y + height - 64
    chart_width = right - left
    chart_height = bottom - top
    y_max = rounded_axis_max(float(np.nanmax(shares)))
    for tick in np.linspace(0, y_max, 5):
        tick_y = bottom - chart_height * tick / y_max
        parts.append(f'<line x1="{left:.1f}" y1="{tick_y:.1f}" x2="{right:.1f}" y2="{tick_y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        tick_label = f"{tick:.0f}%" if float(tick).is_integer() else f"{tick:.1f}%"
        parts.append(svg_text(left - 8, tick_y + 4, tick_label, size=11, anchor="end", fill="#6B7280"))
    parts.append(f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{bottom:.1f}" stroke="#9CA3AF"/>')
    parts.append(f'<line x1="{left:.1f}" y1="{bottom:.1f}" x2="{right:.1f}" y2="{bottom:.1f}" stroke="#9CA3AF"/>')

    category_width = chart_width / len(dimension.labels)
    gap = max(1.5, category_width * 0.03)
    bar_width = max(3, (category_width * 0.78 - 2 * gap) / len(GROUPS))
    for category_index, label in enumerate(dimension.labels):
        center = left + (category_index + 0.5) * category_width
        group_width = len(GROUPS) * bar_width + (len(GROUPS) - 1) * gap
        start = center - group_width / 2
        for group_index, group in enumerate(GROUPS):
            value = shares[group_index, category_index]
            bar_height = chart_height * value / y_max
            bar_x = start + group_index * (bar_width + gap)
            bar_y = bottom - bar_height
            parts.append(
                f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height:.1f}" fill="{GROUP_COLOR[group]}" opacity="0.92">'
                f'<title>{GROUP_LABEL[group]}: {value:.1f}%</title></rect>'
            )
        parts.append(multiline_text(center, bottom + 18, label, size=11))
    return "\n".join(parts)


def format_thousands(population: float) -> str:
    thousands = population / 1000
    if thousands >= 10_000:
        return f"{thousands:,.0f}k"
    return f"{thousands:,.1f}k"


def render_profile_svg(
    path: Path,
    state: str,
    period: int | str,
    sizes: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    width, height = 1400, 1730
    period_label = str(period) if period != "pooled" else "Person-weighted pool, 1970-2010"
    subtitle = PROXY_LABEL[int(period)] if period != "pooled" else "Migration proxy varies by census; observations are weighted by person, not by year"
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{html.escape(state)} migration-group socioeconomic profile, {html.escape(period_label)}</title>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        svg_text(54, 54, f"{state}: immigrants, emigrants, and stayers", size=29, weight=600),
        svg_text(54, 84, period_label, size=19, fill="#374151"),
        svg_text(54, 111, subtitle, size=14, fill="#6B7280"),
    ]
    legend_y = 150
    for group_index, group in enumerate(GROUPS):
        legend_x = 54 + group_index * 420
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 14}" width="18" height="18" rx="2" fill="{GROUP_COLOR[group]}"/>')
        noun = "Pooled N" if period == "pooled" else "N"
        parts.append(svg_text(
            legend_x + 29,
            legend_y,
            f"{GROUP_LABEL[group]}  |  {noun} = {format_thousands(sizes[group_index])}",
            size=15,
        ))

    panel_width, panel_height = 650, 355
    start_x, start_y = 54, 190
    horizontal_gap, vertical_gap = 42, 20
    for dimension_index, dimension in enumerate(DIMENSIONS):
        row, column = divmod(dimension_index, 2)
        x = start_x + column * (panel_width + horizontal_gap)
        y = start_y + row * (panel_height + vertical_gap)
        parts.append(render_panel(x, y, panel_width, panel_height, dimension, values[dimension.key]))

    foot_y = 1692
    parts.append(svg_text(54, foot_y, "Bars are percentages within migration group and panel; legend sizes use census person weights.", size=13, fill="#4B5563"))
    parts.append(svg_text(54, foot_y + 21, "Income: no income plus national weighted quintiles among positive-income adults (1980 uses main-job income).", size=13, fill="#4B5563"))
    if period in (1970, "pooled"):
        parts.append(svg_text(775, foot_y + 21, "1970: Guanabara -> RJ; Fernando de Noronha -> PE; MT and GO retain historical boundaries.", size=12, fill="#6B7280"))
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def render_all(
    figures: Path,
    group_sizes: np.ndarray,
    distributions: dict[str, np.ndarray],
) -> int:
    count = 0
    state_year_dir = figures / "state_year"
    pooled_dir = figures / "state_pooled"
    for year_index, year in enumerate(YEARS):
        for state_index, state in enumerate(STATES):
            sizes = group_sizes[year_index, state_index]
            if sizes.sum() <= 0:
                continue
            values = {
                dimension.key: distributions[dimension.key][year_index, state_index]
                for dimension in DIMENSIONS
            }
            render_profile_svg(
                state_year_dir / str(year) / f"{state}.svg",
                state,
                year,
                sizes,
                values,
            )
            count += 1

    pooled_sizes = group_sizes.sum(axis=0)
    for state_index, state in enumerate(STATES):
        if pooled_sizes[state_index].sum() <= 0:
            continue
        values = {
            dimension.key: distributions[dimension.key][:, state_index].sum(axis=0)
            for dimension in DIMENSIONS
        }
        render_profile_svg(
            pooled_dir / f"{state}.svg",
            state,
            "pooled",
            pooled_sizes[state_index],
            values,
        )
        count += 1
    return count


def write_figure_readme(figures: Path, plot_count: int) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    text = f"""# Migration-group socioeconomic profiles

This directory contains **{plot_count} survey-weighted SVG profiles** comparing
immigrants, emigrants, and stayers.

- `state_year/<year>/<UF>.svg`: one profile for each observed state and census.
- `state_pooled/<UF>.svg`: person-weighted pool across all available censuses.

Every panel is normalized separately within migration group. The plot legend
reports the full classified group population in thousands using `person_weight`.

## Migration definitions

- **1970:** previous state of residence. A missing previous state is treated as
  no interstate move. Guanabara is merged into RJ and Fernando de Noronha into
  PE. The historical MT and GO boundaries remain.
- **1980:** state of birth.
- **1991, 2000, 2010:** state of residence five years before the census.

The common analysis universe is age five or older. International migrants and
records without an identifiable internal origin are excluded from the
three-group comparison. Each internal mover appears as an immigrant in the
destination profile and as an emigrant in the origin profile.

## Dimensions

Age, national income rank, sex, literacy, years of schooling, household size,
rooms, and refrigerator access are shown. Years of schooling are only available
in harmonized form for 1991 and 2000; household size is unavailable in 1991.

Income is plotted as no income plus national, survey-weighted positive-income
quintiles for adults in each census. This makes pooled plots comparable despite
currency changes. The 1980 source contains main-job rather than total personal
income, which is stated in every plot.

Reusable numeric results are written to
`data/processed/censo_microdados/migration_profiles` by
`code/censo_microdados/migration_profiles.py`.
"""
    (figures / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.plots_only:
        print(f"Reading cached aggregates from {args.aggregates}", flush=True)
        group_sizes, distributions, _, diagnostics = read_aggregates(args.aggregates)
    else:
        dataset = ds.dataset(args.persons, format="parquet", partitioning="hive")
        missing = sorted(set(SCAN_COLUMNS) - set(dataset.schema.names))
        if missing:
            raise RuntimeError(f"Person dataset is missing required columns: {missing}")
        group_sizes, distributions, cutoffs, diagnostics = aggregate(dataset, args.batch_size)
        write_aggregates(
            args.aggregates, group_sizes, distributions, cutoffs, diagnostics
        )
        print(f"Wrote aggregate tables to {args.aggregates}", flush=True)

    plot_count = render_all(args.figures, group_sizes, distributions)
    write_figure_readme(args.figures, plot_count)
    print(f"Wrote {plot_count} SVG plots to {args.figures}", flush=True)
    print(f"Diagnostics: {diagnostics}", flush=True)


if __name__ == "__main__":
    main()
