r"""Build weighted 5x5 Brazilian macro-region migration matrices.

Rows are origin regions and columns are current-residence regions. For each
census and for a person-weighted pool of all censuses, the script writes:

* weighted population counts;
* emigrant shares (each cell divided by its origin-row total); and
* immigrant shares (each cell divided by its destination-column total).

The main diagonal includes people who did not move between regions, including
within-state and within-region movers. Historical origin measures follow the
same proxies used by ``migration_profiles.py`` and ``migration_types.py``.

Run from the repository root::

    .venv\Scripts\python.exe -u code\censo_microdados\regional_migration_matrices.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from migration_profiles import STATES, YEARS, map_current, map_origin


REGION_KEYS = ("N", "NE", "CO", "SE", "S")
REGION_LABELS = {
    "N": "North",
    "NE": "Northeast",
    "CO": "Center-West",
    "SE": "Southeast",
    "S": "South",
}
REGION_STATES = {
    "N": frozenset(("AC", "AP", "AM", "PA", "RO", "RR", "TO")),
    "NE": frozenset(("AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE")),
    "CO": frozenset(("DF", "GO", "MT", "MS")),
    "SE": frozenset(("ES", "MG", "RJ", "SP")),
    "S": frozenset(("PR", "SC", "RS")),
}
REGION_INDEX = {region: index for index, region in enumerate(REGION_KEYS)}
STATE_TO_REGION_INDEX = np.array(
    [
        next(
            REGION_INDEX[region]
            for region, members in REGION_STATES.items()
            if state in members
        )
        for state in STATES
    ],
    dtype=np.int8,
)

PERIOD_DEFINITION = {
    1970: "Previous-state/municipality lifetime proxy",
    1980: "Birth-state/municipality lifetime proxy",
    1991: "Five-year internal migration",
    2000: "Five-year internal migration",
    2010: "Five-year internal migration",
}

SCAN_COLUMNS = (
    "current_uf",
    "person_weight",
    "age_years",
    "born_muni_code",
    "birth_uf_code",
    "last_origin_uf_code",
    "origin_5yr_uf_code",
    "migrant_5yr",
    "internal_migrant_5yr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persons",
        type=Path,
        default=Path("data/processed/censo_microdados/persons"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/censo_microdados/regional_migration_matrices"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/regional_migration_matrices.md"),
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    return parser.parse_args()


def origin_states(
    frame: pd.DataFrame,
    year: int,
    universe: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return classified mask and origin-state indices for the analysis universe.

    For 1970, a missing previous-residence state denotes a nonmover and is
    assigned the current state. For 1980, only people with an identified
    Brazilian birth state are classified. For 1991--2010, definite nonmovers
    are assigned the current state, internal migrants use their five-year
    origin, and international or otherwise unidentified movers are excluded.
    """
    origin = np.full(len(frame), -1, dtype=np.int16)

    if year == 1970:
        raw = frame["last_origin_uf_code"].astype("string").str.strip()
        mapped = map_origin(raw, year)
        missing = raw.isna().to_numpy() | (raw == "").fillna(False).to_numpy()
        origin[missing] = current[missing]
        observed_domestic = mapped >= 0
        origin[observed_domestic] = mapped[observed_domestic]
        classified = universe & (origin >= 0)
        excluded_international_or_unknown = universe & ~classified
    elif year == 1980:
        mapped = map_origin(frame["birth_uf_code"], year)
        origin[:] = mapped
        classified = universe & (origin >= 0)
        excluded_international_or_unknown = universe & ~classified
    else:
        mapped = map_origin(frame["origin_5yr_uf_code"], year)
        migrant = frame["migrant_5yr"].astype("boolean")
        internal = frame["internal_migrant_5yr"].astype("boolean")
        internal_true = internal.fillna(False).to_numpy(dtype=bool)
        nonmover = (~migrant).fillna(False).to_numpy(dtype=bool)

        # In 1991, people born in their current municipality can have
        # structurally blank five-year-origin fields. They are definite
        # nonmovers for this state/region analysis.
        if year == 1991:
            born_current_muni = (
                frame["born_muni_code"].astype("string").str.strip() == "1"
            ).fillna(False).to_numpy(dtype=bool)
            nonmover |= migrant.isna().to_numpy() & born_current_muni

        origin[nonmover] = current[nonmover]
        valid_internal = internal_true & (mapped >= 0)
        origin[valid_internal] = mapped[valid_internal]
        classified = universe & (nonmover | valid_internal) & (origin >= 0)
        excluded_international_or_unknown = universe & ~classified

    diagnostics = {
        "classified_sample_rows": int(classified.sum()),
        "excluded_international_or_unknown_sample_rows": int(
            excluded_international_or_unknown.sum()
        ),
    }
    return classified, origin, diagnostics


def aggregate(
    dataset: ds.Dataset, batch_size: int
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    counts = np.zeros(
        (len(YEARS), len(REGION_KEYS), len(REGION_KEYS)), dtype=np.float64
    )
    diagnostics: dict[str, dict[str, float | int]] = {}

    for year_index, year in enumerate(YEARS):
        print(f"Building {year} regional matrix...", flush=True)
        year_diagnostics: dict[str, float | int] = {
            "rows_scanned": 0,
            "analysis_universe_sample_rows": 0,
            "classified_sample_rows": 0,
            "excluded_international_or_unknown_sample_rows": 0,
            "classified_weighted_population": 0.0,
        }
        scanner = dataset.scanner(
            columns=list(SCAN_COLUMNS),
            filter=ds.field("census_year") == year,
            batch_size=batch_size,
            use_threads=True,
        )
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            frame = batch.to_pandas()
            year_diagnostics["rows_scanned"] += len(frame)
            age = frame["age_years"].to_numpy(dtype=float, na_value=np.nan)
            weight = frame["person_weight"].to_numpy(dtype=float, na_value=np.nan)
            current = map_current(frame["current_uf"], year)
            universe = (
                np.isfinite(age)
                & (age >= 5)
                & (age <= 120)
                & np.isfinite(weight)
                & (weight > 0)
                & (current >= 0)
            )
            year_diagnostics["analysis_universe_sample_rows"] += int(universe.sum())
            classified, origin, batch_diagnostics = origin_states(
                frame, year, universe, current
            )
            for key, value in batch_diagnostics.items():
                year_diagnostics[key] += value
            if not classified.any():
                continue

            classified_weight = weight[classified]
            origin_region = STATE_TO_REGION_INDEX[origin[classified]]
            destination_region = STATE_TO_REGION_INDEX[current[classified]]
            flat = origin_region * len(REGION_KEYS) + destination_region
            counts[year_index] += np.bincount(
                flat,
                weights=classified_weight,
                minlength=len(REGION_KEYS) ** 2,
            ).reshape(len(REGION_KEYS), len(REGION_KEYS))
            year_diagnostics["classified_weighted_population"] += float(
                classified_weight.sum()
            )
            if batch_number % 20 == 0:
                print(f"  processed {batch_number} batches", flush=True)

        diagnostics[str(year)] = year_diagnostics
    return counts, diagnostics


def share_matrices(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_totals = counts.sum(axis=1, keepdims=True)
    column_totals = counts.sum(axis=0, keepdims=True)
    emigrant = np.divide(
        counts,
        row_totals,
        out=np.full_like(counts, np.nan),
        where=row_totals > 0,
    ) * 100
    immigrant = np.divide(
        counts,
        column_totals,
        out=np.full_like(counts, np.nan),
        where=column_totals > 0,
    ) * 100
    return emigrant, immigrant


def matrix_frame(matrix: np.ndarray, row_name: str = "origin_region") -> pd.DataFrame:
    frame = pd.DataFrame(matrix, index=REGION_KEYS, columns=REGION_KEYS)
    frame.index.name = row_name
    return frame


def markdown_matrix(matrix: np.ndarray, metric: str) -> str:
    if metric == "weighted_population":
        values = [[f"{value:,.0f}" for value in row] for row in matrix]
    else:
        values = [[f"{value:.3f}%" for value in row] for row in matrix]
    header = "| Origin \\ Destination | " + " | ".join(REGION_KEYS) + " |"
    separator = "|---|" + "|".join("---:" for _ in REGION_KEYS) + "|"
    rows = [
        f"| {origin} | " + " | ".join(values[index]) + " |"
        for index, origin in enumerate(REGION_KEYS)
    ]
    return "\n".join((header, separator, *rows))


def validation_summary(matrix: np.ndarray) -> dict[str, object]:
    emigrant, immigrant = share_matrices(matrix)
    diagonal = np.diag(matrix)
    row_max = matrix.max(axis=1)
    column_max = matrix.max(axis=0)
    return {
        "emigrant_row_sums_percent": emigrant.sum(axis=1).tolist(),
        "immigrant_column_sums_percent": immigrant.sum(axis=0).tolist(),
        "diagonal_is_largest_in_each_origin_row": bool(np.all(diagonal >= row_max)),
        "diagonal_is_largest_in_each_destination_column": bool(
            np.all(diagonal >= column_max)
        ),
    }


def write_outputs(
    output: Path,
    report: Path,
    counts: np.ndarray,
    diagnostics: dict[str, dict[str, float | int]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    matrices_dir = output / "matrices"
    matrices_dir.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    periods: list[tuple[str, np.ndarray]] = [
        (str(year), counts[index]) for index, year in enumerate(YEARS)
    ]
    periods.append(("pooled", counts.sum(axis=0)))

    long_rows: list[dict[str, object]] = []
    validations: dict[str, dict[str, object]] = {}
    report_parts = [
        "# Brazilian macro-region migration matrices",
        "",
        "Rows are origin regions and columns are census-residence destinations. "
        "The universe is people age 5--120 with positive person weight and a "
        "classifiable Brazilian origin. Counts are survey-weighted persons.",
        "",
        "Emigrant shares divide each cell by its origin-row total; immigrant "
        "shares divide each cell by its destination-column total. Therefore, "
        "emigrant-share rows and immigrant-share columns each sum to 100%.",
        "",
        "Region order: N (North), NE (Northeast), CO (Center-West), SE "
        "(Southeast), S (South). The pooled period sums person weights across "
        "censuses rather than giving each census equal weight.",
        "",
        "> Comparability: 1970 uses previous state of residence and 1980 uses "
        "state of birth. The 1991, 2000, and 2010 matrices use residence five "
        "years before the census.",
    ]

    for period, matrix in periods:
        emigrant, immigrant = share_matrices(matrix)
        validations[period] = validation_summary(matrix)
        matrix_frame(matrix).to_csv(
            matrices_dir / f"{period}_weighted_population.csv", float_format="%.6f"
        )
        matrix_frame(emigrant).to_csv(
            matrices_dir / f"{period}_emigrant_share_percent.csv", float_format="%.9f"
        )
        matrix_frame(immigrant).to_csv(
            matrices_dir / f"{period}_immigrant_share_percent.csv", float_format="%.9f"
        )

        for origin_index, origin in enumerate(REGION_KEYS):
            for destination_index, destination in enumerate(REGION_KEYS):
                long_rows.append(
                    {
                        "period": period,
                        "origin_region": origin,
                        "destination_region": destination,
                        "weighted_population": matrix[origin_index, destination_index],
                        "emigrant_share_percent": emigrant[
                            origin_index, destination_index
                        ],
                        "immigrant_share_percent": immigrant[
                            origin_index, destination_index
                        ],
                    }
                )

        title = "All censuses pooled" if period == "pooled" else f"Census {period}"
        report_parts.extend(
            (
                "",
                f"## {title}",
                "",
                "### Weighted population",
                "",
                markdown_matrix(matrix, "weighted_population"),
                "",
                "### Emigrant shares (origin-row percentages)",
                "",
                markdown_matrix(emigrant, "emigrant_share_percent"),
                "",
                "### Immigrant shares (destination-column percentages)",
                "",
                markdown_matrix(immigrant, "immigrant_share_percent"),
            )
        )

    with (output / "regional_migration_flows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)

    metadata = {
        "analysis_universe": (
            "People age 5-120 with positive person weight, valid current Brazilian "
            "region, and classifiable Brazilian origin"
        ),
        "orientation": "Rows are origin regions; columns are destination regions",
        "region_order": list(REGION_KEYS),
        "region_labels": REGION_LABELS,
        "region_states": {
            key: sorted(states) for key, states in REGION_STATES.items()
        },
        "period_definitions": PERIOD_DEFINITION,
        "pooled_definition": "Sum of person weights across all five censuses",
        "emigrant_share_denominator": (
            "Total classified weighted population in the origin-region row"
        ),
        "immigrant_share_denominator": (
            "Total classified weighted population in the destination-region column"
        ),
        "diagonal_definition": (
            "People whose origin and destination macro-regions coincide, including "
            "nonmovers and within-region movers"
        ),
        "comparability_note": (
            "1970 and 1980 are previous-residence/birthplace lifetime proxies; "
            "1991-2010 are comparable five-year migration measures"
        ),
        "diagnostics": diagnostics,
        "validation": validations,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report.write_text("\n".join(report_parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = ds.dataset(args.persons, format="parquet", partitioning="hive")
    missing = sorted(set(SCAN_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise RuntimeError(f"Person dataset is missing required columns: {missing}")
    counts, diagnostics = aggregate(dataset, args.batch_size)
    write_outputs(args.output, args.report, counts, diagnostics)
    print(f"Wrote regional matrices to {args.output}", flush=True)
    print(f"Wrote readable report to {args.report}", flush=True)


if __name__ == "__main__":
    main()
