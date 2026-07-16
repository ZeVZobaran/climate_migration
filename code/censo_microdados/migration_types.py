r"""Classify internal migrants into eight parsimonious movement types.

The classification is mutually exclusive and uses person expansion weights.
It follows the agricultural-frontier state definition already used in
``code/empirical_anl.py``.

Run from the repository root::

    .venv\Scripts\python.exe -u code\censo_microdados\migration_types.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from migration_profiles import (
    STATES,
    YEARS,
    map_current,
    map_origin,
)


NORTHEAST = frozenset(("MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"))
SOUTHEAST = frozenset(("MG", "ES", "RJ", "SP"))
SOUTH = frozenset(("PR", "SC", "RS"))

# Existing project definition in code/empirical_anl.py.
AGRI_FRONTIER = frozenset(("RO", "PA", "MT", "MS", "AP", "RR", "AM", "AC", "TO"))

TYPE_KEYS = (
    "local",
    "neighboring_state",
    "northeast_to_southeast",
    "south_to_agri_frontier",
    "return_agri_frontier_to_south",
    "return_southeast_to_northeast",
    "brasilia_go_inflow",
    "other",
)
TYPE_LABELS = {
    "local": "Local / within-state",
    "neighboring_state": "Neighboring-state circulation",
    "northeast_to_southeast": "Northeast -> Southeast",
    "south_to_agri_frontier": "South -> agricultural frontier",
    "return_agri_frontier_to_south": "Return: agricultural frontier -> South",
    "return_southeast_to_northeast": "Return: Southeast -> Northeast",
    "brasilia_go_inflow": "Long-distance inflow to DF/GO",
    "other": "Other",
}
TYPE_INDEX = {key: i for i, key in enumerate(TYPE_KEYS)}
STATE_INDEX = {state: i for i, state in enumerate(STATES)}

PERIOD_DEFINITION = {
    1970: "Previous-state/municipality lifetime proxy",
    1980: "Birth-state/municipality lifetime proxy",
    1991: "Five-year internal migration",
    2000: "Five-year internal migration",
    2010: "Five-year internal migration",
}

# Undirected land-border pairs. This captures interstate circulation that is
# geographically local even when it crosses an official macro-region boundary
# (for example PR-SP, MA-PA, or DF-GO).
NEIGHBOR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("AC", "AM"), ("AC", "RO"),
        ("AL", "BA"), ("AL", "PE"), ("AL", "SE"),
        ("AM", "MT"), ("AM", "PA"), ("AM", "RO"), ("AM", "RR"),
        ("AP", "PA"),
        ("BA", "ES"), ("BA", "GO"), ("BA", "MG"), ("BA", "PE"),
        ("BA", "PI"), ("BA", "SE"), ("BA", "TO"),
        ("CE", "PB"), ("CE", "PE"), ("CE", "PI"), ("CE", "RN"),
        ("DF", "GO"),
        ("ES", "MG"), ("ES", "RJ"),
        ("GO", "MG"), ("GO", "MS"), ("GO", "MT"), ("GO", "TO"),
        ("MA", "PA"), ("MA", "PI"), ("MA", "TO"),
        ("MG", "MS"), ("MG", "RJ"), ("MG", "SP"),
        ("MS", "MT"), ("MS", "PR"), ("MS", "SP"),
        ("MT", "PA"), ("MT", "RO"), ("MT", "TO"),
        ("PA", "RR"), ("PA", "TO"),
        ("PB", "PE"), ("PB", "RN"),
        ("PE", "PI"),
        ("PI", "TO"),
        ("PR", "SC"), ("PR", "SP"),
        ("RJ", "SP"),
        ("RO", "MT"),
        ("RS", "SC"),
        ("SC", "PR"),
        ("SE", "BA"),
        ("SP", "PR"),
        ("TO", "MT"),
    )
)

SCAN_COLUMNS = (
    "current_uf",
    "person_weight",
    "age_years",
    "born_muni_code",
    "birth_uf_code",
    "last_origin_uf_code",
    "origin_5yr_uf_code",
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
        default=Path("data/processed/censo_microdados/migration_types"),
    )
    parser.add_argument("--batch-size", type=int, default=500_000)
    return parser.parse_args()


def state_membership(indices: np.ndarray, members: frozenset[str]) -> np.ndarray:
    member_indices = np.array([STATE_INDEX[state] for state in members], dtype=np.int16)
    return np.isin(indices, member_indices)


def assign_type(origin: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Return one of eight type indices for every valid origin-destination pair."""
    result = np.full(len(origin), TYPE_INDEX["other"], dtype=np.int8)

    local = origin == destination
    origin_labels = np.array(STATES, dtype=object)[origin]
    destination_labels = np.array(STATES, dtype=object)[destination]
    neighboring = np.fromiter(
        (
            frozenset((str(origin_uf), str(destination_uf))) in NEIGHBOR_PAIRS
            for origin_uf, destination_uf in zip(
                origin_labels, destination_labels, strict=True
            )
        ),
        dtype=bool,
        count=len(origin),
    )
    brasilia_go = np.isin(destination_labels, ("DF", "GO")) & ~neighboring
    ne_to_se = state_membership(origin, NORTHEAST) & state_membership(destination, SOUTHEAST)
    south_to_frontier = state_membership(origin, SOUTH) & state_membership(destination, AGRI_FRONTIER)
    frontier_to_south = state_membership(origin, AGRI_FRONTIER) & state_membership(destination, SOUTH)
    se_to_ne = state_membership(origin, SOUTHEAST) & state_membership(destination, NORTHEAST)

    result[brasilia_go] = TYPE_INDEX["brasilia_go_inflow"]
    result[neighboring] = TYPE_INDEX["neighboring_state"]
    result[se_to_ne] = TYPE_INDEX["return_southeast_to_northeast"]
    result[frontier_to_south] = TYPE_INDEX["return_agri_frontier_to_south"]
    result[south_to_frontier] = TYPE_INDEX["south_to_agri_frontier"]
    result[ne_to_se] = TYPE_INDEX["northeast_to_southeast"]
    result[local] = TYPE_INDEX["local"]
    return result


def migrant_mask_and_origin(
    frame: pd.DataFrame,
    year: int,
    universe: np.ndarray,
    current: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if year == 1970:
        raw = frame["last_origin_uf_code"].astype("string").str.strip()
        origin = map_origin(raw, year)
        observed = raw.notna().to_numpy() & (raw != "").fillna(False).to_numpy()
        migrant = universe & observed & (origin >= 0)
    elif year == 1980:
        origin = map_origin(frame["birth_uf_code"], year)
        # Code 1 means born in the current municipality; code 8 means born
        # elsewhere. Combining the latter with birth UF identifies lifetime
        # intra- and interstate migrants.
        born_elsewhere = (
            frame["born_muni_code"].astype("string").str.strip() == "8"
        ).fillna(False).to_numpy()
        migrant = universe & born_elsewhere & (origin >= 0)
    else:
        origin = map_origin(frame["origin_5yr_uf_code"], year)
        internal = frame["internal_migrant_5yr"].astype("boolean").fillna(False).to_numpy(dtype=bool)
        migrant = universe & internal & (origin >= 0)
    return migrant, origin


def aggregate(dataset: ds.Dataset, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict]:
    counts = np.zeros((len(YEARS), len(TYPE_KEYS)), dtype=np.float64)
    corridors = np.zeros(
        (len(YEARS), len(STATES), len(STATES), len(TYPE_KEYS)), dtype=np.float64
    )
    diagnostics: dict[str, dict[str, float | int]] = {}

    for year_index, year in enumerate(YEARS):
        print(f"Classifying {year} migrants...", flush=True)
        rows_scanned = 0
        migrant_rows = 0
        weighted_migrants = 0.0
        scanner = dataset.scanner(
            columns=list(SCAN_COLUMNS),
            filter=ds.field("census_year") == year,
            batch_size=batch_size,
            use_threads=True,
        )
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            frame = batch.to_pandas()
            rows_scanned += len(frame)
            age = frame["age_years"].to_numpy(dtype=float, na_value=np.nan)
            weight = frame["person_weight"].to_numpy(dtype=float, na_value=np.nan)
            destination = map_current(frame["current_uf"], year)
            universe = (
                np.isfinite(age)
                & (age >= 5)
                & (age <= 120)
                & np.isfinite(weight)
                & (weight > 0)
                & (destination >= 0)
            )
            migrant, origin = migrant_mask_and_origin(
                frame, year, universe, destination
            )
            if not migrant.any():
                continue
            migrant_rows += int(migrant.sum())
            weighted_migrants += float(weight[migrant].sum())
            origin_migrant = origin[migrant]
            destination_migrant = destination[migrant]
            weight_migrant = weight[migrant]
            type_index = assign_type(origin_migrant, destination_migrant)

            counts[year_index] += np.bincount(
                type_index, weights=weight_migrant, minlength=len(TYPE_KEYS)
            )
            flat = (
                ((origin_migrant * len(STATES) + destination_migrant) * len(TYPE_KEYS))
                + type_index
            )
            corridors[year_index] += np.bincount(
                flat,
                weights=weight_migrant,
                minlength=len(STATES) * len(STATES) * len(TYPE_KEYS),
            ).reshape(len(STATES), len(STATES), len(TYPE_KEYS))
            if batch_number % 20 == 0:
                print(f"  processed {batch_number} batches", flush=True)

        diagnostics[str(year)] = {
            "rows_scanned": rows_scanned,
            "migrant_sample_rows": migrant_rows,
            "weighted_migrants": weighted_migrants,
        }
    return counts, corridors, diagnostics


def write_outputs(
    output: Path,
    counts: np.ndarray,
    corridors: np.ndarray,
    diagnostics: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "migration_type_shares.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "census_year", "migration_type", "label", "weighted_migrants",
            "share_percent", "period_definition",
        ))
        for year_index, year in enumerate(YEARS):
            denominator = counts[year_index].sum()
            for type_index, key in enumerate(TYPE_KEYS):
                population = counts[year_index, type_index]
                share = 100 * population / denominator if denominator > 0 else math.nan
                writer.writerow((
                    year, key, TYPE_LABELS[key], population, share,
                    PERIOD_DEFINITION[year],
                ))

    with (output / "migration_type_corridors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "census_year", "origin_uf", "destination_uf", "migration_type",
            "label", "weighted_migrants",
        ))
        for year_index, year in enumerate(YEARS):
            for origin_index, origin in enumerate(STATES):
                for destination_index, destination in enumerate(STATES):
                    for type_index, key in enumerate(TYPE_KEYS):
                        population = corridors[
                            year_index, origin_index, destination_index, type_index
                        ]
                        if population > 0:
                            writer.writerow((
                                year, origin, destination, key, TYPE_LABELS[key],
                                population,
                            ))

    metadata = {
        "analysis_universe": "Internal migrants age 5+ with positive person weight",
        "type_order": list(TYPE_KEYS),
        "type_labels": TYPE_LABELS,
        "regions": {
            "northeast": sorted(NORTHEAST),
            "southeast": sorted(SOUTHEAST),
            "south": sorted(SOUTH),
            "agricultural_frontier": sorted(AGRI_FRONTIER),
        },
        "definitions": {
            "local": "Origin and destination UF are the same",
            "neighboring_state": (
                "Origin and destination UFs share a land border, after assigning "
                "the four named long-distance/reverse directional flows"
            ),
            "northeast_to_southeast": "Northeast origin and Southeast destination",
            "south_to_agri_frontier": "PR/SC/RS origin and agricultural-frontier destination",
            "return_agri_frontier_to_south": "Agricultural-frontier origin and PR/SC/RS destination",
            "return_southeast_to_northeast": "Southeast origin and Northeast destination",
            "brasilia_go_inflow": (
                "Destination is DF or GO and the origin UF is not adjacent to the "
                "destination; neighboring DF/GO flows remain neighboring-state circulation"
            ),
            "other": "Every other identified internal origin-destination combination",
        },
        "period_definitions": PERIOD_DEFINITION,
        "comparability_note": (
            "1970 and 1980 are lifetime/previous-residence proxies. 1991-2010 "
            "are comparable five-year migration flows."
        ),
        "geography_note": (
            "The agricultural frontier follows the existing project definition: "
            "RO, PA, MT, MS, AP, RR, AM, AC, TO. In 1970 Guanabara is merged "
            "into RJ and Fernando de Noronha into PE; MT and GO retain historical boundaries."
        ),
        "diagnostics": diagnostics,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset = ds.dataset(args.persons, format="parquet", partitioning="hive")
    missing = sorted(set(SCAN_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise RuntimeError(f"Person dataset is missing required columns: {missing}")
    counts, corridors, diagnostics = aggregate(dataset, args.batch_size)
    write_outputs(args.output, counts, corridors, diagnostics)
    print(f"Wrote migration-type results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
