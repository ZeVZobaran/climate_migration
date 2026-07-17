"""Build the 1991, 2000, and 2010 person-level characteristics datasets.

These are repeated cross-sections, not longitudinal panels.  Each output row
is one sampled census person.  The script retains only predetermined/basic
characteristics and five-year origin/destination geography, then adds lagged
municipal population, migration-system definitions, and Morten--Oliveira
mesoregion travel measures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_census_parquet import F, P00, dbf_chunks, fixed_chunks


YEARS = (1991, 2000, 2010)
MO_YEAR = {1991: 1990, 2000: 2000, 2010: 2010}
PRIOR_FLOW_YEAR = {1991: 1980, 2000: 1991, 2010: 2000}
CHOICE_ALTERNATIVES = ("stayer", "city", "agri_frontier", "other")
NORTHEAST = frozenset(("21", "22", "23", "24", "25", "26", "27", "28", "29"))
SOUTH = frozenset(("41", "42", "43"))
NORTH = frozenset(("11", "12", "13", "14", "15", "16", "17"))
FRONTIER_DEFINITIONS = {
    "north_only": NORTH,
    "main_north_mt": NORTH | {"51"},
    "broad_north_mt_ms": NORTH | {"50", "51"},
}

SOURCE_COLUMNS = (
    "current_uf_code",
    "current_municipality_code",
    "current_meso_code",
    "urban_code",
    "origin_5yr_uf_code",
    "origin_5yr_muni_code",
    "migrant_5yr",
    "internal_migrant_5yr",
    "person_weight",
    "sex_code",
    "sex",
    "age_years",
    "race_code",
    "education_years",
    "education_level_code",
    "source_file",
    "source_row_in_file",
)

STRING_COLUMNS = (
    "person_record_id",
    "current_uf",
    "source_file",
    "current_municipality_code",
    "current_municipality_name",
    "origin_5yr_uf",
    "origin_5yr_municipality_code",
    "origin_5yr_municipality_name",
    "current_urban_code",
    "current_urban_rural",
    "origin_5yr_urban_code",
    "origin_5yr_urban_rural",
    "current_mesoregion_code",
    "origin_5yr_mesoregion_code",
    "origin_5yr_type",
    "origin_region_group",
    "previous_population_municipality_code",
    "previous_population_municipality_name",
    "previous_population_assignment_method",
    "migration_system_500k_north_only",
    "migration_system_500k_main_north_mt",
    "migration_system_500k_broad_north_mt_ms",
    "migration_system_1m_north_only",
    "migration_system_1m_main_north_mt",
    "migration_system_1m_broad_north_mt_ms",
    "sex_code",
    "sex",
    "race_code",
    "race",
    "education_level_code",
    "education_attainment",
)
BOOLEAN_COLUMNS = (
    "source_migrant_5yr",
    "source_internal_migrant_5yr",
    "five_year_internal_od_observed",
    "stayer_5yr",
    "internal_migrant_5yr",
    "origin_northeast",
    "origin_south",
    "destination_frontier_north_only",
    "destination_frontier_main_north_mt",
    "destination_frontier_broad_north_mt_ms",
    "destination_large_city_500k",
    "destination_large_city_1m",
    "previous_population_imputed",
    "age_5plus",
    "age_25plus",
)
FLOAT_COLUMNS = (
    "person_weight",
    "previous_census_population",
    "previous_population_predecessor_overlap_share",
    "mo_fm_road",
    "mo_fm_road_rad",
    "mo_dist_km",
    "age_years",
    "education_years",
    "prev_corridor_stayer",
    "prev_corridor_city",
    "prev_corridor_agri_frontier",
    "prev_corridor_other",
    "min_tt_to_stayer",
    "min_tt_to_city",
    "min_tt_to_agri_frontier",
    "min_tt_to_other",
    "med_tt_to_stayer",
    "med_tt_to_city",
    "med_tt_to_agri_frontier",
    "med_tt_to_other",
)


def schema() -> pa.Schema:
    fields = [pa.field("census_year", pa.int32())]
    fields.extend(pa.field(column, pa.string()) for column in STRING_COLUMNS)
    fields.append(pa.field("source_row_in_file", pa.int64()))
    fields.extend(pa.field(column, pa.bool_()) for column in BOOLEAN_COLUMNS)
    fields.append(pa.field("previous_census_year", pa.int16()))
    fields.append(pa.field("mo_travel_time_year", pa.int16()))
    fields.extend(pa.field(column, pa.float64()) for column in FLOAT_COLUMNS)
    return pa.schema(fields)


TARGET_SCHEMA = schema()
OUTPUT_COLUMNS = TARGET_SCHEMA.names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persons",
        type=Path,
        default=Path("data/processed/censo_microdados/persons"),
    )
    parser.add_argument(
        "--population-crosswalk",
        type=Path,
        default=Path(
            "data/processed/censo_microdados/characteristics_panels/"
            "municipality_population_crosswalk.parquet"
        ),
    )
    parser.add_argument(
        "--travel-times",
        type=Path,
        default=Path("data/morten_oliveira_final_tables/tt_mesospeed_10.dta"),
    )
    parser.add_argument(
        "--corridor-flows",
        type=Path,
        default=Path("data/morten_oliveira_final_tables/N_od_meso.dta"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/censo_microdados/characteristics_panels"),
    )
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument(
        "--sample-files",
        type=int,
        help="Process only this many state files per year for validation.",
    )
    return parser.parse_args()


def geography_lookup_1991() -> pd.DataFrame:
    path = next(
        Path("data/censo_microdados/Microdados_Censo_Demografico_1991_Amostra").glob(
            "**/DTB Municipios 1991.xls"
        )
    )
    frame = pd.read_excel(path, dtype="string")
    code = frame["COD MUN"].str.zfill(7)
    result = pd.DataFrame(
        {
            "municipality_code": code,
            "source_key": code.str[:6],
            "mesoregion_code": frame["UF"].str.zfill(2)
            + frame["MESORREG"].str.zfill(2),
        }
    )
    return result.drop_duplicates("municipality_code")


def geography_lookup_2000() -> pd.DataFrame:
    path = next(
        Path("data/censo_microdados/Censo_Microdados_2000").glob(
            "**/Divisao Territorial Brasileira.xls"
        )
    )
    frame = pd.read_excel(path, sheet_name=0, dtype="string")
    result = pd.DataFrame(
        {
            "municipality_code": frame.iloc[:, 6].str.zfill(7),
            "mesoregion_code": frame.iloc[:, 2].str.zfill(4),
        }
    ).dropna()
    result["source_key"] = result["municipality_code"]
    return result.drop_duplicates("municipality_code")


def geography_lookup_2010() -> pd.DataFrame:
    path = next(
        Path("data/censo_microdados/Censo_Microdados_2010").glob(
            "**/Unidades da Federação, Mesorregiões, microrregiões e municípios 2010.xls"
        )
    )
    frame = pd.read_excel(path, sheet_name=0, header=2, dtype="string")
    result = pd.DataFrame(
        {
            "municipality_code": frame.iloc[:, 6].str.zfill(7),
            "mesoregion_code": frame.iloc[:, 2].str.zfill(4),
        }
    ).dropna()
    result["source_key"] = result["municipality_code"]
    return result.drop_duplicates("municipality_code")


def load_geographies() -> dict[int, pd.DataFrame]:
    result = {
        1991: geography_lookup_1991(),
        2000: geography_lookup_2000(),
        2010: geography_lookup_2010(),
    }
    expected = {1991: 4491, 2000: 5507, 2010: 5565}
    for year, frame in result.items():
        if len(frame) != expected[year]:
            raise RuntimeError(
                f"Unexpected {year} municipality lookup size: {len(frame)} != {expected[year]}"
            )
    return result


def standardized_uf(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip()
    return value.where(value.str.len() <= 2, value.str[:2]).str.zfill(2)


def canonical_municipality(
    year: int,
    uf: pd.Series,
    source_code: pd.Series,
    source_to_canonical: pd.Series,
) -> pd.Series:
    code = source_code.astype("string").str.strip()
    if year == 1991:
        key = standardized_uf(uf) + code.str.zfill(4)
    else:
        key = code.str.zfill(7)
    return key.map(source_to_canonical).astype("string")


def supplemental_origin_zone(
    year: int, source: Path, expected_rows: int, batch_size: int
) -> pd.Series:
    if year == 1991:
        parts = [
            frame["MIMO86ZN"].astype("string").str.strip()
            for frame in dbf_chunks(source, ["MIMO86ZN"], batch_size, None)
        ]
        result = pd.concat(parts, ignore_index=True)
    elif year == 2000:
        field = P00["status_5yr"]
        parts = [
            frame["status_5yr"].astype("string").str.strip().map(
                {"1": "1", "2": "2", "3": "1", "4": "2"}
            )
            for frame in fixed_chunks(
                source,
                {
                    # Include a nonblank field so pandas does not discard a
                    # record whose selected migration field is blank.
                    "uf": P00["uf"],
                    "status_5yr": F(field.start, field.width),
                },
                batch_size,
                None,
            )
        ]
        result = pd.concat(parts, ignore_index=True).astype("string")
    else:
        result = pd.Series(pd.NA, index=range(expected_rows), dtype="string")
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Supplement row mismatch for {source}: {len(result):,} != {expected_rows:,}"
        )
    result.index = np.arange(1, expected_rows + 1)
    return result


def race_label(code: pd.Series) -> pd.Series:
    return code.astype("string").str.strip().map(
        {
            "1": "white",
            "2": "black",
            "3": "yellow_asian",
            "4": "brown_pardo",
            "5": "indigenous",
        }
    ).astype("string")


def current_zone(year: int, code: pd.Series) -> pd.Series:
    value = code.astype("string").str.strip()
    if year == 1991:
        return pd.Series(
            np.select(
                [value.isin(("1", "2", "3")), value.isin(("4", "5", "6", "7", "8"))],
                ["urban", "rural"],
                default=None,
            ),
            index=value.index,
            dtype="string",
        )
    return value.map({"1": "urban", "2": "rural"}).astype("string")


def education_attainment(
    year: int, years: pd.Series, level: pd.Series
) -> pd.Series:
    schooling = pd.to_numeric(years, errors="coerce")
    if year in (1991, 2000):
        valid = schooling.between(0, 19)
        return pd.Series(
            np.select(
                [
                    valid & schooling.lt(8),
                    valid & schooling.between(8, 10),
                    valid & schooling.between(11, 14),
                    valid & schooling.ge(15),
                ],
                [
                    "less_than_fundamental_complete",
                    "fundamental_complete_secondary_incomplete",
                    "secondary_complete_tertiary_incomplete",
                    "tertiary_complete",
                ],
                default=None,
            ),
            index=years.index,
            dtype="string",
        )
    return level.astype("string").str.strip().map(
        {
            "1": "less_than_fundamental_complete",
            "2": "fundamental_complete_secondary_incomplete",
            "3": "secondary_complete_tertiary_incomplete",
            "4": "tertiary_complete",
        }
    ).astype("string")


def system_category(
    observed: pd.Series,
    stayer: pd.Series,
    large: pd.Series,
    frontier: pd.Series,
) -> pd.Series:
    return pd.Series(
        np.select(
            [
                observed & stayer.fillna(False),
                observed & ~stayer.fillna(False) & large,
                observed & ~stayer.fillna(False) & ~large & frontier,
                observed & ~stayer.fillna(False) & ~large & ~frontier,
            ],
            ["stayer", "large_city", "agricultural_frontier", "other"],
            default=None,
        ),
        index=observed.index,
        dtype="string",
    )


def build_choice_set_accessibility(
    year: int,
    geography: pd.DataFrame,
    population: pd.DataFrame,
    travel_year: pd.DataFrame,
    corridor_all: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the full mesoregion choice set into four alternative measures.

    The main migration-system definition is used: a city mesoregion contains
    at least one municipality above the 500,000 lagged-population threshold;
    the agricultural frontier is North plus Mato Grosso.  Stayer has first
    precedence and is the origin mesoregion itself.  City then precedes the
    frontier, matching the person-level system definition.
    """
    muni_to_meso = geography.set_index("municipality_code")["mesoregion_code"]
    destination = population[["municipality_code", "large_city_500k"]].copy()
    destination["mesoregion_code"] = destination["municipality_code"].map(
        muni_to_meso
    )
    if destination["mesoregion_code"].isna().any():
        raise RuntimeError(f"Missing municipality-to-mesoregion mapping in {year}")
    meso_city = destination.groupby("mesoregion_code")["large_city_500k"].any()
    mesoregions = pd.Index(sorted(meso_city.index.astype("string")))

    prior = corridor_all[corridor_all["year"] == PRIOR_FLOW_YEAR[year]].copy()
    prior["origin"] = prior["orig_id_meso"].astype("int64").astype("string")
    prior["destination"] = prior["dest_id_meso"].astype("int64").astype("string")
    if prior.duplicated(["origin", "destination"]).any():
        raise RuntimeError(f"Duplicate prior corridor cells for {year}")
    if len(prior) != len(mesoregions) ** 2:
        raise RuntimeError(f"Incomplete prior corridor matrix for {year}")

    travel = travel_year.copy()
    travel["origin"] = travel["orig_id_meso"].astype("int64").astype("string")
    travel["destination"] = travel["dest_id_meso"].astype("int64").astype("string")
    if travel.duplicated(["origin", "destination"]).any():
        raise RuntimeError(f"Duplicate travel-time cells for {year}")
    cells = prior[["origin", "destination", "N_od_flow_all"]].merge(
        travel[["origin", "destination", "fm_road"]],
        on=["origin", "destination"],
        how="left",
        validate="one_to_one",
    )
    if cells[["N_od_flow_all", "fm_road"]].isna().any().any():
        raise RuntimeError(f"Missing corridor or travel cell for {year}")

    destination_city = cells["destination"].map(meso_city).fillna(False)
    destination_frontier = cells["destination"].str[:2].isin(
        FRONTIER_DEFINITIONS["main_north_mt"]
    )
    cells["alternative"] = np.select(
        [
            cells["destination"].eq(cells["origin"]),
            destination_city,
            destination_frontier,
        ],
        ["stayer", "city", "agri_frontier"],
        default="other",
    )
    counts = cells.groupby(["origin", "alternative"]).size().unstack(fill_value=0)
    if not set(CHOICE_ALTERNATIVES).issubset(counts.columns):
        raise RuntimeError(f"A choice alternative is absent in {year}")
    if (counts[list(CHOICE_ALTERNATIVES)] == 0).any().any():
        raise RuntimeError(f"An origin has an empty choice alternative in {year}")

    aggregate = cells.groupby(["origin", "alternative"], observed=True).agg(
        prev_corridor=("N_od_flow_all", "sum"),
        min_tt=("fm_road", "min"),
        med_tt=("fm_road", "median"),
    )
    wide = aggregate.unstack("alternative")
    result = pd.DataFrame(index=mesoregions)
    result.index.name = "origin_mesoregion_code"
    for alternative in CHOICE_ALTERNATIVES:
        for measure in ("prev_corridor", "min_tt", "med_tt"):
            result[f"{measure}_{'to_' if measure != 'prev_corridor' else ''}{alternative}"] = (
                wide[(measure, alternative)].reindex(mesoregions).astype("float64")
            )
    expected_columns = [
        *(f"prev_corridor_{value}" for value in CHOICE_ALTERNATIVES),
        *(f"min_tt_to_{value}" for value in CHOICE_ALTERNATIVES),
        *(f"med_tt_to_{value}" for value in CHOICE_ALTERNATIVES),
    ]
    result = result[expected_columns]
    if result.isna().any().any():
        raise RuntimeError(f"Missing choice-set accessibility aggregate in {year}")
    if not (result[["min_tt_to_stayer", "med_tt_to_stayer"]] == 0).all().all():
        raise RuntimeError(f"Nonzero mesoregion stayer travel time in {year}")
    return result.reset_index()


def to_arrow(frame: pd.DataFrame) -> pa.Table:
    arrays = [pa.array(frame[field.name], type=field.type) for field in TARGET_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=TARGET_SCHEMA)


def enrich_batch(
    frame: pd.DataFrame,
    year: int,
    origin_zone: pd.Series,
    geography: pd.DataFrame,
    population: pd.DataFrame,
    travel: pd.DataFrame,
    accessibility: pd.DataFrame,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["census_year"] = year
    result["source_file"] = frame["source_file"].astype("string")
    result["source_row_in_file"] = pd.to_numeric(
        frame["source_row_in_file"], errors="raise"
    ).astype("int64")
    result["current_uf"] = standardized_uf(frame["current_uf_code"])
    source_stem = (
        result["source_file"]
        .str.replace(r"^.*[\\/]", "", regex=True)
        .str.replace(r"\.[^.]+$", "", regex=True)
        .str.lower()
    )
    result["person_record_id"] = (
        str(year)
        + "-"
        + source_stem
        + "-"
        + result["source_row_in_file"].astype("string")
    )
    result["person_weight"] = pd.to_numeric(frame["person_weight"], errors="coerce")

    source_to_canonical = geography.set_index("source_key")["municipality_code"]
    municipality_to_meso = geography.set_index("municipality_code")["mesoregion_code"]
    result["current_municipality_code"] = canonical_municipality(
        year,
        result["current_uf"],
        frame["current_municipality_code"],
        source_to_canonical,
    )
    raw_origin_uf = standardized_uf(frame["origin_5yr_uf_code"])
    origin_candidate = canonical_municipality(
        year,
        raw_origin_uf,
        frame["origin_5yr_muni_code"],
        source_to_canonical,
    )

    source_migrant = frame["migrant_5yr"].astype("boolean")
    source_internal = frame["internal_migrant_5yr"].astype("boolean")
    age = pd.to_numeric(frame["age_years"], errors="coerce")
    age_5plus = age.ge(5) & age.le(120)
    same_source = source_migrant.eq(False).fillna(False) & age_5plus
    internal_source = source_internal.eq(True).fillna(False) & age_5plus
    result["origin_5yr_municipality_code"] = pd.Series(
        pd.NA, index=frame.index, dtype="string"
    )
    result.loc[same_source, "origin_5yr_municipality_code"] = result.loc[
        same_source, "current_municipality_code"
    ]
    result.loc[internal_source, "origin_5yr_municipality_code"] = origin_candidate.loc[
        internal_source
    ]
    result["origin_5yr_uf"] = result["origin_5yr_municipality_code"].str[:2]

    population_index = population.set_index("municipality_code")
    result["current_municipality_name"] = result[
        "current_municipality_code"
    ].map(population_index["municipality_name"])
    result["origin_5yr_municipality_name"] = result[
        "origin_5yr_municipality_code"
    ].map(population_index["municipality_name"])
    result["current_mesoregion_code"] = result["current_municipality_code"].map(
        municipality_to_meso
    )
    result["origin_5yr_mesoregion_code"] = result[
        "origin_5yr_municipality_code"
    ].map(municipality_to_meso)

    result["current_urban_code"] = frame["urban_code"].astype("string").str.strip()
    result["current_urban_rural"] = current_zone(year, result["current_urban_code"])
    source_rows = result["source_row_in_file"].to_numpy(dtype=np.int64)
    result["origin_5yr_urban_code"] = origin_zone.reindex(source_rows).reset_index(
        drop=True
    )
    result["origin_5yr_urban_rural"] = result["origin_5yr_urban_code"].map(
        {"1": "urban", "2": "rural"}
    ).astype("string")

    result["source_migrant_5yr"] = source_migrant
    result["source_internal_migrant_5yr"] = source_internal
    result["age_years"] = age
    result["age_5plus"] = age_5plus
    result["age_25plus"] = age.ge(25) & age.le(120)
    observed = (
        result["age_5plus"]
        & result["current_municipality_code"].notna()
        & result["origin_5yr_municipality_code"].notna()
    )
    result["five_year_internal_od_observed"] = observed
    same = result["origin_5yr_municipality_code"].eq(
        result["current_municipality_code"]
    )
    result["stayer_5yr"] = same.where(observed).astype("boolean")
    result["internal_migrant_5yr"] = (~same).where(observed).astype("boolean")
    foreign = source_migrant.eq(True).fillna(False) & source_internal.eq(False).fillna(False)
    result["origin_5yr_type"] = pd.Series(
        np.select(
            [
                observed & same,
                observed & ~same,
                foreign,
            ],
            ["same_municipality", "internal_migrant", "foreign_country"],
            default="unobserved",
        ),
        index=frame.index,
        dtype="string",
    )

    valid_origin = result["origin_5yr_uf"].notna()
    result["origin_northeast"] = result["origin_5yr_uf"].isin(NORTHEAST).where(
        valid_origin
    ).astype("boolean")
    result["origin_south"] = result["origin_5yr_uf"].isin(SOUTH).where(
        valid_origin
    ).astype("boolean")
    result["origin_region_group"] = pd.Series(
        np.select(
            [result["origin_northeast"].fillna(False), result["origin_south"].fillna(False)],
            ["northeast", "south"],
            default=None,
        ),
        index=frame.index,
        dtype="string",
    )
    result.loc[valid_origin & result["origin_region_group"].isna(), "origin_region_group"] = (
        "other_brazil"
    )

    for key, states in FRONTIER_DEFINITIONS.items():
        result[f"destination_frontier_{key}"] = result["current_uf"].isin(states)

    for column in (
        "previous_census_year",
        "previous_population",
        "previous_municipality_code",
        "previous_municipality_name",
        "population_assignment_method",
        "predecessor_overlap_share",
        "large_city_500k",
        "large_city_1m",
    ):
        result[column] = result["current_municipality_code"].map(population_index[column])
    result = result.rename(
        columns={
            "previous_population": "previous_census_population",
            "previous_municipality_code": "previous_population_municipality_code",
            "previous_municipality_name": "previous_population_municipality_name",
            "population_assignment_method": "previous_population_assignment_method",
            "predecessor_overlap_share": "previous_population_predecessor_overlap_share",
            "large_city_500k": "destination_large_city_500k",
            "large_city_1m": "destination_large_city_1m",
        }
    )
    result["previous_population_imputed"] = result[
        "previous_population_assignment_method"
    ].eq("predecessor_max_area_overlap")

    for threshold, large_column in (
        ("500k", "destination_large_city_500k"),
        ("1m", "destination_large_city_1m"),
    ):
        for frontier_key in FRONTIER_DEFINITIONS:
            result[f"migration_system_{threshold}_{frontier_key}"] = system_category(
                observed,
                result["stayer_5yr"],
                result[large_column].fillna(False),
                result[f"destination_frontier_{frontier_key}"],
            )

    result["mo_travel_time_year"] = MO_YEAR[year]
    travel_key = (
        result["origin_5yr_mesoregion_code"].fillna("")
        + "-"
        + result["current_mesoregion_code"].fillna("")
    )
    for source_column in ("fm_road", "fm_road_rad", "dist_km"):
        result[f"mo_{source_column}"] = travel_key.map(travel[source_column]).where(
            observed
        )
    accessibility_index = accessibility.set_index("origin_mesoregion_code")
    for column in accessibility_index.columns:
        result[column] = result["origin_5yr_mesoregion_code"].map(
            accessibility_index[column]
        ).where(observed)

    result["sex_code"] = frame["sex_code"].astype("string").str.strip()
    result["sex"] = frame["sex"].astype("string")
    result["race_code"] = frame["race_code"].astype("string").str.strip()
    result["race"] = race_label(result["race_code"])
    result["education_years"] = pd.to_numeric(
        frame["education_years"], errors="coerce"
    ).where(lambda value: value.between(0, 19))
    result["education_level_code"] = frame["education_level_code"].astype(
        "string"
    ).str.strip()
    result["education_attainment"] = education_attainment(
        year, frame["education_years"], result["education_level_code"]
    )
    return result[OUTPUT_COLUMNS]


def validate_batch(frame: pd.DataFrame, year: int) -> None:
    if frame["person_record_id"].isna().any():
        raise RuntimeError("Missing person record ID")
    if frame["current_municipality_code"].isna().any():
        raise RuntimeError(f"Unmapped current municipality in {year}")
    if frame["previous_census_population"].isna().any():
        raise RuntimeError(f"Missing previous-census destination population in {year}")
    observed = frame["five_year_internal_od_observed"].fillna(False)
    if frame.loc[observed, "current_mesoregion_code"].isna().any():
        raise RuntimeError(f"Missing current mesoregion in observed {year} OD")
    if frame.loc[observed, "origin_5yr_mesoregion_code"].isna().any():
        raise RuntimeError(f"Missing origin mesoregion in observed {year} OD")
    if frame.loc[observed, "mo_fm_road"].isna().any():
        raise RuntimeError(f"Missing Morten--Oliveira road measure in observed {year} OD")
    accessibility_columns = [
        f"{measure}{'to_' if measure != 'prev_corridor_' else ''}{alternative}"
        for measure in ("prev_corridor_", "min_tt_", "med_tt_")
        for alternative in CHOICE_ALTERNATIVES
    ]
    if frame.loc[observed, accessibility_columns].isna().any().any():
        raise RuntimeError(f"Missing choice-set accessibility in observed {year} OD")
    if frame.loc[~observed, accessibility_columns].notna().any().any():
        raise RuntimeError(f"Choice-set accessibility outside observed {year} OD")
    for threshold in ("500k", "1m"):
        column = f"migration_system_{threshold}_main_north_mt"
        if frame.loc[observed, column].isna().any():
            raise RuntimeError(f"Missing {column} in observed {year} OD")


def main() -> None:
    args = parse_args()
    years = tuple(args.years)
    if not set(years).issubset(YEARS):
        raise ValueError(f"Supported years are {YEARS}")
    geographies = load_geographies()
    population_all = pd.read_parquet(args.population_crosswalk)
    travel_all = pd.read_stata(args.travel_times)
    corridor_all = pd.read_stata(args.corridor_flows)
    args.output.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, dict[str, object]] = {}

    for year in years:
        print(f"Building {year} characteristics dataset...", flush=True)
        population = population_all[
            population_all["target_census_year"] == year
        ].copy()
        travel_year = travel_all[travel_all["year"] == MO_YEAR[year]].copy()
        travel_year["key"] = (
            travel_year["orig_id_meso"].astype("int64").astype("string")
            + "-"
            + travel_year["dest_id_meso"].astype("int64").astype("string")
        )
        if travel_year["key"].duplicated().any():
            raise RuntimeError(f"Duplicate Morten--Oliveira OD keys for {year}")
        travel = travel_year.set_index("key")[["fm_road", "fm_road_rad", "dist_km"]]
        accessibility = build_choice_set_accessibility(
            year, geographies[year], population, travel_year, corridor_all
        )
        accessibility.to_parquet(
            args.output / f"migration_system_accessibility_{year}.parquet",
            index=False,
            compression="zstd",
        )

        files = sorted((args.persons / f"census_year={year}").glob("**/*.parquet"))
        if args.sample_files:
            files = files[: args.sample_files]
        target = args.output / f"persons_{year}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        temporary.unlink(missing_ok=True)
        writer = pq.ParquetWriter(
            temporary,
            TARGET_SCHEMA,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        rows = 0
        system_counts: dict[str, int] = {}
        origin_zone_nonmissing = 0
        observed_rows = 0
        travel_rows = 0
        invalid_source_rows_excluded = 0
        try:
            for file_number, path in enumerate(files, start=1):
                parquet = pq.ParquetFile(path)
                first = next(
                    parquet.iter_batches(columns=["source_file"], batch_size=1)
                ).to_pandas()
                source = Path(first.iloc[0, 0])
                zone = supplemental_origin_zone(
                    year, source, parquet.metadata.num_rows, args.batch_size
                )
                for batch in parquet.iter_batches(
                    columns=list(SOURCE_COLUMNS), batch_size=args.batch_size
                ):
                    enriched = enrich_batch(
                        batch.to_pandas(),
                        year,
                        zone,
                        geographies[year],
                        population,
                        travel,
                        accessibility,
                    )
                    # A small number of fixed-width source files end with a
                    # DOS Ctrl-Z record. It is not a census person and has no
                    # UF, municipality, or weight. Exclude and count it rather
                    # than allowing the sentinel to contaminate the panel.
                    invalid_source = enriched["current_municipality_code"].isna()
                    invalid_source_rows_excluded += int(invalid_source.sum())
                    enriched = enriched.loc[~invalid_source].reset_index(drop=True)
                    if enriched.empty:
                        continue
                    validate_batch(enriched, year)
                    writer.write_table(to_arrow(enriched))
                    rows += len(enriched)
                    origin_zone_nonmissing += int(
                        enriched["origin_5yr_urban_rural"].notna().sum()
                    )
                    observed_rows += int(
                        enriched["five_year_internal_od_observed"].sum()
                    )
                    travel_rows += int(enriched["mo_fm_road"].notna().sum())
                    counts = enriched[
                        "migration_system_500k_main_north_mt"
                    ].value_counts()
                    for key, count in counts.items():
                        system_counts[str(key)] = system_counts.get(str(key), 0) + int(
                            count
                        )
                print(
                    f"  {file_number:02d}/{len(files):02d} {path.name}: "
                    f"{parquet.metadata.num_rows:,} rows",
                    flush=True,
                )
        finally:
            writer.close()
        temporary.replace(target)
        diagnostics[str(year)] = {
            "rows": rows,
            "state_files": len(files),
            "five_year_internal_od_observed_rows": observed_rows,
            "origin_urban_rural_nonmissing_rows": origin_zone_nonmissing,
            "morten_oliveira_nonmissing_rows": travel_rows,
            "invalid_nonperson_source_rows_excluded": invalid_source_rows_excluded,
            "migration_system_500k_main_north_mt": system_counts,
            "output": str(target),
        }

    metadata = {
        "artifact_type": "Person-level repeated census cross-sections (not a longitudinal panel)",
        "census_years": list(years),
        "row_unit": "One sampled census person",
        "migration_period": "Municipality of residence five years before each census",
        "analysis_universe_note": (
            "All sampled persons are retained. Five-year internal OD and migration-system "
            "variables are null for people under age five, foreign origins, and unidentified origins."
        ),
        "migration_system_precedence": [
            "stayer",
            "large_city",
            "agricultural_frontier",
            "other",
        ],
        "large_city_definitions": {
            "500k": "Destination predecessor-census population >= 500,000",
            "1m": "Destination predecessor-census population >= 1,000,000",
        },
        "frontier_definitions": {
            key: sorted(states) for key, states in FRONTIER_DEFINITIONS.items()
        },
        "main_frontier_definition": "main_north_mt",
        "main_large_city_definition": "500k",
        "origin_regions": {
            "northeast": sorted(NORTHEAST),
            "south": sorted(SOUTH),
        },
        "education_attainment": {
            "less_than_fundamental_complete": "0-7 years in 1991/2000; 2010 level 1",
            "fundamental_complete_secondary_incomplete": "8-10 years; 2010 level 2",
            "secondary_complete_tertiary_incomplete": "11-14 years; 2010 level 3",
            "tertiary_complete": "15-19 years; 2010 level 4",
        },
        "origin_zone_availability": {
            "1991": "MIMO86ZN: zone of residence in 1986",
            "2000": "Derived from V4241/status_5yr: codes 1/3 urban, 2/4 rural",
            "2010": "Not collected for the five-year residence; deliberately null",
        },
        "travel_time": {
            "source": "Morten--Oliveira tt_mesospeed_10.dta",
            "year_mapping": {str(year): MO_YEAR[year] for year in years},
            "retained_columns": ["fm_road", "fm_road_rad", "dist_km"],
        },
        "migration_system_choice_set": {
            "alternatives": list(CHOICE_ALTERNATIVES),
            "corridor_source": "Morten--Oliveira N_od_meso.dta:N_od_flow_all",
            "prior_flow_year_mapping": {
                str(year): PRIOR_FLOW_YEAR[year] for year in years
            },
            "travel_source": "Morten--Oliveira tt_mesospeed_10.dta:fm_road",
            "travel_year_mapping": {str(year): MO_YEAR[year] for year in years},
            "definition": (
                "For each origin mesoregion, sum prior flows and calculate the "
                "unweighted minimum and median travel time over all destination "
                "mesoregions in each alternative. Stayer is the origin mesoregion; "
                "city means a mesoregion containing any >=500k lagged-population "
                "municipality; agricultural frontier is North plus Mato Grosso "
                "after city precedence; other is the remainder."
            ),
        },
        "excluded_post_decision_variables": [
            "occupation",
            "employment",
            "income",
            "industry",
            "hours worked",
            "workplace",
        ],
        "diagnostics": diagnostics,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
