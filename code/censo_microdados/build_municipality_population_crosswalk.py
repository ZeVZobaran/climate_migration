"""Build lagged municipal-population and predecessor crosswalks.

The large-city definitions in the characteristics panels use the destination
municipality's population in the preceding census: 1980 for the 1991 panel,
1991 for 2000, and 2000 for 2010.  Population comes from IBGE SIDRA table 202.

SIDRA reports a missing value when a municipality did not yet exist in the
preceding census.  For those cases this script overlays the target-census
municipal polygon on the preceding-census polygons, selects the predecessor
with the largest intersection area, and assigns that predecessor's population.
The predecessor code and overlap share remain in the output for auditing.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd


YEAR_PAIRS = ((1991, 1980), (2000, 1991), (2010, 2000))
SIDRA_URL = (
    "https://apisidra.ibge.gov.br/values/t/202/n6/all/v/93/p/{year}"
    "?formato=json"
)
ANALYSIS_CRS = "EPSG:5880"  # SIRGAS 2000 / Brazil Polyconic; areas in m2.


def normalized_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return " ".join(text.upper().replace("'", " ").replace("-", " ").split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population-root",
        type=Path,
        default=Path("data/censo_microdados/ibge_population"),
    )
    parser.add_argument(
        "--boundary-root",
        type=Path,
        default=Path("data/censo_microdados/ibge_boundaries"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/censo_microdados/characteristics_panels/"
            "municipality_population_crosswalk.parquet"
        ),
    )
    return parser.parse_args()


def load_population(root: Path, year: int) -> pd.DataFrame:
    path = root / f"sidra_table_202_{year}.json"
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame.from_records(records[1:])
    result = pd.DataFrame(
        {
            "municipality_code": frame["D1C"].astype("string"),
            "municipality_name": frame["D1N"].astype("string"),
            "population": pd.to_numeric(frame["V"].replace("...", pd.NA)),
        }
    )
    if result["municipality_code"].duplicated().any():
        raise RuntimeError(f"SIDRA {year} contains duplicate municipality codes")
    return result


def read_zip(path: Path, year: int) -> gpd.GeoDataFrame:
    frame = gpd.read_file("zip://" + str(path.resolve()))
    if year == 1980:
        frame = frame.rename(
            columns={"codigo": "municipality_code", "nome": "municipality_name"}
        )
    elif year == 1991:
        frame = frame.rename(
            columns={
                "BR91POLY_I": "municipality_code",
                "NOMEMUNICP": "municipality_name",
            }
        )
    elif year == 2000:
        frame = frame.rename(
            columns={"GEOCODIGO": "municipality_code", "NOME": "municipality_name"}
        )
        # The distributed 2000 state files omit their .prj files. Coordinates
        # are geographic SAD69, as documented for this IBGE municipal mesh.
        if frame.crs is None:
            frame = frame.set_crs("EPSG:4618")
    elif year == 2010:
        frame = frame.rename(
            columns={
                "CD_GEOCODM": "municipality_code",
                "NM_MUNICIP": "municipality_name",
            }
        )
    else:
        raise ValueError(year)
    frame["municipality_code"] = (
        frame["municipality_code"].astype("string").str.strip().str.zfill(7)
    )
    return frame[["municipality_code", "municipality_name", "geometry"]]


def load_boundaries(root: Path, year: int) -> gpd.GeoDataFrame:
    parts = [read_zip(path, year) for path in sorted((root / str(year)).glob("*.zip"))]
    if not parts:
        raise FileNotFoundError(f"No {year} boundary archives under {root}")
    frame = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    frame = frame.to_crs(ANALYSIS_CRS)
    frame["geometry"] = frame.geometry.make_valid()
    # Historical files can contain multiple features for one municipality.
    frame = frame.dissolve(by="municipality_code", as_index=False, aggfunc="first")
    if frame["municipality_code"].duplicated().any():
        raise RuntimeError(f"Duplicate {year} municipality polygons after dissolve")
    return frame


def predecessor_assignments(
    missing_codes: pd.Index,
    current_boundaries: gpd.GeoDataFrame,
    previous_boundaries: gpd.GeoDataFrame,
    previous_population: pd.DataFrame,
) -> pd.DataFrame:
    current = current_boundaries[
        current_boundaries["municipality_code"].isin(missing_codes)
    ].rename(columns={"municipality_code": "current_municipality_code"})
    previous = previous_boundaries.rename(
        columns={
            "municipality_code": "previous_boundary_code",
            "municipality_name": "previous_boundary_name",
        }
    )
    if len(current) != len(missing_codes):
        absent = sorted(set(missing_codes) - set(current["current_municipality_code"]))
        raise RuntimeError(f"Target boundary is missing municipality codes: {absent[:20]}")

    current = current.copy()
    previous = previous.copy()
    current["uf"] = current["current_municipality_code"].str[:2]
    previous["uf"] = previous["previous_boundary_code"].str[:2]
    previous_geometry = previous.set_index("previous_boundary_code").geometry
    previous_values = previous_population.set_index("municipality_code")["population"]
    population_lookup = previous_population.copy()
    population_lookup["uf"] = population_lookup["municipality_code"].str[:2]
    population_lookup["normalized_name"] = population_lookup["municipality_name"].map(
        lambda value: normalized_name(str(value).rsplit(" - ", 1)[0])
    )
    population_lookup = population_lookup[population_lookup["population"].notna()]
    if population_lookup.duplicated(["uf", "normalized_name"]).any():
        duplicates = population_lookup.loc[
            population_lookup.duplicated(["uf", "normalized_name"], keep=False),
            ["uf", "normalized_name"],
        ]
        raise RuntimeError(f"Duplicate populated municipality names: {duplicates.head()}")
    population_by_name = population_lookup.set_index(["uf", "normalized_name"])

    boundary_population = previous[
        ["previous_boundary_code", "previous_boundary_name", "uf"]
    ].copy()
    boundary_population["previous_municipality_code"] = boundary_population[
        "previous_boundary_code"
    ]
    boundary_population["previous_population"] = boundary_population[
        "previous_boundary_code"
    ].map(previous_values)
    for row_index in boundary_population.index[
        boundary_population["previous_population"].isna()
    ]:
        row = boundary_population.loc[row_index]
        key = (row["uf"], normalized_name(row["previous_boundary_name"]))
        if key in population_by_name.index:
            matched = population_by_name.loc[key]
            boundary_population.at[row_index, "previous_municipality_code"] = matched[
                "municipality_code"
            ]
            boundary_population.at[row_index, "previous_population"] = matched[
                "population"
            ]

    # Resolve renamed northern-Goiás municipalities for the 1980->1991
    # transition. Their 1980 boundary uses a 52 code, while SIDRA carries the
    # historical population under the successor 17 code and current name.
    unresolved_go = boundary_population[
        boundary_population["previous_population"].isna()
        & (boundary_population["uf"] == "52")
    ]
    if not unresolved_go.empty and (current["current_municipality_code"].str[:2] == "17").any():
        old_go = previous[
            previous["previous_boundary_code"].isin(
                unresolved_go["previous_boundary_code"]
            )
        ][["previous_boundary_code", "geometry"]]
        successor = current_boundaries[
            current_boundaries["municipality_code"].str[:2] == "17"
        ].copy()
        successor["successor_population"] = successor["municipality_code"].map(
            previous_values
        )
        successor = successor[successor["successor_population"].notna()]
        successor_geometry = successor.set_index("municipality_code").geometry
        transition = gpd.sjoin(
            old_go,
            successor[["municipality_code", "geometry"]],
            how="inner",
            predicate="intersects",
        ).reset_index(drop=True)
        transition["intersection_area"] = [
            geometry.intersection(successor_geometry[code]).area
            for geometry, code in zip(
                transition.geometry, transition["municipality_code"], strict=True
            )
        ]
        transition = transition.sort_values(
            ["previous_boundary_code", "intersection_area", "municipality_code"],
            ascending=[True, False, True],
        ).drop_duplicates("previous_boundary_code")
        successor_code = transition.set_index("previous_boundary_code")[
            "municipality_code"
        ]
        mask = boundary_population["previous_boundary_code"].isin(successor_code.index)
        boundary_population.loc[mask, "previous_municipality_code"] = boundary_population.loc[
            mask, "previous_boundary_code"
        ].map(successor_code)
        boundary_population.loc[mask, "previous_population"] = boundary_population.loc[
            mask, "previous_municipality_code"
        ].map(previous_values)

    boundary_population = boundary_population.set_index("previous_boundary_code")

    candidates = gpd.sjoin(
        current,
        previous[
            ["previous_boundary_code", "previous_boundary_name", "uf", "geometry"]
        ],
        how="left",
        predicate="intersects",
        lsuffix="current",
        rsuffix="previous",
    )
    # State borders differ slightly by vintage. Restricting to the same UF
    # prevents a thin cross-border sliver from ever becoming a predecessor.
    same_state = candidates["uf_current"] == candidates["uf_previous"]
    # Tocantins was created from northern Goiás in 1988, between the 1980
    # and 1991 censuses.  Its 1991 municipalities therefore legitimately
    # overlap 1980 polygons whose codes begin with 52.
    tocantins_transition = (
        (candidates["uf_current"] == "17")
        & (candidates["uf_previous"] == "52")
    )
    candidates = candidates[same_state | tocantins_transition].copy().reset_index(drop=True)
    candidates["previous_municipality_code"] = candidates[
        "previous_boundary_code"
    ].map(boundary_population["previous_municipality_code"])
    candidates["previous_population"] = candidates["previous_boundary_code"].map(
        boundary_population["previous_population"]
    )
    candidates = candidates[candidates["previous_population"].notna()].copy()
    if candidates.empty:
        raise RuntimeError("No populated predecessor candidates found")

    candidates["intersection_area"] = [
        geometry.intersection(previous_geometry[code]).area
        for geometry, code in zip(
            candidates.geometry,
            candidates["previous_boundary_code"],
            strict=True,
        )
    ]
    candidates["target_area"] = candidates.geometry.area
    candidates["predecessor_overlap_share"] = (
        candidates["intersection_area"] / candidates["target_area"]
    )
    candidates = candidates.sort_values(
        ["current_municipality_code", "intersection_area", "previous_boundary_code"],
        ascending=[True, False, True],
    )
    chosen = candidates.drop_duplicates("current_municipality_code", keep="first")
    if len(chosen) != len(missing_codes):
        absent = sorted(set(missing_codes) - set(chosen["current_municipality_code"]))
        raise RuntimeError(f"No populated predecessor found for: {absent[:20]}")
    return chosen[
        [
            "current_municipality_code",
            "previous_municipality_code",
            "previous_boundary_code",
            "previous_population",
            "predecessor_overlap_share",
        ]
    ].reset_index(drop=True)


def build_pair(
    target_year: int,
    previous_year: int,
    populations: dict[int, pd.DataFrame],
    boundaries: dict[int, gpd.GeoDataFrame],
) -> pd.DataFrame:
    target = populations[target_year].loc[
        populations[target_year]["population"].notna(),
        ["municipality_code", "municipality_name"],
    ].copy()
    previous = populations[previous_year]
    previous_index = previous.set_index("municipality_code")
    target["previous_population"] = target["municipality_code"].map(
        previous_index["population"]
    )
    target["previous_municipality_code"] = target["municipality_code"].where(
        target["previous_population"].notna()
    )
    target["previous_boundary_code"] = target["previous_municipality_code"]
    target["population_assignment_method"] = "direct_code_match"
    target["predecessor_overlap_share"] = 1.0

    missing = pd.Index(target.loc[target["previous_population"].isna(), "municipality_code"])
    if len(missing):
        assigned = predecessor_assignments(
            missing,
            boundaries[target_year],
            boundaries[previous_year],
            previous,
        ).set_index("current_municipality_code")
        mask = target["municipality_code"].isin(missing)
        target.loc[mask, "previous_population"] = target.loc[
            mask, "municipality_code"
        ].map(assigned["previous_population"])
        target.loc[mask, "previous_municipality_code"] = target.loc[
            mask, "municipality_code"
        ].map(assigned["previous_municipality_code"])
        target.loc[mask, "previous_boundary_code"] = target.loc[
            mask, "municipality_code"
        ].map(assigned["previous_boundary_code"])
        target.loc[mask, "predecessor_overlap_share"] = target.loc[
            mask, "municipality_code"
        ].map(assigned["predecessor_overlap_share"])
        target.loc[mask, "population_assignment_method"] = (
            "predecessor_max_area_overlap"
        )

    target["previous_municipality_name"] = target[
        "previous_municipality_code"
    ].map(previous_index["municipality_name"])
    target["target_census_year"] = target_year
    target["previous_census_year"] = previous_year
    target["large_city_500k"] = target["previous_population"] >= 500_000
    target["large_city_1m"] = target["previous_population"] >= 1_000_000
    if target["previous_population"].isna().any():
        raise RuntimeError(f"Unassigned previous population remains for {target_year}")
    return target[
        [
            "target_census_year",
            "previous_census_year",
            "municipality_code",
            "municipality_name",
            "previous_population",
            "previous_municipality_code",
            "previous_boundary_code",
            "previous_municipality_name",
            "population_assignment_method",
            "predecessor_overlap_share",
            "large_city_500k",
            "large_city_1m",
        ]
    ]


def main() -> None:
    args = parse_args()
    years = sorted({year for pair in YEAR_PAIRS for year in pair})
    populations = {year: load_population(args.population_root, year) for year in years}
    boundaries = {year: load_boundaries(args.boundary_root, year) for year in years}
    output = pd.concat(
        [
            build_pair(target, previous, populations, boundaries)
            for target, previous in YEAR_PAIRS
        ],
        ignore_index=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False, compression="zstd")
    output.to_csv(args.output.with_suffix(".csv"), index=False)

    diagnostics = {
        "source": "IBGE SIDRA table 202, resident population",
        "source_urls": {str(year): SIDRA_URL.format(year=year) for year in years},
        "population_rule": {
            str(target): previous for target, previous in YEAR_PAIRS
        },
        "fallback_rule": (
            "For a municipality absent in the preceding census, assign the "
            "population of the same-UF preceding-census polygon with the largest "
            "area overlap."
        ),
        "rows": int(len(output)),
        "assignment_counts": {
            f"{year}:{method}": int(count)
            for (year, method), count in output.groupby(
                ["target_census_year", "population_assignment_method"]
            ).size().items()
        },
        "minimum_predecessor_overlap_share": float(
            output.loc[
                output["population_assignment_method"]
                == "predecessor_max_area_overlap",
                "predecessor_overlap_share",
            ].min()
        ),
    }
    args.output.with_name("municipality_population_crosswalk_metadata.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
