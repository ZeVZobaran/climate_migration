"""Build annual MapBiomas Collection 11 land-cover outcomes on study AMCs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


NATIVE_VEGETATION_CLASSES = {3, 4, 5, 6, 7, 11, 12, 13, 29, 32, 49, 50, 84}
FOREST_CLASSES = {3, 4, 5, 6, 7, 49}
NONFOREST_NATIVE_CLASSES = {11, 12, 13, 29, 32, 50, 84}
AGRICULTURE_CLASSES = {20, 35, 39, 40, 41, 46, 47, 48, 62}
PSEUDO_WATER_GEOCODES = {"4300001", "4300002"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-xlsx",
        type=Path,
        default=Path(
            "data/mapbiomas/raw/"
            "MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.11-"
            "MUNICIPALITIES_STATES_BIOMES.xlsx"
        ),
    )
    parser.add_argument(
        "--municipality-treatment",
        type=Path,
        default=Path(
            "data/processed/amcs/polocentro_1975_municipality_treatment.parquet"
        ),
    )
    parser.add_argument(
        "--amc-gpkg",
        type=Path,
        default=Path("data/censo_microdados/amc/AMC_1970_2010_simplified.gpkg"),
    )
    parser.add_argument(
        "--ms-2024-zip",
        type=Path,
        default=Path("data/ibge/municipality_boundaries_2024/MS_Municipios_2024.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/polocentro_environment"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def area_metrics(raw: pd.DataFrame, year_columns: list[str]) -> dict[str, pd.DataFrame]:
    class_id = raw["class"].astype(int)
    metrics = {
        "territory_ha": pd.Series(True, index=raw.index),
        "not_observed_ha": class_id.eq(0),
        "mapped_territory_ha": class_id.ne(0),
        "native_vegetation_ha": class_id.isin(NATIVE_VEGETATION_CLASSES),
        "forest_ha": class_id.isin(FOREST_CLASSES),
        "nonforest_native_vegetation_ha": class_id.isin(
            NONFOREST_NATIVE_CLASSES
        ),
        "natural_water_ha": class_id.eq(33),
        "natural_nonvegetated_ha": class_id.eq(23),
        "anthropic_use_ha": raw["class_level_0"].eq("Antropic"),
        "farming_ha": raw["class_level_1"].eq("3. Farming"),
        "pasture_ha": class_id.eq(15),
        "agriculture_ha": class_id.isin(AGRICULTURE_CLASSES),
        "soybean_ha": class_id.eq(39),
        "sugarcane_ha": class_id.eq(20),
        "rice_ha": class_id.eq(40),
        "cotton_ha": class_id.eq(62),
        "coffee_ha": class_id.eq(46),
        "forest_plantation_ha": class_id.eq(9),
        "urban_area_ha": class_id.eq(24),
        "mining_ha": class_id.eq(30),
    }
    output = {}
    for name, mask in metrics.items():
        selected = raw.loc[mask, ["geocode", *year_columns]]
        output[name] = selected.groupby("geocode", observed=True)[year_columns].sum()
    return output


def build_municipality_panel(raw: pd.DataFrame, year_columns: list[str]) -> pd.DataFrame:
    metric_tables = area_metrics(raw, year_columns)
    pieces = []
    for name, table in metric_tables.items():
        table = table.rename(columns={column: int(column[1:]) for column in year_columns})
        pieces.append(table.stack().rename(name))
    panel = pd.concat(pieces, axis=1).reset_index(names=["geocode", "year"])

    lookup = (
        raw[["geocode", "municipality", "state", "region"]]
        .drop_duplicates("geocode")
        .set_index("geocode")
    )
    panel = panel.join(lookup, on="geocode")
    panel = add_derived_metrics(panel, "geocode")
    return panel.sort_values(["geocode", "year"]).reset_index(drop=True)


def paraiso_weights(ms_zip: Path, amc_gpkg: Path) -> pd.DataFrame:
    municipalities = gpd.read_file(f"zip://{ms_zip.as_posix()}")
    paraiso = municipalities[
        municipalities["CD_MUN"].astype(str).eq("5006275")
    ].to_crs("EPSG:5880")
    if len(paraiso) != 1:
        raise RuntimeError("Could not isolate Paraíso das Águas in 2024 IBGE boundaries")
    amcs = gpd.read_file(amc_gpkg).to_crs("EPSG:5880")
    amcs["amc_code"] = amcs["code_amc"].astype(int).astype(str)
    overlap = gpd.overlay(
        paraiso[["geometry"]], amcs[["amc_code", "geometry"]], how="intersection"
    )
    overlap["allocation_area_km2"] = overlap.area / 1_000_000
    overlap["allocation_weight"] = (
        overlap["allocation_area_km2"] / overlap["allocation_area_km2"].sum()
    )
    overlap["geocode"] = "5006275"
    overlap["allocation_method"] = "2024 polygon area share across 2010 AMCs"
    return overlap[
        [
            "geocode",
            "amc_code",
            "allocation_weight",
            "allocation_area_km2",
            "allocation_method",
        ]
    ]


def build_crosswalk(
    raw_codes: set[str], treatment_path: Path, ms_zip: Path, amc_gpkg: Path
) -> pd.DataFrame:
    treatment = pd.read_parquet(treatment_path)
    treatment["municipality_code"] = treatment["municipality_code"].astype("string")
    treatment["amc_code"] = treatment["amc_code"].astype("string")
    base = treatment[["municipality_code", "amc_code"]].rename(
        columns={"municipality_code": "geocode"}
    )
    base = base[base["geocode"].isin(raw_codes)].copy()
    base["allocation_weight"] = 1.0
    base["allocation_area_km2"] = np.nan
    base["allocation_method"] = "direct 2010 municipality code"

    parent_codes = {
        "1504752": "1506807",  # Mojuí dos Campos <- Santarém
        "4212650": "4209409",  # Pescaria Brava <- Laguna
        "4220000": "4207007",  # Balneário Rincão <- Içara
        "4314548": "4302105",  # Pinto Bandeira <- Bento Gonçalves
    }
    parent_map = treatment.set_index("municipality_code")["amc_code"].to_dict()
    extra_rows = []
    for current, parent in parent_codes.items():
        extra_rows.append(
            {
                "geocode": current,
                "amc_code": parent_map[parent],
                "allocation_weight": 1.0,
                "allocation_area_km2": np.nan,
                "allocation_method": f"post-2010 municipality assigned to parent {parent}",
            }
        )
    # Boa Esperança do Norte was created from Nova Ubiratã and Sorriso; both
    # belong to AMC 1098, so no within-AMC allocation ambiguity arises.
    boa_parents = [parent_map["5106240"], parent_map["5107925"]]
    if len(set(boa_parents)) != 1:
        raise RuntimeError("Boa Esperança do Norte parent AMCs unexpectedly differ")
    extra_rows.append(
        {
            "geocode": "5101837",
            "amc_code": boa_parents[0],
            "allocation_weight": 1.0,
            "allocation_area_km2": np.nan,
            "allocation_method": "post-2010 municipality; both parents are AMC 1098",
        }
    )
    extras = pd.DataFrame(extra_rows)
    paraiso = paraiso_weights(ms_zip, amc_gpkg)
    crosswalk = pd.concat([base, extras, paraiso], ignore_index=True)

    represented = set(crosswalk["geocode"])
    if represented != raw_codes:
        raise RuntimeError(
            f"MapBiomas crosswalk mismatch; missing={sorted(raw_codes-represented)}, "
            f"unexpected={sorted(represented-raw_codes)}"
        )
    weight_check = crosswalk.groupby("geocode")["allocation_weight"].sum()
    if not np.allclose(weight_check.to_numpy(), 1.0, atol=1e-10):
        raise RuntimeError("Municipality-to-AMC allocation weights do not sum to one")
    return crosswalk.sort_values(["geocode", "amc_code"]).reset_index(drop=True)


def add_derived_metrics(panel: pd.DataFrame, geography: str) -> pd.DataFrame:
    panel = panel.sort_values([geography, "year"]).copy()
    denominator = panel["mapped_territory_ha"].replace(0, np.nan)
    share_areas = [
        "native_vegetation_ha",
        "forest_ha",
        "nonforest_native_vegetation_ha",
        "natural_water_ha",
        "anthropic_use_ha",
        "farming_ha",
        "pasture_ha",
        "agriculture_ha",
        "soybean_ha",
        "urban_area_ha",
    ]
    for area in share_areas:
        panel[area.replace("_ha", "_share_of_mapped")] = panel[area] / denominator

    groups = panel.groupby(geography, observed=True)
    panel["native_vegetation_change_ha"] = groups["native_vegetation_ha"].diff()
    panel["native_vegetation_net_loss_ha"] = -panel["native_vegetation_change_ha"]
    panel["anthropic_use_change_ha"] = groups["anthropic_use_ha"].diff()
    native_1985 = panel[geography].map(
        panel.loc[panel["year"].eq(1985)].set_index(geography)["native_vegetation_ha"]
    )
    anthropic_1985 = panel[geography].map(
        panel.loc[panel["year"].eq(1985)].set_index(geography)["anthropic_use_ha"]
    )
    panel["native_vegetation_net_loss_since_1985_ha"] = (
        native_1985 - panel["native_vegetation_ha"]
    )
    panel["native_vegetation_net_loss_share_of_1985"] = (
        panel["native_vegetation_net_loss_since_1985_ha"]
        / native_1985.replace(0, np.nan)
    )
    panel["anthropic_expansion_since_1985_ha"] = (
        panel["anthropic_use_ha"] - anthropic_1985
    )
    panel["is_first_mapbiomas_year"] = panel["year"].eq(1985)
    return panel


def build_amc_panel(
    municipality: pd.DataFrame, crosswalk: pd.DataFrame
) -> pd.DataFrame:
    allocated = municipality.merge(crosswalk, on="geocode", how="inner", validate="m:m")
    area_columns = [column for column in municipality if column.endswith("_ha")]
    # Derived municipality changes/shares are recalculated after AMC aggregation.
    base_area_columns = [
        column
        for column in area_columns
        if "change" not in column and "since_1985" not in column and "net_loss" not in column
    ]
    for column in base_area_columns:
        allocated[column] = allocated[column] * allocated["allocation_weight"]

    amc = allocated.groupby(["amc_code", "year"], observed=True)[base_area_columns].sum()
    diagnostics = allocated.groupby(["amc_code", "year"], observed=True).agg(
        source_mapbiomas_geocodes=("geocode", "nunique"),
        source_municipality_equivalents=("allocation_weight", "sum"),
        includes_split_municipality=(
            "allocation_method",
            lambda values: any("area share" in value for value in values),
        ),
    )
    amc = amc.join(diagnostics).reset_index()
    return add_derived_metrics(amc, "amc_code").sort_values(
        ["amc_code", "year"]
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {args.source_xlsx}", flush=True)
    raw = pd.read_excel(args.source_xlsx, sheet_name="COVERAGE_11")
    raw["geocode"] = raw["geocode"].astype(str).str.replace(r"\.0$", "", regex=True)
    raw = raw[~raw["geocode"].isin(PSEUDO_WATER_GEOCODES)].copy()
    year_columns = sorted(
        [column for column in raw if str(column).startswith("y") and str(column)[1:].isdigit()]
    )
    expected_years = [f"y{year}" for year in range(1985, 2026)]
    if year_columns != expected_years:
        raise RuntimeError(f"Unexpected MapBiomas years: {year_columns}")
    if raw["geocode"].nunique() != 5570:
        raise RuntimeError(f"Unexpected MapBiomas municipality count: {raw.geocode.nunique()}")

    municipality = build_municipality_panel(raw, year_columns)
    crosswalk = build_crosswalk(
        set(raw["geocode"]),
        args.municipality_treatment,
        args.ms_2024_zip,
        args.amc_gpkg,
    )
    amc = build_amc_panel(municipality, crosswalk)

    if len(municipality) != 5570 * 41:
        raise RuntimeError(f"Unexpected municipality-year rows: {len(municipality)}")
    if amc["amc_code"].nunique() != 3799 or len(amc) != 3799 * 41:
        raise RuntimeError(f"Unexpected AMC-year coverage: {amc.shape}")
    municipality_share = municipality["native_vegetation_share_of_mapped"]
    amc_share = amc["native_vegetation_share_of_mapped"]
    tolerance = 1e-10
    if municipality_share.min() < -tolerance or municipality_share.max() > 1 + tolerance:
        raise RuntimeError(
            "Municipality native-vegetation share materially outside [0,1]: "
            f"min={municipality_share.min()}, max={municipality_share.max()}"
        )
    if amc_share.min() < -tolerance or amc_share.max() > 1 + tolerance:
        raise RuntimeError(
            "AMC native-vegetation share materially outside [0,1]: "
            f"min={amc_share.min()}, max={amc_share.max()}"
        )
    national_municipality = municipality.groupby("year")["territory_ha"].sum()
    national_amc = amc.groupby("year")["territory_ha"].sum()
    if not np.allclose(
        national_municipality.to_numpy(), national_amc.to_numpy(), rtol=0, atol=1e-4
    ):
        raise RuntimeError("National area changed during municipality-to-AMC allocation")

    municipality_path = (
        args.output_dir / "mapbiomas_collection11_municipality_year.parquet"
    )
    amc_path = args.output_dir / "mapbiomas_collection11_amc_year.parquet"
    amc_csv_path = args.output_dir / "mapbiomas_collection11_amc_year.csv.gz"
    crosswalk_path = (
        args.output_dir / "mapbiomas_collection11_municipality_to_amc.parquet"
    )
    municipality.to_parquet(municipality_path, index=False)
    amc.to_parquet(amc_path, index=False)
    amc.to_csv(amc_csv_path, index=False, compression="gzip", float_format="%.8f")
    crosswalk.to_parquet(crosswalk_path, index=False)
    crosswalk.to_csv(
        args.output_dir / "mapbiomas_collection11_municipality_to_amc.csv",
        index=False,
        float_format="%.12f",
    )

    metadata = {
        "created": date.today().isoformat(),
        "dataset": "MapBiomas Brazil Collection 11 land-cover outcomes",
        "years": [1985, 2025],
        "year_count": 41,
        "source_unit": "Current municipality x biome x leaf land-cover class",
        "municipality_output_unit": "Current municipality x year",
        "amc_output_unit": "1970-2010 AMC x year",
        "merge_key": ["amc_code (string)", "year (integer)"],
        "amc_coverage": 3799,
        "amc_unavailable": {
            "amc_code": "7061",
            "reason": (
                "Fernando de Noronha (IBGE 2605459) is not included as a municipality "
                "in the Collection 11 source workbook; it is a standalone, non-"
                "POLOCENTRO AMC."
            ),
        },
        "native_vegetation_classes": sorted(NATIVE_VEGETATION_CLASSES),
        "important_interpretation": (
            "Annual native_vegetation_net_loss_ha is the signed year-to-year net "
            "stock reduction. It is not gross deforestation: gains/recovery can offset "
            "losses within an AMC-year. Use the pixel-transition/deforestation module "
            "for gross clearing events."
        ),
        "timing_warning": (
            "The series begins in 1985, ten years after POLOCENTRO assignment. It "
            "provides post-treatment environmental outcomes but no untreated pre-1975 "
            "land-cover baseline or pre-trend."
        ),
        "boundary_harmonization": {
            "direct_codes": 5564,
            "post_2010_single_amc": [
                "1504752",
                "4212650",
                "4220000",
                "4314548",
                "5101837",
            ],
            "paraiso_das_aguas_5006275": crosswalk[
                crosswalk["geocode"].eq("5006275")
            ].to_dict("records"),
            "caution": (
                "Paraíso das Águas statistics are split across three 2010 AMCs in "
                "proportion to 2024 municipal polygon area. This preserves totals but "
                "assumes land-cover composition is uniform within that municipality."
            ),
        },
        "excluded_nonmunicipal_geocodes": sorted(PSEUDO_WATER_GEOCODES),
        "source_file": {
            "path": args.source_xlsx.as_posix(),
            "bytes": args.source_xlsx.stat().st_size,
            "sha256": sha256(args.source_xlsx),
            "official_download_page": (
                "https://brasil.mapbiomas.org/downloads/estatisticas/"
            ),
            "official_drive_id": "1ZRobxa-pK4AYo3kYqMZ6Fww1TMULO25M",
        },
        "ibge_boundary_source": (
            "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
            "malhas_territoriais/malhas_municipais/municipio_2024/UFs/MS/"
            "MS_Municipios_2024.zip"
        ),
        "license": "MapBiomas CC BY 4.0",
        "outputs": [
            municipality_path.as_posix(),
            amc_path.as_posix(),
            amc_csv_path.as_posix(),
            crosswalk_path.as_posix(),
        ],
    }
    (args.output_dir / "mapbiomas_collection11_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {municipality_path} ({len(municipality):,} rows)", flush=True)
    print(f"Wrote {amc_path} ({len(amc):,} rows)", flush=True)


if __name__ == "__main__":
    main()
