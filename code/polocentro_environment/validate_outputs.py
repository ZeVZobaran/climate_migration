"""Compact validation and descriptive summary for environment extensions."""

from pathlib import Path

import pandas as pd


output = Path("data/processed/polocentro_environment")
gaez_wide = pd.read_parquet(
    output / "gaez_v4_6190_amc_crop_suitability_wide.parquet"
)
gaez_long = pd.read_parquet(output / "gaez_v4_6190_amc_crop_suitability.parquet")
mapbiomas = pd.read_parquet(output / "mapbiomas_collection11_amc_year.parquet")
treatment = pd.read_parquet(
    "data/processed/amcs/polocentro_1975_amc_treatment.parquet"
)[["amc_code", "polocentro_operational_core", "polocentro_operational_area_share"]]
treatment["amc_code"] = treatment["amc_code"].astype(str)
gaez_wide = gaez_wide.merge(treatment, on="amc_code", validate="1:1")
mapbiomas = mapbiomas.merge(treatment, on="amc_code", validate="m:1")

print("FILES")
for path in sorted(output.iterdir()):
    print(path.name, path.stat().st_size)

print("GAEZ rows/amcs", len(gaez_long), gaez_long.amc_code.nunique(), len(gaez_wide))
gaez_columns = [
    "gaez_soy_high_mean_0_100",
    "gaez_soy_low_mean_0_100",
    "gaez_core8_high_mean_0_100",
    "gaez_core8_low_mean_0_100",
]
print("GAEZ ranges")
print(gaez_wide[gaez_columns].agg(["min", "mean", "max"]).round(3).to_string())
print("treated AMCs", int(treatment.polocentro_operational_core.sum()))
print("GAEZ treatment-group means")
print(
    gaez_wide.groupby("polocentro_operational_core")[
        ["gaez_soy_high_mean_0_100", "gaez_core8_high_mean_0_100"]
    ]
    .mean()
    .round(3)
    .to_string()
)

print(
    "MapBiomas rows/years/amcs",
    len(mapbiomas),
    int(mapbiomas.year.min()),
    int(mapbiomas.year.max()),
    mapbiomas.amc_code.nunique(),
)
print("MapBiomas treatment-group unweighted AMC means")
print(
    mapbiomas[mapbiomas.year.isin([1985, 1991, 2000, 2010, 2025])]
    .groupby(["polocentro_operational_core", "year"])[
        [
            "native_vegetation_share_of_mapped",
            "anthropic_use_share_of_mapped",
            "agriculture_share_of_mapped",
            "pasture_share_of_mapped",
        ]
    ]
    .mean()
    .round(4)
    .to_string()
)

print("National million ha, excluding Fernando de Noronha")
print(
    (
        mapbiomas.groupby("year")
        [["native_vegetation_ha", "anthropic_use_ha", "territory_ha"]]
        .sum()
        .loc[[1985, 2025]]
        / 1e6
    )
    .round(3)
    .to_string()
)
print("Treated-AMC million ha")
print(
    (
        mapbiomas[mapbiomas.polocentro_operational_core]
        .groupby("year")
        [["native_vegetation_ha", "anthropic_use_ha", "territory_ha"]]
        .sum()
        .loc[[1985, 2025]]
        / 1e6
    )
    .round(3)
    .to_string()
)
print(
    "Native share range",
    mapbiomas.native_vegetation_share_of_mapped.min(),
    mapbiomas.native_vegetation_share_of_mapped.max(),
)
print("Paraíso das Águas allocation")
print(
    pd.read_parquet(output / "mapbiomas_collection11_municipality_to_amc.parquet")
    .query("geocode == '5006275'")
    .to_string(index=False)
)
