"""Aggregate FAO GAEZ v4 crop-suitability rasters to 1970--2010 AMCs.

The input rasters are the continuous all-land crop suitability index (sx),
historical CRU TS 3.2 climate for 1961--1990, rainfed production, no CO2
fertilization, at high and low management/input levels. Raster values run
from 0 to 10,000 and are rescaled here to 0--100.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from exactextract import exact_extract


CROPS = {
    "soy": "Soybean",
    "mze": "Maize",
    "rcd": "Dryland rice",
    "phb": "Phaseolus bean",
    "whe": "Wheat",
    "cot": "Cotton",
    "cof": "Coffee",
    "suc": "Sugarcane",
}
SOURCE_BASE = (
    "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/"
    "res05/CRUTS32/Hist"
)
FILE_RE = re.compile(r"^sx(?P<input>[HL])r0_(?P<crop>[a-z0-9]+)\.tif$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amc-gpkg",
        type=Path,
        default=Path("data/censo_microdados/amc/AMC_1970_2010_simplified.gpkg"),
    )
    parser.add_argument(
        "--raster-dir", type=Path, default=Path("data/gaez_v4/raw/6190")
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


def source_url(path: Path, input_code: str) -> str:
    return f"{SOURCE_BASE}/6190{input_code}/{path.name}"


def extract_one(path: Path, amcs: gpd.GeoDataFrame) -> pd.DataFrame:
    match = FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected GAEZ filename: {path.name}")
    input_code = match.group("input")
    crop_code = match.group("crop")
    if crop_code not in CROPS:
        raise ValueError(f"Unselected crop in raster directory: {crop_code}")

    with rasterio.open(path) as source:
        if source.count != 1 or source.width != 4320 or source.height != 2160:
            raise RuntimeError(f"Unexpected GAEZ raster dimensions: {path}")
        if source.nodata != -9:
            raise RuntimeError(f"Unexpected GAEZ nodata value: {path}")
        if str(source.dtypes[0]) != "int16":
            raise RuntimeError(f"Unexpected GAEZ raster dtype: {path}")

    result = exact_extract(
        path.as_posix(),
        amcs,
        [
            "mean(coverage_weight=area_spherical_m2)",
            "stdev(coverage_weight=area_spherical_m2)",
            "min",
            "max",
            "count(coverage_weight=area_spherical_km2)",
        ],
        include_cols=["amc_code"],
        output="pandas",
        strategy="raster-sequential",
    ).rename(
        columns={
            "mean": "mean_suitability_0_100",
            "stdev": "sd_suitability_0_100",
            "min": "min_suitability_0_100",
            "max": "max_suitability_0_100",
            "count": "valid_raster_area_km2",
        }
    )
    for column in [
        "mean_suitability_0_100",
        "sd_suitability_0_100",
        "min_suitability_0_100",
        "max_suitability_0_100",
    ]:
        result[column] = result[column] / 100.0
    result.insert(1, "crop_code", crop_code)
    result.insert(2, "crop_name", CROPS[crop_code])
    result.insert(3, "input_level", "high" if input_code == "H" else "low")
    result["water_supply"] = "rainfed"
    result["climate_source"] = "CRUTS32 historical"
    result["climate_period"] = "1961-1990"
    result["gaez_variable"] = "sx: suitability index, all land in grid cell"
    result["source_file"] = path.name
    result["source_url"] = source_url(path, input_code)
    return result


def build_wide(long: pd.DataFrame) -> pd.DataFrame:
    means = long.pivot(
        index="amc_code",
        columns=["crop_code", "input_level"],
        values="mean_suitability_0_100",
    )
    means.columns = [f"gaez_{crop}_{level}_mean_0_100" for crop, level in means]
    wide = means.reset_index()
    for level in ("high", "low"):
        columns = [f"gaez_{crop}_{level}_mean_0_100" for crop in CROPS]
        wide[f"gaez_core8_{level}_mean_0_100"] = wide[columns].mean(axis=1)
        wide[f"gaez_core8_{level}_max_0_100"] = wide[columns].max(axis=1)
        wide[f"gaez_core8_{level}_min_0_100"] = wide[columns].min(axis=1)
    wide["gaez_core8_high_minus_low_mean_0_100"] = (
        wide["gaez_core8_high_mean_0_100"] - wide["gaez_core8_low_mean_0_100"]
    )
    return wide


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    amcs = gpd.read_file(args.amc_gpkg).to_crs("EPSG:4326")
    amcs["amc_code"] = amcs["code_amc"].astype(int).astype(str)
    if len(amcs) != 3800 or amcs["amc_code"].nunique() != 3800:
        raise RuntimeError("Unexpected AMC geography dimensions")
    amcs = amcs[["amc_code", "geometry"]]

    rasters = sorted(args.raster_dir.glob("sx?r0_*.tif"))
    expected = {
        f"sx{input_code}r0_{crop}.tif"
        for crop in CROPS
        for input_code in ("H", "L")
    }
    found = {path.name for path in rasters}
    if found != expected:
        raise RuntimeError(
            f"GAEZ raster set mismatch; missing={sorted(expected-found)}, "
            f"unexpected={sorted(found-expected)}"
        )

    pieces = []
    for path in rasters:
        print(f"Extracting {path.name}", flush=True)
        pieces.append(extract_one(path, amcs))
    long = pd.concat(pieces, ignore_index=True).sort_values(
        ["amc_code", "crop_code", "input_level"]
    )
    long["amc_code"] = long["amc_code"].astype("string")
    wide = build_wide(long)

    expected_rows = 3800 * len(CROPS) * 2
    if len(long) != expected_rows:
        raise RuntimeError(f"Unexpected long-table rows: {len(long)}")
    if long["mean_suitability_0_100"].isna().any():
        raise RuntimeError("Missing AMC suitability means")
    if not long["mean_suitability_0_100"].between(0, 100).all():
        raise RuntimeError("Suitability mean outside 0--100")
    if len(wide) != 3800 or wide["amc_code"].nunique() != 3800:
        raise RuntimeError("Unexpected wide-table AMC coverage")

    long_path = args.output_dir / "gaez_v4_6190_amc_crop_suitability.parquet"
    wide_path = args.output_dir / "gaez_v4_6190_amc_crop_suitability_wide.parquet"
    csv_path = args.output_dir / "gaez_v4_6190_amc_crop_suitability.csv.gz"
    long.to_parquet(long_path, index=False)
    wide.to_parquet(wide_path, index=False)
    long.to_csv(csv_path, index=False, compression="gzip", float_format="%.8f")

    metadata = {
        "created": date.today().isoformat(),
        "dataset": "FAO/IIASA GAEZ v4 AMC crop suitability",
        "unit_of_observation_long": "AMC x crop x input level",
        "unit_of_observation_wide": "AMC",
        "merge_key": "amc_code (string)",
        "amc_count": 3800,
        "long_rows": len(long),
        "crops": CROPS,
        "climate_period": "1961-1990 30-year historical normal; not annual data",
        "climate_source": "CRU TS 3.2",
        "water_supply": "Rainfed",
        "input_levels": ["high", "low"],
        "variable": "sx continuous all-land suitability index",
        "source_scale": "0-10000, rescaled in outputs to 0-100",
        "spatial_resolution": "5 arc-minutes (about 9 km at the equator)",
        "aggregation": (
            "Exact polygon/raster intersection with spherical-area weighting; "
            "nodata pixels excluded"
        ),
        "primary_recommendation": (
            "Use crop-specific high-input rainfed suitability (especially soybean) "
            "as the main predetermined control; use low input and the core-eight "
            "composite for robustness. Do not interpret this as an annual panel."
        ),
        "license": "CC BY-NC-SA 4.0",
        "source_pages": [
            "https://www.fao.org/gaez/gaezv4/en",
            "https://gaez-v4-data.fao.org/data/GAEZ%20v4%20data%20repository%20user%20guide.pdf",
            "https://gaez-services.fao.org/server/rest/services/res05/ImageServer",
        ],
        "files": {
            path.name: {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "url": source_url(path, FILE_RE.match(path.name).group("input")),
            }
            for path in rasters
        },
        "outputs": [long_path.as_posix(), wide_path.as_posix(), csv_path.as_posix()],
    }
    (args.output_dir / "gaez_v4_6190_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {long_path} ({len(long):,} rows)", flush=True)
    print(f"Wrote {wide_path} ({len(wide):,} rows)", flush=True)


if __name__ == "__main__":
    main()
