"""Run streaming integrity checks on the three characteristics datasets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("data/processed/censo_microdados/characteristics_panels")
YEARS = (1991, 2000, 2010)
FRONTIERS = ("north_only", "main_north_mt", "broad_north_mt_ms")
THRESHOLDS = ("500k", "1m")
VALID_SYSTEMS = {"stayer", "large_city", "agricultural_frontier", "other"}
ALTERNATIVES = ("stayer", "city", "agri_frontier", "other")
PRIOR_FLOW_YEAR = {1991: 1980, 2000: 1991, 2010: 2000}
ACCESSIBILITY_COLUMNS = [
    *(f"prev_corridor_{value}" for value in ALTERNATIVES),
    *(f"min_tt_to_{value}" for value in ALTERNATIVES),
    *(f"med_tt_to_{value}" for value in ALTERNATIVES),
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    build_metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    corridor_all = pd.read_stata(
        "data/morten_oliveira_final_tables/N_od_meso.dta",
        columns=["year", "orig_id_meso", "dest_id_meso", "N_od_flow_all"],
    )
    report: dict[str, object] = {"years": {}, "status": "in_progress"}
    for year in YEARS:
        path = ROOT / f"persons_{year}.parquet"
        parquet = pq.ParquetFile(path)
        expected_rows = build_metadata["diagnostics"][str(year)]["rows"]
        require(
            parquet.metadata.num_rows == expected_rows,
            f"{year}: Parquet rows differ from build metadata",
        )
        accessibility = pd.read_parquet(
            ROOT / f"migration_system_accessibility_{year}.parquet"
        )
        require(len(accessibility) == 137, f"{year}: accessibility lookup is not 137 origins")
        require(
            not accessibility["origin_mesoregion_code"].duplicated().any(),
            f"{year}: duplicate accessibility origins",
        )
        require(
            accessibility[ACCESSIBILITY_COLUMNS].notna().all().all(),
            f"{year}: missing accessibility lookup values",
        )
        for alternative in ALTERNATIVES:
            require(
                accessibility[f"min_tt_to_{alternative}"].le(
                    accessibility[f"med_tt_to_{alternative}"]
                ).all(),
                f"{year}: minimum exceeds median for {alternative}",
            )
        require(
            accessibility[["min_tt_to_stayer", "med_tt_to_stayer"]].eq(0).all().all(),
            f"{year}: stayer travel accessibility is not zero",
        )
        prior = corridor_all[corridor_all["year"] == PRIOR_FLOW_YEAR[year]].copy()
        prior["origin"] = prior["orig_id_meso"].astype("int64").astype("string")
        raw_total = prior.groupby("origin")["N_od_flow_all"].sum()
        raw_stayer = prior.loc[
            prior["orig_id_meso"].eq(prior["dest_id_meso"])
        ].set_index("origin")["N_od_flow_all"]
        lookup = accessibility.set_index("origin_mesoregion_code")
        require(
            np.allclose(
                lookup[[f"prev_corridor_{value}" for value in ALTERNATIVES]].sum(axis=1),
                raw_total.reindex(lookup.index),
            ),
            f"{year}: alternative corridor totals do not exhaust raw flows",
        )
        require(
            np.allclose(lookup["prev_corridor_stayer"], raw_stayer.reindex(lookup.index)),
            f"{year}: stayer corridor does not equal raw diagonal flow",
        )
        rows = observed_rows = travel_rows = zone_rows = 0
        characteristics = {"sex": 0, "race": 0, "education_attainment": 0}
        systems = {
            f"{threshold}_{frontier}": {key: 0 for key in sorted(VALID_SYSTEMS)}
            for threshold in THRESHOLDS
            for frontier in FRONTIERS
        }
        seen_sources: set[str] = set()
        active_source: str | None = None
        last_source_row = 0

        for batch in parquet.iter_batches(batch_size=300_000):
            frame = batch.to_pandas()
            rows += len(frame)
            require(frame["current_municipality_code"].notna().all(), f"{year}: missing D municipality")
            require(frame["current_mesoregion_code"].notna().all(), f"{year}: missing D mesoregion")
            require(frame["previous_census_population"].notna().all(), f"{year}: missing lagged population")
            require(
                frame["destination_large_city_500k"].eq(
                    frame["previous_census_population"].ge(500_000)
                ).all(),
                f"{year}: 500k flag disagrees with population",
            )
            require(
                frame["destination_large_city_1m"].eq(
                    frame["previous_census_population"].ge(1_000_000)
                ).all(),
                f"{year}: 1m flag disagrees with population",
            )
            require(
                (~frame["destination_frontier_north_only"] | frame["destination_frontier_main_north_mt"]).all(),
                f"{year}: frontier nesting failure (North -> main)",
            )
            require(
                (~frame["destination_frontier_main_north_mt"] | frame["destination_frontier_broad_north_mt_ms"]).all(),
                f"{year}: frontier nesting failure (main -> broad)",
            )

            observed = frame["five_year_internal_od_observed"].fillna(False)
            observed_rows += int(observed.sum())
            require(frame.loc[observed, "origin_5yr_municipality_code"].notna().all(), f"{year}: observed O municipality missing")
            require(frame.loc[observed, "origin_5yr_mesoregion_code"].notna().all(), f"{year}: observed O mesoregion missing")
            require(frame.loc[observed, "mo_fm_road"].notna().all(), f"{year}: observed travel measure missing")
            require(frame.loc[~observed, "mo_fm_road"].isna().all(), f"{year}: travel measure outside valid OD universe")
            expected_accessibility = frame["origin_5yr_mesoregion_code"].to_frame().join(
                lookup[ACCESSIBILITY_COLUMNS],
                on="origin_5yr_mesoregion_code",
            )
            for column in ACCESSIBILITY_COLUMNS:
                require(
                    np.allclose(
                        frame[column].to_numpy(dtype="float64", na_value=np.nan),
                        expected_accessibility[column].to_numpy(dtype="float64", na_value=np.nan),
                        equal_nan=True,
                    ),
                    f"{year}: person-level {column} disagrees with origin lookup",
                )
            travel_rows += int(frame["mo_fm_road"].notna().sum())
            zone_rows += int(frame["origin_5yr_urban_rural"].notna().sum())
            if year == 2010:
                require(frame["origin_5yr_urban_rural"].isna().all(), "2010: origin zone must be unavailable")

            stayer = frame["stayer_5yr"].fillna(False)
            for threshold in THRESHOLDS:
                large = frame[
                    "destination_large_city_500k" if threshold == "500k" else "destination_large_city_1m"
                ].fillna(False)
                for frontier in FRONTIERS:
                    column = f"migration_system_{threshold}_{frontier}"
                    value = frame[column]
                    require(value.loc[observed].isin(VALID_SYSTEMS).all(), f"{year}: invalid or missing {column}")
                    require(value.loc[~observed].isna().all(), f"{year}: {column} outside OD universe")
                    frontier_flag = frame[f"destination_frontier_{frontier}"].fillna(False)
                    expected = pd.Series("other", index=frame.index, dtype="string")
                    expected.loc[~stayer & large] = "large_city"
                    expected.loc[~stayer & ~large & frontier_flag] = "agricultural_frontier"
                    expected.loc[stayer] = "stayer"
                    require(
                        value.loc[observed].astype("string").eq(expected.loc[observed]).all(),
                        f"{year}: precedence failure in {column}",
                    )
                    for key, count in value.value_counts().items():
                        systems[f"{threshold}_{frontier}"][str(key)] += int(count)

            for column in characteristics:
                characteristics[column] += int(frame[column].notna().sum())

            # Source rows must be strictly consecutive within each contiguous
            # source file, and person IDs must include that file's stem.
            for source, group in frame.groupby("source_file", sort=False):
                source = str(source)
                if source != active_source:
                    require(source not in seen_sources, f"{year}: source file is noncontiguous: {source}")
                    seen_sources.add(source)
                    active_source = source
                    last_source_row = 0
                source_rows = group["source_row_in_file"].astype("int64")
                require(int(source_rows.iloc[0]) == last_source_row + 1, f"{year}: source-row gap/duplicate in {source}")
                require(source_rows.is_monotonic_increasing, f"{year}: source rows not increasing")
                require(int(source_rows.iloc[-1]) - int(source_rows.iloc[0]) + 1 == len(source_rows), f"{year}: source rows not consecutive")
                last_source_row = int(source_rows.iloc[-1])
                stem = Path(source).stem.lower()
                expected_id = str(year) + "-" + stem + "-" + source_rows.astype("string")
                require(
                    group["person_record_id"].reset_index(drop=True).eq(expected_id.reset_index(drop=True)).all(),
                    f"{year}: person_record_id mismatch in {source}",
                )

        require(rows == expected_rows, f"{year}: streamed row count mismatch")
        require(observed_rows == travel_rows, f"{year}: OD/travel coverage mismatch")
        report["years"][str(year)] = {
            "rows": rows,
            "source_files": len(seen_sources),
            "five_year_internal_od_observed_rows": observed_rows,
            "morten_oliveira_rows": travel_rows,
            "origin_urban_rural_nonmissing_rows": zone_rows,
            "characteristic_nonmissing_rows": characteristics,
            "migration_system_counts": systems,
            "choice_set_accessibility": {
                "origin_mesoregions": int(len(accessibility)),
                "prior_flow_year": PRIOR_FLOW_YEAR[year],
                "columns": ACCESSIBILITY_COLUMNS,
                "person_rows_with_values": observed_rows,
            },
        }

    crosswalk = pd.read_parquet(ROOT / "municipality_population_crosswalk.parquet")
    require(crosswalk["previous_population"].notna().all(), "Population crosswalk has missing values")
    require(
        not crosswalk.duplicated(["target_census_year", "municipality_code"]).any(),
        "Population crosswalk has duplicate target municipalities",
    )
    fallback = crosswalk[
        crosswalk["population_assignment_method"] == "predecessor_max_area_overlap"
    ]
    report["population_crosswalk"] = {
        "rows": int(len(crosswalk)),
        "fallback_rows": int(len(fallback)),
        "minimum_fallback_overlap_share": float(fallback["predecessor_overlap_share"].min()),
        "median_fallback_overlap_share": float(fallback["predecessor_overlap_share"].median()),
        "fallback_large_city_rows": int(fallback["large_city_500k"].sum()),
    }
    report["status"] = "passed"
    (ROOT / "validation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
