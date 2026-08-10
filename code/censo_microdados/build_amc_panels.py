"""Build migration, socioeconomic, and GDP panels on 1970-2010 AMCs.

The script creates four analysis tables suitable for program-evaluation work:

1. destination AMC x assigned origin UF x census-year migration cells;
2. destination AMC x census-year resident characteristics;
3. municipality x benchmark-year IPEA GDP and sector value added;
4. an outer AMC x year panel joining census characteristics and aggregate GDP.

The 1970 and 1980 migration measures are explicitly proxies. In 1970 a person
is an interstate migrant when residence in the current UF is under five years
and the previous UF differs. In 1980 the same duration restriction is combined
with UF of birth. The later censuses use the fixed-date five-year UF. People
aged five or older who are not identified as interstate migrants are assigned
to the current-UF residual cell; that cell is not interpreted as "stayers"
because it includes intrastate migrants.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pdfplumber
import pyarrow.dataset as ds
import pyreadr

from migration_profiles import (
    INCOME_COLUMN,
    INCOME_UPPER_EXCLUSIVE,
    STATE_TO_INDEX,
    STATES,
    map_current,
    map_origin,
)


CENSUS_YEARS = (1970, 1980, 1991, 2000, 2010)
GDP_YEARS = (1959, 1970, 1975, 1980, 1985, 1996, 2000, 2005, 2010)
# IPEA's 1959 workbook uses an inactive IBGE code for Cococi. The geobr
# genealogy retains Cococi under its special historical code, which belongs to
# the same 1970-2010 AMC as Taua. Keep the recode explicit and auditable.
GDP_HISTORICAL_CODE_ALIASES = {"2310298": "2399903"}
MIGRATION_MEASURE = {
    1970: "previous_state_recent_proxy",
    1980: "birth_state_recent_proxy",
    1991: "fixed_date_5yr",
    2000: "fixed_date_5yr",
    2010: "fixed_date_5yr",
}
SUCCESSOR_UFS_1970 = {
    "RO": {"11"}, "AC": {"12"}, "AM": {"13"}, "RR": {"14"},
    "PA": {"15"}, "AP": {"16"}, "MA": {"21"}, "PI": {"22"},
    "CE": {"23"}, "RN": {"24"}, "PB": {"25"}, "PE": {"26"},
    "FN": {"26"}, "AL": {"27"}, "SE": {"28"}, "BA": {"29"},
    "MG": {"31"}, "ES": {"32"}, "RJ": {"33"}, "GB": {"33"},
    "SP": {"35"}, "PR": {"41"}, "SC": {"42"}, "RS": {"43"},
    "MT": {"50", "51"}, "GO": {"17", "52"}, "DF": {"53"},
}
# Historical names/codes that changed and therefore cannot be recovered by an
# exact name match against the genealogy. Values are the documented successor
# municipality codes shown in IPEA's 1970 correspondence table. Keeping this
# short override explicit makes every non-exact link auditable.
MANUAL_1970_SUCCESSOR = {
    "02501": "1303205", "02503": "1303601", "92207": "1716653",
    "92415": "1721307", "10104": "2106201", "10704": "2111409",
    "10609": "2108108", "12208": "2203420", "17612": "2403251",
    "17712": "2407609", "17816": "2411429", "20601": "2500700",
    "20702": "2502607", "20307": "2516409", "23101": "2602902",
    "23010": "2607604", "23102": "2607901", "22107": "2614303",
    "26304": "2804458", "30602": "2900504", "30714": "2919504",
    "42002": "3131802", "40302": "3155603", "51301": "3301009",
    "61403": "3507803", "63301": "3509601", "62006": "3515186",
    "62804": "3520905", "63610": "3543501", "72107": "4108205",
    "72301": "4120903", "71514": "4125308", "80208": "4212809",
    "85302": "4317608", "90202": "5105507", "92804": "5208301",
}
SOURCE_COLUMNS = (
    "current_uf", "current_uf_code", "current_municipality_code",
    "current_micro_code", "person_weight", "age_years", "sex", "race_code",
    "literacy_code", "education_years", "education_level_code", "income_main",
    "income_total", "urban_code", "household_size", "rooms", "bedrooms",
    "bathrooms", "refrigerator_code", "automobile_code", "electricity_code",
    "labor_force_code", "employment_status_code", "years_uf", "birth_uf_code",
    "last_origin_uf_code", "origin_5yr_uf_code", "internal_migrant_5yr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persons", type=Path,
                        default=Path("data/processed/censo_microdados/persons"))
    parser.add_argument("--amc-gpkg", type=Path, default=Path(
        "data/censo_microdados/amc/AMC_1970_2010_simplified.gpkg"))
    parser.add_argument("--geobr-crosswalk", type=Path, default=Path(
        "data/censo_microdados/amc/geobr_Crosswalk_pre.rds"))
    parser.add_argument("--ipea-1970-codebook", type=Path, default=Path(
        "data/censo_microdados/amc/codAMC0070.pdf"))
    parser.add_argument("--boundaries", type=Path,
                        default=Path("data/censo_microdados/ibge_boundaries"))
    parser.add_argument("--gdp", type=Path,
                        default=Path("data/ipea/PIB_Municipal"))
    parser.add_argument("--income-cutoffs", type=Path, default=Path(
        "data/processed/censo_microdados/migration_profiles/"
        "income_quintile_cutoffs.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/processed/amcs"))
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--sample-batches", type=int)
    parser.add_argument("--rebuild-crosswalk", action="store_true")
    parser.add_argument(
        "--gdp-only", action="store_true",
        help="Rebuild only municipal GDP, the combined AMC-year panel, and metadata.",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", " ", text.upper()).strip()


def clean_code(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def load_amcs(path: Path) -> tuple[gpd.GeoDataFrame, dict[str, str], pd.DataFrame]:
    frame = gpd.read_file(path)
    frame["amc_code"] = frame["code_amc"].map(lambda value: str(int(value)))
    rows = []
    municipality_to_amc: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        codes = [code.strip() for code in str(row.list_code_muni_2010).split(",")]
        ufs = sorted({code[:2] for code in codes})
        for code in codes:
            if code in municipality_to_amc:
                raise RuntimeError(f"Duplicate 2010 municipality in AMC file: {code}")
            municipality_to_amc[code] = row.amc_code
        rows.append({
            "amc_code": row.amc_code,
            "municipality_count_2010": len(codes),
            "destination_uf_2010_set": ",".join(ufs),
            "municipality_codes_2010": ",".join(codes),
        })
    lookup = pd.DataFrame(rows)
    # geobr retains the historical/special code 2399903 in addition to the
    # 5,565 municipality polygons in IBGE's 2010 boundary release.
    if len(frame) != 3800 or len(municipality_to_amc) != 5566:
        raise RuntimeError("Unexpected 1970-2010 AMC geography dimensions")
    return frame[["amc_code", "geometry"]], municipality_to_amc, lookup


def parse_ipea_1970_names(path: Path) -> pd.DataFrame:
    """Extract the 1970 area/municipality identifier and name from IPEA's PDF."""
    rows: list[dict[str, object]] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            lines: defaultdict[float, list[dict[str, object]]] = defaultdict(list)
            for word in page.extract_words(x_tolerance=1, y_tolerance=2):
                if 55 <= word["top"] < 570:
                    lines[round(float(word["top"]), 1)].append(word)
            for top, words in lines.items():
                words.sort(key=lambda word: float(word["x0"]))
                code_tokens = [
                    str(word["text"]) for word in words
                    if 674 <= float(word["x0"]) < 708
                ]
                if not (
                    len(code_tokens) >= 2
                    and re.fullmatch(r"\d{2}", code_tokens[0])
                    and re.fullmatch(r"\d{3}", code_tokens[1])
                ):
                    continue
                name = " ".join(
                    str(word["text"]) for word in words if float(word["x0"]) >= 708
                )
                rows.append({
                    "source_key": code_tokens[0] + code_tokens[1],
                    "municipality_name_1970": name.strip(" +"),
                    "normalized_name": normalize_name(name),
                    "codebook_page": page_number,
                    "codebook_top": top,
                })
    result = pd.DataFrame(rows).drop_duplicates("source_key")
    if len(rows) != len(result) or len(result) < 3950:
        raise RuntimeError(
            f"Unexpected IPEA 1970 codebook extraction: {len(rows)} rows, "
            f"{len(result)} unique keys"
        )
    return result


def historical_1970_key_to_uf(persons: ds.Dataset, batch_size: int) -> dict[str, str]:
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    scanner = persons.scanner(
        columns=["current_uf", "current_micro_code", "current_municipality_code"],
        filter=ds.field("census_year") == 1970,
        batch_size=batch_size,
    )
    for batch in scanner.to_batches():
        frame = batch.to_pandas()
        key = clean_code(frame["current_micro_code"]).str[:2] + clean_code(
            frame["current_municipality_code"]
        )
        pairs = pd.DataFrame({"source_key": key, "current_uf": frame["current_uf"]})
        for (key, uf), count in pairs.dropna().value_counts().items():
            counts[(str(key), str(uf))] += int(count)
    # A handful of source files end with one malformed non-person record whose
    # character positions resemble a municipality in another UF. The modal UF
    # for each historical key is robust to those sentinels and is audited again
    # through the person-level mapping rate.
    by_key: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    for (key, uf), count in counts.items():
        by_key[key].append((count, uf))
    return {key: max(values)[1] for key, values in by_key.items()}


def build_1970_crosswalk(
    persons: ds.Dataset,
    pdf_path: Path,
    rds_path: Path,
    municipality_to_amc: dict[str, str],
    batch_size: int,
) -> pd.DataFrame:
    names = parse_ipea_1970_names(pdf_path)
    key_to_uf = historical_1970_key_to_uf(persons, batch_size)
    names["current_uf"] = names["source_key"].map(key_to_uf)

    genealogy = next(iter(pyreadr.read_r(str(rds_path)).values())).copy()
    genealogy = genealogy[genealogy["exist_d1970"].astype("string").str.strip() == "1"]
    genealogy["normalized_name"] = genealogy["muname1970"].map(normalize_name)
    genealogy["successor_code"] = clean_code(genealogy["code2020"])
    genealogy["successor_uf"] = genealogy["successor_code"].str[:2]
    genealogy["amc_code"] = genealogy["successor_code"].map(municipality_to_amc)
    genealogy = genealogy[genealogy["amc_code"].notna()]

    by_name_uf: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for row in genealogy.itertuples(index=False):
        by_name_uf[(row.normalized_name, row.successor_uf)].add(row.amc_code)

    all_names_by_uf: defaultdict[str, set[str]] = defaultdict(set)
    for normalized_name, successor_uf in by_name_uf:
        all_names_by_uf[successor_uf].add(normalized_name)

    # Administrative zones in Guanabara and Distrito Federal are not separate
    # municipalities. They all inherit the Rio de Janeiro/Brasilia AMC.
    forced_area_code = {
        "53": municipality_to_amc["3304557"],
        "94": municipality_to_amc["5300108"],
    }
    records = []
    unresolved = []
    for row in names.itertuples(index=False):
        current_uf = row.current_uf
        if row.source_key in MANUAL_1970_SUCCESSOR:
            successor = MANUAL_1970_SUCCESSOR[row.source_key]
            amc = municipality_to_amc.get(successor)
            if amc is None:
                raise RuntimeError(
                    f"Manual 1970 successor is absent from AMC lookup: "
                    f"{row.source_key} -> {successor}"
                )
            method = "documented_successor_override"
        elif row.source_key[:2] in forced_area_code:
            amc = forced_area_code[row.source_key[:2]]
            method = "historical_administrative_zone"
        else:
            allowed = SUCCESSOR_UFS_1970.get(str(current_uf), set())
            candidates = set().union(*(
                by_name_uf.get((row.normalized_name, uf), set()) for uf in allowed
            )) if allowed else set()
            if len(candidates) == 1:
                amc = next(iter(candidates))
                method = "genealogy_exact_name_and_successor_uf"
            else:
                # Historical spelling changes are resolved conservatively by a
                # unique close match within the permitted successor UF(s).
                from difflib import SequenceMatcher
                scored = []
                for uf in allowed:
                    for candidate_name in all_names_by_uf[uf]:
                        score = SequenceMatcher(None, row.normalized_name, candidate_name).ratio()
                        if score >= 0.72:
                            scored.append((score, candidate_name, uf))
                scored.sort(reverse=True)
                candidate_amcs = set()
                if scored:
                    best = scored[0][0]
                    for score, candidate_name, uf in scored:
                        if score < best - 0.025:
                            break
                        candidate_amcs.update(by_name_uf[(candidate_name, uf)])
                if len(candidate_amcs) == 1 and scored and scored[0][0] >= 0.78:
                    amc = next(iter(candidate_amcs))
                    method = "genealogy_fuzzy_name_and_successor_uf"
                else:
                    amc = None
                    method = "unresolved"
                    unresolved.append({
                        "source_key": row.source_key,
                        "name": row.municipality_name_1970,
                        "current_uf": current_uf,
                        "candidate_amcs": sorted(candidates or candidate_amcs),
                        "best_matches": scored[:3],
                    })
        records.append({
            "census_year": 1970,
            "source_key": row.source_key,
            "municipality_code_vintage": pd.NA,
            "municipality_name_vintage": row.municipality_name_1970,
            "current_uf": current_uf,
            "amc_code": amc,
            "mapping_method": method,
        })
    represented = {record["source_key"] for record in records}
    # The 1970 files enumerate Brasilia and Guanabara administrative zones,
    # while the IPEA municipality codebook has only their municipality-level
    # entry. Add the remaining observed zones to their containing AMC.
    for key, current_uf in key_to_uf.items():
        if key in represented or key[:2] not in forced_area_code:
            continue
        records.append({
            "census_year": 1970,
            "source_key": key,
            "municipality_code_vintage": pd.NA,
            "municipality_name_vintage": (
                "BRASILIA_ADMINISTRATIVE_ZONE" if key[:2] == "94"
                else "GUANABARA_ADMINISTRATIVE_ZONE"
            ),
            "current_uf": current_uf,
            "amc_code": forced_area_code[key[:2]],
            "mapping_method": "historical_administrative_zone",
        })
    result = pd.DataFrame(records)
    if unresolved:
        # Keep the audit trail, but do not silently accept ambiguous fuzzy links.
        print(f"1970 unresolved codebook entries: {len(unresolved)}")
        for item in unresolved[:30]:
            print("  ", item)
    return result


def read_historical_boundaries(root: Path, year: int) -> gpd.GeoDataFrame:
    parts = []
    code_column = {
        1980: "codigo", 1991: "BR91POLY_I", 2000: "GEOCODIGO", 2010: "CD_GEOCODM"
    }[year]
    name_column = {
        1980: "nome", 1991: "NOMEMUNICP", 2000: "NOME", 2010: "NM_MUNICIP"
    }[year]
    for path in sorted((root / str(year)).glob("*.zip")):
        frame = gpd.read_file("zip://" + str(path.resolve()))
        if year == 2000 and frame.crs is None:
            frame = frame.set_crs("EPSG:4618")
        frame = frame.rename(columns={
            code_column: "municipality_code_vintage",
            name_column: "municipality_name_vintage",
        })
        parts.append(frame[["municipality_code_vintage", "municipality_name_vintage", "geometry"]])
    if not parts:
        raise FileNotFoundError(f"No {year} boundary files under {root}")
    result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    result["municipality_code_vintage"] = clean_code(
        result["municipality_code_vintage"]
    ).str.zfill(7)
    return result.dissolve(by="municipality_code_vintage", as_index=False, aggfunc="first")


def build_later_crosswalks(
    amcs: gpd.GeoDataFrame,
    municipality_to_amc: dict[str, str],
    boundary_root: Path,
) -> pd.DataFrame:
    amc_projected = amcs.to_crs("EPSG:5880")
    outputs = []
    for year in (1980, 1991, 2000, 2010):
        municipalities = read_historical_boundaries(boundary_root, year).to_crs("EPSG:5880")
        municipalities["amc_code"] = municipalities["municipality_code_vintage"].map(
            municipality_to_amc
        )
        municipalities["mapping_method"] = "direct_continuing_code"
        missing = municipalities["amc_code"].isna()
        if missing.any():
            points = municipalities.loc[missing, ["municipality_code_vintage", "geometry"]].copy()
            points.geometry = points.geometry.representative_point()
            joined = gpd.sjoin(points, amc_projected, how="left", predicate="within")
            # Simplification can put a point on two polygons. Resolve those few
            # cases by maximum intersection area using the original polygon.
            for code, candidates in joined.groupby("municipality_code_vintage"):
                candidate_codes = candidates["amc_code"].dropna().astype("string").unique()
                if len(candidate_codes) == 1:
                    chosen = str(candidate_codes[0])
                else:
                    geometry = municipalities.set_index("municipality_code_vintage").at[code, "geometry"]
                    areas = {
                        candidate: geometry.intersection(
                            amc_projected.set_index("amc_code").at[candidate, "geometry"]
                        ).area
                        for candidate in candidate_codes
                    }
                    chosen = max(areas, key=areas.get) if areas else None
                mask = municipalities["municipality_code_vintage"] == code
                municipalities.loc[mask, "amc_code"] = chosen
                municipalities.loc[mask, "mapping_method"] = "representative_point_to_amc"
        invalid_special = municipalities["municipality_code_vintage"].isin(
            {"0000000", "9999910", "9999920", "4300001", "4300002"}
        )
        municipalities.loc[invalid_special, "mapping_method"] = "invalid_special_code"
        municipalities.loc[invalid_special, "amc_code"] = pd.NA
        valid = municipalities[~invalid_special]
        if valid["amc_code"].isna().any():
            missing_codes = valid.loc[valid["amc_code"].isna(), "municipality_code_vintage"].tolist()
            raise RuntimeError(f"Unmapped {year} municipalities: {missing_codes}")
        if year in (1980, 1991):
            source_key = municipalities["municipality_code_vintage"].str[:6]
        else:
            source_key = municipalities["municipality_code_vintage"]
        output = pd.DataFrame({
            "census_year": year,
            "source_key": source_key,
            "municipality_code_vintage": municipalities["municipality_code_vintage"],
            "municipality_name_vintage": municipalities["municipality_name_vintage"],
            "current_uf": municipalities["municipality_code_vintage"].str[:2],
            "amc_code": municipalities["amc_code"],
            "mapping_method": municipalities["mapping_method"],
        })
        output = output.drop_duplicates("source_key")
        # The 1991 Ceará person file uses the official Itapipoca key 230640,
        # while the historical boundary layer transposes the last two digits
        # as 230604. Retain the boundary row for spatial auditability and add
        # the person-file alias explicitly.
        if year == 1991:
            itapipoca = output[output["source_key"] == "230604"].copy()
            if len(itapipoca) != 1:
                raise RuntimeError("Expected one 1991 Itapipoca boundary record (230604)")
            itapipoca["source_key"] = "230640"
            itapipoca["mapping_method"] = "documented_source_code_alias"
            output = pd.concat([output, itapipoca], ignore_index=True)
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True)


def source_key(frame: pd.DataFrame, year: int) -> pd.Series:
    muni = clean_code(frame["current_municipality_code"])
    if year == 1970:
        return clean_code(frame["current_micro_code"]).str[:2] + muni
    if year in (1980, 1991):
        return clean_code(frame["current_uf_code"]).str[:2] + muni.str.zfill(4)
    return muni.str.zfill(7)


def boolean_indicator(code: pd.Series, yes: set[str], no: set[str]) -> tuple[np.ndarray, np.ndarray]:
    value = clean_code(code)
    valid = value.isin(yes | no).to_numpy(dtype=bool)
    indicator = value.isin(yes).to_numpy(dtype=bool)
    return valid, indicator


def add_binary(
    output: pd.DataFrame,
    prefix: str,
    weight: np.ndarray,
    eligible: np.ndarray,
    valid: np.ndarray,
    value: np.ndarray,
) -> None:
    keep = eligible & valid
    output[f"{prefix}__den"] = np.where(keep, weight, 0.0)
    output[f"{prefix}__num"] = np.where(keep & value, weight, 0.0)


def add_mean(
    output: pd.DataFrame,
    prefix: str,
    weight: np.ndarray,
    eligible: np.ndarray,
    values: np.ndarray,
    lower: float,
    upper: float,
) -> None:
    keep = eligible & np.isfinite(values) & (values >= lower) & (values <= upper)
    output[f"{prefix}__den"] = np.where(keep, weight, 0.0)
    output[f"{prefix}__num"] = np.where(keep, weight * np.nan_to_num(values), 0.0)


def contributions(
    frame: pd.DataFrame,
    year: int,
    cutoffs: np.ndarray,
    universe: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(frame)
    weight = pd.to_numeric(frame["person_weight"], errors="coerce").to_numpy(float)
    valid_weight = np.isfinite(weight) & (weight > 0)
    if universe is not None:
        valid_weight &= universe
    weight = np.where(valid_weight, weight, 0.0)
    age = pd.to_numeric(frame["age_years"], errors="coerce").to_numpy(float)
    valid_age = np.isfinite(age) & (age >= 0) & (age <= 120)
    age15 = valid_age & (age >= 15)
    age25 = valid_age & (age >= 25)
    output = pd.DataFrame({
        "weighted_population": weight,
        "sample_persons": valid_weight.astype(np.int64),
    })
    add_mean(output, "mean_age", weight, valid_weight, age, 0, 120)
    for label, mask in {
        "share_age_0_4": valid_age & (age < 5),
        "share_age_5_14": valid_age & (age >= 5) & (age < 15),
        "share_age_15_24": valid_age & (age >= 15) & (age < 25),
        "share_age_25_64": valid_age & (age >= 25) & (age < 65),
        "share_age_65plus": valid_age & (age >= 65),
    }.items():
        add_binary(output, label, weight, valid_weight, valid_age, mask)

    sex = frame["sex"].astype("string")
    add_binary(
        output, "female_share", weight, valid_weight,
        sex.isin(("female", "male")).to_numpy(),
        sex.eq("female").fillna(False).to_numpy(),
    )
    race = clean_code(frame["race_code"])
    race_valid = race.isin(("1", "2", "3", "4", "5")).to_numpy()
    for code, label in (("1", "white"), ("2", "black"), ("3", "asian"),
                        ("4", "pardo"), ("5", "indigenous")):
        add_binary(output, f"race_{label}_share", weight, valid_weight,
                   race_valid, race.eq(code).fillna(False).to_numpy())

    literacy_yes_no = {
        1970: ({"1"}, {"2"}), 1980: ({"2"}, {"4", "6"}),
        1991: ({"1"}, {"2"}), 2000: ({"1"}, {"2"}), 2010: ({"1"}, {"2"}),
    }[year]
    literacy_valid, literacy_yes = boolean_indicator(
        frame["literacy_code"], *literacy_yes_no
    )
    add_binary(output, "literacy_share_age15plus", weight, valid_weight & age15,
               literacy_valid, literacy_yes)

    schooling = pd.to_numeric(frame["education_years"], errors="coerce").to_numpy(float)
    add_mean(output, "mean_education_years_age25plus", weight,
             valid_weight & age25, schooling, 0, 30)
    if year in (1991, 2000):
        attainment = np.select(
            [np.isfinite(schooling) & (schooling < 8),
             np.isfinite(schooling) & (schooling >= 8) & (schooling <= 10),
             np.isfinite(schooling) & (schooling >= 11) & (schooling <= 14),
             np.isfinite(schooling) & (schooling >= 15) & (schooling <= 30)],
            ["less_fundamental", "fundamental", "secondary", "tertiary"], default="",
        )
    elif year == 2010:
        attainment = clean_code(frame["education_level_code"]).map({
            "1": "less_fundamental", "2": "fundamental", "3": "secondary",
            "4": "tertiary",
        }).fillna("").to_numpy()
    else:
        attainment = np.full(n, "", dtype=object)
    attainment_valid = attainment != ""
    for label in ("less_fundamental", "fundamental", "secondary", "tertiary"):
        add_binary(output, f"education_{label}_share_age25plus", weight,
                   valid_weight & age25, attainment_valid, attainment == label)

    income = pd.to_numeric(frame[INCOME_COLUMN[year]], errors="coerce").to_numpy(float)
    income_valid = (
        np.isfinite(income) & (income >= 0) &
        (income < INCOME_UPPER_EXCLUSIVE[year])
    )
    add_mean(output, "mean_income_age15plus", weight, valid_weight & age15,
             income, 0, INCOME_UPPER_EXCLUSIVE[year] - 1)
    add_mean(output, "mean_positive_income_age15plus", weight,
             valid_weight & age15 & income_valid & (income > 0), income, 0,
             INCOME_UPPER_EXCLUSIVE[year] - 1)
    add_binary(output, "zero_income_share_age15plus", weight, valid_weight & age15,
               income_valid, income == 0)
    positive = income_valid & (income > 0)
    left = np.searchsorted(cutoffs, np.nan_to_num(income), side="left")
    right = np.searchsorted(cutoffs, np.nan_to_num(income), side="right")
    quintile = 1 + ((left + right) // 2)
    for q in range(1, 6):
        add_binary(output, f"positive_income_q{q}_share_age15plus", weight,
                   valid_weight & age15, positive, quintile == q)

    for column, label, upper in (
        ("household_size", "mean_household_size", 30),
        ("rooms", "mean_rooms", 50), ("bedrooms", "mean_bedrooms", 30),
        ("bathrooms", "mean_bathrooms", 20),
    ):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        add_mean(output, label, weight, valid_weight, values, 0, upper)

    appliance_maps = {
        "refrigerator": {
            1970: ({"1"}, {"2"}), 1980: ({"1"}, {"8"}),
            1991: ({"1", "2", "3", "4"}, {"0"}),
            2000: ({"1"}, {"2"}), 2010: ({"1"}, {"2"}),
        },
        "automobile": {
            1970: ({"1"}, {"2"}), 1980: ({"1", "3"}, {"8"}),
            1991: (set(), set()), 2000: (set(), set()),
            2010: ({"1"}, {"2"}),
        },
        "electricity": {
            1970: ({"1"}, {"2"}), 1980: ({"2"}, {"4", "8"}),
            1991: ({"1", "2"}, {"3", "4"}),
            2000: ({"1"}, {"2"}), 2010: ({"1", "2"}, {"3"}),
        },
    }
    source_column = {
        "refrigerator": "refrigerator_code",
        "automobile": "automobile_code", "electricity": "electricity_code",
    }
    for label, mappings in appliance_maps.items():
        valid, yes = boolean_indicator(frame[source_column[label]], *mappings[year])
        add_binary(output, f"{label}_share", weight, valid_weight, valid, yes)

    urban_maps = {
        1970: ({"0"}, {"1", "2"}), 1980: (set(), set()),
        1991: ({"1", "2", "3"}, {"4", "5", "6", "7", "8"}),
        2000: ({"1"}, {"2"}), 2010: ({"1"}, {"2"}),
    }
    valid, yes = boolean_indicator(frame["urban_code"], *urban_maps[year])
    add_binary(output, "urban_share", weight, valid_weight, valid, yes)

    lf_valid, lf_yes = boolean_indicator(frame["labor_force_code"], {"1"}, {"2"})
    add_binary(output, "labor_force_share_age15plus", weight, valid_weight & age15,
               lf_valid, lf_yes)
    emp_valid, emp_yes = boolean_indicator(frame["employment_status_code"], {"1"}, {"2"})
    add_binary(output, "employment_share_age15plus", weight, valid_weight & age15,
               emp_valid, emp_yes)
    return output


def collapse(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True)
    numeric = [column for column in combined.columns if column not in keys]
    return combined.groupby(keys, observed=True, dropna=False)[numeric].sum().reset_index()


def finalize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    prefixes = sorted({column[:-5] for column in result if column.endswith("__den")})
    for prefix in prefixes:
        denominator = result.pop(prefix + "__den")
        numerator = result.pop(prefix + "__num")
        result[prefix] = numerator.div(denominator.where(denominator > 0))
        result[prefix + "_valid_weighted_population"] = denominator
    return result


def migration_assignment(
    frame: pd.DataFrame, year: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_index = map_current(frame["current_uf"], year)
    age = pd.to_numeric(frame["age_years"], errors="coerce").to_numpy(float)
    weight = pd.to_numeric(frame["person_weight"], errors="coerce").to_numpy(float)
    universe = (
        np.isfinite(age) & (age >= 5) & (age <= 120) &
        np.isfinite(weight) & (weight > 0) & (current_index >= 0)
    )
    if year == 1970:
        origin_index = map_origin(frame["last_origin_uf_code"], year)
        duration = pd.to_numeric(frame["years_uf"], errors="coerce").to_numpy(float)
        migrant = universe & np.isfinite(duration) & (duration >= 0) & (duration < 5)
        migrant &= (origin_index >= 0) & (origin_index != current_index)
    elif year == 1980:
        origin_index = map_origin(frame["birth_uf_code"], year)
        duration = pd.to_numeric(frame["years_uf"], errors="coerce").to_numpy(float)
        migrant = universe & np.isfinite(duration) & (duration >= 0) & (duration < 5)
        migrant &= (origin_index >= 0) & (origin_index != current_index)
    else:
        origin_index = map_origin(frame["origin_5yr_uf_code"], year)
        internal = frame["internal_migrant_5yr"].astype("boolean").fillna(False).to_numpy(bool)
        migrant = universe & internal & (origin_index >= 0) & (origin_index != current_index)
    assigned = current_index.copy()
    assigned[migrant] = origin_index[migrant]
    return universe, migrant, assigned


def build_census_tables(
    persons: ds.Dataset,
    crosswalk: pd.DataFrame,
    amc_lookup: pd.DataFrame,
    cutoffs: dict[int, np.ndarray],
    batch_size: int,
    sample_batches: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    flow_years = []
    characteristic_years = []
    diagnostics: dict[str, object] = {}
    amc_meta = amc_lookup.set_index("amc_code")
    for year in CENSUS_YEARS:
        print(f"Aggregating census {year}...", flush=True)
        year_crosswalk = crosswalk[crosswalk["census_year"] == year]
        source_to_amc = year_crosswalk.set_index("source_key")["amc_code"]
        flow_parts: list[pd.DataFrame] = []
        characteristic_parts: list[pd.DataFrame] = []
        rows = mapped_rows = flow_universe_rows = migrant_rows = 0
        scanner = persons.scanner(
            columns=list(SOURCE_COLUMNS),
            filter=ds.field("census_year") == year,
            batch_size=batch_size,
            use_threads=True,
        )
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            if sample_batches is not None and batch_number > sample_batches:
                break
            frame = batch.to_pandas()
            rows += len(frame)
            frame["amc_code"] = source_key(frame, year).map(source_to_amc)
            mapped = frame["amc_code"].notna().to_numpy()
            mapped_rows += int(mapped.sum())
            resident = contributions(frame, year, cutoffs[year])
            resident["census_year"] = year
            resident["amc_code"] = frame["amc_code"]
            resident = resident[mapped]
            characteristic_parts.append(
                resident.groupby(["census_year", "amc_code"], observed=True).sum().reset_index()
            )

            universe, migrant, assigned = migration_assignment(frame, year)
            universe &= mapped
            flow_universe_rows += int(universe.sum())
            migrant_rows += int((migrant & mapped).sum())
            flow = contributions(frame, year, cutoffs[year], universe=universe)
            flow["census_year"] = year
            flow["amc_code"] = frame["amc_code"]
            flow["origin_uf"] = pd.Series(
                np.where(assigned >= 0, np.array(STATES, dtype=object)[np.maximum(assigned, 0)], None),
                dtype="string",
            )
            flow["weighted_interstate_migrants"] = np.where(
                migrant & mapped,
                pd.to_numeric(frame["person_weight"], errors="coerce").fillna(0), 0.0,
            )
            flow["weighted_same_current_uf_residual"] = np.where(
                universe & ~migrant,
                pd.to_numeric(frame["person_weight"], errors="coerce").fillna(0), 0.0,
            )
            flow["sample_interstate_migrants"] = (migrant & mapped).astype(np.int64)
            flow["sample_same_current_uf_residual"] = (
                universe & ~migrant
            ).astype(np.int64)
            flow = flow[universe]
            flow_parts.append(
                flow.groupby(["census_year", "amc_code", "origin_uf"], observed=True).sum().reset_index()
            )
            if batch_number % 10 == 0:
                flow_parts = [collapse(flow_parts, ["census_year", "amc_code", "origin_uf"])]
                characteristic_parts = [collapse(
                    characteristic_parts, ["census_year", "amc_code"]
                )]
                print(f"  {batch_number} batches, {rows:,} rows", flush=True)
        flows = finalize_metrics(collapse(
            flow_parts, ["census_year", "amc_code", "origin_uf"]
        ))
        chars = finalize_metrics(collapse(
            characteristic_parts, ["census_year", "amc_code"]
        ))
        destination_total = flows.groupby(["census_year", "amc_code"])[
            "weighted_population"
        ].transform("sum")
        flows["population_share_of_destination_age5plus"] = (
            flows["weighted_population"] / destination_total
        )
        flows["interstate_migrant_share_of_destination_age5plus"] = (
            flows["weighted_interstate_migrants"] / destination_total
        )
        flows["migration_measure"] = MIGRATION_MEASURE[year]
        flows["contains_same_current_uf_residual"] = (
            flows["weighted_same_current_uf_residual"] > 0
        )
        flows["contains_interstate_migrants"] = (
            flows["weighted_interstate_migrants"] > 0
        )
        flows["mixed_migrant_and_residual_cell"] = (
            flows["contains_same_current_uf_residual"]
            & flows["contains_interstate_migrants"]
        )
        flows = flows.merge(amc_lookup, on="amc_code", how="left", validate="many_to_one")
        chars = chars.merge(amc_lookup, on="amc_code", how="left", validate="many_to_one")
        flow_years.append(flows)
        characteristic_years.append(chars)
        diagnostics[str(year)] = {
            "person_rows_scanned": rows,
            "person_rows_mapped_to_amc": mapped_rows,
            "mapping_rate": mapped_rows / rows if rows else math.nan,
            "age5plus_flow_universe_rows": flow_universe_rows,
            "interstate_migrant_sample_rows": migrant_rows,
            "flow_cells": len(flows),
            "amc_characteristic_rows": len(chars),
        }
    return (
        pd.concat(flow_years, ignore_index=True),
        pd.concat(characteristic_years, ignore_index=True),
        diagnostics,
    )


def read_gdp_workbook(path: Path, year: int) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Séries", engine="xlrd", dtype=object)
    result = pd.DataFrame({
        "year": year,
        "uf": raw.iloc[:, 0].astype("string").str.strip(),
        "municipality_code": raw.iloc[:, 1].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(7),
        "municipality_name": raw.iloc[:, 2].astype("string").str.strip(),
        "gdp_total_2010_brl_thousand": pd.to_numeric(raw.iloc[:, 3], errors="coerce"),
        "va_industry_2010_brl_thousand": pd.to_numeric(raw.iloc[:, 4], errors="coerce"),
        "va_services_private_2010_brl_thousand": pd.to_numeric(raw.iloc[:, 5], errors="coerce"),
        "va_public_administration_2010_brl_thousand": pd.to_numeric(raw.iloc[:, 6], errors="coerce"),
        "va_agriculture_2010_brl_thousand": pd.to_numeric(raw.iloc[:, 7], errors="coerce"),
    })
    values = [column for column in result if column.endswith("_thousand")]
    result = result[result[values].notna().any(axis=1)].copy()
    if result["municipality_code"].duplicated().any():
        raise RuntimeError(f"Duplicate municipality codes in {path}")
    return result


def build_gdp_tables(
    root: Path,
    municipality_to_amc: dict[str, str],
    characteristics: pd.DataFrame,
    amc_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    parts = []
    diagnostics = {}
    for year in GDP_YEARS:
        path = root / f"PIB_Munis_{year}.xls"
        frame = read_gdp_workbook(path, year)
        frame["municipality_code_amc_lookup"] = frame["municipality_code"].replace(
            GDP_HISTORICAL_CODE_ALIASES
        )
        frame["amc_code"] = frame["municipality_code_amc_lookup"].map(municipality_to_amc)
        if frame["amc_code"].isna().any():
            codes = frame.loc[frame["amc_code"].isna(), "municipality_code"].tolist()
            raise RuntimeError(f"GDP {year} has municipalities outside AMC lookup: {codes}")
        parts.append(frame)
        diagnostics[str(year)] = {
            "municipality_rows": len(frame),
            "total_gdp_nonmissing": int(frame["gdp_total_2010_brl_thousand"].notna().sum()),
            "nonmissing_by_measure": {
                column: int(frame[column].notna().sum())
                for column in frame.columns if column.endswith("_thousand")
            },
        }
    municipality = pd.concat(parts, ignore_index=True)
    value_columns = [column for column in municipality if column.endswith("_thousand")]
    positive_gdp = municipality["gdp_total_2010_brl_thousand"].where(
        municipality["gdp_total_2010_brl_thousand"] > 0
    )
    for column in value_columns[1:]:
        municipality[column.replace("_2010_brl_thousand", "_share_of_gdp")] = (
            municipality[column] / positive_gdp
        )
    aggregates = []
    for (year, amc), group in municipality.groupby(["year", "amc_code"], observed=True):
        row = {
            "year": int(year), "amc_code": amc,
            "gdp_municipalities_observed": int(len(group)),
        }
        for column in value_columns:
            count = int(group[column].notna().sum())
            row[column] = group[column].sum(min_count=1)
            row[column + "_municipality_coverage"] = count
        aggregates.append(row)
    amc_gdp = pd.DataFrame(aggregates)
    census = characteristics.rename(columns={"census_year": "year"})
    panel = amc_gdp.merge(census, on=["year", "amc_code"], how="outer", validate="one_to_one")
    # Census rows already carry these fields. Fill them for GDP-only years too.
    panel = panel.drop(columns=[
        column for column in amc_lookup.columns if column != "amc_code" and column in panel
    ]).merge(amc_lookup, on="amc_code", how="left", validate="many_to_one")
    positive_amc_gdp = panel["gdp_total_2010_brl_thousand"].where(
        panel["gdp_total_2010_brl_thousand"] > 0
    )
    for column in value_columns[1:]:
        panel[column.replace("_2010_brl_thousand", "_share_of_gdp")] = (
            panel[column] / positive_amc_gdp
        )
    panel["gdp_per_capita_2010_brl"] = (
        panel["gdp_total_2010_brl_thousand"] * 1000
        / panel["weighted_population"].where(panel["weighted_population"] > 0)
    )
    panel["has_gdp"] = panel["gdp_municipalities_observed"].notna()
    panel["has_census_characteristics"] = panel["weighted_population"].notna()
    return municipality, panel, diagnostics


def validate_outputs(
    flows: pd.DataFrame,
    characteristics: pd.DataFrame,
    municipality_gdp: pd.DataFrame,
    amc_panel: pd.DataFrame,
    sampled: bool,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    if flows.duplicated(["census_year", "amc_code", "origin_uf"]).any():
        raise RuntimeError("Duplicate flow keys")
    if characteristics.duplicated(["census_year", "amc_code"]).any():
        raise RuntimeError("Duplicate characteristic keys")
    if municipality_gdp.duplicated(["year", "municipality_code"]).any():
        raise RuntimeError("Duplicate municipality GDP keys")
    if amc_panel.duplicated(["year", "amc_code"]).any():
        raise RuntimeError("Duplicate AMC panel keys")
    if (flows["weighted_interstate_migrants"] > flows["weighted_population"] + 1e-6).any():
        raise RuntimeError("Interstate migrants exceed flow-cell population")
    shares = flows.groupby(["census_year", "amc_code"])[
        "population_share_of_destination_age5plus"
    ].sum()
    if not sampled and not np.allclose(shares, 1.0, atol=1e-8):
        raise RuntimeError("Flow shares do not sum to one")
    checks.update({
        "flow_rows": len(flows),
        "characteristics_rows": len(characteristics),
        "municipality_gdp_rows": len(municipality_gdp),
        "amc_panel_rows": len(amc_panel),
        "flow_share_min": float(shares.min()),
        "flow_share_max": float(shares.max()),
        "gdp_years": sorted(municipality_gdp["year"].unique().tolist()),
        "contains_2015_gdp": bool((municipality_gdp["year"] == 2015).any()),
    })
    return checks


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    required_sources = (args.amc_gpkg,) if args.gdp_only else (
        args.amc_gpkg, args.geobr_crosswalk, args.ipea_1970_codebook
    )
    for required in required_sources:
        if not required.exists():
            raise FileNotFoundError(f"Required AMC source is missing: {required}")

    amcs, municipality_to_amc, amc_lookup = load_amcs(args.amc_gpkg)
    metadata_path = args.output / "metadata.json"
    if args.gdp_only:
        flow_path = args.output / "amc_origin_uf_year_flows.parquet"
        characteristic_path = args.output / "amc_year_characteristics.parquet"
        for required in (flow_path, characteristic_path):
            if not required.exists():
                raise FileNotFoundError(f"GDP-only rebuild requires existing output: {required}")
        flows = pd.read_parquet(flow_path)
        characteristics = pd.read_parquet(characteristic_path)
        if metadata_path.exists():
            previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            census_diagnostics = previous_metadata.get("census_diagnostics", {})
        else:
            census_diagnostics = {}
    else:
        persons = ds.dataset(args.persons, format="parquet", partitioning="hive")
        missing_columns = sorted(set(SOURCE_COLUMNS) - set(persons.schema.names))
        if missing_columns:
            raise RuntimeError(f"Person dataset lacks columns: {missing_columns}")
        crosswalk_path = args.output / "municipality_to_amc_crosswalk.parquet"
        if crosswalk_path.exists() and not args.rebuild_crosswalk:
            print(f"Reusing {crosswalk_path}", flush=True)
            crosswalk = pd.read_parquet(crosswalk_path)
        else:
            print("Building historical municipality-to-AMC crosswalk...", flush=True)
            crosswalk_1970 = build_1970_crosswalk(
                persons, args.ipea_1970_codebook, args.geobr_crosswalk,
                municipality_to_amc, args.batch_size,
            )
            crosswalk_later = build_later_crosswalks(
                amcs, municipality_to_amc, args.boundaries
            )
            crosswalk = pd.concat([crosswalk_1970, crosswalk_later], ignore_index=True)
            crosswalk.to_parquet(crosswalk_path, index=False, compression="zstd")

        cutoff_frame = pd.read_csv(args.income_cutoffs)
        cutoffs = {
            int(row.census_year): np.array((row.q20, row.q40, row.q60, row.q80), dtype=float)
            for row in cutoff_frame.itertuples(index=False)
        }
        flows, characteristics, census_diagnostics = build_census_tables(
            persons, crosswalk, amc_lookup, cutoffs, args.batch_size, args.sample_batches
        )
    municipality_gdp, amc_panel, gdp_diagnostics = build_gdp_tables(
        args.gdp, municipality_to_amc, characteristics, amc_lookup
    )

    outputs = {
        "municipality_year_gdp.parquet": municipality_gdp,
        "amc_year_panel.parquet": amc_panel,
    }
    if not args.gdp_only:
        outputs = {
            "amc_origin_uf_year_flows.parquet": flows,
            "amc_year_characteristics.parquet": characteristics,
            **outputs,
        }
    for name, frame in outputs.items():
        path = args.output / name
        frame.to_parquet(path, index=False, compression="zstd")
        print(f"Wrote {len(frame):,} rows to {path}", flush=True)

    validation = validate_outputs(
        flows, characteristics, municipality_gdp, amc_panel,
        sampled=args.sample_batches is not None,
    )
    metadata = {
        "geography": {
            "definition": "Ehrl/geobr minimum comparable areas, 1970-2010",
            "amc_count": 3800,
            "source_gpkg": str(args.amc_gpkg),
            "historical_genealogy": str(args.geobr_crosswalk),
            "historical_1970_codebook": str(args.ipea_1970_codebook),
        },
        "migration": {
            "universe": "Age 5+, positive person weight, mapped destination AMC",
            "1970": "years in current UF 0-4 and previous UF differs",
            "1980": "years in current UF 0-4 and birth UF differs",
            "1991_2010": "fixed-date five-year UF differs",
            "same_uf_note": (
                "The current-UF cell is a residual containing stayers, intrastate "
                "migrants, and cases not identified as interstate migrants."
            ),
        },
        "income": {
            "raw_measure_by_year": INCOME_COLUMN,
            "warning": "Raw income levels are not comparable across census currencies/years.",
            "national_positive_income_cutoffs_source": str(args.income_cutoffs),
        },
        "gdp": {
            "years": list(GDP_YEARS),
            "excluded_year": 2015,
            "units": "R$ thousand, constant 2010 prices",
            "aggregation": "Municipality values summed within AMC; component coverage retained.",
            "historical_code_aliases": GDP_HISTORICAL_CODE_ALIASES,
        },
        "census_diagnostics": census_diagnostics,
        "gdp_diagnostics": gdp_diagnostics,
        "validation": validation,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
