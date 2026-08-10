"""Build auditable POLOCENTRO 1975 municipality and AMC treatment vectors.

The legal source is Decreto 75.320/1975, Article 2.  Banco Central Circular
259/1975, Annex 1 supplies the operational widths and surface descriptions.
Where those sources differ, both decree-literal and operational variants are
retained.  The reconstructed polygons are intended for municipal/AMC analysis,
not parcel-level eligibility decisions.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid
from shapely.geometry import LineString, Point, Polygon, box, shape
from shapely.ops import nearest_points, unary_union


CRS = "EPSG:5880"
LEGAL_URL = (
    "https://www2.camara.leg.br/legin/fed/decret/1970-1979/"
    "decreto-75320-29-janeiro-1975-423871-publicacaooriginal-1-pe.html"
)
CIRCULAR_URL = (
    "https://normativos.bcb.gov.br/Lists/Normativos/Attachments/40950/"
    "Circ_0259_v1_O.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amc-gpkg", type=Path, default=Path(
        "data/censo_microdados/amc/AMC_1970_2010_simplified.gpkg"
    ))
    parser.add_argument("--boundary-root", type=Path, default=Path(
        "data/censo_microdados/ibge_boundaries/2010"
    ))
    parser.add_argument("--transport", type=Path, default=Path(
        "data/censo_microdados/amc/Transporte_v2017.zip"
    ))
    parser.add_argument("--localities", type=Path, default=Path(
        "data/censo_microdados/amc/Localidades_v2017.zip"
    ))
    parser.add_argument("--waterways", type=Path, default=Path(
        "data/censo_microdados/amc/polocentro_waterways.json"
    ))
    parser.add_argument("--crosswalk", type=Path, default=Path(
        "data/processed/amcs/municipality_to_amc_crosswalk.parquet"
    ))
    parser.add_argument("--output", type=Path, default=Path(
        "data/processed/amcs"
    ))
    return parser.parse_args()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def zip_layer(path: Path, layer: str) -> str:
    return "zip://" + path.resolve().as_posix() + "!" + layer


def read_municipalities(root: Path) -> gpd.GeoDataFrame:
    parts = []
    archives = sorted(root.glob("*_municipios.zip"))
    if len(archives) != 27:
        raise FileNotFoundError(f"Expected 27 state/DF municipality archives, found {len(archives)}")
    for archive in archives:
        frame = gpd.read_file("zip://" + archive.resolve().as_posix())
        frame = frame.rename(columns={"CD_GEOCODM": "municipality_code", "NM_MUNICIP": "municipality_name"})
        frame["municipality_code"] = frame["municipality_code"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(7)
        frame["uf"] = frame["municipality_code"].str[:2]
        parts.append(frame[["municipality_code", "municipality_name", "uf", "geometry"]])
    result = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    result = result[~result["municipality_code"].isin(["4300001", "4300002"])]
    return result.to_crs(CRS)


def read_seats(localities: Path, municipalities: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    cities = gpd.read_file(zip_layer(localities, "loc_cidade_p.shp"))
    capitals = gpd.read_file(zip_layer(localities, "loc_capital_p.shp"))
    seats = gpd.GeoDataFrame(
        pd.concat([cities[["nome", "geometry"]], capitals[["nome", "geometry"]]], ignore_index=True),
        geometry="geometry", crs=cities.crs,
    ).to_crs(CRS)
    seats = gpd.sjoin(seats[["nome", "geometry"]], municipalities[
        ["municipality_code", "municipality_name", "uf", "geometry"]
    ], how="inner", predicate="within").drop(columns="index_right")
    seats["normalized_name"] = seats["nome"].map(normalize)
    # Spatial location is the identifier; name agreement is a useful audit.
    seats["name_matches_polygon"] = (
        seats["normalized_name"] == seats["municipality_name"].map(normalize)
    )
    seats = seats.sort_values("name_matches_polygon", ascending=False).drop_duplicates("municipality_code")
    return gpd.GeoDataFrame(seats, geometry="geometry", crs=CRS)


def seat(seats: gpd.GeoDataFrame, name: str, uf: str) -> Point:
    match = seats[(seats["uf"] == uf) & (seats["normalized_name"] == normalize(name))]
    if len(match) != 1:
        raise RuntimeError(f"Could not uniquely locate {name}/{uf}: {len(match)} matches")
    return match.geometry.iloc[0]


def read_transport(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads = gpd.read_file(zip_layer(path, "tra_trecho_rodoviario_l.shp")).to_crs(CRS)
    rail = gpd.read_file(zip_layer(path, "tra_trecho_ferroviario_l.shp")).to_crs(CRS)
    return roads, rail


def between_features(
    features: gpd.GeoDataFrame,
    start: Point,
    end: Point,
    max_deviation_km: float,
) -> gpd.GeoSeries:
    chord = LineString([start, end])
    length = chord.length
    points = features.geometry.representative_point()
    projection = points.map(chord.project) / length
    distance = points.distance(chord)
    keep = projection.between(-0.04, 1.04) & (distance <= max_deviation_km * 1000)
    selected = features.loc[keep, "geometry"]
    if selected.empty:
        raise RuntimeError("No transport features selected between endpoints")
    return selected


def network_path(features: gpd.GeoSeries, start: Point, end: Point, snap_m: float = 250) -> object:
    adjacency: dict[tuple[int, int], list[tuple[float, tuple[int, int], object]]] = {}

    def node(coordinate: tuple[float, float]) -> tuple[int, int]:
        return (round(coordinate[0] / snap_m), round(coordinate[1] / snap_m))

    for geometry in features:
        parts = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
        for part in parts:
            coordinates = list(part.coords)
            for first, second in zip(coordinates[:-1], coordinates[1:]):
                a, b = node(first), node(second)
                segment = LineString([first, second])
                adjacency.setdefault(a, []).append((segment.length, b, segment))
                adjacency.setdefault(b, []).append((segment.length, a, segment))
    nodes = list(adjacency)
    if not nodes:
        raise RuntimeError("Empty transport graph")

    def nearest(target: Point) -> tuple[int, int]:
        return min(nodes, key=lambda value: (value[0] * snap_m - target.x) ** 2 + (value[1] * snap_m - target.y) ** 2)

    source, target = nearest(start), nearest(end)
    distance = {source: 0.0}
    previous: dict[tuple[int, int], tuple[tuple[int, int], object]] = {}
    queue = [(0.0, source)]
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current == target:
            break
        if current_distance != distance.get(current):
            continue
        for length, neighbor, geometry in adjacency[current]:
            candidate = current_distance + length
            if candidate < distance.get(neighbor, float("inf")):
                distance[neighbor] = candidate
                previous[neighbor] = (current, geometry)
                heapq.heappush(queue, (candidate, neighbor))
    if target not in distance:
        raise RuntimeError(f"No connected transport path (snap={snap_m}m)")
    path = []
    current = target
    while current != source:
        current, geometry = previous[current]
        path.append(geometry)
    return unary_union(path)


def road_axis(roads: gpd.GeoDataFrame, ref: str, start: Point, end: Point, deviation: float = 120) -> object:
    selected = roads[roads["codtrechor"].astype("string") == ref]
    candidates = between_features(selected, start, end, deviation)
    for snap in (100, 250, 500, 1_000, 2_000):
        try:
            return network_path(candidates, start, end, snap)
        except RuntimeError:
            continue
    # BC250 occasionally leaves gaps or changes the reference field at urban
    # crossings. Preserve the official road geometry on a tighter chord filter
    # instead of replacing it with a straight line.
    return unary_union(between_features(selected, start, end, min(deviation, 20)).tolist())


def road_intersection(roads: gpd.GeoDataFrame, first: str, second: str, near: Point) -> Point:
    a = unary_union(roads[roads["codtrechor"].astype("string") == first].geometry.tolist())
    b = unary_union(roads[roads["codtrechor"].astype("string") == second].geometry.tolist())
    candidates = a.intersection(b)
    if not candidates.is_empty:
        return nearest_points(near, candidates)[1]
    p1, p2 = nearest_points(a, b)
    return Point((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)


def osm_waterways(path: Path) -> gpd.GeoDataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for element in payload["elements"]:
        coordinates = [(point["lon"], point["lat"]) for point in element.get("geometry", [])]
        if len(coordinates) < 2:
            continue
        rows.append({
            "name": element.get("tags", {}).get("name", ""),
            "geometry": LineString(coordinates),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(CRS)


def water(waters: gpd.GeoDataFrame, name: str, bounds: tuple[float, float, float, float] | None = None) -> object:
    chosen = waters[waters["name"].map(normalize) == normalize(name)]
    if bounds is not None:
        clip = gpd.GeoSeries([box(*bounds)], crs="EPSG:4326").to_crs(CRS).iloc[0]
        chosen = chosen[chosen.intersects(clip)]
    if chosen.empty:
        raise RuntimeError(f"No waterway geometry for {name}")
    return unary_union(chosen.geometry.tolist())


def valid_polygon(geometry: object) -> object:
    geometry = make_valid(geometry)
    if geometry.geom_type == "GeometryCollection":
        polygons = [part for part in geometry.geoms if part.geom_type in ("Polygon", "MultiPolygon")]
        geometry = unary_union(polygons)
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    return geometry


def polygon_from_points(points: list[Point]) -> object:
    return valid_polygon(Polygon([(point.x, point.y) for point in points]))


def build_areas(
    roads: gpd.GeoDataFrame,
    rail: gpd.GeoDataFrame,
    waters: gpd.GeoDataFrame,
    seats: gpd.GeoDataFrame,
    municipalities: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    rows = []

    def add(name: str, state_1975: str, geometry: object, method: str, confidence: str,
            circular_area_ha: int | None, variant: str = "both") -> None:
        rows.append({
            "area_name": name,
            "state_1975": state_1975,
            "source_variant": variant,
            "construction_method": method,
            "boundary_confidence": confidence,
            "circular_approx_area_ha": circular_area_ha,
            "reconstructed_area_ha": geometry.area / 10_000,
            "geometry": valid_polygon(geometry),
        })

    # Explicit road and railway corridors.
    axis = road_axis(roads, "BR-365", seat(seats, "Patrocínio", "31"), seat(seats, "Canápolis", "31"))
    add("Triângulo Mineiro", "MG", axis.buffer(40_000), "BR-365, 40 km each side, Patrocínio–Canápolis", "high", 800_000)

    paracatu = seat(seats, "Paracatu", "31")
    end_356 = road_intersection(roads, "BR-040", "BR-356", paracatu)
    end_365 = road_intersection(roads, "BR-040", "BR-365", paracatu)
    add("Vão do Paracatu", "MG", road_axis(roads, "BR-040", paracatu, end_356, 80).buffer(20_000),
        "Decree-literal BR-040 corridor to BR-356, 20 km each side", "high", 500_000, "decree")
    add("Vão do Paracatu", "MG", road_axis(roads, "BR-040", paracatu, end_365, 80).buffer(20_000),
        "Circular 259 BR-040 corridor to BR-365, 20 km each side", "high", 500_000, "operational")

    cg, tl = seat(seats, "Campo Grande", "50"), seat(seats, "Três Lagoas", "50")
    rail_candidates = between_features(rail, cg, tl, 90)
    rail_axis = None
    for snap in (250, 500, 1_000, 2_000, 5_000):
        try:
            rail_axis = network_path(rail_candidates, cg, tl, snap)
            break
        except RuntimeError:
            continue
    if rail_axis is None:
        rail_axis = unary_union(rail_candidates.tolist())
    add("Campo Grande–Três Lagoas", "MT (current MS)", rail_axis.buffer(20_000),
        "Noroeste railway corridor, 20 km each side", "high", 1_400_000)

    bodo_line = LineString([
        seat(seats, "Aquidauana", "50"), seat(seats, "Bodoquena", "50"), seat(seats, "Bonito", "50")
    ])
    add("Bodoquena", "MT (current MS)", bodo_line.buffer(20_000),
        "Aquidauana–Bodoquena–Bonito road approximation, 20 km each side", "medium", 400_000)

    axis = road_axis(roads, "BR-153", seat(seats, "Porangatu", "52"), seat(seats, "Gurupi", "17"))
    add("Gurupi", "GO (current GO/TO)", axis.buffer(20_000),
        "BR-153 corridor, 40 km total width, Porangatu–Gurupi", "high", None)

    paraim = water(waters, "Rio Paraim", (-48, -16, -46, -14))
    br020 = unary_union(roads[roads["codtrechor"].astype("string") == "BR-020"].geometry.tolist())
    paraim_crossing = nearest_points(paraim, br020)[1]
    axis = road_axis(roads, "BR-020", paraim_crossing, seat(seats, "Posse", "52"), 80)
    add("Paranã", "GO", axis.buffer(20_000), "BR-020 corridor, 20 km each side, Rio Paraim–Posse", "high", 560_000)

    axis = road_axis(roads, "BR-158", seat(seats, "Piranhas", "52"), seat(seats, "Aragarças", "52"), 80)
    add("Piranhas", "GO", axis.buffer(10_000), "BR-158 corridor, 10 km each side, Piranhas–Aragarças", "high", 200_000)

    # Parecis: official one-sided northern strip from Sacré to Sumidouro.
    br364 = unary_union(roads[roads["codtrechor"].astype("string") == "BR-364"].geometry.tolist())
    sacre = water(waters, "Rio Sacre", (-59, -15, -57, -12.5))
    sumidouro = water(waters, "Ribeirão Sumidouro", (-57, -16, -55, -14))
    west = nearest_points(sacre, br364)[1]
    east = nearest_points(sumidouro, br364)[1]
    parecis_axis = road_axis(roads, "BR-364", west, east, 100)
    # The axis may be multipart; retain its northern half of the ordinary buffer.
    mid_y = (west.y + east.y) / 2
    north_clip = box(min(west.x, east.x) - 100_000, mid_y - 10_000,
                     max(west.x, east.x) + 100_000, max(west.y, east.y) + 100_000)
    add("Parecis", "MT", parecis_axis.buffer(40_000).intersection(north_clip),
        "North-side BR-364 strip, Sacré–Sumidouro, 40 km", "medium", 800_000)

    # River-bounded areas. These polygons follow official anchors; straight
    # closures are explicitly lower-confidence at municipal scale.
    sete = seat(seats, "Sete Lagoas", "31")
    paraopeba = water(waters, "Rio Paraopeba", (-46, -21, -43, -17))
    sf = water(waters, "Rio São Francisco", (-46, -21, -42, -16))
    velhas = water(waters, "Rio das Velhas", (-46, -21, -42, -16))
    p_south = nearest_points(sete, paraopeba)[1]
    v_south = nearest_points(sete, velhas)[1]
    p_sf = nearest_points(paraopeba, sf)[0]
    sf_p = nearest_points(paraopeba, sf)[1]
    v_sf = nearest_points(velhas, sf)[0]
    sf_v = nearest_points(velhas, sf)[1]
    alto = polygon_from_points([p_south, p_sf, sf_p, sf_v, v_sf, v_south])
    add("Alto-Médio São Francisco", "MG", alto, "Polygon anchored to Paraopeba, São Francisco, Velhas and Sete Lagoas", "medium", 3_000_000)

    # Xavantina lies in Cocalinho between the Cristalino and Água Preta. The
    # named minor tributary is incomplete in the national vector, so calibrate
    # a local circle to the official 200,000 ha around the documented crossing.
    x_center = gpd.GeoSeries([Point(-51.42, -14.15)], crs="EPSG:4326").to_crs(CRS).iloc[0]
    x_radius = math.sqrt(2_000_000_000 / math.pi)
    cocalinho = municipalities[municipalities["municipality_name"].map(normalize) == "COCALINHO"].geometry.iloc[0]
    add("Xavantina", "MT", x_center.buffer(x_radius).intersection(cocalinho),
        "Official-area-calibrated local polygon between Cristalino and Água Preta in Cocalinho", "low", 200_000)

    # Pirineus: approximate the four official boundaries with their nearest
    # intersections and retain the stated 520,000 ha as an audit benchmark.
    df = gpd.read_file("zip://" + (Path("data/censo_microdados/ibge_boundaries/2010/df_municipios.zip").resolve().as_posix())).to_crs(CRS).geometry.iloc[0]
    br080 = unary_union(roads[roads["codtrechor"].astype("string") == "BR-080"].geometry.tolist())
    maranhao = water(waters, "Rio Maranhão", (-50, -17, -46, -13))
    almas = water(waters, "Rio das Almas", (-51, -17, -46, -12))
    q1 = nearest_points(df.boundary, br080)[0]
    q2 = nearest_points(br080, almas)[0]
    q3a, q3b = nearest_points(almas, maranhao)
    q4 = nearest_points(maranhao, df.boundary)[0]
    pirineus = polygon_from_points([q1, q2, q3a, q3b, q4])
    add("Pirineus", "GO", pirineus, "Polygon anchored to BR-080, DF boundary, Maranhão and das Almas rivers", "medium", 520_000)

    # Rio Verde: official sources define the boundaries but the homonymous
    # river network is ambiguous nationally. Use the documented service and
    # infrastructure municipalities, then calibrate to the stated 1.2m ha.
    rv_points = [seat(seats, name, "52") for name in ("Rio Verde", "Jataí", "Montividiu", "Santa Helena de Goiás", "Perolândia")]
    rv_hull = unary_union(rv_points).convex_hull
    radius = 1_000.0
    target = 12_000_000_000
    for _ in range(32):
        trial = rv_hull.buffer(radius)
        radius *= math.sqrt(target / trial.area)
    add("Rio Verde", "GO", rv_hull.buffer(radius),
        "Official-area-calibrated polygon around Rio Verde, Jataí, Montividiu, Santa Helena and Perolândia", "low", 1_200_000)

    permitted_ufs = {
        "Triângulo Mineiro": {"31"},
        "Alto-Médio São Francisco": {"31"},
        "Vão do Paracatu": {"31"},
        "Campo Grande–Três Lagoas": {"50"},
        "Bodoquena": {"50"},
        "Xavantina": {"51"},
        "Parecis": {"51"},
        "Gurupi": {"52", "17"},
        "Paranã": {"52"},
        "Pirineus": {"52"},
        "Piranhas": {"52"},
        "Rio Verde": {"52"},
    }
    state_masks = {
        name: unary_union(municipalities[municipalities["uf"].isin(ufs)].geometry.tolist())
        for name, ufs in permitted_ufs.items()
    }
    for row in rows:
        row["geometry"] = valid_polygon(row["geometry"].intersection(state_masks[row["area_name"]]))
        row["reconstructed_area_ha"] = row["geometry"].area / 10_000
    result = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    return result


def assignment(
    geography: gpd.GeoDataFrame,
    id_column: str,
    areas: gpd.GeoDataFrame,
    seats: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    geo = geography[[id_column, "geometry"]].copy()
    geo["geography_area_km2"] = geo.area / 1_000_000
    variants = {
        "decree": areas[areas["source_variant"].isin(["both", "decree"])],
        "operational": areas[areas["source_variant"].isin(["both", "operational"])],
    }
    output = geo[[id_column, "geography_area_km2"]].copy()
    for label, subset in variants.items():
        union = unary_union(subset.geometry.tolist())
        intersections = geo.geometry.intersection(union).area
        output[f"polocentro_{label}_area_share"] = (intersections / geo.area).clip(0.0, 1.0)
        # One-hectare floor removes numerical border slivers between state and
        # municipality layers while remaining negligible at this scale.
        output[f"polocentro_{label}_any_overlap"] = intersections > 10_000.0
        output[f"polocentro_{label}_majority_area"] = output[f"polocentro_{label}_area_share"] >= 0.5
        if seats is not None:
            seat_map = seats.set_index(id_column).geometry
            output[f"polocentro_{label}_seat_inside"] = output[id_column].map(seat_map).map(
                lambda point: bool(point is not None and union.covers(point))
            )
            area_names = []
            for identifier in output[id_column]:
                point = seat_map.get(identifier)
                hit = subset[subset.geometry.map(lambda geometry: bool(point is not None and geometry.covers(point)))][
                    "area_name"
                ].drop_duplicates().tolist()
                area_names.append("|".join(hit))
            output[f"polocentro_{label}_seat_area_names"] = area_names
            output[f"polocentro_{label}_core"] = (
                output[f"polocentro_{label}_seat_inside"]
                | (output[f"polocentro_{label}_area_share"] >= 0.10)
            )
    names: list[set[str]] = [set() for _ in range(len(geo))]
    spatial_index = geo.sindex
    for area in areas.itertuples():
        positions = spatial_index.query(area.geometry, predicate="intersects")
        if len(positions) == 0:
            continue
        overlap = geo.geometry.iloc[positions].intersection(area.geometry).area.to_numpy()
        for position in positions[overlap > 10_000]:
            names[int(position)].add(area.area_name)
    output["polocentro_area_names"] = ["|".join(sorted(value)) for value in names]
    return output


def main() -> None:
    args = parse_args()
    for path in (args.amc_gpkg, args.transport, args.localities, args.waterways, args.crosswalk):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    municipalities = read_municipalities(args.boundary_root)
    seats = read_seats(args.localities, municipalities)
    roads, rail = read_transport(args.transport)
    waters = osm_waterways(args.waterways)
    areas = build_areas(roads, rail, waters, seats, municipalities)

    crosswalk = pd.read_parquet(args.crosswalk)
    current = crosswalk[crosswalk["census_year"] == 2010][
        ["source_key", "amc_code"]
    ].rename(columns={"source_key": "municipality_code"})
    municipalities = municipalities.merge(current, on="municipality_code", how="left", validate="one_to_one")
    if municipalities["amc_code"].isna().any():
        raise RuntimeError("Some 2010 municipalities lack an AMC")

    muni = assignment(municipalities, "municipality_code", areas, seats)
    muni = muni.merge(municipalities[["municipality_code", "municipality_name", "uf", "amc_code"]],
                      on="municipality_code", how="left", validate="one_to_one")
    outside_program_states = ~muni["uf"].isin(["17", "31", "50", "51", "52"])
    for column in muni.columns:
        if column.startswith("polocentro_") and column.endswith("_area_share"):
            muni.loc[outside_program_states, column] = 0.0
        elif column.startswith("polocentro_") and muni[column].dtype == bool:
            muni.loc[outside_program_states, column] = False
        elif column.startswith("polocentro_") and column.endswith("_area_names"):
            muni.loc[outside_program_states, column] = ""
    ordered = ["municipality_code", "municipality_name", "uf", "amc_code"]
    muni = muni[ordered + [column for column in muni.columns if column not in ordered]]

    amcs = gpd.read_file(args.amc_gpkg).rename(columns={"code_amc": "amc_code"}).to_crs(CRS)
    amcs["amc_code"] = amcs["amc_code"].round().astype("Int64").astype("string")
    amc_seats = seats.merge(current, on="municipality_code", how="inner")
    # For seat_inside at AMC level, use any current municipal seat in the AMC.
    amc_assignment = assignment(amcs, "amc_code", areas)
    for label in ("decree", "operational"):
        union = unary_union(areas[areas["source_variant"].isin(["both", label])].geometry.tolist())
        inside = amc_seats.assign(inside=amc_seats.geometry.map(union.covers)).groupby("amc_code")["inside"].any()
        amc_assignment[f"polocentro_{label}_any_seat_inside"] = amc_assignment["amc_code"].map(inside).fillna(False)
        amc_assignment[f"polocentro_{label}_core"] = (
            amc_assignment[f"polocentro_{label}_any_seat_inside"]
            | (amc_assignment[f"polocentro_{label}_area_share"] >= 0.10)
        )
    permitted_amcs = set(municipalities.loc[
        municipalities["uf"].isin(["17", "31", "50", "51", "52"]), "amc_code"
    ])
    outside_program_amcs = ~amc_assignment["amc_code"].isin(permitted_amcs)
    for column in amc_assignment.columns:
        if column.startswith("polocentro_") and column.endswith("_area_share"):
            amc_assignment.loc[outside_program_amcs, column] = 0.0
        elif column.startswith("polocentro_") and amc_assignment[column].dtype == bool:
            amc_assignment.loc[outside_program_amcs, column] = False
        elif column.startswith("polocentro_") and column.endswith("_area_names"):
            amc_assignment.loc[outside_program_amcs, column] = ""

    areas_out = args.output / "polocentro_1975_areas.gpkg"
    if areas_out.exists():
        areas_out.unlink()
    # Keep the equal-area working CRS in the analytical GeoPackage. Reprojecting
    # very detailed state-clipped rings to geographic coordinates can introduce
    # sub-millimetre self-intersections in GDAL serialization.
    areas.to_file(areas_out, layer="reconstructed_areas", driver="GPKG")
    muni.to_parquet(args.output / "polocentro_1975_municipality_treatment.parquet", index=False, compression="zstd")
    muni.to_csv(args.output / "polocentro_1975_municipality_treatment.csv", index=False, encoding="utf-8-sig")
    amc_assignment.to_parquet(args.output / "polocentro_1975_amc_treatment.parquet", index=False, compression="zstd")
    amc_assignment.to_csv(args.output / "polocentro_1975_amc_treatment.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "program": "Programa de Desenvolvimento dos Cerrados (POLOCENTRO)",
        "assignment_date": "1975-01-29",
        "legal_source": LEGAL_URL,
        "operational_source": CIRCULAR_URL,
        "unit_note": "Municipality vector uses 2010 municipality polygons/codes; AMC vector is compatible with the 1970-2010 panel.",
        "recommended_primary": "polocentro_operational_core (seat inside or at least 10% of land area covered); retain area_share as the preferred continuous exposure.",
        "robustness": "Use seat_inside/any_seat_inside, majority_area and any_overlap variants; compare decree and operational Vão do Paracatu definitions.",
        "parcel_warning": "Reconstruction is suitable for municipality/AMC analysis, not parcel-level eligibility.",
        "area_diagnostics": areas.drop(columns="geometry").to_dict("records"),
        "counts": {
            "municipalities": len(muni),
            "amcs": len(amc_assignment),
            "municipality_operational_seat_inside": int(muni["polocentro_operational_seat_inside"].sum()),
            "municipality_operational_core": int(muni["polocentro_operational_core"].sum()),
            "municipality_operational_majority_area": int(muni["polocentro_operational_majority_area"].sum()),
            "municipality_operational_any_overlap": int(muni["polocentro_operational_any_overlap"].sum()),
            "amc_operational_any_seat_inside": int(amc_assignment["polocentro_operational_any_seat_inside"].sum()),
            "amc_operational_core": int(amc_assignment["polocentro_operational_core"].sum()),
            "amc_operational_majority_area": int(amc_assignment["polocentro_operational_majority_area"].sum()),
            "amc_operational_any_overlap": int(amc_assignment["polocentro_operational_any_overlap"].sum()),
        },
    }
    (args.output / "polocentro_1975_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
