"""Draw the Banco Central operational POLOCENTRO areas over 1970--2010 AMCs.

By default the script creates two separate 16:9 figures: a Brazil-wide locator
map and a POLOCENTRO-region map with legible AMC boundaries. Most presentation
choices live in ``polocentro_map_config.json``. Run this file with ``--help``
for quick command-line overrides.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box
from shapely.ops import unary_union


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "areas": "data/processed/amcs/polocentro_1975_areas.gpkg",
        "amcs": "data/censo_microdados/amc/AMC_1970_2010_simplified.gpkg",
        "treatment": "data/processed/amcs/polocentro_1975_amc_treatment.parquet",
        "output_dir": "figs/polocentro_maps",
        "output_stems": {
            "brazil": "polocentro_bcb_operational_brazil",
            "zoom": "polocentro_bcb_operational_amcs_zoom",
        },
    },
    "figure": {
        "width_inches": 13.333,
        "height_inches": 7.5,
        "dpi": 300,
        "formats": ["png", "pdf"],
        "background": "#FCFBF7",
        "tight_bbox": False,
    },
    "views": {
        "brazil": {
            "title": "POLOCENTRO operational areas in Brazil",
            "subtitle": "Banco Central Circular 259/1975 operational geography over harmonized 1970–2010 AMCs",
            "padding_fraction": 0.025,
            "show_area_labels": False,
            "show_north_arrow": False,
            "show_scale_bar": True,
            "scale_bar_km": 500,
            "scale_bar_position_axes": [0.08, 0.06],
            "amc_boundary_width": 0.12,
            "operational_boundary_width": 1.55,
            "area_label_size": 8.0,
            "tight_bbox": True,
            "legend_inside": True,
            "legend_location": "lower left",
            "legend_anchor_axes": [0.055, 0.17],
            "legend_columns": 1,
            "show_zoom_extent": True,
            "zoom_extent_linewidth": 2.2,
        },
        "zoom": {
            "title": "POLOCENTRO operational areas and AMC exposure",
            "subtitle": "Detailed view of the Banco Central operational geography",
            "padding_fraction": 0.015,
            "show_area_labels": True,
            "show_north_arrow": True,
            "show_scale_bar": True,
            "scale_bar_km": 200,
            "scale_bar_position_axes": [0.055, 0.93],
            "amc_boundary_width": 0.38,
            "operational_boundary_width": 1.9,
            "area_label_size": 9.0,
            "tight_bbox": True,
            "legend_inside": True,
            "legend_location": "lower right",
            "legend_anchor_axes": [0.985, 0.025],
            "legend_columns": 1,
        },
    },
    "presentation": {
        "source_note": (
            "Sources: Banco Central Circular 259/1975, Annex 1; Decreto 75.320/1975, "
            "art. 2; IBGE/IPEA AMC boundaries. Operational boundaries are historical reconstructions."
        ),
        "show_title": False,
        "show_subtitle": False,
        "show_source_note": False,
        "show_counts": False,
        "label_offsets_km": {
            "Triângulo Mineiro": [20, -20],
            "Vão do Paracatu": [5, 15],
            "Campo Grande–Três Lagoas": [0, -20],
            "Bodoquena": [20, 25],
            "Gurupi": [0, 15],
            "Paranã": [15, 5],
            "Piranhas": [-10, 0],
            "Parecis": [-5, 15],
            "Alto-Médio São Francisco": [15, -10],
            "Xavantina": [0, 0],
            "Pirineus": [5, 0],
            "Rio Verde": [10, -5],
        },
    },
    "classification": {"full_share_tolerance": 1e-8},
    "colors": {
        "background": "#FCFBF7",
        "amc_boundary": "#9C9B95",
        "amc_context_fill": "#F0EFEA",
        "amc_partial": "#9CCAD1",
        "amc_full": "#247B8A",
        "operational_fill": "#315C9A",
        "operational_boundary": "#214A84",
        "zoom_extent": "#D36B2D",
        "text": "#202124",
        "muted_text": "#5F6368",
        "label_halo": "#FCFBF7",
    },
    "style": {
        "program_fill_alpha": 0.09,
        "amc_partial_alpha": 0.78,
        "amc_full_alpha": 0.90,
        "font_family": "DejaVu Sans",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("polocentro_map_config.json"),
        help="JSON file with presentation choices.",
    )
    parser.add_argument(
        "--view",
        choices=["both", "brazil", "zoom"],
        default="both",
        help="Generate both figures or only one view.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the output directory.")
    parser.add_argument("--dpi", type=int, help="Override raster resolution.")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        help="Output formats, for example: --formats png pdf",
    )
    parser.add_argument("--no-labels", action="store_true", help="Hide area names.")
    parser.add_argument("--no-counts", action="store_true", help="Hide AMC coverage counts.")
    return parser.parse_args()


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config = deep_merge(config, json.load(handle))
    return config


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir)
    if args.dpi is not None:
        config["figure"]["dpi"] = args.dpi
    if args.formats is not None:
        config["figure"]["formats"] = args.formats
    if args.no_labels:
        for view in config["views"].values():
            view["show_area_labels"] = False
    if args.no_counts:
        config["presentation"]["show_counts"] = False
    return config


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def operational_areas(areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Exclude the decree-only Paracatu geometry and retain the BCB definition."""
    return areas[areas["source_variant"].isin(["both", "operational"])].copy()


def load_data(config: dict[str, Any]) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    paths = config["paths"]
    areas = operational_areas(gpd.read_file(resolve_project_path(paths["areas"])))
    amcs = gpd.read_file(resolve_project_path(paths["amcs"]))
    treatment = pd.read_parquet(resolve_project_path(paths["treatment"]))

    amcs = amcs.rename(columns={"code_amc": "amc_code"})
    amcs["amc_code"] = amcs["amc_code"].round().astype("Int64").astype("string")
    treatment["amc_code"] = treatment["amc_code"].astype("string")
    amcs = amcs.merge(treatment, on="amc_code", how="left", validate="one_to_one")
    return areas, amcs.to_crs(areas.crs)


def padded_bounds(bounds: Any, fraction: float) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = map(float, bounds)
    padding_x = (xmax - xmin) * fraction
    padding_y = (ymax - ymin) * fraction
    return xmin - padding_x, ymin - padding_y, xmax + padding_x, ymax + padding_y


def crop_to_bounds(frame: gpd.GeoDataFrame, bounds: tuple[float, ...]) -> gpd.GeoDataFrame:
    return frame[frame.intersects(box(*bounds))].copy()


def exposed_amcs(
    amcs: gpd.GeoDataFrame,
    tolerance: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    share = amcs["polocentro_operational_area_share"].fillna(0.0)
    overlap = amcs["polocentro_operational_any_overlap"].fillna(False).astype(bool)
    full = amcs[overlap & (share >= 1.0 - tolerance)]
    partial = amcs[overlap & (share < 1.0 - tolerance)]
    return full, partial


def draw_map_layers(
    ax: plt.Axes,
    amcs: gpd.GeoDataFrame,
    areas: gpd.GeoDataFrame,
    bounds: tuple[float, ...],
    view_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    colors = config["colors"]
    style = config["style"]
    tolerance = float(config["classification"]["full_share_tolerance"])
    context = crop_to_bounds(amcs, bounds)
    full, partial = exposed_amcs(context, tolerance)

    context.plot(ax=ax, facecolor=colors["amc_context_fill"], edgecolor="none", zorder=1)
    areas.plot(
        ax=ax,
        facecolor=colors["operational_fill"],
        edgecolor="none",
        alpha=float(style["program_fill_alpha"]),
        zorder=2,
    )
    if not partial.empty:
        partial.plot(
            ax=ax,
            facecolor=colors["amc_partial"],
            edgecolor="none",
            alpha=float(style["amc_partial_alpha"]),
            zorder=3,
        )
    if not full.empty:
        full.plot(
            ax=ax,
            facecolor=colors["amc_full"],
            edgecolor="none",
            alpha=float(style["amc_full_alpha"]),
            zorder=4,
        )
    context.boundary.plot(
        ax=ax,
        color=colors["amc_boundary"],
        linewidth=float(view_config["amc_boundary_width"]),
        alpha=0.82,
        zorder=5,
    )
    areas.boundary.plot(
        ax=ax,
        color=colors["operational_boundary"],
        linewidth=float(view_config["operational_boundary_width"]),
        zorder=6,
    )


def add_area_labels(
    ax: plt.Axes,
    areas: gpd.GeoDataFrame,
    view_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    offsets = config["presentation"].get("label_offsets_km", {})
    for name, group in areas.groupby("area_name", sort=False):
        point = unary_union(group.geometry.tolist()).representative_point()
        offset_x, offset_y = offsets.get(name, [0, 0])
        label = ax.text(
            point.x + float(offset_x) * 1_000,
            point.y + float(offset_y) * 1_000,
            name,
            ha="center",
            va="center",
            fontsize=float(view_config["area_label_size"]),
            color=config["colors"]["text"],
            zorder=10,
        )
        label.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=2.8,
                    foreground=config["colors"]["label_halo"],
                ),
                path_effects.Normal(),
            ]
        )


def add_north_arrow(ax: plt.Axes, config: dict[str, Any]) -> None:
    color = config["colors"]["text"]
    ax.annotate(
        "N",
        xy=(0.955, 0.92),
        xytext=(0.955, 0.835),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=10,
        color=color,
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": color},
        zorder=20,
    )


def add_scale_bar(
    ax: plt.Axes,
    view_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    length_km = float(view_config["scale_bar_km"])
    position = view_config["scale_bar_position_axes"]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + float(position[0]) * (xmax - xmin)
    y0 = ymin + float(position[1]) * (ymax - ymin)
    length_m = length_km * 1_000
    cap = 0.008 * (ymax - ymin)
    color = config["colors"]["text"]

    ax.plot([x0, x0 + length_m], [y0, y0], color=color, linewidth=2.0, zorder=20)
    ax.plot([x0, x0], [y0 - cap, y0 + cap], color=color, linewidth=1.2, zorder=20)
    ax.plot(
        [x0 + length_m, x0 + length_m],
        [y0 - cap, y0 + cap],
        color=color,
        linewidth=1.2,
        zorder=20,
    )
    ax.text(
        x0 + length_m / 2,
        y0 + cap * 1.7,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        zorder=20,
    )


def legend_handles(config: dict[str, Any], view_config: dict[str, Any]) -> list[Any]:
    colors = config["colors"]
    handles = [
        Patch(facecolor=colors["amc_full"], edgecolor="none", label="AMC fully inside"),
        Patch(facecolor=colors["amc_partial"], edgecolor="none", label="AMC partially inside"),
        Line2D(
            [0],
            [0],
            color=colors["operational_boundary"],
            linewidth=2.2,
            label="BCB operational boundary",
        ),
    ]
    if view_config.get("show_zoom_extent", False):
        handles.append(
            Line2D(
                [0],
                [0],
                color=colors["zoom_extent"],
                linewidth=float(view_config.get("zoom_extent_linewidth", 2.2)),
                linestyle=(0, (5, 3)),
                label="Detailed view extent",
            )
        )
    return handles


def add_zoom_extent(
    ax: plt.Axes,
    zoom_bounds: tuple[float, float, float, float],
    view_config: dict[str, Any],
    config: dict[str, Any],
) -> None:
    xmin, ymin, xmax, ymax = zoom_bounds
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="none",
            edgecolor=config["colors"]["zoom_extent"],
            linewidth=float(view_config.get("zoom_extent_linewidth", 2.2)),
            linestyle=(0, (5, 3)),
            zorder=15,
        )
    )


def coverage_counts(amcs: gpd.GeoDataFrame, config: dict[str, Any]) -> str:
    tolerance = float(config["classification"]["full_share_tolerance"])
    full, partial = exposed_amcs(amcs, tolerance)
    return f"{len(full)} AMCs fully inside · {len(partial)} AMCs partially inside"


def view_bounds(
    view_name: str,
    areas: gpd.GeoDataFrame,
    amcs: gpd.GeoDataFrame,
    padding: float,
) -> tuple[float, float, float, float]:
    source_bounds = amcs.total_bounds if view_name == "brazil" else areas.total_bounds
    return padded_bounds(source_bounds, padding)


def render_view(
    view_name: str,
    areas: gpd.GeoDataFrame,
    amcs: gpd.GeoDataFrame,
    config: dict[str, Any],
) -> plt.Figure:
    figure_config = config["figure"]
    view_config = config["views"][view_name]
    colors = config["colors"]
    fig = plt.figure(
        figsize=(float(figure_config["width_inches"]), float(figure_config["height_inches"])),
        facecolor=figure_config.get("background", colors["background"]),
    )
    presentation = config["presentation"]
    has_header = any(
        (
            presentation["show_title"],
            presentation["show_subtitle"],
            presentation["show_counts"],
        )
    )
    has_source_note = bool(presentation["show_source_note"])
    legend_inside = bool(view_config.get("legend_inside", False))
    if legend_inside and not has_source_note:
        axes_bottom = 0.02
    else:
        axes_bottom = 0.145 if has_source_note else 0.115
    axes_top = 0.835 if has_header else 0.985
    ax = fig.add_axes([0.025, axes_bottom, 0.95, axes_top - axes_bottom])
    bounds = view_bounds(
        view_name,
        areas,
        amcs,
        float(view_config["padding_fraction"]),
    )

    draw_map_layers(ax, amcs, areas, bounds, view_config, config)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    if view_name == "brazil" and view_config.get("show_zoom_extent", False):
        zoom_config = config["views"]["zoom"]
        zoom_bounds = view_bounds(
            "zoom",
            areas,
            amcs,
            float(zoom_config["padding_fraction"]),
        )
        add_zoom_extent(ax, zoom_bounds, view_config, config)
    if view_config["show_area_labels"]:
        add_area_labels(ax, areas, view_config, config)
    if view_config["show_north_arrow"]:
        add_north_arrow(ax, config)
    if view_config["show_scale_bar"]:
        add_scale_bar(ax, view_config, config)

    if presentation["show_title"]:
        fig.text(
            0.045,
            0.945,
            view_config["title"],
            ha="left",
            va="top",
            fontsize=20,
            color=colors["text"],
        )
    if presentation["show_subtitle"]:
        fig.text(
            0.045,
            0.895,
            view_config["subtitle"],
            ha="left",
            va="top",
            fontsize=10.5,
            color=colors["muted_text"],
        )
    if presentation["show_counts"]:
        fig.text(
            0.955,
            0.895,
            coverage_counts(amcs, config),
            ha="right",
            va="top",
            fontsize=9.2,
            color=colors["muted_text"],
        )
    legend_kwargs = {
        "handles": legend_handles(config, view_config),
        "frameon": False,
        "handlelength": 2.8,
        "columnspacing": 2.1,
        "fontsize": 9.4,
    }
    if legend_inside:
        ax.legend(
            loc=view_config.get("legend_location", "lower left"),
            bbox_to_anchor=tuple(view_config.get("legend_anchor_axes", [0.055, 0.17])),
            ncol=int(view_config.get("legend_columns", 1)),
            **legend_kwargs,
        )
    else:
        fig.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.078 if has_source_note else 0.035),
            ncol=int(view_config.get("legend_columns", 3)),
            **legend_kwargs,
        )
    if has_source_note:
        fig.text(
            0.045,
            0.022,
            presentation["source_note"],
            ha="left",
            va="bottom",
            fontsize=7.7,
            color=colors["muted_text"],
        )
    return fig


def save_figure(fig: plt.Figure, view_name: str, config: dict[str, Any]) -> list[Path]:
    output_dir = resolve_project_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = config["paths"]["output_stems"][view_name]
    outputs = []
    for extension in config["figure"]["formats"]:
        output_path = output_dir / f"{stem}.{extension}"
        kwargs: dict[str, Any] = {"facecolor": fig.get_facecolor()}
        tight_bbox = config["views"][view_name].get(
            "tight_bbox", config["figure"].get("tight_bbox", False)
        )
        if tight_bbox:
            kwargs["bbox_inches"] = "tight"
        if extension.lower() == "png":
            kwargs["dpi"] = int(config["figure"]["dpi"])
        fig.savefig(output_path, **kwargs)
        outputs.append(output_path.resolve())
    plt.close(fig)
    return outputs


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    plt.rcParams.update(
        {
            "font.family": config["style"]["font_family"],
            "font.size": 9.5,
            "text.color": config["colors"]["text"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    areas, amcs = load_data(config)
    views = ["brazil", "zoom"] if args.view == "both" else [args.view]
    for view_name in views:
        figure = render_view(view_name, areas, amcs, config)
        for output in save_figure(figure, view_name, config):
            print(output)


if __name__ == "__main__":
    main()
