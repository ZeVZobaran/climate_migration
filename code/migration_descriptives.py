# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:55 2026

@author: c337191
"""

import math
import re
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
sns.set(style="whitegrid")

path = r'D:\Users\c337191\Documents\climate_migration'
# %% get data
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:5880").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

regions = regions.rename(columns={
    'NM_MESO': 'name',
    'CD_GEOCME': 'id'
    })
regions['id']

df = pd.read_parquet(
    f'{path}/data/formatted_composit/climate_mig_meso.parquet'
    )

df = df[[
    'orig_id', 'dest_id', 'pair_id', 'orig_abbrv', 'orig_name', 'dest_abbrv', 'dest_name',
    'decade', 'N_od_flow_wm', 'N_od_flow_all',
    'orig_comp_rel_within', 'dest_comp_rel_within', 
    'orig_hot_within', 'orig_flood_within',
    'orig_cold_within','orig_dry_within',
    'dest_hot_within', 'dest_flood_within',
    'dest_cold_within', 'dest_dry_within',
    'orig_gdp_capita', 'dest_gdp_capita', 
    'orig_pop', 'dest_pop', 
    'orig_gdp_ind_share', 'dest_gdp_ind_share',
    'orig_gdp_agr_share', 'dest_gdp_agr_share',
    'orig_gdp_serv_share', 'dest_gdp_serv_share'
    ]]
# %% Defaults
# Most of these are chat-made and unneeded, since my data is already
# quite clean. Still, I'll keep them for cleanliness

DEFAULT_FLOW = "N_od_flow_all"
DEFAULT_PERIOD = "decade"
DEFAULT_ORIG = "orig_id"
DEFAULT_DEST = "dest_id"

DEFAULT_ATTRS = (
    'pop',
    "comp_rel_within",
    'hot_within', 
    'flood_within',
    'cold_within',
    'dry_within',
    "gdp_capita",
    'gdp_ind_share',
    'gdp_agr_share',
    'gdp_serv_share'
)

DEFAULT_RANK_ATTRS = (
    'pop',
    "comp_rel_within",
    'hot_within', 
    'flood_within',
    'cold_within',
    'dry_within',
    "gdp_capita",
    'gdp_ind_share',
    'gdp_agr_share',
    'gdp_serv_share'
)


# ============================================================
# Core helpers
# ============================================================

def _first_nonnull(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else np.nan


def _weighted_mean(x, w):
    ok = x.notna() & w.notna() & (w > 0)
    if ok.sum() == 0:
        return np.nan
    return np.average(x[ok], weights=w[ok])


def _weighted_corr(x, y, w):
    ok = x.notna() & y.notna() & w.notna() & (w > 0)
    x, y, w = x[ok], y[ok], w[ok]

    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)

    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)

    if vx <= 0 or vy <= 0:
        return np.nan

    return cov / np.sqrt(vx * vy)


def _safe_corr(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 2:
        return np.nan
    if x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return x[ok].corr(y[ok])


def _side_col(side, attr):
    return f"{side}_{attr}"


def clean_od_panel(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
    exclude_self=True,
):
    """
    Ensures one row per origin-destination-period.
    Sums flows if duplicate OD-period rows exist.
    Keeps first non-null value for other columns.
    """
    d = df.copy()
    d[flow_col] = pd.to_numeric(d[flow_col], errors="coerce").fillna(0)

    if exclude_self:
        d = d[d[orig_col] != d[dest_col]].copy()

    keys = [period_col, orig_col, dest_col]

    agg = {flow_col: "sum"}
    for c in d.columns:
        if c not in keys and c != flow_col:
            agg[c] = _first_nonnull

    return d.groupby(keys, as_index=False).agg(agg)


def region_attributes_from_od(
    df,
    attrs=DEFAULT_ATTRS,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    Builds a region-period attribute table from observed origin and destination columns.

    Uses:
    orig_gdp_capita, dest_gdp_capita
    orig_comp_rel_within, dest_comp_rel_within
    orig_comp_rel_between, dest_comp_rel_between
    etc.
    """
    frames = []

    for side, id_col in [("orig", orig_col), ("dest", dest_col)]:
        cols = [period_col, id_col]

        for meta in ["name", "abbrv"]:
            c = _side_col(side, meta)
            if c in df.columns:
                cols.append(c)

        for attr in attrs:
            c = _side_col(side, attr)
            if c in df.columns:
                cols.append(c)

        temp = df[cols].copy()
        temp = temp.rename(columns={id_col: "region_id"})

        rename = {}
        for c in temp.columns:
            prefix = f"{side}_"
            if c.startswith(prefix):
                rename[c] = c.replace(prefix, "")

        temp = temp.rename(columns=rename)
        frames.append(temp)

    reg = pd.concat(frames, ignore_index=True, sort=False)

    agg = {
        c: _first_nonnull
        for c in reg.columns
        if c not in [period_col, "region_id"]
    }

    reg = reg.groupby([period_col, "region_id"], as_index=False).agg(agg)

    # Rank variables within each decade
    rankable = [c for c in attrs if c in reg.columns]

    for c in rankable:
        reg[f"{c}_rank_pct"] = (
            reg.groupby(period_col)[c]
            .rank(method="average", pct=True)
        )

    return reg


def add_region_ranks(
    df,
    attrs=DEFAULT_ATTRS,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    Adds rank-percentile variables to OD dataframe for origins and destinations.
    Does not overwrite existing orig_/dest_ levels.
    """
    d = df.copy()

    reg = region_attributes_from_od(
        d,
        attrs=attrs,
        orig_col=orig_col,
        dest_col=dest_col,
        period_col=period_col,
    )

    rank_cols = [c for c in reg.columns if c.endswith("_rank_pct")]

    for side, id_col in [("orig", orig_col), ("dest", dest_col)]:
        temp = reg[[period_col, "region_id"] + rank_cols].copy()
        temp = temp.rename(columns={"region_id": id_col})

        rename = {c: f"{side}_{c}" for c in rank_cols}
        temp = temp.rename(columns=rename)

        d = d.merge(temp, on=[period_col, id_col], how="left")

    return d


def complete_flow_panel(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
    exclude_self=True,
):
    """
    Completes missing OD-period cells with zero flow.
    Useful for persistence, entry, and survival.
    """
    periods = sorted(df[period_col].dropna().unique())
    ids = sorted(set(df[orig_col].dropna()) | set(df[dest_col].dropna()))

    full = pd.MultiIndex.from_product(
        [periods, ids, ids],
        names=[period_col, orig_col, dest_col],
    ).to_frame(index=False)

    if exclude_self:
        full = full[full[orig_col] != full[dest_col]].copy()

    flows = (
        df.groupby([period_col, orig_col, dest_col], as_index=False)[flow_col]
        .sum()
    )

    out = full.merge(flows, on=[period_col, orig_col, dest_col], how="left")
    out[flow_col] = out[flow_col].fillna(0)

    return out

def _make_average_panel(panel, region_col="region_id", period_col="decade", value_cols=None):
    """
    Adds 'Average' rows across periods for selected value columns.
    """
    p = panel.copy()
    if value_cols is None:
        value_cols = [c for c in p.columns if c not in [region_col, period_col]]

    avg = (
        p.groupby(region_col, as_index=False)[value_cols]
        .mean(numeric_only=True)
    )
    avg[period_col] = "Average"

    out = pd.concat([p, avg], ignore_index=True, sort=False)
    return out

def _panel_period_order(panel, period_col="decade"):
    vals = list(panel[period_col].dropna().unique())
    base = sorted([v for v in vals if v != "Average"])
    if "Average" in vals:
        base = base + ["Average"]
    return base

def _ensure_str_id(df, col):
    out = df.copy()
    out[col] = out[col].astype(str)
    return out

# %% 1. Overall correlation between origins and destinations

def origin_destination_correlations(
    df,
    attrs=DEFAULT_RANK_ATTRS,
    flow_col=DEFAULT_FLOW,
    period_col=DEFAULT_PERIOD,
):
    """
    Flow-weighted and unweighted correlation between origin and destination attributes.
    """
    d = clean_od_panel(df, flow_col=flow_col)

    rows = []

    for period, g in [("all", d)] + list(d.groupby(period_col)):
        for attr in attrs:
            oc = _side_col("orig", attr)
            dc = _side_col("dest", attr)

            if oc not in g.columns or dc not in g.columns:
                print(attr)
                continue

            rows.append({
                period_col: period,
                "attribute": attr,
                "weighted_corr": _weighted_corr(g[oc], g[dc], g[flow_col]),
                "unweighted_corr": _safe_corr(g[oc], g[dc]),
                "n_pairs": len(g),
                "total_flow": g[flow_col].sum(),
            })

    return pd.DataFrame(rows)


# %% 2. Concentration in receiver, sender and pairs

def receiver_concentration(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    For each destination, computes concentration of inflows across origins.
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    g = (
        d.groupby([period_col, dest_col, orig_col], as_index=False)[flow_col]
        .sum()
    )

    g["total_inflow"] = g.groupby([period_col, dest_col])[flow_col].transform("sum")
    g["origin_share"] = np.where(
        g["total_inflow"] > 0,
        g[flow_col] / g["total_inflow"],
        np.nan,
    )

    out = (
        g.groupby([period_col, dest_col])
        .agg(
            total_inflow=("total_inflow", "first"),
            receiver_hhi=("origin_share", lambda x: np.nansum(x ** 2)),
            top_origin_share=("origin_share", "max"),
            n_positive_origins=(flow_col, lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    out["effective_origins"] = 1 / out["receiver_hhi"]

    return out.sort_values([period_col, "receiver_hhi"], ascending=[True, False])

# ============================================================
# 3. Concentration in sender regions
# ============================================================

def sender_concentration(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    For each origin, computes concentration of outflows across destinations.
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    g = (
        d.groupby([period_col, orig_col, dest_col], as_index=False)[flow_col]
        .sum()
    )

    g["total_outflow"] = g.groupby([period_col, orig_col])[flow_col].transform("sum")
    g["dest_share"] = np.where(
        g["total_outflow"] > 0,
        g[flow_col] / g["total_outflow"],
        np.nan,
    )

    out = (
        g.groupby([period_col, orig_col])
        .agg(
            total_outflow=("total_outflow", "first"),
            sender_hhi=("dest_share", lambda x: np.nansum(x ** 2)),
            top_dest_share=("dest_share", "max"),
            n_positive_dests=(flow_col, lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    out["effective_destinations"] = 1 / out["sender_hhi"]

    return out.sort_values([period_col, "sender_hhi"], ascending=[True, False])

# ============================================================
# 4. Network in senders-receivers: HHI
# ============================================================

def sender_receiver_network_hhi(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    Returns:
    - global OD concentration by decade
    - local partner concentration by region-decade
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    # Global OD concentration
    g = d.copy()
    g["period_total_flow"] = g.groupby(period_col)[flow_col].transform("sum")
    g["od_share"] = np.where(
        g["period_total_flow"] > 0,
        g[flow_col] / g["period_total_flow"],
        np.nan,
    )

    global_hhi = (
        g.groupby(period_col)
        .agg(
            total_flow=("period_total_flow", "first"),
            global_od_hhi=("od_share", lambda x: np.nansum(x ** 2)),
            n_positive_corridors=(flow_col, lambda x: (x > 0).sum()),
        )
        .reset_index()
    )

    global_hhi["effective_corridors"] = 1 / global_hhi["global_od_hhi"]

    # Local partner concentration
    out = d[[period_col, orig_col, dest_col, flow_col]].copy()
    out["out_total"] = out.groupby([period_col, orig_col])[flow_col].transform("sum")
    out["out_share"] = np.where(out["out_total"] > 0, out[flow_col] / out["out_total"], 0)
    out = out.rename(columns={orig_col: "region_id", dest_col: "partner_id"})
    out = out[[period_col, "region_id", "partner_id", "out_share"]]

    inn = d[[period_col, orig_col, dest_col, flow_col]].copy()
    inn["in_total"] = inn.groupby([period_col, dest_col])[flow_col].transform("sum")
    inn["in_share"] = np.where(inn["in_total"] > 0, inn[flow_col] / inn["in_total"], 0)
    inn = inn.rename(columns={dest_col: "region_id", orig_col: "partner_id"})
    inn = inn[[period_col, "region_id", "partner_id", "in_share"]]

    local = out.merge(inn, on=[period_col, "region_id", "partner_id"], how="outer")
    local[["out_share", "in_share"]] = local[["out_share", "in_share"]].fillna(0)

    local["raw_partner_weight"] = 0.5 * (local["out_share"] + local["in_share"])
    local["weight_sum"] = local.groupby([period_col, "region_id"])["raw_partner_weight"].transform("sum")

    local["partner_weight"] = np.where(
        local["weight_sum"] > 0,
        local["raw_partner_weight"] / local["weight_sum"],
        np.nan,
    )

    local_hhi = (
        local.groupby([period_col, "region_id"])
        .agg(
            network_hhi=("partner_weight", lambda x: np.nansum(x ** 2)),
            top_partner_share=("partner_weight", "max"),
            n_partners=("partner_id", "nunique"),
        )
        .reset_index()
    )

    local_hhi["effective_partners"] = 1 / local_hhi["network_hhi"]

    return {
        "global_od_hhi": global_hhi,
        "local_network_hhi": local_hhi,
    }

# %% 5. Graph-like / cluster analysis for migration basins # TODO ADD INDICADOR DE DIREÇÃO

# basin plotter
def basin_membership_from_summary(
    basin_summary,
    period_col="decade",
    basin_col="basin_id",
    regions_col="regions",
):
    """
    Converts basin_summary with a list-valued `regions` column into
    one row per region-decade-basin.
    """
    mem = basin_summary[[period_col, basin_col, "basin_size", "internal_weight", regions_col]].copy()

    mem = mem.explode(regions_col)
    mem = mem.rename(columns={regions_col: "region_id"})

    return mem


def plot_migration_basins_by_decade(
    regions,
    basin_summary,
    region_id_col="id",
    period_col="decade",
    basin_col="basin_id",
    regions_col="regions",
    figsize_per_panel=(6, 6),
    ncols=2,
    cmap_name="tab20",
    missing_color="lightgrey",
    edgecolor="white",
    linewidth=0.25,
    label_largest=True,
    max_labels=8,
):
    """
    Plots migration basins by decade.

    Parameters
    ----------
    regions : geopandas.GeoDataFrame
        Must contain region_id_col and geometry.
    basin_summary : DataFrame
        Usually results['5_migration_basins']['basin_summary'].
        Must contain period_col, basin_col, and a list-valued regions_col.
    """

    mem = basin_membership_from_summary(
        basin_summary,
        period_col=period_col,
        basin_col=basin_col,
        regions_col=regions_col,
    )

    # Harmonize ID types to avoid failed merges
    geo = regions.copy()
    geo[region_id_col] = geo[region_id_col].astype(str)
    mem["region_id"] = mem["region_id"].astype(str)

    periods = sorted(mem[period_col].dropna().unique())

    n = len(periods)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        constrained_layout=True,
    )

    axes = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap(cmap_name)

    for ax, period in zip(axes, periods):
        period_mem = mem[mem[period_col] == period].copy()

        plot_df = geo.merge(
            period_mem,
            left_on=region_id_col,
            right_on="region_id",
            how="left",
        )

        basin_ids = sorted(period_mem[basin_col].dropna().unique())
        basin_to_color = {
            b: cmap(i % cmap.N)
            for i, b in enumerate(basin_ids)
        }

        plot_df["_color"] = plot_df[basin_col].map(basin_to_color).fillna(missing_color)

        plot_df.plot(
            ax=ax,
            color=plot_df["_color"],
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

        ax.set_title(f"Migration basins — {period}", fontsize=13)
        ax.set_axis_off()

        # Label largest basins at approximate centroids
        if label_largest:
            largest = (
                period_mem
                .drop_duplicates(basin_col)
                .sort_values("internal_weight", ascending=False)
                .head(max_labels)
            )

            for _, row in largest.iterrows():
                b = row[basin_col]
                subset = plot_df[plot_df[basin_col] == b]

                if subset.empty:
                    continue

                point = subset.geometry.unary_union.representative_point()

                ax.text(
                    point.x,
                    point.y,
                    str(b),
                    ha="center",
                    va="center",
                    fontsize=9,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75,
                    ),
                )

        legend_items = [
            Patch(
                facecolor=basin_to_color[b],
                edgecolor="none",
                label=f"Basin {b}",
            )
            for b in basin_ids[:20]
        ]

        if len(basin_ids) > 0:
            ax.legend(
                handles=legend_items,
                loc="lower left",
                fontsize=8,
                frameon=True,
                title="Basins",
                title_fontsize=9,
            )

    # Hide empty subplot slots
    for ax in axes[len(periods):]:
        ax.set_axis_off()

    return fig, axes


def migration_basins(
    df, regions_df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
    min_flow_share=0.001,
):
    """
    Simple migration-basin detection.

    Treats migration as an undirected weighted graph:
    weight(i,j) = flow(i->j) + flow(j->i)

    Requires networkx.
    """

    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    node_rows = []
    basin_rows = []

    for period, g in d.groupby(period_col):
        total_flow = g[flow_col].sum()
        min_flow = total_flow * min_flow_share

        edges = g[g[flow_col] >= min_flow].copy()

        edges["a"] = np.minimum(edges[orig_col], edges[dest_col])
        edges["b"] = np.maximum(edges[orig_col], edges[dest_col])

        undirected = (
            edges.groupby(["a", "b"], as_index=False)[flow_col]
            .sum()
        )

        G = nx.Graph()

        all_nodes = sorted(set(g[orig_col]) | set(g[dest_col]))
        G.add_nodes_from(all_nodes)

        for _, row in undirected.iterrows():
            if row["a"] != row["b"] and row[flow_col] > 0:
                G.add_edge(row["a"], row["b"], weight=row[flow_col])

        if G.number_of_edges() == 0:
            continue

        communities = list(nx.community.greedy_modularity_communities(G, weight="weight"))

        for basin_id, nodes in enumerate(communities):
            nodes = sorted(nodes)

            for n in nodes:
                node_rows.append({
                    period_col: period,
                    "region_id": n,
                    "basin_id": basin_id,
                    "basin_size": len(nodes),
                })

            internal_weight = sum(
                edata.get("weight", 0)
                for _, _, edata in G.subgraph(nodes).edges(data=True)
            )

            basin_rows.append({
                period_col: period,
                "basin_id": basin_id,
                "basin_size": len(nodes),
                "internal_weight": internal_weight,
                "regions": nodes,
            })

    basin_summary = pd.DataFrame(basin_rows)
    fig, axes = plot_migration_basins_by_decade(
        regions=regions,
        basin_summary=basin_summary,
        region_id_col="id",
        period_col="decade",
        ncols=2,
        label_largest=True,
        max_labels=8,
        )

    return {
        "region_basins": pd.DataFrame(node_rows),
        "basin_summary": pd.DataFrame(basin_rows),
        'map': (fig, axes)
    }

# %% 6. Persistence in origins, destinations, corridors

def migration_persistence(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    Computes persistence across consecutive decades:
    - origin outflows
    - destination inflows
    - OD corridor flows
    """
    d = complete_flow_panel(df, flow_col, orig_col, dest_col, period_col)

    # Origin persistence
    out = (
        d.groupby([period_col, orig_col], as_index=False)[flow_col]
        .sum()
        .sort_values([orig_col, period_col])
    )

    out["lag_flow"] = out.groupby(orig_col)[flow_col].shift(1)

    origin_persistence = (
        out.dropna(subset=["lag_flow"])
        .groupby(period_col)
        .apply(lambda g: pd.Series({
            "origin_outflow_corr": _safe_corr(np.log1p(g[flow_col]), np.log1p(g["lag_flow"])),
            "n_origins": g[orig_col].nunique(),
        }))
        .reset_index()
    )

    # Destination persistence
    inn = (
        d.groupby([period_col, dest_col], as_index=False)[flow_col]
        .sum()
        .sort_values([dest_col, period_col])
    )

    inn["lag_flow"] = inn.groupby(dest_col)[flow_col].shift(1)

    destination_persistence = (
        inn.dropna(subset=["lag_flow"])
        .groupby(period_col)
        .apply(lambda g: pd.Series({
            "destination_inflow_corr": _safe_corr(np.log1p(g[flow_col]), np.log1p(g["lag_flow"])),
            "n_destinations": g[dest_col].nunique(),
        }))
        .reset_index()
    )

    # Corridor persistence
    cor = d.sort_values([orig_col, dest_col, period_col]).copy()
    cor["lag_flow"] = cor.groupby([orig_col, dest_col])[flow_col].shift(1)

    corridor_persistence = (
        cor.dropna(subset=["lag_flow"])
        .groupby(period_col)
        .apply(lambda g: pd.Series({
            "corridor_flow_corr": _safe_corr(np.log1p(g[flow_col]), np.log1p(g["lag_flow"])),
            "link_survival_rate": ((g["lag_flow"] > 0) & (g[flow_col] > 0)).sum() / max((g["lag_flow"] > 0).sum(), 1),
            "link_entry_rate": ((g["lag_flow"] == 0) & (g[flow_col] > 0)).sum() / max((g["lag_flow"] == 0).sum(), 1),
            "n_corridors": len(g),
        }))
        .reset_index()
    )

    return {
        "origin_persistence": origin_persistence,
        "destination_persistence": destination_persistence,
        "corridor_persistence": corridor_persistence,
    }

# %% 7. Reciprocity in migration


def migration_reciprocity(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    For each unordered pair {i,j}, compares i->j and j->i.

    reciprocity = min(flow_ij, flow_ji) / max(flow_ij, flow_ji)
    one_wayness = abs(flow_ij - flow_ji) / (flow_ij + flow_ji)
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    d["a"] = np.minimum(d[orig_col], d[dest_col])
    d["b"] = np.maximum(d[orig_col], d[dest_col])
    d["direction"] = np.where(d[orig_col] == d["a"], "ab", "ba")

    p = (
        d.groupby([period_col, "a", "b", "direction"], as_index=False)[flow_col]
        .sum()
        .pivot_table(
            index=[period_col, "a", "b"],
            columns="direction",
            values=flow_col,
            fill_value=0,
        )
        .reset_index()
    )

    if "ab" not in p.columns:
        p["ab"] = 0
    if "ba" not in p.columns:
        p["ba"] = 0

    p = p.rename(columns={"ab": "flow_a_to_b", "ba": "flow_b_to_a"})

    p["gross_pair_flow"] = p["flow_a_to_b"] + p["flow_b_to_a"]
    p = p[p["gross_pair_flow"] > 0].copy()

    p["reciprocity"] = (
        np.minimum(p["flow_a_to_b"], p["flow_b_to_a"])
        / np.maximum(p["flow_a_to_b"], p["flow_b_to_a"])
    )

    p["one_wayness"] = (
        np.abs(p["flow_a_to_b"] - p["flow_b_to_a"])
        / p["gross_pair_flow"]
    )

    p["dominant_origin"] = np.where(
        p["flow_a_to_b"] >= p["flow_b_to_a"], p["a"], p["b"]
    )

    p["dominant_destination"] = np.where(
        p["flow_a_to_b"] >= p["flow_b_to_a"], p["b"], p["a"]
    )

    summary = (
        p.groupby(period_col)
        .apply(lambda g: pd.Series({
            "weighted_reciprocity": _weighted_mean(g["reciprocity"], g["gross_pair_flow"]),
            "weighted_one_wayness": _weighted_mean(g["one_wayness"], g["gross_pair_flow"]),
            "n_unordered_pairs": len(g),
            "total_gross_pair_flow": g["gross_pair_flow"].sum(),
        }))
        .reset_index()
    )

    return {
        "pair_reciprocity": p,
        "period_summary": summary,
    }

# %% 8. Degree of redistribution in population due to migration

def migration_redistribution(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    raw_abs_net_over_gross = sum_i |inflow_i - outflow_i| / gross_migration

    redistribution_index = 0.5 * raw_abs_net_over_gross
    This bounded version lies between 0 and 1.
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    out = (
        d.groupby([period_col, orig_col], as_index=False)[flow_col]
        .sum()
        .rename(columns={orig_col: "region_id", flow_col: "outflow"})
    )

    inn = (
        d.groupby([period_col, dest_col], as_index=False)[flow_col]
        .sum()
        .rename(columns={dest_col: "region_id", flow_col: "inflow"})
    )

    reg = out.merge(inn, on=[period_col, "region_id"], how="outer").fillna(0)

    reg["net_migration"] = reg["inflow"] - reg["outflow"]
    reg["abs_net_migration"] = reg["net_migration"].abs()

    summary = (
        reg.groupby(period_col)
        .agg(
            total_abs_net=("abs_net_migration", "sum"),
            gross_migration=("outflow", "sum"),
        )
        .reset_index()
    )

    summary["raw_abs_net_over_gross"] = (
        summary["total_abs_net"] / summary["gross_migration"]
    )

    summary["redistribution_index"] = 0.5 * summary["raw_abs_net_over_gross"]

    return {
        "region_net_migration": reg,
        "period_summary": summary,
    }

# %% 9. Outside option ranks: GDP and climate

def outside_option_ranks(
    df,
    attrs=DEFAULT_RANK_ATTRS,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    For each origin-decade, compares origin attributes to migration-weighted
    destination attributes.

    Example:
    origin GDP rank vs weighted average GDP rank of destinations.
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)
    d = add_region_ranks(d)

    rows = []

    for (period, origin), g in d.groupby([period_col, orig_col]):
        total_out = g[flow_col].sum()

        for attr in attrs:
            oc = _side_col("orig", attr)
            dc = _side_col("dest", attr)
            orank = _side_col("orig", f"{attr}_rank_pct")
            drank = _side_col("dest", f"{attr}_rank_pct")

            if oc not in g.columns or dc not in g.columns:
                continue

            rows.append({
                period_col: period,
                orig_col: origin,
                "attribute": attr,
                "total_outflow": total_out,
                "origin_value": _first_nonnull(g[oc]),
                "weighted_dest_value": _weighted_mean(g[dc], g[flow_col]),
                "origin_rank_pct": _first_nonnull(g[orank]) if orank in g.columns else np.nan,
                "weighted_dest_rank_pct": _weighted_mean(g[drank], g[flow_col]) if drank in g.columns else np.nan,
            })

    out = pd.DataFrame(rows)

    out["value_gap_origin_minus_dest"] = (
        out["origin_value"] - out["weighted_dest_value"]
    )

    out["rank_gap_origin_minus_dest"] = (
        out["origin_rank_pct"] - out["weighted_dest_rank_pct"]
    )


    period_mean = out[
        ['decade', 'attribute', 'rank_gap_origin_minus_dest']
                         ].groupby(['decade', 'attribute']).mean()
    period_std = out[
        ['decade', 'attribute', 'rank_gap_origin_minus_dest']
                         ].groupby(['decade', 'attribute']).std()
    period_summary = period_mean.rename(columns={'rank_gap_origin_minus_dest': 'Mean O-D rank gap'})
    period_summary['Std. Dev'] = period_std['rank_gap_origin_minus_dest']


    return {
        'detailed': out,
        'period_summary': period_summary
        }



# %% 10. Migration on GDP and climate rank: 3x3 matrices

def _add_tercile(reg, attr, period_col=DEFAULT_PERIOD):
    """
    Adds low/medium/high terciles within each decade.
    """
    x = reg.copy()
    col = f"{attr}_tercile"

    def make_tercile(s):
        out = pd.Series(index=s.index, dtype="object")
        ok = s.notna()

        if ok.sum() < 3:
            out.loc[ok] = np.nan
            return out

        out.loc[ok] = pd.qcut(
            s.loc[ok].rank(method="first"),
            q=3,
            labels=["low", "medium", "high"],
        ).astype(str)

        return out

    x[col] = x.groupby(period_col)[attr].transform(make_tercile)
    return x[[period_col, "region_id", col]]


def migration_rank_matrix_3x3(
    df,
    attrs=DEFAULT_RANK_ATTRS,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    For each attribute and decade:
    origin tercile x destination tercile migration-flow shares.
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)
    reg = region_attributes_from_od(d, attrs)

    rows = []

    for attr in attrs:
        if attr not in reg.columns:
            continue

        terc = _add_tercile(reg, attr, period_col)

        orig_terc = terc.rename(columns={
            "region_id": orig_col,
            f"{attr}_tercile": "origin_tercile",
        })

        dest_terc = terc.rename(columns={
            "region_id": dest_col,
            f"{attr}_tercile": "destination_tercile",
        })

        g = (
            d.merge(orig_terc, on=[period_col, orig_col], how="left")
             .merge(dest_terc, on=[period_col, dest_col], how="left")
        )

        tab = (
            g.groupby([period_col, "origin_tercile", "destination_tercile"], as_index=False)[flow_col]
            .sum()
        )

        tab["attribute"] = attr
        tab["period_total_flow"] = tab.groupby(period_col)[flow_col].transform("sum")
        tab["flow_share"] = tab[flow_col] / tab["period_total_flow"]

        rows.append(tab)

    tidy = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    pivots = {}

    if not tidy.empty:
        for (period, attr), g in tidy.groupby([period_col, "attribute"]):
            pivots[(period, attr)] = (
                g.pivot_table(
                    index="origin_tercile",
                    columns="destination_tercile",
                    values="flow_share",
                    fill_value=0,
                )
                .reindex(
                    index=["low", "medium", "high"],
                    columns=["low", "medium", "high"],
                )
            )
    
    # all-period pivot table
    t = tidy.groupby([
        'origin_tercile', 'destination_tercile', 'attribute'
        ]).mean()[['flow_share']]
    idx = pd.IndexSlice
    for attr in attrs:
        t_attr = t.loc[idx[:, :, attr]]
        all_period = t_attr.pivot_table(
            index="origin_tercile",
            columns="destination_tercile",
            values="flow_share",
            fill_value=0,
        ).reindex(
            index=["low", "medium", "high"],
            columns=["low", "medium", "high"],)
        pivots[('all', attr)] = all_period

    return {
        "tidy": tidy,
        "pivots": pivots,
    }


# %% 11. Trapped populations: exit rate vs GDP/climate

def trapped_population_exit_rates(
    df,
    attrs=DEFAULT_RANK_ATTRS,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    """
    Exit rate = total outflow from origin / origin population.

    Returns:
    - origin-decade exit rates
    - simple correlations with GDP/climate attributes
    """
    d = clean_od_panel(df, flow_col, orig_col, dest_col, period_col)

    keep_attrs = [
        _side_col("orig", attr)
        for attr in attrs
        if _side_col("orig", attr) in d.columns
    ]

    out = (
        d.groupby([period_col, orig_col], as_index=False)
        .agg(
            total_outflow=(flow_col, "sum"),
            orig_pop=("orig_pop", _first_nonnull),
            **{c: (c, _first_nonnull) for c in keep_attrs}
        )
    )

    out["exit_rate"] = out["total_outflow"] / out["orig_pop"]

    corr_rows = []

    for period, g in out.groupby(period_col):
        for attr in attrs:
            c = _side_col("orig", attr)
            if c not in g.columns:
                continue

            corr_rows.append({
                period_col: period,
                "attribute": attr,
                "corr_exit_rate_attribute": _safe_corr(g["exit_rate"], g[c]),
                "pop_weighted_corr": _weighted_corr(g["exit_rate"], g[c], g["orig_pop"]),
                "n_origins": len(g),
            })

    return {
        "exit_rates": out,
        "correlations": pd.DataFrame(corr_rows),
    }

# %% 12. Top receivers / senders

def top_receiver_sender_tables(
    df,
    flow_col="N_od_flow_all",
    orig_col="orig_id",
    dest_col="dest_id",
    period_col="decade",
    orig_name_col="orig_name",
    dest_name_col="dest_name",
    orig_pop_col="orig_pop",
    dest_pop_col="dest_pop",
    exclude_self=True,
    top_n=10,
):
    """
    Produces top receiver/sender tables averaged across periods.

    Receiver absolute:
        average inflow by destination across decades.

    Receiver population percentage:
        average inflow / destination population across decades.

    Sender absolute:
        average outflow by origin across decades.

    Sender population percentage:
        average outflow / origin population across decades.
    """

    d = df.copy()
    d[flow_col] = pd.to_numeric(d[flow_col], errors="coerce").fillna(0)

    if exclude_self:
        d = d[d[orig_col] != d[dest_col]].copy()

    # ----------------------------
    # Receiver table: destination-decade inflows
    # ----------------------------
    receivers = (
        d.groupby([period_col, dest_col], as_index=False)
        .agg(
            inflow=(flow_col, "sum"),
            dest_pop=(dest_pop_col, "first"),
            dest_name=(dest_name_col, "first"),
        )
    )

    receivers["inflow_pop_pct"] = 100 * receivers["inflow"] / receivers["dest_pop"]

    receivers_avg = (
        receivers.groupby(dest_col, as_index=False)
        .agg(
            region_name=("dest_name", "first"),
            avg_inflow=("inflow", "mean"),
            avg_inflow_pop_pct=("inflow_pop_pct", "mean"),
            total_inflow=("inflow", "sum"),
            n_decades=(period_col, "nunique"),
        )
        .rename(columns={dest_col: "region_id"})
    )

    top_receivers_abs = (
        receivers_avg
        .sort_values("avg_inflow", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_receivers_pct = (
        receivers_avg
        .sort_values("avg_inflow_pop_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # ----------------------------
    # Sender table: origin-decade outflows
    # ----------------------------
    senders = (
        d.groupby([period_col, orig_col], as_index=False)
        .agg(
            outflow=(flow_col, "sum"),
            orig_pop=(orig_pop_col, "first"),
            orig_name=(orig_name_col, "first"),
        )
    )

    senders["outflow_pop_pct"] = 100 * senders["outflow"] / senders["orig_pop"]

    senders_avg = (
        senders.groupby(orig_col, as_index=False)
        .agg(
            region_name=("orig_name", "first"),
            avg_outflow=("outflow", "mean"),
            avg_outflow_pop_pct=("outflow_pop_pct", "mean"),
            total_outflow=("outflow", "sum"),
            n_decades=(period_col, "nunique"),
        )
        .rename(columns={orig_col: "region_id"})
    )

    top_senders_abs = (
        senders_avg
        .sort_values("avg_outflow", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_senders_pct = (
        senders_avg
        .sort_values("avg_outflow_pop_pct", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "top_receivers_absolute": top_receivers_abs,
        "top_receivers_population_pct": top_receivers_pct,
        "top_senders_absolute": top_senders_abs,
        "top_senders_population_pct": top_senders_pct,
        "all_receivers_average": receivers_avg,
        "all_senders_average": senders_avg,
        "receiver_decade_panel": receivers,
        "sender_decade_panel": senders,
    }

# %% MAPS: Functions

def plot_panel_maps(
    regions,
    panel,
    value_col,
    title,
    region_id_col="id",
    panel_region_col="region_id",
    period_col="decade",
    ncols=3,
    figsize_per_panel=(5.2, 5.2),
    cmap="viridis",
    center=None,                 # use 0 for diverging maps like net migration / gaps
    missing_color="lightgrey",
    edgecolor="white",
    linewidth=0.25,
    legend_label=None,
    highlight_region_id=None,
    highlight_edgecolor="black",
    highlight_linewidth=1.8,
):
    """
    Generic small-multiple choropleth by decade + optional Average.
    """
    geo = regions.copy()
    geo = _ensure_str_id(geo, region_id_col)
    p = panel.copy()
    p = _ensure_str_id(p, panel_region_col)

    plot_df = geo.merge(
        p[[panel_region_col, period_col, value_col]],
        left_on=region_id_col,
        right_on=panel_region_col,
        how="left",
    )

    periods = _panel_period_order(plot_df, period_col=period_col)
    n = len(periods)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    vals = plot_df[value_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        raise ValueError(f"No valid values found for {value_col}")

    if center is not None:
        vmax = np.nanmax(np.abs(vals))
        norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=center, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    for ax, period in zip(axes, periods):
        g = plot_df[plot_df[period_col] == period].copy()

        g.plot(
            column=value_col,
            ax=ax,
            cmap=cmap,
            norm=norm,
            edgecolor=edgecolor,
            linewidth=linewidth,
            missing_kwds={"color": missing_color},
        )

        if highlight_region_id is not None:
            highlight_region_id = str(highlight_region_id)
            focal = geo[geo[region_id_col] == highlight_region_id]
            if len(focal):
                focal.boundary.plot(
                    ax=ax,
                    color=highlight_edgecolor,
                    linewidth=highlight_linewidth,
                )

        ax.set_title(f"{title} — {period}", fontsize=12)
        ax.set_axis_off()

    for ax in axes[len(periods):]:
        ax.set_axis_off()

    cbar = fig.colorbar(sm, ax=axes[:len(periods)], shrink=0.75)
    if legend_label is not None:
        cbar.set_label(legend_label)

    return fig, axes


def build_region_fundamentals_panel(
    df,
    attrs=DEFAULT_ATTRS,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD
):
    """
    Returns one row per region-decade with fundamentals.
    """
    pieces = []
    attrs = list(attrs)

    for side, id_col in [("orig", orig_col), ("dest", dest_col)]:
        cols = [period_col, id_col]
        for base in ["name", "abbrv"] + attrs:
            c = f"{side}_{base}"
            if c in df.columns:
                cols.append(c)

        temp = df[cols].copy()
        temp = temp.rename(columns={id_col: "region_id"})

        rename = {}
        for c in temp.columns:
            if c.startswith(f"{side}_"):
                rename[c] = c.replace(f"{side}_", "")
        temp = temp.rename(columns=rename)

        pieces.append(temp)

    reg = pd.concat(pieces, ignore_index=True, sort=False)

    agg = {c: _first_nonnull for c in reg.columns if c not in [period_col, "region_id"]}
    reg = reg.groupby([period_col, "region_id"], as_index=False).agg(agg)

    return reg


def build_migration_panel(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    d = clean_od_panel(df, flow_col=flow_col, orig_col=orig_col, dest_col=dest_col, period_col=period_col)

    reg = build_region_fundamentals_panel(d, attrs=["pop"], orig_col=orig_col, dest_col=dest_col, period_col=period_col)
    reg = reg.rename(columns={"pop": "region_pop"})

    outflows = (
        d.groupby([period_col, orig_col], as_index=False)[flow_col]
        .sum()
        .rename(columns={orig_col: "region_id", flow_col: "outflow"})
    )

    inflows = (
        d.groupby([period_col, dest_col], as_index=False)[flow_col]
        .sum()
        .rename(columns={dest_col: "region_id", flow_col: "inflow"})
    )

    mig = outflows.merge(inflows, on=[period_col, "region_id"], how="outer").fillna(0)
    mig = mig.merge(reg[[period_col, "region_id", "region_pop"]], on=[period_col, "region_id"], how="left")

    mig["gross_migration"] = mig["inflow"] + mig["outflow"]
    mig["net_migration"] = mig["inflow"] - mig["outflow"]

    for c in ["inflow", "outflow", "gross_migration", "net_migration"]:
        mig[f"{c}_pct_pop"] = 100 * mig[c] / mig["region_pop"]

    return mig

def build_hhi_panels(
    df,
    flow_col=DEFAULT_FLOW,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
):
    d = clean_od_panel(df, flow_col=flow_col, orig_col=orig_col, dest_col=dest_col, period_col=period_col)

    # Sender HHI
    s = d.groupby([period_col, orig_col, dest_col], as_index=False)[flow_col].sum()
    s["total_outflow"] = s.groupby([period_col, orig_col])[flow_col].transform("sum")
    s["dest_share"] = np.where(s["total_outflow"] > 0, s[flow_col] / s["total_outflow"], np.nan)

    sender_hhi = (
        s.groupby([period_col, orig_col])
        .agg(sender_hhi=("dest_share", lambda x: np.nansum(x ** 2)))
        .reset_index()
        .rename(columns={orig_col: "region_id"})
    )

    # Receiver HHI
    r = d.groupby([period_col, dest_col, orig_col], as_index=False)[flow_col].sum()
    r["total_inflow"] = r.groupby([period_col, dest_col])[flow_col].transform("sum")
    r["orig_share"] = np.where(r["total_inflow"] > 0, r[flow_col] / r["total_inflow"], np.nan)

    receiver_hhi = (
        r.groupby([period_col, dest_col])
        .agg(receiver_hhi=("orig_share", lambda x: np.nansum(x ** 2)))
        .reset_index()
        .rename(columns={dest_col: "region_id"})
    )

    return sender_hhi, receiver_hhi

def build_outside_option_panel(
    df,
    attrs=DEFAULT_ATTRS,
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
    flow_col=DEFAULT_FLOW
):
    attrs = list(attrs)
    d = clean_od_panel(df, flow_col=flow_col, orig_col=orig_col, period_col=period_col)

    rows = []

    for (period, origin), g in d.groupby([period_col, orig_col]):
        total_out = g[flow_col].sum()

        row = {
            period_col: period,
            "region_id": origin,
            "total_outflow": total_out,
        }

        for attr in attrs:
            oc = f"orig_{attr}"
            dc = f"dest_{attr}"

            if oc in g.columns and dc in g.columns:
                row[f"origin_{attr}"] = _first_nonnull(g[oc])
                row[f"weighted_dest_{attr}"] = _weighted_mean(g[dc], g[flow_col])
                row[f"outside_option_gap_{attr}"] = row[f"origin_{attr}"] - row[f"weighted_dest_{attr}"]

        rows.append(row)

    return pd.DataFrame(rows)

def build_partner_flow_panel(
    df,
    focal_region_id,
    role="receiver",               # "receiver" or "sender"
    orig_col=DEFAULT_ORIG,
    dest_col=DEFAULT_DEST,
    period_col=DEFAULT_PERIOD,
    flow_col=DEFAULT_FLOW,
    partner_name_col=None,
):
    """
    Returns partner-region maps for one focal region:
    - role='receiver': map origins sending migrants into focal destination
    - role='sender'  : map destinations receiving migrants from focal origin
    """
    d = clean_od_panel(df, flow_col=flow_col, orig_col=orig_col, dest_col=dest_col, period_col=period_col)
    focal_region_id = str(focal_region_id)
    d[orig_col] = d[orig_col].astype(str)
    d[dest_col] = d[dest_col].astype(str)

    if role == "receiver":
        g = d[d[dest_col] == focal_region_id].copy()
        g["region_id"] = g[orig_col]
        if partner_name_col is None:
            partner_name_col = "orig_name"
        total_col = "total_inflow_to_focal"
        title_stub = "Origins sending to"
    elif role == "sender":
        g = d[d[orig_col] == focal_region_id].copy()
        g["region_id"] = g[dest_col]
        if partner_name_col is None:
            partner_name_col = "dest_name"
        total_col = "total_outflow_from_focal"
        title_stub = "Destinations receiving from"
    else:
        raise ValueError("role must be 'receiver' or 'sender'")

    panel = (
        g.groupby([period_col, "region_id"], as_index=False)
        .agg(
            partner_flow=(flow_col, "sum"),
            partner_name=(partner_name_col, "first"),
        )
    )

    panel[total_col] = panel.groupby(period_col)["partner_flow"].transform("sum")
    panel["partner_share_pct"] = 100 * panel["partner_flow"] / panel[total_col]

    panel_avg = (
        panel.groupby("region_id", as_index=False)
        .agg(
            partner_flow=("partner_flow", "mean"),
            partner_share_pct=("partner_share_pct", "mean"),
            partner_name=("partner_name", "first"),
        )
    )
    panel_avg[period_col] = "Average"

    panel = pd.concat([panel, panel_avg], ignore_index=True, sort=False)
    panel.attrs["title_stub"] = title_stub
    panel.attrs["role"] = role
    panel.attrs["focal_region_id"] = focal_region_id

    return panel

def plot_focal_partner_maps(
    regions,
    partner_panel,
    focal_region_id,
    value_col="partner_flow",      # "partner_flow" or "partner_share_pct"
    region_id_col="id",
    period_col="decade",
    cmap="Reds",
    ncols=3,
):
    title_stub = partner_panel.attrs.get("title_stub", "Partner flows")
    title = f"{title_stub} {focal_region_id}"

    return plot_panel_maps(
        regions=regions,
        panel=partner_panel,
        value_col=value_col,
        title=title,
        region_id_col=region_id_col,
        panel_region_col="region_id",
        period_col=period_col,
        ncols=ncols,
        cmap=cmap,
        center=None,
        legend_label=value_col,
        highlight_region_id=focal_region_id,
    )

# %% MAPS: Wrappers

def plot_fundamental_maps(
    regions,
    df,
    region_id_col="id",
    period_col="decade",
    add_average=True,
):
    reg = build_region_fundamentals_panel(df, period_col=period_col)

    if add_average:
        value_cols = [c for c in reg.columns if c not in ["region_id", period_col, "name", "abbrv"]]
        reg = _make_average_panel(reg, region_col="region_id", period_col=period_col, value_cols=value_cols)

    metric_specs = {
        "comp_rel_within": {"title": "Composite climate index", "cmap": "viridis", "center": None},
        "heat_within":     {"title": "Heat index",              "cmap": "viridis", "center": None},
        "flood_within":    {"title": "Flood index",             "cmap": "viridis", "center": None},
        "cold_within":     {"title": "Cold index",              "cmap": "viridis", "center": None},
        "dry_within":      {"title": "Dry index",               "cmap": "viridis", "center": None},
        "pop":             {"title": "Population",              "cmap": "Blues",   "center": None},
        "gdp_capita":      {"title": "GDP per capita",          "cmap": "YlGnBu",  "center": None},
        "gdp_agr_share":   {"title": "Agriculture share",       "cmap": "Greens",  "center": None},
        "gdp_ind_share":   {"title": "Industry share",          "cmap": "Oranges", "center": None},
        "gdp_serv_share":  {"title": "Services share",          "cmap": "Purples", "center": None},
    }

    figs = {}

    for var, spec in metric_specs.items():
        if var not in reg.columns:
            continue

        fig, axes = plot_panel_maps(
            regions=regions,
            panel=reg[["region_id", period_col, var]],
            value_col=var,
            title=spec["title"],
            region_id_col=region_id_col,
            panel_region_col="region_id",
            period_col=period_col,
            cmap=spec["cmap"],
            center=spec["center"],
            legend_label=var,
        )
        figs[var] = (fig, axes)
        plt.close()

    return figs

def plot_migration_maps(
    regions,
    df,
    region_id_col="id",
    period_col="decade",
    add_average=True,
):
    mig = build_migration_panel(df, period_col=period_col)

    if add_average:
        value_cols = [c for c in mig.columns if c not in ["region_id", period_col]]
        mig = _make_average_panel(mig, region_col="region_id", period_col=period_col, value_cols=value_cols)

    metric_specs = {
        "inflow":                 {"title": "Immigration",              "cmap": "Blues", "center": None},
        "outflow":                {"title": "Emigration",               "cmap": "Reds",  "center": None},
        "gross_migration":        {"title": "Gross migration",          "cmap": "Purples", "center": None},
        "net_migration":          {"title": "Net migration",            "cmap": "RdBu_r", "center": 0},
        "inflow_pct_pop":         {"title": "Immigration (% pop)",      "cmap": "Blues", "center": None},
        "outflow_pct_pop":        {"title": "Emigration (% pop)",       "cmap": "Reds",  "center": None},
        "gross_migration_pct_pop":{"title": "Gross migration (% pop)",  "cmap": "Purples", "center": None},
        "net_migration_pct_pop":  {"title": "Net migration (% pop)",    "cmap": "RdBu_r", "center": 0},
    }

    figs = {}

    for var, spec in metric_specs.items():
        fig, axes = plot_panel_maps(
            regions=regions,
            panel=mig[["region_id", period_col, var]],
            value_col=var,
            title=spec["title"],
            region_id_col=region_id_col,
            panel_region_col="region_id",
            period_col=period_col,
            cmap=spec["cmap"],
            center=spec["center"],
            legend_label=var,
        )
        figs[var] = (fig, axes)
        plt.close()

    return figs

def plot_n_links_maps(
    regions,
    df,
    region_id_col="id",
    period_col="decade",
    add_average=True,
):
    sender_hhi, receiver_hhi = build_hhi_panels(df, period_col=period_col)
    sender_hhi['n_links'] = 1/sender_hhi['sender_hhi']
    receiver_hhi['n_links'] = 1/receiver_hhi['receiver_hhi']

    figs = {}

    for name, panel in {
        "sender_hhi": sender_hhi,
        "receiver_hhi": receiver_hhi,
    }.items():

        if add_average:
            panel = _make_average_panel(panel, region_col="region_id", period_col=period_col, value_cols=['n_links'])

        title = "Sender Effective Links" if name == "sender_hhi" else "Receiver Effective Links"

        fig, axes = plot_panel_maps(
            regions=regions,
            panel=panel[["region_id", period_col, 'n_links']],
            value_col='n_links',
            title=title,
            region_id_col=region_id_col,
            panel_region_col="region_id",
            period_col=period_col,
            cmap="magma",
            center=None,
            legend_label=name,
        )
        figs[name] = (fig, axes)
        plt.show()
        plt.close()

    return figs

def plot_outside_option_maps(
    regions,
    df,
    region_id_col="id",
    period_col="decade",
    add_average=True,
):
    oop = build_outside_option_panel(
        df,
        attrs=["gdp_capita", "comp_rel_within"],
        period_col=period_col,
    )

    if add_average:
        value_cols = [c for c in oop.columns if c not in ["region_id", period_col]]
        oop = _make_average_panel(oop, region_col="region_id", period_col=period_col, value_cols=value_cols)

    metric_specs = {
        "outside_option_gap_gdp_capita": {
            "title": "Outside-option GDP pc gap (origin - weighted destination)",
            "cmap": "RdBu_r",
            "center": 0,
        },
        "outside_option_gap_comp_rel_within": {
            "title": "Outside-option climate gap (origin - weighted destination)",
            "cmap": "RdBu_r",
            "center": 0,
        },
    }

    figs = {}

    for var, spec in metric_specs.items():
        if var not in oop.columns:
            continue

        fig, axes = plot_panel_maps(
            regions=regions,
            panel=oop[["region_id", period_col, var]],
            value_col=var,
            title=spec["title"],
            region_id_col=region_id_col,
            panel_region_col="region_id",
            period_col=period_col,
            cmap=spec["cmap"],
            center=spec["center"],
            legend_label=var,
        )
        figs[var] = (fig, axes)
        plt.close()

    return figs

def plot_top_region_corridor_maps(
    regions,
    df,
    top_tables=None,
    region_id_col="id",
    period_col="decade",
    value_col="partner_flow",      # "partner_flow" or "partner_share_pct"
    ncols=3,
):
    """
    Creates maps of partners for top 10 senders/receivers.
    Returns nested dict of figures.
    """
    if top_tables is None:
        top_tables = top_receiver_sender_tables(df, period_col=period_col)

    configs = {
        "top_receivers_absolute":      {"role": "receiver", "name_col": "region_name"},
        "top_receivers_population_pct":{"role": "receiver", "name_col": "region_name"},
        "top_senders_absolute":        {"role": "sender",   "name_col": "region_name"},
        "top_senders_population_pct":  {"role": "sender",   "name_col": "region_name"},
    }

    out = {}

    for key, cfg in configs.items():
        tab = top_tables[key].copy()
        tab["region_id"] = tab["region_id"].astype(str)

        out[key] = {}

        for _, row in tab.iterrows():
            focal_id = row["region_id"]
            focal_name = row[cfg["name_col"]]

            partner_panel = build_partner_flow_panel(
                df=df,
                focal_region_id=focal_id,
                role=cfg["role"],
                period_col=period_col,
            )

            fig, axes = plot_focal_partner_maps(
                regions=regions,
                partner_panel=partner_panel,
                focal_region_id=focal_id,
                value_col=value_col,
                region_id_col=region_id_col,
                period_col=period_col,
                cmap="Reds" if cfg["role"] == "receiver" else "Blues",
                ncols=ncols,
            )

            fig.suptitle(f"{focal_name} ({focal_id})", fontsize=14, y=1.02)
            out[key][focal_id] = (fig, axes)
            plt.close()

    return out

def run_all_map_plots(
    regions,
    df,
    region_id_col="id",
    period_col="decade",
):
    """
    Runs everything and returns nested dictionaries of figures.
    """
    outputs = {}

    outputs["fundamentals"] = plot_fundamental_maps(
        regions=regions,
        df=df,
        region_id_col=region_id_col,
        period_col=period_col,
    )

    outputs["migration"] = plot_migration_maps(
        regions=regions,
        df=df,
        region_id_col=region_id_col,
        period_col=period_col,
    )

    outputs["hhi"] = plot_hhi_maps(
        regions=regions,
        df=df,
        region_id_col=region_id_col,
        period_col=period_col,
    )

    outputs["outside_option"] = plot_outside_option_maps(
        regions=regions,
        df=df,
        region_id_col=region_id_col,
        period_col=period_col,
    )

    top_tables = top_receiver_sender_tables(df, period_col=period_col)
    outputs["top_tables"] = top_tables

    outputs["top_region_corridors"] = plot_top_region_corridor_maps(
        regions=regions,
        df=df,
        top_tables=top_tables,
        region_id_col=region_id_col,
        period_col=period_col,
        value_col="partner_flow",   # change to "partner_share_pct" if preferred
    )

    return outputs

# %% MAPS: Saver
def _safe_filename(x, max_len=140):
    """
    Makes strings safe for filenames.
    """
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:max_len]


def _is_fig_tuple(obj):
    """
    Detects objects of the form (fig, axes).
    """
    return (
        isinstance(obj, tuple)
        and len(obj) >= 1
        and hasattr(obj[0], "savefig")
    )


def save_figure(
    fig,
    path_without_ext,
    formats=("png",),
    dpi=220,
    bbox_inches="tight",
):
    """
    Saves one matplotlib figure in one or more formats.
    """
    path_without_ext = Path(path_without_ext)

    for fmt in formats:
        outpath = path_without_ext.with_suffix(f".{fmt}")
        outpath.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            outpath,
            dpi=dpi,
            bbox_inches=bbox_inches,
        )


def save_dataframe(
    df,
    path_without_ext,
    index=False,
):
    """
    Saves one dataframe as CSV.
    """
    path_without_ext = Path(path_without_ext)
    outpath = path_without_ext.with_suffix(".csv")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(outpath, index=index)


def save_nested_map_outputs(
    obj,
    out_dir,
    name_prefix="",
    formats=("png",),
    dpi=220,
    close_figures=True,
    save_tables=True,
):
    """
    Recursively saves nested dictionaries containing:
    - matplotlib figure tuples: (fig, axes)
    - pandas DataFrames
    - nested dictionaries

    Designed for outputs from run_all_map_plots().
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    if _is_fig_tuple(obj):
        fig = obj[0]

        fname = _safe_filename(name_prefix or "figure")
        path = out_dir / fname

        save_figure(
            fig,
            path,
            formats=formats,
            dpi=dpi,
        )

        saved.extend([str(path.with_suffix(f".{fmt}")) for fmt in formats])

        if close_figures:
            plt.close(fig)

        return saved

    if isinstance(obj, pd.DataFrame):
        if save_tables:
            fname = _safe_filename(name_prefix or "table")
            path = out_dir / fname
            save_dataframe(obj, path)
            saved.append(str(path.with_suffix(".csv")))

        return saved

    if isinstance(obj, dict):
        for key, val in obj.items():
            safe_key = _safe_filename(key)

            # Put nested groups in folders, but figures directly get prefixed names
            if _is_fig_tuple(val) or isinstance(val, pd.DataFrame):
                child_prefix = safe_key if not name_prefix else f"{name_prefix}_{safe_key}"
                saved.extend(
                    save_nested_map_outputs(
                        val,
                        out_dir=out_dir,
                        name_prefix=child_prefix,
                        formats=formats,
                        dpi=dpi,
                        close_figures=close_figures,
                        save_tables=save_tables,
                    )
                )
            else:
                child_dir = out_dir / safe_key
                saved.extend(
                    save_nested_map_outputs(
                        val,
                        out_dir=child_dir,
                        name_prefix="",
                        formats=formats,
                        dpi=dpi,
                        close_figures=close_figures,
                        save_tables=save_tables,
                    )
                )

        return saved

    return saved

def save_all_map_outputs(
    all_maps,
    root_dir="outputs/maps",
    formats=("png",),
    dpi=220,
    close_figures=True,
    save_tables=True,
):
    """
    Saves all maps and tables produced by run_all_map_plots().

    Expected folder structure:
        root_dir/
            fundamentals/
            migration/
            hhi/
            outside_option/
            top_tables/
            top_region_corridors/
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for group_name, group_obj in all_maps.items():
        group_dir = root_dir / _safe_filename(group_name)

        saved_files.extend(
            save_nested_map_outputs(
                group_obj,
                out_dir=group_dir,
                name_prefix="",
                formats=formats,
                dpi=dpi,
                close_figures=close_figures,
                save_tables=save_tables,
            )
        )

    saved_index = pd.DataFrame({"path": saved_files})
    saved_index.to_csv(root_dir / "_saved_files_index.csv", index=False)

    return saved_index
# %% MAPS: Runner
all_maps = run_all_map_plots(
    regions=regions,
    df=df,
    region_id_col="id",
    period_col="decade",
)

saved = save_all_map_outputs(
    all_maps,
    root_dir=f"{path}/figs/descriptive_maps",
    formats=("png",),      # or ("png", "pdf")
    dpi=220,
    close_figures=True,
    save_tables=True,
)

saved.head()


# %% Convenience wrapper


def run_all_migration_objects(
    df,
    flow_col=DEFAULT_FLOW
    ):
    """
    Runs all requested high-level migration objects.
    """
    results = {
        "1_origin_destination_correlations": origin_destination_correlations(df, flow_col=flow_col),
        "2_receiver_concentration": receiver_concentration(df, flow_col=flow_col),
        "3_sender_concentration": sender_concentration(df, flow_col=flow_col),
        "4_sender_receiver_network_hhi": sender_receiver_network_hhi(df, flow_col=flow_col),
        "5_migration_basins": migration_basins(df, regions, flow_col=flow_col),
        "6_persistence": migration_persistence(df, flow_col=flow_col),
        "7_reciprocity": migration_reciprocity(df, flow_col=flow_col),
        "8_redistribution": migration_redistribution(df, flow_col=flow_col),
        "9_outside_option_ranks": outside_option_ranks(df, flow_col=flow_col),
        "10_rank_matrices_3x3": migration_rank_matrix_3x3(df, flow_col=flow_col),
        "11_trapped_population_exit_rates": trapped_population_exit_rates(df, flow_col=flow_col),
        "12_top_sender_receiver": top_receiver_sender_tables(df, flow_col=flow_col),
    }

    return results

#results = run_all_migration_objects(df)
mtx = migration_rank_matrix_3x3(df)

# %%

attr = 'gdp_serv_share'
mtx['pivots'][(1980, attr)]
mtx['pivots'][(1990, attr)]
mtx['pivots'][(2000, attr)]
mtx['pivots'][(2010, attr)]
mtx['pivots'][('all', attr)]

# %%
# slide 1
slide_1_1_df = results["12_top_sender_receiver"]['top_receivers_absolute'][['region_name', 'avg_inflow']]
slide_1_2_df = results["12_top_sender_receiver"]['top_receivers_population_pct'][['region_name', 'avg_inflow_pop_pct']]

# slide 2
slide_2_1_df = results["12_top_sender_receiver"]['top_senders_absolute'][['region_name', 'avg_outflow']]
slide_2_2_df = results["12_top_sender_receiver"]['top_senders_population_pct'][['region_name', 'avg_outflow_pop_pct']]

# slide 3
slide_3_df = pd.concat([
    results['2_receiver_concentration'].groupby('dest_id').mean()[['effective_origins']].describe(),
    results['3_sender_concentration'].groupby('orig_id').mean()[['effective_destinations']].describe()], axis=1)

# slide 4
slide_4_df = results['6_persistence']['origin_persistence'][['decade', 'origin_outflow_corr']]
slide_4_df = pd.merge(
    slide_4_df,
    results['6_persistence']['destination_persistence'][['decade', 'destination_inflow_corr']],
    on='decade')

# slide 5
slide_5_df = results['4_sender_receiver_network_hhi']['global_od_hhi'][
    ['decade','effective_corridors']
    ]
slide_5_df = pd.merge(
    slide_5_df,
    results['6_persistence']['corridor_persistence'][['decade', 'corridor_flow_corr', 'link_survival_rate']],
    on='decade'
    )

# slide 6
results['5_migration_basins']['map'][1].show()

# slide 7
slide_7_df = results['7_reciprocity']['period_summary'][['decade', 'weighted_reciprocity', 'weighted_one_wayness']]
slide_7_df = pd.merge(
    slide_7_df,
    results['8_redistribution']['period_summary'][['decade', 'redistribution_index']],
    on='decade'
    )

# slide 8
results['11_trapped_population_exit_rates']['correlations'].groupby(
    'attribute')[['corr_exit_rate_attribute', 'pop_weighted_corr']].mean()[1:]
results['10_rank_matrices_3x3']['pivots'][('all', 'gdp_capita')]
results['10_rank_matrices_3x3']['pivots'][('all', 'comp_rel_within')]

# %%

def pretty_latex_df(df, label, caption, columns_names=None, reset_index=False):
    df_use = df
    if not(columns_names is None):
        df_use.columns = columns_names
    if reset_index:
        df_use = df.reset_index()
    
    df_latex = df_use.to_latex(
        index=False,
        float_format="%.1f",
        bold_rows=True,
        label=label,
        caption=caption
        )
    print(df_latex)
    return df_latex

slide_1_latex = pretty_latex_df(
    slide_1_2_df, 'top_receivers', 'Top Destination Regions',  
    ['Region', 'Average Inflow as Population Share']
    )
slide_2_latex = pretty_latex_df(
    slide_2_2_df, 'top_senders', 'Top Origin Regions',  
    ['Region', 'Average Outflow as Population Share']
    )
slide_3_latex = pretty_latex_df(
    slide_3_df, 'hhi_links', 'HHI implied linkages',  
    ['Effective Origins', 'Effective Destinations'], reset_index=True
    )
slide_4_latex = pretty_latex_df(
    slide_4_df*100, 'od_persistence', 'Persistence in Origins and Destinations',  
    ['Decade', 'Origin Outflow Autocorrelation', 'Destination Inflow Autocorrelation']
    )
slide_5_latex = pretty_latex_df(
    slide_5_df*100, 'corridors', 'Migration Corridor Analysis',  
    ['Decade', 'Effective Corridors', 'Flow Autocorrelation', 'Average Survival Rate']
    )
slide_7_latex = pretty_latex_df(
    slide_7_df*100, 'redistr', 'Distributive Effects of Migrations',  
    ['Decade', 'Reciprocity', 'One-Wayness', 'Redistribution Index']
    )




results['11_trapped_population_exit_rates']['correlations'].groupby(
    'attribute')[['corr_exit_rate_attribute', 'pop_weighted_corr']].mean()[1:]
results['10_rank_matrices_3x3']['pivots'][('all', 'gdp_capita')]
results['10_rank_matrices_3x3']['pivots'][('all', 'comp_rel_within')]















