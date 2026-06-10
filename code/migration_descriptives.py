# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:30:55 2026

@author: c337191
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math

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

df = pd.read_parquet(
    f'{path}/data/formatted_composit/climate_mig_meso.parquet'
    )

df = df[[
    'orig_id', 'dest_id', 'N_od_flow_wm', 'N_od_flow_all', 'decade',
           'orig_comp_rel_within', 'dest_comp_rel_within',
           'orig_comp_rel_between', 'dest_comp_rel_between', 'pair_id', 'orig_abbrv', 'orig_name',
           'orig_gdp_capita', 'dest_abbrv', 'dest_name', 'dest_gdp_capita',
           'orig_pop', 'dest_pop'
    ]]

# %% Defaults
# Most of these are chat-made and unneeded, since my data is already
# quite clean. Still, I'll keep them for cleanliness

DEFAULT_FLOW = "N_od_flow_all"
DEFAULT_PERIOD = "decade"
DEFAULT_ORIG = "orig_id"
DEFAULT_DEST = "dest_id"

DEFAULT_ATTRS = (
    "gdp_capita",
    "pop",
    "comp_rel_within",
    "comp_rel_between",
)

DEFAULT_RANK_ATTRS = (
    "gdp_capita",
    "comp_rel_within",
    "comp_rel_between",
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


# %%
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

# %%
# %% 5. Graph-like / cluster analysis for migration basins

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
    reg = region_attributes_from_od(d)

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
    print('here')
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

results = run_all_migration_objects(df)

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















