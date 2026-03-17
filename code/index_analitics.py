# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:04:01 2026

@author: c337191
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.colors as colors
plt.ion()
path = r'D:/Users/c337191/Documents/EnvEcon'
index_path = f'{path}/data/climate_indexes'
# %% index downloads
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:4326").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

exposure_z_index = pd.read_parquet(f'{index_path}/exposure_z_scores.parquet')

# these ended up being a dead end - just map distance to equator
livability_index = pd.read_parquet(f'{index_path}/livability.parquet')
stability_index = pd.read_parquet(f'{index_path}/stability.parquet')

# %%

corr = exposure_z_index[["heat", "flood", 'drought_anomaly', 'drought_absolute', 'exposure_index']].corr()

exposure_z_index['ex_abs'] = exposure_z_index[["flood", 'drought_anomaly', 'heat']].mean(axis=1)

# generating PCA-based index
exposure_z_index["pca_index"] = np.nan
exposure_z_index["pca_ex_abs"] = np.nan

# PCA
pca = PCA(n_components=1)
pc = pca.fit_transform(exposure_z_index[["flood", 'drought_anomaly', 'drought_absolute', 'heat']].values)
exposure_z_index.loc[:, "pca_index"] = pc.T[0]*-1  #rotation agnostic
# PCA ex abs dryness
pca = PCA(n_components=1)
pc = pca.fit_transform(exposure_z_index[["flood", 'drought_anomaly', 'heat']].values)
exposure_z_index.loc[:, "pca_ex_abs"] = pc.T[0]

corr = exposure_z_index[["exposure_index", 'ex_abs', 'pca_index', 'pca_ex_abs']].corr()
exposure_z_index[['year', "exposure_index", 'ex_abs', 'pca_index', 'pca_ex_abs']].groupby('year').mean().plot()
plt.show()

# averaging and taking PCs seems similar


# final exposure index: distance from 0 squared
# idea: big exposure much worse than small exposure
exposure_z_index[["exposure_index", 'ex_abs', 'pca_index', 'pca_ex_abs']] = \
    exposure_z_index[["exposure_index", 'ex_abs', 'pca_index', 'pca_ex_abs']]**2

# %% full time-sample vizualization

def plot_climate_regions(climate_df, regions_df, var, year='all',
                         legend='Exposure Index (z-score)', center='infer'):
    if year == 'all':
        df_use = climate_df[['region', var]].groupby('region').mean().reset_index()
    else:
        df_use = climate_df.loc[
            climate_df['year']==year, ['region', var]
            ].groupby('region').mean().reset_index()

    # Keep only what we need
    plot_gdf = regions_df.merge(
        df_use,
        left_on="CD_GEOCME",
        right_on="region",
        how="left"
    )
    # centering 0
    if center == 0:
        norm = colors.TwoSlopeNorm(
        vmin=plot_gdf[var].min(),
        vcenter=0,
        vmax=plot_gdf[var].max()
        )
    else:
        norm = colors.TwoSlopeNorm(
        vmin=0,
        vcenter=plot_gdf[var].median(),
        vmax=plot_gdf[var].quantile(0.99)
        )
        

    fig, ax = plt.subplots(figsize=(4, 4))
    plot_gdf.plot(
        column=var,
        ax=ax,
        legend=True,
        legend_kwds={
            "label": legend,
            "shrink": 0.6
            },
        edgecolor="black",
        linewidth=0.3,
        missing_kwds={"color": "lightgrey", "label": "Missing"},
        cmap='seismic',
        norm=norm
    )
    return fig, ax
#%%

fig, fig_ax = plot_climate_regions(
    exposure_z_index, regions, 'pca_index', legend='Heat, floods and aridity\nexposure index'
    )
exposure_z_index[['region', 'pca_index']].groupby('region').mean().min()

fig_ax.set_title(
    'Extreme Climate Exposure \n1961-2024',
    fontsize=12
    )
fig_ax.set_axis_off()
fig_ax.set_aspect("equal")
# =============================================================================
# cbar = fig_ax.axes[1]
# yticks = np.arange(0, 5, 0.5)
# cbar.set_yticks(yticks)
# =============================================================================
plt.show()

# preserve the simple mean index as a robustness check
exposure_index = exposure_z_index.drop(
    ['heat', 'drought_anomaly', 'flood', 'drought_absolute',
     'ex_abs', 'pca_ex_abs', 'abbrevs', 'names'], axis=1)

exposure_index.columns = ['CD_GEOCME', 'year', 'NM_MESO', 'mean_exp', 'pca_exp']

exposure_index.to_parquet(
    f'{path}/data/climate_indexes/exposure_pca_mean.parquet', index=False
    )

# ID polígono das secas region

ord_drought = exposure_z_index[['region', 'NM_MESO', 'drought_absolute']].groupby(['region', 'NM_MESO']).mean().reset_index().sort_values('drought_absolute')

print(ord_drought.iloc[-20:].sort_values('region'))






