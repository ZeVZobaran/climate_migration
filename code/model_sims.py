# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:37:56 2026

@author: c337191
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.ion()
path = r'D:/Users/c337191/Documents/EnvEcon'

# %% get data
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:5880").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

exposure_index = pd.read_parquet(f'{path}//data/climate_indexes/exposure_pca_mean.parquet')

population = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='pop_mesorreg_interpol'
    )
gdp_per_capita = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_capita_mesorreg'
    )

# %% making IPEA dfs tidy
def make_ipea_tidy(df, var):
    df['CD_GEOCME'] = pd.to_numeric(df['CD_GEOCME'])
    id_cols = ['Sigla', 'CD_GEOCME', 'NM_MESO']
    year_cols = [c for c in df.columns if c not in id_cols]
    df_tidy = (
    df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year",
        value_name=var
        )
    )
    df_tidy["year"] = df_tidy["year"].astype(int)
    df_tidy = df_tidy.sort_values(["CD_GEOCME", "year"]).reset_index(drop=True)
    return df_tidy

gdp_per_capita = make_ipea_tidy(gdp_per_capita, 'gdp_capita')
population = make_ipea_tidy(population, 'pop')

# %% viz gdp per capita

def plot_regions(data_df, regions_df, var, year='all',
                 legend='index', center='infer', title='placeholder'):
    if year == 'all':
        df_use = data_df[['CD_GEOCME', var]].groupby('CD_GEOCME').mean().reset_index()
    else:
        df_use = data_df.loc[
            data_df['year']==year, ['CD_GEOCME', var]
            ].groupby('CD_GEOCME').mean().reset_index()

    # Keep only what we need
    plot_gdf = regions_df.merge(
        df_use,
        left_on="CD_GEOCME",
        right_on="CD_GEOCME",
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
        vmin=plot_gdf[var].quantile(0.01),
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
    ax.set_title(
        title,
        fontsize=12
        )
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax

fig_gdp = plot_regions(gdp_per_capita, regions, 'gdp_capita', 2010,
                   legend='', title='GDP per Capita by Region\n2010')
fig_exp = plot_regions(exposure_index, regions, 'pca_exp', 2010,
                   legend='', title='Climate Exposure by Region\n2010')
fig_pop = plot_regions(population, regions, 'pop', 2010,
                   legend='', title='Total Population by Region\n2010')

plt.show()

# %% Implementing a standard Roback model
# U = u_i*C_i^1-sigma*h_i^sigma
# Y = AiLi

# set sigma to 0.2 --> use 0.8 of income in consumption, 0.2 in housing
sigma = 0.2

# Ai = gdp_capita, normalize RM_SP = 1
sprm = gdp_per_capita[gdp_per_capita['CD_GEOCME']==3512][['year', 'gdp_capita']]
sprm = sprm.rename(columns={'gdp_capita': 'sp_baseline'})
gdp_per_capita = gdp_per_capita.merge(sprm, on="year", how="left")
gdp_per_capita['a_i'] = gdp_per_capita['gdp_capita']/gdp_per_capita['sp_baseline']
ai_df = gdp_per_capita[['CD_GEOCME', 'NM_MESO', 'year', 'a_i']]

# Ui = exposure_index
# averaging thourgh intervals in gdp / population data
model_years = ai_df["year"].unique()
exposure_matched = (
    exposure_index[exposure_index["year"].isin(model_years)]
    .groupby(["CD_GEOCME", "year"], as_index=False)
    .agg({
        "mean_exp": "mean",
        "pca_exp": "mean"
    })
    )

# population shares
population['pop_share'] = (
    population['pop'] /
    population.groupby('year')['pop'].transform('sum')
    )
pop_shares = population.drop(['Sigla', 'NM_MESO', 'pop'], axis=1)

ai_df =  ai_df.merge(
    pop_shares,
    on=["CD_GEOCME", "year"],
    how="left"
    )

model_df = ai_df.merge(
    exposure_matched,
    on=["CD_GEOCME", "year"],
    how="left"
    ).dropna().reset_index(drop=True)

# transform exposure into amenity
# lambda: how strongly exposure affects utility
lam = 2
model_df['u_i'] = np.exp(-lam * model_df['pca_exp'])
index = 'u_i'

# %% computing equilibrium pop shares
# 1st exercise: no migration costs, no size heterogeneity

model_df['sum_term_1'] = (
    model_df[index]**(1/sigma) *
    model_df['a_i']**((1-sigma)/sigma)
)

sum_by_year = model_df.groupby('year')['sum_term_1'].transform('sum')

model_df['v_1'] = (1-sigma)**(1-sigma) * (sum_by_year**sigma)

model_df['pred_pop_share_1'] = (
    (((1-sigma)**(1-sigma)) / model_df['v_1'])**(1/sigma)
    * model_df['sum_term_1']
)

# sanity check
model_df[['year', 'pred_pop_share_1']].groupby('year').sum()
# diagnostics
model_df['pop_error_1'] = (model_df['pred_pop_share_1']-model_df['pop_share'])/model_df['pop_share']

plot_df = model_df.sort_values('pop_error_1')

fig_err_1 = plot_regions(plot_df, regions, 'pop_error_1', 2010,
                   legend='pred - act', title='Model 1 Error')
plt.show()

# First results:
    # Model basically throws everyone at a single region every year
    # even accounting for that, plenty of black holes remain
    # many places with many times over real error
    # very bad!

# %% exercise 2: add size heterogeneity
# basically multiply by T_i for Li
# and multiply by T_i inside the sum for V

regions_area = regions.copy()
regions_area ['area_m2'] = regions_area .geometry.area
# convert to km²
regions_area ['T_i'] = regions_area ['area_m2'] / 1e6
size_df = regions_area[['CD_GEOCME', 'T_i']].copy()

model_df = model_df.merge(size_df, on='CD_GEOCME', how='left')

model_df['eq_term'] = (
    model_df['T_i'] *
    model_df[index]**(1/sigma) *
    model_df['a_i']**((1-sigma)/sigma)
    )

model_df['pred_pop_share_2'] = (
    model_df['eq_term'] /
    model_df.groupby('year')['eq_term'].transform('sum')
    )

# sanity:
model_df.groupby('year')['pred_pop_share_2'].sum()

# diagnostics
model_df['pop_error_2'] = (model_df['pred_pop_share_2']-model_df['pop_share'])/model_df['pop_share']

plot_df = model_df.sort_values('pop_error_2')

fig_err_2 = plot_regions(plot_df, regions, 'pop_error_2', 2010,
                   legend='pred - act', title='Model 2 Error')
plt.show()

model_df[['pop_error_1', 'year']].groupby('year').mean()
model_df[['pop_error_2', 'year']].groupby('year').mean()

err_1 = model_df[['pop_error_1', 'NM_MESO']].groupby('NM_MESO').mean()
err_2 = model_df[['pop_error_2', 'NM_MESO']].groupby('NM_MESO').mean()

# Second results:
    # Similar, only now we believe loads of people should move north due to 
    # low rents
    # MODEL SUCKS

#%% Now adding migration costs
# My guess: the model will become EVEN WORSE




# %% maybe useful in future
# compiling SP Metro, RJ Metro and BH Metro (only regs > 5mn people)
main_metros = [3107, 3512, 3306]
# compiling regions with > 1 on relative aridity score
poli_secas = [
    2204, # piaui
    2301, 2302, 2304, 2305, 2306, 2307, # ceara
    2401, 2402, 2403, 2404, # RN
    2501, 2502, # Paraiba
    2601, 2602, 2603, # Pernambuco
    2701, 2702, # Alagoas
    2801 # Sergipe
    ]

nordeste_exba = [
    2204, 2201, 2202, 2203,  # piaui
    2301, 2302, 2304, 2305, 2306, 2307, 2303, # ceara
    2401, 2402, 2403, 2404, # RN
    2501, 2502, 2503, 2504, # Paraiba
    2601, 2602, 2603, 2604, 2605, # Pernambuco
    2701, 2702, 2703, # Alagoas
    2801, 2802, 2803, # Sergipe
    2101, 2012, 2103, 2104, 2105 # Maranhão
    ]


# TODO generate relative sizes
# TODO compile GDP per capita
# TODO fit model!

# after that, modelling migration
# and at last, empirics