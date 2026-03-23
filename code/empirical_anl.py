# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:52:17 2026

@author: c337191
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import pyfixest as pf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

sns.set(style="whitegrid")

path = r'D:\Users\c337191\Documents\climate_migration'

# %% get data
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:5880").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

exposure_index = pd.read_parquet(f'{path}//data/climate_indexes/exposure_pca_mean.parquet')
exposure_subindex = pd.read_parquet(f'{path}//data/climate_indexes//exposure_z_scores.parquet')

population = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='pop_mesorreg_interpol'
    )
gdp_per_capita = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_capita_mesorreg'
    )

gdp = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_tot_mes'
    )
mo_meso = pd.read_stata(
    f'{path}/data/morten_oliveira_final_tables/N_od_meso.dta'
    )

mo_meso_tt = pd.read_stata(
    f'{path}/data/morten_oliveira_final_tables/tt_mesospeed_10.dta'
    )

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
gdp = make_ipea_tidy(gdp, 'gdp')

gdp_per_capita['log_gdppc'] = np.log(gdp_per_capita['gdp_capita'])
gdp_per_capita["dlog_gdppc"] = (
    gdp_per_capita.groupby("CD_GEOCME")["log_gdppc"]
      .diff()
      )
gdp_per_capita["year_lag"] = gdp_per_capita.groupby("CD_GEOCME")["year"].shift(1)
gdp_per_capita["dlog_gdppc_ann"] = gdp_per_capita["dlog_gdppc"] /(gdp_per_capita["year"] - gdp_per_capita["year_lag"])

# add spread
# normalize by year!
gdp_per_capita["log_gdppc_z"] = (
    gdp_per_capita.groupby("year")["log_gdppc"]
    .transform(lambda x: (x - x.mean()) / x.std())
)
gdp_per_capita["delta_gdppc_spread"] = (
    gdp_per_capita.groupby("CD_GEOCME")["log_gdppc_z"]
      .diff()
      )

population["log_pop"] = np.log(population["pop"])
population["dlog_pop"] = (
    population.groupby("CD_GEOCME")["log_pop"]
      .diff()
      )
population["year_lag"] = population.groupby("CD_GEOCME")["year"].shift(1)
population["dlog_pop_ann"] = population["dlog_pop"] /(population["year"] - population["year_lag"])


gdp_cap_years = gdp_per_capita['year'].unique()
exposure_matched = (
    exposure_index[exposure_index["year"].isin(gdp_cap_years)]
    .groupby(["CD_GEOCME", "year"], as_index=False)
    .agg({
        "mean_exp": "mean",
        "pca_exp": "mean"
    })
    )

climate_subindex_matched = (
    exposure_subindex[exposure_subindex["year"].isin(gdp_cap_years)]
    .groupby(["region", "year"], as_index=False)
    .agg({
        "drought_anomaly": "mean",
        "drought_absolute": "mean",
        'heat': "mean",
        'flood': "mean"
    })
    )
# name complying
climate_subindex_matched = climate_subindex_matched.rename({'region': 'CD_GEOCME'}, axis=1)

# df carrying info we will use from now on (mostly)
model_df =  gdp_per_capita.merge(
    population,
    on=['Sigla', 'NM_MESO', "CD_GEOCME", "year"],
    how="left"
    )

model_df =  model_df.merge(
    exposure_matched,
    on=["CD_GEOCME", "year"],
    how="left"
    ).dropna().reset_index(drop=True)

model_df =  model_df.merge(
    climate_subindex_matched,
    on=["CD_GEOCME", "year"],
    how="left"
    ).dropna().reset_index(drop=True)

model_df = model_df.drop(['year_lag_x', 'year_lag_y', 'dlog_gdppc', 'dlog_pop'],
                         axis=1)

# %% Summary statistics
# key data
# Migration: meso-meso migration, state-state migration (later)
# Population change, GDPPC change: Mean, dev, qunatiles
# Climate: N_Meso_Year per bin, stability
# Travel cost: Mean, dev, quantile, change over time
# GDP_Spread: change over time (plot variance change over time?)

# scatter plots:
    # pop change vs climate
    # pop change vs travel time
    # gdppc change vs pop change
    # gdppc diff vs pop change
    # gdppc diff vs climate
    # gdppc diff vs tt

def scatter_plot(df, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(7,5))
    
    sns.scatterplot(data=df, x=x, y=y, alpha=0.5)
    sns.regplot(data=df, x=x, y=y, scatter=False)  # adds fit line
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    plt.tight_layout()
    plt.show()

scatter_plot(
    model_df,
    x="mean_exp",
    y="pca_exp",
    xlabel="Mean Exposure",
    ylabel="PCA Exposure",
    title="Climate Index Consistency"
)
# mean seems to be "stricter"; PCA thinks some places that are mean-exposed
# are fine. Lets go with mean.
climate_index = 'mean_exp'
model_df_mean = model_df[['year', 'CD_GEOCME', 'dlog_pop_ann', climate_index,
                          'dlog_gdppc_ann', 'log_gdppc_z', 'delta_gdppc_spread']].groupby('CD_GEOCME').mean()

# Climate drives smaller pop growth?
# Seems so!
scatter_plot(
    model_df_mean,
    x=climate_index ,
    y="dlog_pop_ann",
    xlabel="Climate Exposure",
    ylabel="Population Growth (log, annualized)",
    title="Population Change vs Climate Exposure"
)

# Population growth increases or reduces GDP per capita growth?
# Increases!
scatter_plot(
    model_df_mean,
    x="dlog_pop_ann",
    y="dlog_gdppc_ann",
    xlabel="Population Growth (log, annualized)",
    ylabel="GDP per Capita Growth (log, annualized)",
    title="GDP per Capita Growth vs Population Change"
)

# Population change impacts releative inequality?
# Places with higher growth become relatively _much_ richer!
scatter_plot(
    model_df_mean,
    y="dlog_pop_ann",
    x="delta_gdppc_spread",
    ylabel="Population Growth (log, annualized)",
    xlabel="Relative GDP per Capita Change",
    title="Inequality vs Population Change"
)

# Bad climate drives relative inequality change?
# Zero relation at first glance
scatter_plot(
    model_df_mean,
    x=climate_index,
    y="delta_gdppc_spread",
    xlabel="Climate Exposure",
    ylabel="Relative Inequality",
    title="Inequality vs Climate Exposure"
)

# Bad climate drives gdppc growth?
# Also little to no impact overall
scatter_plot(
    model_df_mean,
    x=climate_index,
    y="dlog_gdppc_ann",
    xlabel="Climate Exposure",
    ylabel="GDP per Capita Growth",
    title="Does Climate Affect Growth?"
)


# Partial zobaran victory!

# some checks

# Is there gdp per capita convergence going on?
# Yes! Same for population
model_df["log_gdppc_lag"] = model_df.groupby("CD_GEOCME")["log_gdppc"].shift(1)
model_df["log_pop_lag"] = model_df.groupby("CD_GEOCME")["log_pop"].shift(1)
scatter_plot(
    model_df,
    x="log_gdppc_lag",
    y="dlog_gdppc_ann",
    xlabel="Initial GDP per Capita (log)",
    ylabel="GDP Growth",
    title="Conditional Convergence"
)
scatter_plot(
    model_df,
    x="log_pop_lag",
    y="dlog_pop_ann",
    xlabel="Initial Population (log)",
    ylabel="Population Growth",
    title="Convergence / Scale Effects"
)
# Further evidence of sharp inequality decrease in sample:
# Veeeeery much driven by 1990-200 and 2005-2010
model_df[['year', 'log_gdppc']].groupby('year').var().plot()

# Richer places experience higher pop growth (or vice versa!)
scatter_plot(
    model_df_mean,
    x="log_gdppc_z",
    y="dlog_pop_ann",
    xlabel="Relative Income (Z-score)",
    ylabel="Population Growth",
    title="Do People Move Toward Richer Regions?"
)

#%% Regs on mesorregions
# Advantage: more region
# Disadvantage: less data (i think climate nullifies that tho)
# relevant cols:
    # 'N_od_flow_wm' (working males), 'N_od_flow_all'
    # orig_id_meso, dest_id_meso

# First re estimate their instrument
    # log_fm_empty - fast marching empty map; log_fm_mst_pie - fast marching post brasilia
    # log_fm_road - fm through existing roads -- dependent
    # Event study IV: log_fm_road ~ a_t(log_fm_empty-log_fm_mst_pie) FE_ot FE_dt FE_od

# Keep only needed columns
df = mo_meso_tt.copy()
cols = ["orig_id", "dest_id", "year", "log_fm_empty",
        "log_fm_mst_pie", "log_fm_road"]
df = df[cols].dropna().copy()
# Make sure year is integer-like
df["year"] = df["year"].astype(int)

# Instrument intensity: predicted reduction in travel time
df["z"] = df["log_fm_empty"] - df["log_fm_mst_pie"]
res = pf.feols(
    "log_fm_road ~ i(year, z, ref=1950) | orig_id^year + dest_id^year + orig_id^dest_id",
    data=df
    )

print(res.summary())

# good news! got exact replication! Alas, variation ceases by 1980s...
# the Brasilia shock lasts from 1950 to 1980, normalizes afterward

# set up two matrixes
# one mapping log fm road for each origin / destination
# and one mapping log instrumented travel time reduciton

years_iv = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
effects = [res.coef()[0], 0, res.coef()[1], res.coef()[2], res.coef()[3],
           res.coef()[4], res.coef()[5], res.coef()[6]]

event_study_df = pd.DataFrame(
    [effects, years_iv],
    index=['effect', 'year']
    ).T

df_tt = df[['orig_id', 'dest_id', 'year', "z"]]
df_tt = df_tt.merge(
    event_study_df, on=['year'], how='left'
    )
df_tt['IV_tt'] = res.predict()

# We now create two indexes: market desirability and bad climate 
# mkt desirabilty: highest GDPs in t-1
# bad climate: mean_exp > 1 (meaning original index > 1)
# lets see 10% best regions each year

top_regions = (
    gdp.sort_values(['year', 'gdp'], ascending=[True, False])
       .groupby('year')
       .head(14)[['year', 'CD_GEOCME']]
       .rename(columns={'CD_GEOCME': 'dest_id'}))

years_map = {
    1920: 1920,
    1939: 1940,
    1949: 1950,
    1959: 1960,
    1970: 1970,
    1980: 1980,
    1990: 1990,
    1991: 1990,
    2000: 2000,
    2010: 2010
    }
top_regions['year'] = top_regions['year'].map(years_map)
top_regions = top_regions[top_regions['year'].isin(years_iv)]
top_regions['is_good_mkt'] = 1
# Index for top econ regions is very stable!

df_tt = df_tt.merge(
    top_regions,
    on=['year', 'dest_id'],
    how='left'
)
df_tt['is_good_mkt'] = df_tt['is_good_mkt'].fillna(0)

# now making index: minimum travel time to any good market
# FIXME this is an obvious point to make progress
df_tt['ivtt_good_mkt'] = df_tt['is_good_mkt']*df_tt['IV_tt']
mkt_access_iv_tt = df_tt.groupby(['orig_id', 'year'])['ivtt_good_mkt'].min().reset_index()
mean_ivtt = df_tt.groupby(['orig_id', 'year'])['IV_tt'].mean().reset_index()

# climate dummy: bad if index higher than 1
model_df['bad_climate'] = 1*(model_df[climate_index]>=1)


# passing travel time data years to the model
model_df['year_tt'] = model_df['year'].map(years_map)

# merging
reg_df = model_df.merge(
    mkt_access_iv_tt,
    how='left',
    left_on=['CD_GEOCME', 'year_tt'],
    right_on=['orig_id', 'year']
    )

reg_df = reg_df.merge(
    mean_ivtt,
    how='left',
    left_on=['CD_GEOCME', 'year_tt'],
    right_on=['orig_id', 'year']
    ).dropna()

# climate data begins at 1961. We miss out on 1940, 1950 and 1970 as datapoints
reg_df['year'] = reg_df['year_y']
reg_df = reg_df.drop([
    'year_x', 'year_y', 'year_tt', 'orig_id_x', 'orig_id_y'
    ], axis=1)

# %% Regressions
# reg 1: dlog_pop_i,t ~ IV_tt_i,t ivtt_goodmkt_it
                      # bad_climate_i,t bad_climate_i,t*IV_tt_it bad_climate_i,t*mkt_access  
                      # gdppc_it-1 gdppc_z_it-1 FE_i FE_t

reg_df = reg_df.sort_values(['CD_GEOCME', 'year'])
# lag within region
reg_df['log_gdppc_lag']   = reg_df.groupby('CD_GEOCME')['log_gdppc'].shift(1)
reg_df['log_gdppc_z_lag'] = reg_df.groupby('CD_GEOCME')['log_gdppc_z'].shift(1)

reg1 = pf.feols(
    "dlog_pop_ann ~ IV_tt + ivtt_good_mkt + "  # movement + mkt access
    "bad_climate + bad_climate:IV_tt + bad_climate:ivtt_good_mkt + " # climate responses
    "log_gdppc_lag + log_gdppc_z_lag | " # controls
    "CD_GEOCME + year", # fixed effects
    data=reg_df,
    vcov={"CRV1": "CD_GEOCME"}   # cluster by region
)
reg1 = pf.feols(
    "dlog_pop_ann ~ IV_tt + ivtt_good_mkt + "  # movement + mkt access
    "bad_climate:IV_tt + log_gdppc_lag +  log_gdppc_z_lag |" # climate responses
    "CD_GEOCME + year", # fixed effects
    data=reg_df,
    vcov={"CRV1": "CD_GEOCME"}   # cluster by region
)

print(reg1.summary())

# this sucks. Lets try with migration data

# %% Reg 2: 
    # mig_ijt ~ IV_tt_ijt diff_climate_ijt badclimate_it badclimate_it*IV_tt
    # diff_log_gdppc_ijt FE_it FE_jt FE_ij

mo_meso['year'] = mo_meso['year'].map(years_map)  # 1991 --> 1990

# keep what we need
mig_odt = mo_meso[['orig_id', 'dest_id', 'year', 'N_od_flow_wm', 'N_od_flow_all', "log_fm_empty"]]

df_odt = df_tt[['orig_id', 'dest_id', 'year', 'IV_tt']].copy()

# building the variables:
region_year = model_df[['CD_GEOCME', 'year_tt', climate_index, 'bad_climate', 'log_gdppc']].copy()
region_year_orig = region_year.rename(columns={
    'CD_GEOCME': 'orig_id',
    climate_index: f'orig_{climate_index}',
    'bad_climate': 'orig_bad_climate',
    'log_gdppc': 'orig_log_gdppc',
    'year_tt': 'year'})

# origin-side merge
df_odt = df_odt.merge(
    region_year_orig,
    on=['orig_id', 'year'],
    how='left'
    )

region_year_dest = region_year.rename(columns={
    'CD_GEOCME': 'dest_id',
    climate_index: f'dest_{climate_index}',
    'bad_climate': 'dest_bad_climate',
    'log_gdppc': 'dest_log_gdppc',
    'year_tt': 'year'})

# destination-side merge
df_odt = df_odt.merge(
    region_year_dest,
    on=['dest_id', 'year'],
    how='left'
    )
df_odt['diff_climate']   = df_odt[f'dest_{climate_index}'] - df_odt[f'orig_{climate_index}']
df_odt['diff_log_gdppc_ijt'] = df_odt['dest_log_gdppc'] - df_odt['orig_log_gdppc']

df_odt = df_odt.merge(
    mig_odt,
    on=['orig_id', 'dest_id', 'year'],
    how='left'
    )

reg_df_2 = df_odt.copy()
reg_df_2 = reg_df_2.dropna(axis=0)
reg_df_2["pair_id"] = reg_df_2["orig_id"].astype(str) + "_" + reg_df_2["dest_id"].astype(str)
reg_df_2["orig_year"]  = reg_df_2["orig_id"].astype(str) + "_" + reg_df_2["year"].astype(str)
reg_df_2["dest_year"]  = reg_df_2["dest_id"].astype(str) + "_" + reg_df_2["year"].astype(str)

# kill zero flows
reg_df_2_nonzero = reg_df_2[reg_df_2['N_od_flow_wm']>0]
reg_df_2_nonzero['log_migration'] = np.log(reg_df_2_nonzero['N_od_flow_all'])
reg_df_2_nonzero['log_migration_wm'] = np.log(reg_df_2_nonzero['N_od_flow_wm'])
reg_df_2_nonzero['level_iv_tt'] = np.exp(reg_df_2_nonzero['IV_tt'])

# Regression: in levels and logs of working males and all population
# weighted by migration 
# regression formula

fml = (
    "Y ~ IV_tt + orig_bad_climate:IV_tt"
    " | orig_id^year + dest_id^year"
    )

reg_all_w = pf.feols(
    fml.replace("Y", "log_migration"),
    data=reg_df_2_nonzero,
    weights="N_od_flow_all",
    vcov={"CRV1": "pair_id"}
)
reg_wm_w = pf.feols(
    fml.replace("Y", "log_migration_wm"),
    data=reg_df_2_nonzero,
    weights="N_od_flow_wm",
    vcov={"CRV1": "pair_id"}
)

print(reg_all_w.summary())
print(reg_wm_w.summary())


fml_index = (
    f"Y ~ IV_tt + orig_{climate_index}:IV_tt"
    " | orig_id^year + dest_id^year"
    )

reg_all_index = pf.feols(
    fml_index.replace("Y", "log_migration"),
    data=reg_df_2_nonzero,
    weights="N_od_flow_all",
    vcov={"CRV1": "pair_id"}
)
print(reg_all_index.summary())

# After taking 1990 in consideration (had fallen due to coding)
# the effect becomes zero. Some indexes even "get it wrong"
# Alas.

# %% PPML version

reg_ppml = pf.fepois(
    "N_od_flow_all ~ IV_tt + orig_bad_climate:IV_tt | "
    "orig_year + dest_year",
    data=reg_df_2,
    vcov={"CRV1": "pair_id"}
)

print(reg_ppml.summary())

# %% No climate regression for next blocks:
fml_noclimate = (
    "Y ~ IV_tt + "
    " | orig_id^year + dest_id^year"
    )

reg_no_climate = pf.feols(
    fml_noclimate.replace("Y", "log_migration"),
    data=reg_df_2_nonzero,
    weights="N_od_flow_all",
    vcov={"CRV1": "pair_id"}
)
reg_no_climate.summary()

# %% Now we use the estimated migration losses
# to estimate GDP per capita differential!

response_df = reg_df_2_nonzero.copy()
# Recovering absolute predicted flows
# We must use a no-roads counterfactual baseline, which we have!
# I use the full index regression to get fuller sample data
coefs = reg_no_climate.coef()
beta_tt = coefs["IV_tt"]

tt_actual = response_df["IV_tt"]
tt_cf = response_df['log_fm_empty']

# fitted linear predictor under actual observed tt
xb_full = reg_all_w.predict()

# counterfactual tt, keeping climate amplification
xb_cf_full = (
    xb_full
    + beta_tt * (tt_cf - tt_actual)
)

# predicted flows
response_df["flow_full"] = np.exp(xb_full)
response_df["flow_cf_full"] = np.exp(xb_cf_full)

# decomposition
response_df["effect_total_tt"] = (
    response_df["flow_full"] - response_df["flow_cf_full"]
)
print(response_df[[
    "effect_total_tt"
]].describe())

# %% regs on full migration effect, not just climate
# inflows (i -> j)
total_inflows = response_df.groupby(["dest_id", "year"])["effect_total_tt"].sum()
# outflows (j -> k)
total_outflows = response_df.groupby(["orig_id", "year"])["effect_total_tt"].sum()

total_net = total_inflows.sub(total_outflows, fill_value=0).reset_index()
total_net.columns = ["region", "year", "net_receival_due_tt"]
total_net['net_receival_due_tt'].describe()
# positive -> place gained people due to Brasília
# negative -> place lost people

# rebuilidng growth and population change data on new year indexes
net_years = [1970, 1980, 1990, 2000, 2010]  # get 1970 for variation calculation
# want to better control columns
gdppc_years_tt = gdp_per_capita[gdp_per_capita['year'].isin(net_years)][['year', 'CD_GEOCME', 'log_gdppc']]
gdppc_years_tt["dlog_gdppc"] = (
    gdppc_years_tt.groupby("CD_GEOCME")["log_gdppc"]
      .diff()
      )
gdppc_years_tt["year_lag"] = gdppc_years_tt.groupby("CD_GEOCME")["year"].shift(1)
gdppc_years_tt["dlog_gdppc_ann"] = gdppc_years_tt["dlog_gdppc"] /(gdppc_years_tt["year"] - gdppc_years_tt["year_lag"])
# Now population
pop_years_tt = population[population['year'].isin(net_years)][['year', 'CD_GEOCME', 'log_pop']]
pop_years_tt["dlog_pop"] = (
    pop_years_tt.groupby("CD_GEOCME")["log_pop"]
      .diff()
      )
pop_years_tt["year_lag"] = pop_years_tt.groupby("CD_GEOCME")["year"].shift(1)
pop_years_tt["dlog_pop_ann"] = pop_years_tt["dlog_pop"] /(pop_years_tt["year"] - pop_years_tt["year_lag"])


# adding most data
stage_2_total = total_net.merge(
    model_df[['CD_GEOCME', 'year_tt', 'pop', 'delta_gdppc_spread',
              'log_gdppc_z', 'log_gdppc', f'{climate_index}']],
    how='left',
    left_on=['region', 'year'],
    right_on=['CD_GEOCME', 'year_tt']
    )

# adding annualized gdp and population change data
stage_2_total = stage_2_total.merge(
    gdppc_years_tt[['CD_GEOCME', 'year', 'dlog_gdppc_ann']],
    how='left',
    left_on=['region', 'year'],
    right_on=['CD_GEOCME', 'year']
    )
stage_2_total = stage_2_total.merge(
    pop_years_tt[['CD_GEOCME', 'year', 'dlog_pop_ann']],
    how='left',
    left_on=['region', 'year'],
    right_on=['CD_GEOCME', 'year']
    )

stage_2_total['net_receival_rate_tt'] = stage_2_total['net_receival_due_tt'] / stage_2_total['pop']
stage_2_total['time_trend'] = stage_2_total['year'] - stage_2_total['year'].min()

fml_s2_t = (
    "Y ~ net_receival_rate_tt + "
    " | region + year + region[time_trend]" # controlling for convergence
    )

reg_s2_lvl_t = pf.feols(
    fml_s2_t.replace("Y", "log_gdppc_z"),
    data=stage_2_total,
    vcov={"CRV1": "region"}
)

reg_s2_dlog_gdppc_t = pf.feols(
    fml_s2_t.replace("Y", "dlog_gdppc_ann"),
    data=stage_2_total,
    vcov={"CRV1": "region"}
)

reg_s2_gdppc_t = pf.feols(
    fml_s2_t.replace("Y", "log_gdppc"),
    data=stage_2_total,
    vcov={"CRV1": "region"}
)

print(reg_s2_lvl_t.summary())
# nothing here
print(reg_s2_dlog_gdppc_t.summary())
# very good! 
print(reg_s2_gdppc_t.summary())
# not much
 
# Decent evidence in favor of the "higher migration --> higher gdp per capita" story!
# Selection vs agglomeration!

# %%

def plot_regions(data_df, regions_df, var, year='all',
                 legend='index', title='placeholder', center='infer'):

    # Keep only what we need
    plot_gdf = regions_df.merge(
        data_df,
        left_on="CD_GEOCME",
        right_on="CD_GEOCME",
        how="left"
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    if center == 0:
        norm = colors.TwoSlopeNorm(
        vmin=plot_gdf[var].quantile(0.05),
        vcenter=0,
        vmax=plot_gdf[var].quantile(0.95)
        )
    else:
        norm = colors.TwoSlopeNorm(
        vmin=plot_gdf[var].quantile(0.05),
        vcenter=plot_gdf[var].median(),
        vmax=plot_gdf[var].quantile(0.95)
        )

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
        cmap='RdBu_r',
        norm=norm
    )
    ax.set_title(
        title,
        fontsize=12
        )
    ax.set_axis_off()
    ax.set_aspect("equal")
    plt.show()
    return fig, ax


stage_2_total['gdp_change_due_mig'] = reg_s2_dlog_gdppc_t.coef()['net_receival_rate_tt']*\
    stage_2_total['net_receival_rate_tt']

interest_outcomes = ['CD_GEOCME', 'gdp_change_due_mig', 'net_receival_rate_tt', 'dlog_gdppc_ann', 'pop']

df_agg_results = stage_2_total[interest_outcomes].groupby('CD_GEOCME').mean().reset_index()

df_agg_results = df_agg_results.merge(
    regions[['CD_GEOCME', 'NM_MESO']], on='CD_GEOCME', how='left'
    )

df_agg_results['growth_share_att_mig'] = df_agg_results['gdp_change_due_mig'] / df_agg_results['dlog_gdppc_ann']

# Winner regions
df_agg_results[df_agg_results['growth_share_att_mig'] > 0].sort_values('growth_share_att_mig')['NM_MESO']

# back of envelope gain of welfare in 2010 reais:
    # using mean population from sample
gdp_gain_ann = ((1+df_agg_results['gdp_change_due_mig']) * df_agg_results['pop']).sum()/df_agg_results['pop'].sum()
(gdp_gain_ann**40 - 1)*100

fig_gdp_pre, _ = plot_regions(df_agg_results, regions, 'dlog_gdppc_ann',
                      title='Δlog GDP per capita \n1970-2010',
                      legend='')

fig_mig, _ = plot_regions(df_agg_results, regions, 'net_receival_rate_tt',
                      title='Net Road Induced Migrant Receival Rate\n1980-2010',
                      legend='as share of local pop.', center=0)

fig_gdp, _ = plot_regions(df_agg_results, regions, 'gdp_change_due_mig',
                      title='Δlog GDP per Capita due Migration \n1970-2010',
                      legend='', center=0)

fig_gdp_share, _ = plot_regions(df_agg_results, regions, 'growth_share_att_mig',
                      title='Migration Induced Growth Relative to Total\n1970-2010',
                      legend='', center=0)

# %% Next:
    # On climate:
        # I'm mostly satisfied with the null
    # On GDPPC:
        # Robustness: state level, more controls, drop best fits, reg on dlogGDP
        # Interpretation: Selection of Migrants vs Agglomeration in Cities
        # Analize micro migration data, wage if avaiable
        # Bilateral effect? If selection yes, if agg no!
        # Formalize a model explaining findings
    # Another avenue: [Blank] Access Motive Test Technology
        # Compile what the theory says should drive migration
        # Show how model implies that these also work through the "higher response to lower cost" channel
        # See if there's data to test them from 1980's on
        # Test them!
        # Issue: F test
        
        

