# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:52:17 2026

@author: c337191
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import statsmodels.formula.api as smf
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

# %%
gdp_per_capita['log_gdppc'] = np.log(gdp_per_capita['gdp_capita'])
gdp_per_capita["dlog_gdppc"] = (
    gdp_per_capita.groupby("CD_GEOCME")["log_gdppc"]
      .diff()
      )
gdp_per_capita["year_lag"] = gdp_per_capita.groupby("CD_GEOCME")["year"].shift(1)
gdp_per_capita["dlog_gdppc_ann"] = gdp_per_capita["dlog_gdppc"] /(gdp_per_capita["year"] - gdp_per_capita["year_lag"])

population["log_pop"] = np.log(population["pop"])
population["dlog_pop"] = (
    population.groupby("CD_GEOCME")["log_pop"]
      .diff()
      )
population["year_lag"] = population.groupby("CD_GEOCME")["year"].shift(1)
population["dlog_pop_ann"] = population["dlog_pop"] /(population["year"] - population["year_lag"])


gdp["log_gdp"] = np.log(gdp["gdp"])
gdp["dlog_gdp"] = (
    gdp.groupby("CD_GEOCME")["log_gdp"]
      .diff()
      )
gdp["year_lag"] = gdp.groupby("CD_GEOCME")["year"].shift(1)
gdp["dlog_gdp_ann"] = gdp["dlog_gdp"] /(gdp["year"] - gdp["year_lag"])



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
    gdp[['Sigla', 'NM_MESO', "CD_GEOCME", "year", 'log_gdp', 'dlog_gdp_ann']],
    on=['Sigla', 'NM_MESO', "CD_GEOCME", "year"],
    how="left"
    )

model_df =  model_df.merge(
    exposure_matched,
    on=["CD_GEOCME", "year"],
    how="left"
    )

model_df =  model_df.merge(
    climate_subindex_matched,
    on=["CD_GEOCME", "year"],
    how="left"
    )

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

# Bad climate drives gdppc growth?
# Also little to no impact overall
scatter_plot(
    model_df_mean,
    x="mean_exp",
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
plt.show()
# Richer places experience higher pop growth (or vice versa!)


#%% Recovering the Morten-Oliveira instrument
# relevant cols:
    # 'N_od_flow_wm' (working males), 'N_od_flow_all'
    # orig_id_meso, dest_id_meso
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

# %% Now calculating the effect of roads on travel time (Morten-Oliveira eq 1)
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

# 1939 to 1940 and 1991 to 1990 on migration data
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

# bad climate: mean_exp > 1 (meaning original index > 1)
# climate dummy: bad if index higher than 1
model_df['bad_climate'] = 1*(model_df[climate_index]>=1)
# passing travel time data years to the model
model_df['year_tt'] = model_df['year'].map(years_map)

# %% Regression on instrumented migration flows:
    # looking at possible climate effect
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

# Creating market access index
reg_df_2['mkt_access'] = reg_df_2['dest_log_gdppc'] / reg_df_2['IV_tt']

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
    f"Y ~ IV_tt + orig_{climate_index}:IV_tt +"
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
    f"N_od_flow_all ~ IV_tt + orig_bad_climate:IV_tt +  | "
    "orig_year + dest_year",
    data=reg_df_2,
    vcov={"CRV1": "pair_id"}
)
print(reg_ppml.summary())

# PPML yields a result! But I dont buy it given no result in normal reg

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
t_iv = reg_no_climate.tidy().loc["IV_tt", "t value"]
f_iv = t_iv ** 2

print("First-stage F-stat:", f_iv)
# Very strong instrument for full migration, as MO had shown 

# %% Now we recover the instrumented migration flow

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

# inflows (i -> j)
tt_inflows = response_df.groupby(["dest_id", "year"])["effect_total_tt"].sum()
# outflows (j -> k)
tt_outflows = response_df.groupby(["orig_id", "year"])["effect_total_tt"].sum()

total_net = tt_inflows.sub(tt_outflows, fill_value=0).reset_index()
total_net.columns = ["region", "year", "net_receival_due_tt"]
# positive -> place gained people due to Brasília
# negative -> place lost people

# This is an obvious one to improve on. Market access gained by TT reduction
# for now, simple mean in overall reduction and in instrumented reduction
response_df['ivtt_red'] = response_df['log_fm_empty'] - response_df['IV_tt']
response_df['distance_to_gdp'] = response_df['dest_log_gdppc'] / response_df['ivtt_red']

total_net['mean_ivtt_red'] = response_df.groupby(["orig_id", "year"])['ivtt_red'].mean().values
# mean time travel reduction weighted by gdp of destionation
total_net['mkt_access_gained'] = response_df.groupby(["orig_id", "year"])['distance_to_gdp'].mean().values

# getting real migration flows
real_inflows = response_df.groupby(["dest_id", "year"])["N_od_flow_all"].sum()
# outflows (j -> k)
real_outflows  = response_df.groupby(["orig_id", "year"])["N_od_flow_all"].sum()

real_net = real_inflows.sub(real_outflows, fill_value=0).reset_index()

total_net['net_receival_real'] = real_net.groupby(["dest_id", "year"])['N_od_flow_all'].sum().values

# %% Building the data for the GDP change analysis

# Building growth data for the avaiable migration timeframes
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
# Now raw GDP
gdp['log_gdp'] = np.log(gdp['gdp'])
gdp_years_tt = gdp[gdp['year'].isin(net_years)][['year', 'CD_GEOCME', 'log_gdp']]
gdp_years_tt["dlog_gdp"] = (
    gdp_years_tt.groupby("CD_GEOCME")["log_gdp"]
      .diff()
      )
gdp_years_tt["year_lag"] = gdp_years_tt.groupby("CD_GEOCME")["year"].shift(1)
gdp_years_tt["dlog_gdp_ann"] = gdp_years_tt["dlog_gdp"] /(gdp_years_tt["year"] - gdp_years_tt["year_lag"])

# adding most data
stage_2_total = total_net.merge(
    model_df[['CD_GEOCME', 'Sigla', 'year_tt', 'pop','log_gdppc',
              f'{climate_index}']],
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

stage_2_total = stage_2_total.merge(
    gdp_years_tt[['CD_GEOCME', 'year', 'dlog_gdp_ann', 'log_gdp']],
    how='left',
    left_on=['CD_GEOCME', 'year'],
    right_on=['CD_GEOCME', 'year']
    )

stage_2_total['net_receival_rate_tt'] = stage_2_total['net_receival_due_tt'] / stage_2_total['pop']
stage_2_total['time_trend'] = stage_2_total['year'] - stage_2_total['year'].min()
stage_2_total['net_receival_rate_tt_pos'] = stage_2_total['net_receival_rate_tt'].apply(lambda x: x if x > 0 else 0)
stage_2_total['net_receival_rate_tt_neg'] = stage_2_total['net_receival_rate_tt'].apply(lambda x: x if x < 0 else 0)
stage_2_total['net_receival_rate_real'] = stage_2_total['net_receival_real'] / stage_2_total['pop']

# %%  Design: impacts of instrumented receival rate on log gdppc
# with random effects for region and trends, and fixed effects for years

mod_gdppc = smf.mixedlm(
    "dlog_gdppc_ann ~ net_receival_rate_tt + mkt_access_gained + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)
# And against real receival rate, not instrumented
mod_gdppc_real = smf.mixedlm(
    "dlog_gdp_ann ~ net_receival_rate_real + mkt_access_gained + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)

res_gdppc = mod_gdppc.fit()
print(res_gdppc.summary())
# We find a positive and significant effect, as expected!
# Moreover, we see no effect from market acces gain once the migration is accounted for
# Which is interesting. 
# Rate of growth was that much larger due to migration

res_gdppc_real = mod_gdppc_real.fit()
print(res_gdppc_real.summary())
# We find a larger value when going by simple, non-instrumented receival rates
# indicating endogeneity is real

mod_dlog_gdp = smf.mixedlm(
    "dlog_gdp_ann ~ net_receival_rate_tt + mkt_access_gained + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)

res_dlog_gdp = mod_dlog_gdp.fit()
print(res_dlog_gdp.summary())
# neat positive effect on full gdp also

mod_log = smf.mixedlm(
    "log_gdppc~ net_receival_rate_tt + mkt_access_gained + C(year)",
    data=stage_2_total,
    groups="region",
    re_formula="~time_trend"
)

res_log = mod_log.fit()
print(res_log.summary())
# works on levels also

mod_gdp_lvl = smf.mixedlm(
    "log_gdp ~ net_receival_rate_tt + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)

res_gdp_lvl = mod_gdp_lvl.fit()
print(res_gdp_lvl.summary())
# Neat evidence for lvl gdp effect!

mod_posneg = smf.mixedlm(
    "dlog_gdppc_ann ~ net_receival_rate_tt_pos + net_receival_rate_tt_neg + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)

res_pn = mod_posneg.fit()
print(res_pn.summary())
# Both gaining and loosing people are significant and have the right signal
# Interestingly, loosing people seems to be much worse!

# Lets look at "types of attractors"
# Idea: agri regions benfit from migration due to S curve / abundant resources
# Cities benfit due to agglomerative effects
agri_frontier = ['RO', 'PA', 'MT', 'MS', 'AP', 'RR', 'AM', 'AC', 'TO']
stage_2_total['agri'] = stage_2_total['Sigla'].isin(agri_frontier)
stage_2_total['urban'] = stage_2_total['pop'] > 5e5  # IBGE for large city
stage_2_total['non_urban_agri'] = ~(stage_2_total['urban'] | stage_2_total['agri'])

stage_2_total["region_type"] = np.select(
    [
        stage_2_total["urban"] == 1,
        stage_2_total["agri"] == 1,
        stage_2_total["non_urban_agri"] == 1
    ],
    [
        "urban",
        "agri",
        "other"
    ]
)

mod_agri_urban = smf.mixedlm(
    "dlog_gdppc_ann ~ 0 + net_receival_rate_tt:C(region_type) + C(year)",
    data=stage_2_total.dropna(),
    groups="region",
    re_formula="~time_trend"
)

res_agri_urban = mod_agri_urban.fit()
print(res_agri_urban.summary())
# Significant only for agri. However, few datapoints

# %% initial rob checks: trying to implement the same ideas under all fixed effects

# Fixed effects
fml_s2_t = (
    "Y ~ net_receival_rate_tt + "
    " | region + year + region[time_trend]" # controlling for convergence
    )

robFE_dlog_gdppc = pf.feols(
    fml_s2_t.replace("Y", "dlog_gdppc_ann"),
    data=stage_2_total.dropna(),
    vcov={"CRV1": "region"}
)
robFE_log_gdppc = pf.feols(
    fml_s2_t.replace("Y", "log_gdppc"),
    data=stage_2_total,
    vcov={"CRV1": "region"}
)

robFE_dlog_gdp = pf.feols(
    fml_s2_t.replace("Y", "dlog_gdp_ann"),
    data=stage_2_total.dropna(),
    vcov={"CRV1": "region"}
)

print(robFE_log_gdppc.summary())
print(robFE_dlog_gdppc.summary())
print(robFE_dlog_gdp.summary()) 
# PASS, with values on the same order of magnitude

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

# Effect going by agg gain/loss
stage_2_total['dlog_gdppc_due_mig'] = res_gdppc.params['net_receival_rate_tt']*\
    stage_2_total['net_receival_rate_tt']
stage_2_total['dlog_gdp_due_mig'] = res_dlog_gdp.params['net_receival_rate_tt']*\
    stage_2_total['net_receival_rate_tt']

interest_outcomes = [
    'CD_GEOCME', 'dlog_gdppc_due_mig', 'dlog_gdp_due_mig',
    'net_receival_rate_tt', 'dlog_gdppc_ann', 'mean_exp', 'net_receival_rate_real'
                     ]

total_gdp_change = stage_2_total[stage_2_total['year']==2010]['log_gdppc'].reset_index() - \
    stage_2_total[stage_2_total['year']==1980]['log_gdppc'].reset_index()
# Lots of places lost GDP per capita. WTF

# mean annualized gain due migration and mean net receival rate by region over the period
df_agg_results = stage_2_total[interest_outcomes].groupby('CD_GEOCME').agg(
    {
    'dlog_gdppc_due_mig': 'mean',
    'dlog_gdp_due_mig': 'mean',
    'dlog_gdppc_ann': 'mean',
    'net_receival_rate_tt': 'mean',
    'net_receival_rate_real': 'mean',
    'mean_exp': 'mean'
    }
    ).reset_index()

df_agg_results = df_agg_results.merge(
    regions[['CD_GEOCME', 'NM_MESO']], on='CD_GEOCME', how='left'
    )

fig_gdppc, _ = plot_regions(df_agg_results, regions, 'dlog_gdppc_ann',
                      title='Annualized GDP per capita Growth Rate \n1970-2010',
                      legend='')
fig_gdppc.savefig(f'{path}/figs/dlog_gdppc.png', transparent=True)

fig_clima, _ = plot_regions(df_agg_results, regions, 'mean_exp',
                      title='Mean Extreme Climate Exposure \n1970-2010',
                      legend='')
fig_clima.savefig(f'{path}/figs/climate_exp.png', transparent=True)

fig_mig_real, _ = plot_regions(df_agg_results, regions, 'net_receival_rate_real',
                      title='Net Migrant Receival Rate\n1980-2010, share of population',
                      legend='', center=0)
fig_mig_real.savefig(f'{path}/figs/fig_mig_real.png', transparent=True)

fig_mig, _ = plot_regions(df_agg_results, regions, 'net_receival_rate_tt',
                      title='Net Road Induced Migrant Receival Rate\n1980-2010',
                      legend='as share of local pop.', center=0)

fig_gdp_mig, _ = plot_regions(df_agg_results, regions, 'dlog_gdp_due_mig',
                      title='GDP Growth due Migration \n1970-2010',
                      legend='', center=0)
fig_gdp_mig.savefig(f'{path}/figs/main_result_gdp.png', transparent=True)

fig_gdppc_mig, _ = plot_regions(df_agg_results, regions, 'dlog_gdppc_due_mig',
                      title='GDP per Capita Growth due Migration \n1970-2010',
                      legend='', center=0)
fig_gdppc_mig.savefig(f'{path}/figs/main_result_gdppc.png', transparent=True)


# Growth accounting
gdp_1970 = gdp[gdp['year']==1970][['CD_GEOCME', 'gdp']]
pop_2010 = population[population['year']==2010][['CD_GEOCME', 'pop']]
gdp_2010 = gdp[gdp['year']==2010][['CD_GEOCME', 'gdp']]
gdp_2010.columns = ['CD_GEOCME', 'gdp_2010']

df_accounting = df_agg_results.merge(
    gdp_1970,
    how='left',
    on='CD_GEOCME'
    )
df_accounting = df_accounting.merge(
    gdp_2010,
    how='left',
    on='CD_GEOCME'
    )

df_accounting = df_accounting.merge(
    pop_2010,
    how='left',
    on='CD_GEOCME'
    )

df_accounting['gdp_gain'] = df_accounting['gdp'] * df_agg_results['dlog_gdp_due_mig'] * 40
tot_gdp_gain = (df_accounting['gdp_gain'].sum() /  df_accounting['gdp'].sum())
# We find a high effect in overall GDP. Something like growth of 9.35% from 1970 to 2010

gdppc_ex_mig = (df_accounting['gdp_2010'] - df_accounting['gdp_gain']).sum()/df_accounting['pop'].sum()

gdppc_2010 = df_accounting['gdp_2010'].sum()/df_accounting['pop'].sum()
1-gdppc_ex_mig / gdppc_2010

# Something like 2.1% GDP per capita can be attributed to internal migration reshufling
# Morten-Oliveira find 0.67% gain in aggregate welfare. Mine are significantly higher!
# Note, they find 2.8% gains from migration + trade. My estimation may be "polluted"

# %% Offloading tables 


print(res.summary()) # instrument replication
print(reg_all_index.summary()) # instrument works, climate doesnt
print(reg_all_w.summary())
print(reg_no_climate.summary()) # reg without climate for future work
print(res_gdppc_real.summary()) # effect is higher under true migration,

print(res_gdppc.summary()) # positive effect on gdp per capita; no market acess motive shows up
print(res_dlog_gdp.summary()) # positive effect under gdp level
print(robFE_dlog_gdppc.summary()) # it works under fixed effects also
print(robFE_dlog_gdp.summary()) 



# 1920 to 2010, for 137 mesorregions (less for years 1920 - 1960)
macro_summary = model_df[['dlog_gdppc_ann', 'dlog_pop_ann', 'dlog_gdp_ann']].describe()

# Original data: 1961 to 2024, daily for 0.1x0.1 degrees data on rain, max min temp, relative humidity and evapotranspiration
# Formatted data: ten-year average of exposure indexes
climate_summary = model_df[['mean_exp', 'drought_anomaly', 'drought_absolute', 'heat', 'flood']].describe()

# 1980 to 2010, decennial for meso origin-destination pair
mo_summary = mo_meso[['N_od_flow_all', 'log_fm_empty', 'log_fm_road']].describe()

# %% Next:
    # On climate:
        # I'm mostly satisfied with the null
    # On GDPPC:
        # Robustness: state level, more controls, drop best fits
        # Robustness: reg on log GDP --> pass! For both agg, pos and neg effects
        # Doesnt pass for agri/urban/other
        # Robustness: compare OLS to IV migration
        # Robustness: FE vs RE --> pass!
        # Interpretation: Selection of Migrants vs Agglomeration in Cities
        # Analize micro migration data, wage if avaiable
        # Bilateral effect? If selection yes, if agg no!
        # Agri vs Urban vs Other --> All have the effect!
    # Another avenue: [Blank] Access Motive Test Technology
        # Compile what the theory says should drive migration
        # Show how model implies that these also work through the "higher response to lower cost" channel
        # See if there's data to test them from 1980's on
        # Test them!
        # Issue: F test
        
        

