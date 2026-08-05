# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:31:17 2026

@author: c337191
"""
import pandas as pd
import numpy as np
import geopandas as gpd
import pyfixest as pf
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

sns.set(style="whitegrid")

path = r'C:\Users\josez\OneDrive\Desktop\EPGE\climate_migration'
# %% get data
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:5880").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

# old. We keep for comparison only
exposure_index = pd.read_parquet(f'{path}//data/climate_indexes/exposure_pca_mean.parquet')

mo_meso = pd.read_stata(
    f'{path}/data/morten_oliveira_final_tables/N_od_meso.dta'
    )

relative_index = pd.read_parquet(f'{path}//data/climate_indexes//relative_exposure_index.parquet')

population = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='pop_mesorreg_interpol'
    )
gdp_per_capita = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_capita_mesorreg'
    )
gdp_ind_share = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_ind_share'
    )
gdp_serv_share = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_serv_share'
    )
gdp_agr_share = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='gdp_agr_share'
    )

# %%

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
    df_tidy = df_tidy.rename(columns={
        'CD_GEOCME': 'id',
        'NM_MESO': 'name',
        'Sigla': 'abbrv'
        })
    return df_tidy

tidy_gdppc = make_ipea_tidy(gdp_per_capita, 'gdp_capita')
tidy_pop = make_ipea_tidy(population, 'pop')
tidy_gdp_ind = make_ipea_tidy(gdp_ind_share, 'gdp_ind_share')
tidy_gdp_agr = make_ipea_tidy(gdp_agr_share, 'gdp_agr_share')
tidy_gdp_serv = make_ipea_tidy(gdp_serv_share, 'gdp_serv_share')

# %% Take averages in terms of available census data

def decade_averager(df):
    # We want the decade 1990 to incorporate shocks from 1981
    # up to 1990. So
    df["decade"] = (df["year"] // 10 + 1) * 10

    # Average all numeric columns within each mesoregion-decade
    df_decade = (
        df.groupby(["CD_GEOCME", "NM_MESO", "decade"], as_index=False)
          .mean(numeric_only=True)
    )
    df_decade.drop(columns=['year'])
    return df_decade

# Old indexes that do not interest us anymore. We prefer the within and between definitions
# Keep the composite mean and PCA index for comparison

df_climate_decade = decade_averager(exposure_index)
relative_index['NM_MESO'] = 0
relative_index = relative_index.rename(columns={'region':'CD_GEOCME'})
rel_index_decade = decade_averager(relative_index).drop(columns=['NM_MESO'])
# Renaming indexes
df_climate_decade['comp_rel_within'] = rel_index_decade['composite_within_exposure']
df_climate_decade['hot_within'] = rel_index_decade['Tmax_hot_exposure']
df_climate_decade['dry_within'] = rel_index_decade['wb_dry_90d_exposure']
df_climate_decade['flood_within'] = rel_index_decade['pr_extreme_7d_exposure']
df_climate_decade['cold_within'] = rel_index_decade['Tmin_cold_exposure']

df_climate_decade['comp_rel_between'] = rel_index_decade['composite_between_exposure']
df_climate_decade['hot_between'] = rel_index_decade['Tmax_hot_between_exposure']
df_climate_decade['dry_between'] = rel_index_decade['wb_dry_90d_between_exposure']
df_climate_decade['flood_between'] = rel_index_decade['pr_extreme_7d_between_exposure']
df_climate_decade['cold_between'] = rel_index_decade['Tmin_cold_between_exposure']

# %% Migration data format
df_migration_short = mo_meso[['orig_id', 'dest_id', 'year', 'N_od_flow_wm', 'N_od_flow_all']]

years_map = {
    1980: 1980,
    1991: 1990,
    2000: 2000,
    2010: 2010
    }

df_migration_short['decade'] = df_migration_short['year'].map(years_map)

# merging origin and destination climate indexes
orig_columns_names = {
    df_climate_decade.columns[i]: f'orig_{df_climate_decade.columns[i]}' for i in range(df_climate_decade.shape[1])
    }
orig_columns_names['CD_GEOCME'] = 'orig_id'
orig_columns_names['decade'] = 'decade'
orig_columns_names['year'] = 'year'

df_climate_decade_orig = df_climate_decade.rename(
    columns = orig_columns_names
    )

df_migration_short = df_migration_short.merge(
    df_climate_decade_orig,
    on = ['orig_id', 'decade']
    )
dest_columns_names = {
    df_climate_decade.columns[i]: f'dest_{df_climate_decade.columns[i]}' for i in range(df_climate_decade.shape[1])
    }
dest_columns_names['CD_GEOCME'] = 'dest_id'
dest_columns_names['decade'] = 'decade'
dest_columns_names['year'] = 'year'

df_climate_decade_dest = df_climate_decade.rename(
    columns = dest_columns_names
    )

df_migration_short = df_migration_short.merge(
    df_climate_decade_dest,
    on = ['dest_id', 'decade']
    )

# final: drop same location values
# those are actually non-migrant counts
not_pop_counts = df_migration_short['orig_id']!=df_migration_short['dest_id']

df_model = df_migration_short[not_pop_counts].reset_index(drop=True)

df_model['pair_id'] = df_model['orig_id'].astype(str) + "_" + df_model['dest_id'].astype(str)

df_model_nonzero = df_model[df_model['N_od_flow_all']>0]
df_model_nonzero['log_flow'] = np.log(df_model_nonzero['N_od_flow_all'])
df_model = df_model.drop(columns=['year_x', 'year_y'])


# %% Adding IPEA data

def merge_model_ipea(df_model, df_ipea):
    
    orig_ipea = df_ipea.rename(columns={
        x: f'orig_{x}' for x in df_ipea.columns
        })
    dest_ipea = df_ipea.rename(columns={
        x: f'dest_{x}' for x in df_ipea.columns
        })
    df_merged = df_model.merge(
        orig_ipea,
        left_on = ['orig_id', 'decade'],
        right_on = ['orig_id', 'orig_year']
        )
    df_merged = df_merged.merge(
        dest_ipea,
        left_on = ['dest_id', 'decade'],
        right_on = ['dest_id', 'dest_year']
        )
    columns_drop = [col for col in df_merged.columns if '_x' in col[-2:] or '_y' in col[-2:]]
    df_merged = df_merged.drop(columns_drop, axis=1)
    return df_merged

df_model = merge_model_ipea(df_model, tidy_gdppc)
df_model = merge_model_ipea(df_model, tidy_pop[['id', 'year', 'pop']])

df_model = merge_model_ipea(df_model, tidy_gdp_ind[['id', 'year', 'gdp_ind_share']])
df_model = merge_model_ipea(df_model, tidy_gdp_agr[['id', 'year', 'gdp_agr_share']])
df_model = merge_model_ipea(df_model, tidy_gdp_serv[['id', 'year', 'gdp_serv_share']])


df_model = df_model.drop(columns=['orig_year', 'dest_year'])

df_model['emigration_pct_flow_all'] = df_model['N_od_flow_all'] / df_model['orig_pop']
df_model['immigration_pct_flow_all'] = df_model['N_od_flow_all'] / df_model['dest_pop']

df_model.to_parquet(
    f'{path}/data/formatted_composit/climate_mig_meso.parquet', index=False)

# %% Climate will be taken as origin_climate and dest_climate
# Across climate "types"
# We add random effects and check for robustness with fixed effects

# we regress pop flows on climate outcomes and fixed effects
# we are leveraging the low-mobility approximation from Boryusak
# Later, implement the NLLS estimator they present, as we should have the
# needed data kinda

def run_reg_fe(df, climate_index):
    fml = (
        f"log_flow ~ orig_{climate_index} + dest_{climate_index}"
        " | year + pair_id"
        )
    reg_w = pf.feols(  #Not taking zeros into acc
        fml,
        data=df,
        vcov={"CRV1": "orig_id + dest_id"}
        )
    return reg_w

rows = []
climate_indexes = df_climate_decade.columns[6:15]

for climate_index in climate_indexes:
    reg = run_reg_fe(df_model_nonzero, climate_index)
    # Extract coefficient table
    # Origin variable name
    orig_var = f"orig_{climate_index}"
    dest_var = f"dest_{climate_index}"

    row = {
        "index": climate_index,
        "origin_coef": reg.coef().loc[orig_var],
#        "origin_se": reg.se().loc[orig_var],
        "origin_p": reg.pvalue().loc[orig_var],

        "dest_coef": reg.coef().loc[dest_var],
#        "dest_se": reg.se().loc[dest_var],
        "dest_p": reg.pvalue().loc[dest_var],
        'reg': reg
    }
    rows.append(row)

fe_ols_results = pd.DataFrame(rows)
latex_out_ols = fe_ols_results.drop(columns=['reg']).set_index('index').apply(lambda x: np.round(x, 3))
# The old indexes have the expected behavior
# The new ones are more muddled. The between indexes are much too erratic
# in particular, they always display the wrong sign for origins
# The whithin indexes are more interesting. P values are much too close to 5
# for my taste, but still. I'd venture to say that:
    # aridity events drive people out of origins
    # and flooding drive people out of destinations
    # I do think this is somewhat aligned with Sophie's findings?

# %% PPML version

def run_reg_ppml(df, climate_index):
    reg_ppml = pf.fepois(
        f'N_od_flow_wm ~ orig_{climate_index} + dest_{climate_index}'
        " | year + pair_id",
    data=df,
    vcov={"CRV1": "orig_id + dest_id"}
    )
    return reg_ppml

rows_ppml = []

for climate_index in climate_indexes:
    reg = run_reg_ppml(df_model, climate_index)
    # Extract coefficient table
    # Origin variable name
    orig_var = f"orig_{climate_index}"
    dest_var = f"dest_{climate_index}"

    row = {
        "index": climate_index,
        "origin_coef": reg.coef().loc[orig_var],
#        "origin_se": reg.se().loc[orig_var],
        "origin_p": reg.pvalue().loc[orig_var],

        "dest_coef": reg.coef().loc[dest_var],
#        "dest_se": reg.se().loc[dest_var],
        "dest_p": reg.pvalue().loc[dest_var],
        'reg': reg
    }
    rows_ppml.append(row)
fe_ppml_results = pd.DataFrame(rows_ppml)
latex_out_ppml = fe_ppml_results.drop(columns=['reg']).set_index('index').apply(lambda x: np.round(x, 3))

# PPML version
# This points towards a "stickiness". People are less likely to 
# leave places with bad climate, even as they are more likely to go to places
# with good climate
# This could be due to poverty in origins making migration less likely in general,
# or due to adaptation

# How do I test adaptation?














