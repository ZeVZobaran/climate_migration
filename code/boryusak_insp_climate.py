# -*- coding: utf-8 -*-
"""
Created on Mon May 11 14:31:17 2026

@author: c337191
"""


import pandas as pd
import numpy as np
import geopandas as gpd
import pyfixest as pf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
from scipy.optimize import minimize_scalar
from scipy.optimize._numdiff import approx_derivative
from numpy.linalg import solve

sns.set(style="whitegrid")

path = r'D:\Users\c337191\Documents\climate_migration'
# %% get data
regions = gpd.read_file(f'{path}/data/br_mesorregioes/BRMEE250GC_SIR.shp')
# Climate grids are in lon/lat, so polygons should also be in EPSG:4326
regions = regions.to_crs("EPSG:5880").copy()
regions['CD_GEOCME'] = pd.to_numeric(regions['CD_GEOCME'])

exposure_index = pd.read_parquet(f'{path}//data/climate_indexes/exposure_pca_mean.parquet')
exposure_subindex = pd.read_parquet(f'{path}//data/climate_indexes//exposure_z_scores.parquet')
mo_meso = pd.read_stata(
    f'{path}/data/morten_oliveira_final_tables/N_od_meso.dta'
    )

relative_index = pd.read_parquet(f'{path}//data/climate_indexes//relative_exposure_index.parquet')
rel_comp_index = relative_index[['year', 'region', 'composite_within_exposure', 'composite_between_exposure']]

# %% Take averages in terms of available census data

def decade_averager(df):
    df["decade"] = (df["year"] // 10) * 10

    # Average all numeric columns within each mesoregion-decade
    df_decade = (
        df.groupby(["CD_GEOCME", "NM_MESO", "decade"], as_index=False)
          .mean(numeric_only=True)
    )
    df_decade.drop(columns=['year'])
    return df_decade

exposure_subindex_drop = exposure_subindex[['region', 'year', 'NM_MESO', 'heat',
       'drought_anomaly', 'flood', 'drought_absolute']]
exposure_subindex_drop = exposure_subindex_drop.rename(columns={'region':'CD_GEOCME'})

exp_overall = exposure_index.merge(
    exposure_subindex_drop, on=['CD_GEOCME', 'year', 'NM_MESO']
    )

df_climate_decade = decade_averager(exp_overall)
rel_comp_index['NM_MESO'] = 0
rel_comp_index = rel_comp_index.rename(columns={'region':'CD_GEOCME'})
rel_comp_index_decade = decade_averager(rel_comp_index).drop(columns=['NM_MESO'])
df_climate_decade['comp_rel_within'] = rel_comp_index_decade['composite_within_exposure']
df_climate_decade['comp_rel_between'] = rel_comp_index_decade['composite_between_exposure']


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
df_climate_decade_orig = df_climate_decade.rename(
    columns = {
        'CD_GEOCME': 'orig_id',
        'mean_exp': 'orig_mean_exp',
        'pca_exp': 'orig_pca_exp',
        'heat': 'orig_heat',
       'drought_anomaly': 'orig_drought_anomaly',
       'flood': 'orig_flood',
       'drought_absolute': 'orig_drought_absolute',
       'comp_rel_within': 'orig_comp_rel_within',
       'comp_rel_between': 'orig_comp_rel_between'
        }
    )

df_migration_short = df_migration_short.merge(
    df_climate_decade_orig,
    on = ['orig_id', 'decade']
    )

df_climate_decade_dest = df_climate_decade.rename(
    columns = {
        'CD_GEOCME': 'dest_id',
        'mean_exp': 'dest_mean_exp',
        'pca_exp': 'dest_pca_exp',
        'heat': 'dest_heat',
       'drought_anomaly': 'dest_drought_anomaly',
       'flood': 'dest_flood',
       'drought_absolute': 'dest_drought_absolute',
       'comp_rel_within': 'dest_comp_rel_within',
       'comp_rel_between': 'dest_comp_rel_between'

        }
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
        " | orig_id + dest_id + year"
        )
    
    reg_w = pf.feols(
        fml,
        data=df,
        weights="N_od_flow_all",
        vcov={"CRV1": "pair_id"}
    )
    return reg_w

rows = []
for climate_index in [
        'mean_exp', 'pca_exp',
        'comp_rel_within', 'comp_rel_between'
        ]:
    reg = run_reg_fe(df_model_nonzero, climate_index)
    # Extract coefficient table
    # Origin variable name
    orig_var = f"orig_{climate_index}"
    dest_var = f"dest_{climate_index}"

    row = {
        "index": climate_index,
        "origin_coef": reg.coef().loc[orig_var],
        "origin_se": reg.se().loc[orig_var],
        "origin_p": reg.pvalue().loc[orig_var],

        "dest_coef": reg.coef().loc[dest_var],
        "dest_se": reg.se().loc[dest_var],
        "dest_p": reg.pvalue().loc[dest_var],
        'reg': reg
    }
    rows.append(row)


fe_ols_results = pd.DataFrame(rows)
latex_out = fe_ols_results.drop(columns=['reg']).set_index('index').apply(lambda x: np.round(x, 3))

# Although at each climate index the result isnt too great,
# in aggregate we do capture the expected effect!
# That is, bad climate drives people out of origins (positive sign)
# and bad climate reduces flows into destinations (negative sign)
# with destination effects being both weaker and less precisely identified
# neat!

# ADD: composite indexed made
# The between indexes are not significant for origins! 
# A given region being on the extreme of the national
# average doesnt drive migration. They ARE significant for destinations,
# so that a destination being extreme drives people... towards? it?
# Local adaptation!
# The composite within index is significant on both counts
# but has the wrong? sign?
# I hate it here

# %% Implementing the NLSS estimator from Boryusak
# Getting raw population data
population = pd.read_excel(
    f'{path}/data/ipea/ipea_format.xlsx', sheet_name='pop_mesorreg_interpol'
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
population = make_ipea_tidy(population, 'pop')
population = population[population['year'].isin([1970, 1980, 1990, 2000, 2010])]
population = population.fillna(method='bfill')  # for leste rondoniense migué

population['decade'] = population['year'] + 10
population['orig_id'] = population['CD_GEOCME']

# %%  Getting population change percentage

pop_changes = df_model[
    ['orig_id', 'N_od_flow_all', 'decade']
    ].groupby(['orig_id', 'decade']).sum().reset_index()

pop_changes = pop_changes.merge(
    population[['orig_id', 'decade', 'pop']],
    on=['orig_id', 'decade']
    )

pop_changes['change_pct'] = pop_changes['N_od_flow_all'] / pop_changes['pop']

# %% Getting pi and gamma

population_orig = population.rename(columns={'pop':'orig_pop'})

df_migration_short = df_migration_short.merge(
    population_orig[['orig_id', 'decade', 'orig_pop']],
    on=['orig_id', 'decade']
    )



df_migration_short['pi_od'] = df_migration_short['N_od_flow_all']/df_migration_short['orig_pop']
# values may be larger than one for same-same region!

population_dest = population[['CD_GEOCME', 'year', 'pop']].rename(columns={
    'CD_GEOCME': 'dest_id',
    'year': 'decade',  # now we work with contemporaneous data according to the model!
    'pop': 'dest_pop'
    })

df_migration_short = df_migration_short.merge(
    population_dest[['dest_id', 'decade', 'dest_pop']],
    on=['dest_id', 'decade']
    )
df_migration_short['gamma_od'] = df_migration_short['N_od_flow_all']/df_migration_short['dest_pop']

# %% averaging out and getting PI and GAMMA

df_pi_gamma = df_migration_short[['orig_id', 'dest_id', 'decade', 'pi_od', 'gamma_od']].groupby(
    ['orig_id', 'dest_id']).mean().drop(columns=['decade']).reset_index()

pi_matrix = (
    df_pi_gamma.pivot_table(
        index="orig_id",
        columns="dest_id",
        values="pi_od"
    )
)

# Matrix for gamma_od
gamma_matrix = (
    df_pi_gamma.pivot_table(
        index="orig_id",
        columns="dest_id",
        values="gamma_od"
    )
)

# %% Implementing Boryusak's estimator

def omega_matrix(lam, PI, GAMMA):
    """
    Ω(λ) = I - [I + λ(I - Γ'Π)]^{-1}
    """
    PI = np.asarray(PI, dtype=float)
    GAMMA = np.asarray(GAMMA, dtype=float)

    n = PI.shape[0]
    I = np.eye(n)

    A = I + lam * (I - GAMMA.T @ PI)

    return I - solve(A, I)


def fitted_values(lam, PI, GAMMA, Z, L, intercept=True):
    Z = np.asarray(Z, dtype=float).reshape(-1)

    fitted = omega_matrix(lam, PI, GAMMA) @ Z

    if intercept:
        c = np.mean(L - fitted)
        fitted = fitted + c

    return fitted


def ssr_objective(lam, PI, GAMMA, L, Z, intercept=True):
    L = np.asarray(L, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)
    
    OMEGA = omega_matrix(lam, PI, GAMMA)

    fitted = OMEGA @ Z

    if intercept:
        c = np.mean(L - fitted)
        fitted = fitted + c

    resid = L - fitted

    return resid @ resid


def estimate_theta_over_sigma(
    PI,
    GAMMA,
    L,
    Z,
    bounds=(1e-8, 100),
    intercept=True
):

    L = np.asarray(L, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)

    n = len(L)
    k = 2 if intercept else 1

    # -----------------------------------
    # Estimate lambda
    # -----------------------------------

    result = minimize_scalar(
        ssr_objective,
        bounds=bounds,
        method="bounded",
        args=(PI, GAMMA, L, Z, intercept),
        options={"xatol": 1e-10}
    )

    lam_hat = result.x

    # -----------------------------------
    # Fitted values
    # -----------------------------------

    fitted_no_intercept = omega_matrix(lam_hat, PI, GAMMA) @ Z

    if intercept:
        c_hat = np.mean(L - fitted_no_intercept)
    else:
        c_hat = 0.0

    fitted = fitted_no_intercept + c_hat
    resid = L - fitted

    ssr = resid @ resid

    # -----------------------------------
    # Estimate residual variance
    # -----------------------------------

    sigma2_hat = ssr / (n - k)

    # -----------------------------------
    # Numerical Jacobian wrt lambda
    # -----------------------------------

    def model_prediction(lam):
        pred = omega_matrix(lam[0], PI, GAMMA) @ Z

        if intercept:
            c = np.mean(L - pred)
            pred = pred + c

        return pred

    J = approx_derivative(
        model_prediction,
        x0=np.array([lam_hat]),
        method='3-point'
    )

    # J shape: (n, 1)
    JTJ = J.T @ J

    # -----------------------------------
    # Asymptotic variance
    # Var(beta) = sigma² (J'J)^(-1)
    # -----------------------------------

    vcov = sigma2_hat * np.linalg.inv(JTJ)

    se_lambda = np.sqrt(vcov[0, 0])

    t_stat = lam_hat / se_lambda

    return {
        "theta_over_sigma": lam_hat,
        "std_error": se_lambda,
        "t_stat": t_stat,
        "intercept": c_hat,
        "sigma2_hat": sigma2_hat,
        "ssr": ssr,
        "success": result.success,
        "message": result.message,
        "fitted": fitted,
        "residuals": resid,
        "vcov": vcov,
    }


ids = pi_matrix.index.intersection(gamma_matrix.index)
PI = pi_matrix.loc[ids, ids].values
GAMMA = gamma_matrix.loc[ids, ids].values
L = pop_changes.set_index('orig_id').loc[ids, 'change_pct'].values

df_climate_80_2010 = df_climate_decade[df_climate_decade['decade'].isin([1980, 1990, 2000, 2010])]

nlls_rows = []

for climate_index in [
        'mean_exp', 'pca_exp', 'heat', 'flood', 'drought_anomaly', 'drought_absolute'
        ]:
   
    # Method 1: average out shocks and population changes to capture the overall effect of climate
    decade_climate = df_climate_80_2010[['CD_GEOCME', 'decade', climate_index]].groupby('CD_GEOCME').mean()
    Z = decade_climate.loc[ids, climate_index]
    decade_pop_changes = pop_changes[['orig_id', 'decade', 'change_pct']].groupby('orig_id').mean()
    L = decade_pop_changes.loc[ids, 'change_pct'].values
    
    
    # Method 2: lets check what happens if we use only data from a single decade
    climate_90 = df_climate_80_2010[df_climate_80_2010['decade']==1990]
    Z = climate_90.set_index('CD_GEOCME').loc[ids, climate_index].values
    pop_change_90 = pop_changes[pop_changes['decade']==1990]
    L = pop_change_90.set_index('orig_id').loc[ids, 'change_pct'].values

    res = estimate_theta_over_sigma(
        PI=PI,
        GAMMA=GAMMA,
        L=L,
        Z=Z,
        bounds=(1e-8, 50)
    )
    row = {
        'index': climate_index,
        'lam': res['theta_over_sigma'],
        'se': res['std_error'],
        'const': res['intercept'],
        'success': res['success']
        }
    nlls_rows.append(row)

nlls_results = pd.DataFrame(nlls_rows)
# noncredible results. Gotta go back to the model and understand it properly





