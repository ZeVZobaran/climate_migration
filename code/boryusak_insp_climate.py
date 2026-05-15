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
from numpy.linalg import solve, inv

sns.set(style="whitegrid")

path = r'D:\Users\c337191\Documents\climate_migration'
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
        " | year"
        )

    reg_w = pf.feols(
        fml,
        data=df,
        weights="N_od_flow_all",
        vcov={"CRV1": "pair_id"}
    )
    return reg_w

rows = []
climate_indexes = df_climate_decade.columns[4:]

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
        f'N_od_flow_all ~ orig_{climate_index} + dest_{climate_index}'
        " | year",
    data=df,
    vcov={"CRV1": "pair_id"}
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
# Lend credence to the OG indexes for origin push factor
# Between indexes remain an utter mess
# story gets weirder for within indexes also
# will ignore this for now...
    
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

    PI    : out-migration matrix Π
    GAMMA : in-migration matrix Γ
    lam   : theta / sigma
    """
    PI = np.asarray(PI, dtype=float)
    GAMMA = np.asarray(GAMMA, dtype=float)

    n = PI.shape[0]
    I = np.eye(n)

    A = I + lam * (I - GAMMA.T @ PI)

    return I - solve(A, I)


def nlls_prediction(params, PI, GAMMA, Z):
    """
    Model prediction:
        L_hat = c + Ω(λ) Z

    params = [lambda, intercept]
    """
    lam, intercept = params

    Z = np.asarray(Z, dtype=float).reshape(-1)

    return intercept + omega_matrix(lam, PI, GAMMA) @ Z


def nlls_objective_with_intercept(lam, PI, GAMMA, L, Z):
    """
    SSR objective where intercept is profiled out.

    For each λ:
        c(λ) = mean[L - Ω(λ)Z]
    """
    L = np.asarray(L, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)

    fitted_no_intercept = omega_matrix(lam, PI, GAMMA) @ Z

    intercept = np.mean(L - fitted_no_intercept)

    resid = L - intercept - fitted_no_intercept

    return resid @ resid


def estimate_theta_over_sigma(
    PI,
    GAMMA,
    L,
    Z,
    bounds=(1e-8, 100),
    xatol=1e-10
):
    """
    Estimate:

        L = c + Ω(λ)Z + error

    by NLLS, with λ = theta / sigma.

    Returns λ_hat, intercept_hat, standard errors, fitted values,
    residuals, and variance-covariance matrix.
    """

    PI = np.asarray(PI, dtype=float)
    GAMMA = np.asarray(GAMMA, dtype=float)
    L = np.asarray(L, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float).reshape(-1)

    n = len(L)
    k = 2  # lambda + intercept

    # -------------------------------
    # 1. Estimate lambda
    # -------------------------------

    opt = minimize_scalar(
        nlls_objective_with_intercept,
        bounds=bounds,
        method="bounded",
        args=(PI, GAMMA, L, Z),
        options={"xatol": xatol}
    )

    lam_hat = opt.x

    # -------------------------------
    # 2. Estimate intercept conditional on lambda_hat
    # -------------------------------

    fitted_no_intercept = omega_matrix(lam_hat, PI, GAMMA) @ Z

    intercept_hat = np.mean(L - fitted_no_intercept)

    fitted = intercept_hat + fitted_no_intercept
    resid = L - fitted

    ssr = resid @ resid
    sigma2_hat = ssr / (n - k)

    # -------------------------------
    # 3. Standard errors
    # -------------------------------

    params_hat = np.array([lam_hat, intercept_hat])

    def pred_func(params):
        return nlls_prediction(params, PI, GAMMA, Z)

    J = approx_derivative(
        pred_func,
        x0=params_hat,
        method="3-point"
    )

    # J is n x 2:
    # column 0 = derivative wrt lambda
    # column 1 = derivative wrt intercept

    vcov = sigma2_hat * inv(J.T @ J)

    se_lambda = np.sqrt(vcov[0, 0])
    se_intercept = np.sqrt(vcov[1, 1])

    t_lambda = lam_hat / se_lambda
    t_intercept = intercept_hat / se_intercept

    return {
        "theta_over_sigma": lam_hat,
        "intercept": intercept_hat,

        "std_error_lambda": se_lambda,
        "std_error_intercept": se_intercept,

        "t_lambda": t_lambda,
        "t_intercept": t_intercept,

        "sigma2_hat": sigma2_hat,
        "ssr": ssr,

        "success": opt.success,
        "message": opt.message,

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
        'mean_exp', 'pca_exp', 'comp_rel_within'
        ]:
   
    # Method 1: average out shocks and population changes to capture the overall effect of climate
    decade_climate = df_climate_80_2010[['CD_GEOCME', 'decade', climate_index]].groupby('CD_GEOCME').mean()
    Z = decade_climate.loc[ids, climate_index]
    decade_pop_changes = pop_changes[['orig_id', 'decade', 'change_pct']].groupby('orig_id').mean()
    L = decade_pop_changes.loc[ids, 'change_pct'].values
    
    
    # Method 2: lets check what happens if we use only data from a single decade
    climate_90 = df_climate_80_2010[df_climate_80_2010['decade']==1980]
    Z = climate_90.set_index('CD_GEOCME').loc[ids, climate_index].values
    pop_change_90 = pop_changes[pop_changes['decade']==1980]
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
        'se': res['std_error_lambda'],
        'const': res['intercept'],
        'success': res['success']
        }
    nlls_rows.append(row)

nlls_results = pd.DataFrame(nlls_rows)
# noncredible results. Gotta go back to the model and understand it properly





