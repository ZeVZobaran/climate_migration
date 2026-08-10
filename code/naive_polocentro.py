# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 13:46:19 2026

@author: josez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
import pyfixest as pf
import statsmodels.api as sm
import seaborn as sns

# %% Quickcheck with Polocentro treated regions
path_2 = r'C:\Users\josez\OneDrive\Desktop\EPGE\climate_migration\data\processed\amcs'
# check 1: maximally naive

muni_amc_cross = pd.read_parquet(
    f'{path_2}/municipality_to_amc_crosswalk.parquet'
    )
polocentro_treat = pd.read_parquet(
    f'{path_2}/polocentro_1975_amc_treatment.parquet'
    )
uf_amc_flows = pd.read_parquet(
    f'{path_2}/amc_origin_uf_year_flows.parquet'
    )

amc_traits = pd.read_parquet(
    f'{path_2}/amc_year_characteristics.parquet'
    )
# Just comparing means
uf_amc_flows = pd.merge(
    uf_amc_flows,
    polocentro_treat[['amc_code',
                      'polocentro_operational_core',
                      'polocentro_operational_area_share']],
    on = ['amc_code']
    )

amc_traits = pd.merge(
    amc_traits,
    polocentro_treat[['amc_code',
                      'polocentro_operational_core',
                      'polocentro_operational_area_share']],
    on = ['amc_code']
    )


averages = uf_amc_flows[[
    'census_year', 'polocentro_operational_core', 'weighted_interstate_migrants',
    'interstate_migrant_share_of_destination_age5plus',
    'employment_share_age15plus', 'literacy_share_age15plus',
    'positive_income_q1_share_age15plus',
    'positive_income_q2_share_age15plus',
    'positive_income_q3_share_age15plus',
    'positive_income_q4_share_age15plus',
    'positive_income_q5_share_age15plus',
    'zero_income_share_age15plus'
    ]].groupby(
    ['census_year', 'polocentro_operational_core']).mean()

stds = uf_amc_flows[[
    'census_year', 'polocentro_operational_core',
    'interstate_migrant_share_of_destination_age5plus',
    'employment_share_age15plus', 'literacy_share_age15plus',
    'positive_income_q1_share_age15plus',
    'positive_income_q2_share_age15plus',
    'positive_income_q3_share_age15plus',
    'positive_income_q4_share_age15plus',
    'positive_income_q5_share_age15plus',
    'zero_income_share_age15plus'
    ]].groupby(
    ['census_year', 'polocentro_operational_core']).std()
        
averages['interstate_migrant_share_of_destination_age5plus'].unstack().plot()
# Good!
# Can look also at overall migration, not pair-wise, maybe
#%% First pass regression

year_dummies = pd.get_dummies(uf_amc_flows['census_year'])
for year in year_dummies.columns:
    uf_amc_flows[f'treated_{year}'] = year_dummies[year] * uf_amc_flows['polocentro_operational_core']

reg_ppml = pf.fepois(
    'weighted_interstate_migrants ~ treated_1970 + treated_1991 + treated_2000 + treated_2010'
    " | census_year + amc_code + origin_uf",
    data=uf_amc_flows,
    vcov={"CRV1": "amc_code + origin_uf"}
    )
reg_ppml.summary()
# Looks somewhat good also

# %% Muni GDP
polocentro_treat_munis = pd.read_parquet(
    f'{path_2}/polocentro_1975_municipality_treatment.parquet'
    )

muni_traits = pd.read_parquet(
    f'{path_2}/municipality_year_gdp.parquet'
    )
# Investigar os zeros nisso aqui
muni_traits = pd.merge(
    muni_traits,
    polocentro_treat_munis[['municipality_code', 'polocentro_operational_core']],
    on = ['municipality_code']
    )

gdp_sums = muni_traits[[
    'year', 'polocentro_operational_core',
    'gdp_total_2010_brl_thousand', 'va_industry_2010_brl_thousand',
    'va_services_private_2010_brl_thousand',
    'va_public_administration_2010_brl_thousand',
    'va_agriculture_2010_brl_thousand'
    ]].groupby(
    ['year', 'polocentro_operational_core']).sum().unstack()
gdp_growth = gdp_sums.pct_change()
gdp_growth['va_industry_2010_brl_thousand'].plot()


# %%
muni_traits_nonzero = muni_traits[muni_traits['gdp_total_2010_brl_thousand']>0]

muni_traits_nonzero['log_gdp_total'] = muni_traits_nonzero['gdp_total_2010_brl_thousand'].apply(
    lambda x: np.log(x)
    )

muni_traits_nonzero['log_va_agri'] = np.log(muni_traits_nonzero['va_agriculture_2010_brl_thousand'])
muni_traits_nonzero['log_va_ind'] = np.log(muni_traits_nonzero['va_industry_2010_brl_thousand'])
muni_traits_nonzero['log_va_priv_serv'] = np.log(muni_traits_nonzero['va_services_private_2010_brl_thousand'])
muni_traits_nonzero['log_va_public_adm'] = np.log(muni_traits_nonzero['va_public_administration_2010_brl_thousand'])

muni_years_dummies = pd.get_dummies(muni_traits_nonzero['year'])
for year in muni_years_dummies.columns:
    muni_traits_nonzero[f'treated_{year}'] = muni_years_dummies[year] * muni_traits_nonzero['polocentro_operational_core']

reg_gdp = pf.feols(
    f'log_gdp_total ~ treated_1959 + treated_1970 + '
    'treated_1980 + treated_1985 + treated_1996 + treated_2000 + treated_2005 + treated_2010'
    " | year + municipality_code",
    data=muni_traits_nonzero,
    vcov={"CRV1": "municipality_code"}
    )
reg_gdp.summary()

# No effect on public adm
# Industrial and private services VAs have anticipatory effects 
# Selection?
# Agriculture and total GDP as expected: positive (and lasting!) after 1975, zero before
# they do, however, look trendy. Some selection is at play here

# %% Coef plotting function
def plot_coefficients(
    title: str,
    coefficients: Sequence[float],
    std_vars: Sequence[float],
    x: Sequence | None = None,
    xlabel: str = "",
    ylabel: str = "Coefficient",
    color: str = "tab:blue",
    figsize: tuple[float, float] = (10, 5),
):
    """
    Plot coefficients with shaded ±1 and ±2 standard-error bands.

    Parameters
    ----------
    title
        Plot title.
    coefficients
        Sequence of estimated coefficients.
    std_vars
        Sequence of standard errors or standard deviations.
    x
        Optional x-axis values. Defaults to 0, 1, ..., n-1.
    """

    coefficients = np.asarray(coefficients, dtype=float)
    std_vars = np.asarray(std_vars, dtype=float)

    if coefficients.ndim != 1 or std_vars.ndim != 1:
        raise ValueError("coefficients and std_vars must be one-dimensional.")

    if len(coefficients) != len(std_vars):
        raise ValueError("coefficients and std_vars must have the same length.")

    if np.any(std_vars < 0):
        raise ValueError("std_vars cannot contain negative values.")

    if x is None:
        x = np.arange(len(coefficients))
    else:
        x = np.asarray(x)

    if len(x) != len(coefficients):
        raise ValueError("x must have the same length as coefficients.")

    fig, ax = plt.subplots(figsize=figsize)

    # Lighter outer band: coefficient ± 2 standard errors
    ax.fill_between(
        x,
        coefficients - 2 * std_vars,
        coefficients + 2 * std_vars,
        color=color,
        alpha=0.15,
        label=r"$\pm 2$ standard errors",
    )

    # Darker inner band: coefficient ± 1 standard error
    ax.fill_between(
        x,
        coefficients - std_vars,
        coefficients + std_vars,
        color=color,
        alpha=0.30,
        label=r"$\pm 1$ standard error",
    )

    ax.plot(
        x,
        coefficients,
        color=color,
        linewidth=2,
        marker="o",
        label="Coefficient",
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, ax


# %% PSW Functions
def generate_psm_weights(
    df,
    treatment,
    covariates,
    estimand="ATT",
    trim=None
):
    """
    Estimate propensity scores and generate IPW weights.

    Parameters
    ----------
    df : pd.DataFrame
    treatment : str
        Binary treatment variable (0/1).
    covariates : list[str]
        Variables determining treatment.
    estimand : {"ATT", "ATE"}
    trim : float or None
        If e.g. 0.01, drops observations with pscore < .01 or > .99.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with 'pscore' and 'psm_weight'.
    model : fitted statsmodels Logit
    """
    subset = [treatment] + covariates
    out = df[subset].fillna(0).copy()

    X = sm.add_constant(out[covariates])
    T = out[treatment]

    model = sm.Logit(T, X).fit(disp=False)

    out["pscore"] = model.predict(X)

    # Optional crude common-support trimming
    if trim is not None:
        out = out[
            out["pscore"].between(trim, 1 - trim)
        ].copy()

    p = out["pscore"]
    T = out[treatment]

    if estimand.upper() == "ATE":
        # Treated: 1/p
        # Controls: 1/(1-p)
        out["psm_weight"] = (
            T / p +
            (1 - T) / (1 - p)
        )

    elif estimand.upper() == "ATT":
        # Treated get weight 1.
        # Controls reweighted to resemble treated.
        out["psm_weight"] = (
            T +
            (1 - T) * p / (1 - p)
        )

    else:
        raise ValueError("estimand must be 'ATT' or 'ATE'")

    return out, model

def plot_propensity_scores(
    df,
    treatment,
    pscore="pscore",
    bins=30,
    title="Propensity Score Overlap"
):
    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df[df[treatment] == 0],
        x=pscore,
        bins=bins,
        stat="density",
        alpha=0.4,
        color="tab:blue",
        label="Control"
    )

    sns.histplot(
        data=df[df[treatment] == 1],
        x=pscore,
        bins=bins,
        stat="density",
        alpha=0.4,
        color="tab:orange",
        label="Treated"
    )

    plt.xlabel("Propensity score")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% PSW and common support tests
uf_amc_covs = [
    'weighted_population', 
    'employment_share_age15plus',
    'female_share',
    'mean_age',
    'mean_education_years_age25plus',
    'mean_household_size',
    'positive_income_q1_share_age15plus',
    'positive_income_q2_share_age15plus',
    'positive_income_q3_share_age15plus',
    'positive_income_q4_share_age15plus',
    'positive_income_q5_share_age15plus',
    'race_asian_share',
    'race_black_share', 
    'race_indigenous_share',
    'race_pardo_share',
    'race_white_share',
    'urban_share',
    'zero_income_share_age15plus'
    ]

uf_amc_psw, psw_model = generate_psm_weights(
    df=uf_amc_flows,
    treatment='polocentro_operational_core',
    covariates=uf_amc_covs,
    estimand='ATT', trim=0.01
    )

plot_propensity_scores(
    uf_amc_psw,
    treatment="polocentro_operational_core"
    )
# %%
muni_traits_psw_adj = muni_traits_nonzero.copy()
muni_traits_psw_adj['polocentro_operational_core'] = \
    muni_traits_psw_adj['polocentro_operational_core'].apply(lambda x:
                                                             1 if x else 0)
state_dummies = pd.get_dummies(muni_traits_psw_adj['uf'])
state_dummies = state_dummies.apply(lambda col: col.apply(
    lambda x: 1 if x else 0
    ))
state_dummies = state_dummies[['SP', 'MG', 'GO', 'MS', 'MT', 'TO', 'PA']]
                               
muni_traits_nonzero[state_dummies.columns] = state_dummies

muni_traits_covs = [
    'gdp_total_2010_brl_thousand', 
    'va_industry_2010_brl_thousand',
    'va_services_private_2010_brl_thousand',
    'va_public_administration_2010_brl_thousand',
    'va_agriculture_2010_brl_thousand', 
    'va_industry_share_of_gdp',
    'va_services_private_share_of_gdp',
    ]
muni_traits_covs.extend(state_dummies.columns)

munis_psw, munis_psw_model = generate_psm_weights(
    df=muni_traits_nonzero,
    treatment='polocentro_operational_core',
    covariates=muni_traits_covs,
    estimand='ATT', trim=0.01
    )

plot_propensity_scores(
    munis_psw,
    treatment="polocentro_operational_core"
    )
# %% Regs with PS weights

muni_traits_nonzero = muni_traits_nonzero.merge(
    munis_psw['psm_weight'], left_index=True, right_index=True
    )

reg_gdp_psw = pf.feols(
    'log_gdp_total ~ treated_1959 + treated_1970 + '
    'treated_1980 + treated_1985 + treated_1996 + treated_2000 + treated_2005 + treated_2010'
    " | year + municipality_code",
    data=muni_traits_nonzero,
    weights = 'psm_weight',
    vcov={"CRV1": "municipality_code"}
    )

reg_gdp.summary()
reg_gdp_psw.summary()

uf_amc_flows = uf_amc_flows.merge(
    uf_amc_psw['psm_weight'], left_index=True, right_index=True
    )
reg_ppml_psw = pf.fepois(
    'weighted_interstate_migrants ~ treated_1970 + treated_1991 + treated_2000 + treated_2010'
    " | census_year + amc_code + origin_uf",
    data=uf_amc_flows,
    weights = 'psm_weight',
    vcov={"CRV1": "amc_code + origin_uf"}
    )
reg_ppml_psw.summary()

# %% Quick plots
migration_plot = plot_coefficients(
    'POLOCENTRO naive event study coefficients on interstate migration',
    reg_ppml.coef(), reg_ppml.se(), [-5, 5, 25, 35]
    )

migration_plot = plot_coefficients(
    'POLOCENTRO naive event study coefficients on interstate migration - PSW',
    reg_ppml_psw.coef(), reg_ppml_psw.se(), [-5, 5, 25, 35]
    )


gdp_plot = plot_coefficients(
    'POLOCENTRO naive event study coefficients on log GDP',
    reg_gdp.coef(), reg_gdp.se(), [-16, -5, 5, 10, 21, 25, 30, 35]
    )
gdp_plot = plot_coefficients(
    'POLOCENTRO Event Study Coefficients on municipal log GDP - PSW',
    reg_gdp_psw.coef(), reg_gdp_psw.se(), [-16, -5, 5, 10, 21, 25, 30, 35]
    )













