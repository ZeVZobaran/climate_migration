# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:42:00 2026

@author: josez
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
import pyfixest as pf
import statsmodels.api as sm
import seaborn as sns
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from auxs.plotting import compare_regression_coefficients
from auxs.propensity import generate_psw_weights, plot_propensity_diagnostics

# %% data load
data_path = f'{ROOT}/data/processed'

treat_assign = pd.read_parquet(
    f'{data_path}/amcs/polocentro_1975_amc_treatment.parquet'
    )

controls = pd.read_parquet(
    f'{data_path}/polocentro_event_study/amc_year_controls.parquet'
    )

outcomes = pd.read_parquet(
    f'{data_path}/polocentro_event_study/amc_year_outcomes.parquet'
    )

# %% ES

possible_treat_assigns = {
    'core': "polocentro_operational_core",
    'lax': "polocentro_operational_any_overlap",
    'rigid': "polocentro_operational_majority_area"
    }
# For each treatment assignment, I want to generate PS scores
# and use those as weighs. Also run a nonweighted version for comparison
# And keep it simple wrt FEs. AMC and year will suffice for now

# Relevant state dummies
state_dummies = pd.get_dummies(controls['state_uf'])
state_dummies = state_dummies.apply(lambda col: col.apply(
    lambda x: 1 if x else 0
    ))
state_dummies = state_dummies[['MG', 'GO', 'MS', 'MT', 'TO']]
controls[state_dummies.columns] = state_dummies
# Processing controls and setting stable covariates
controls_1970 = controls[controls['year']==1970].reset_index(drop=True)
covariates = [
'positive_income_q1_share_age15plus_1970',
'positive_income_q3_share_age15plus_1970', 'female_share_1970',
'urban_share_1970', 'share_age_65plus_1970', 'log_population_1970',
'log_gdp_total_1970', 'log_gdp_per_capita_1970',
'va_agriculture_share_of_gdp_1970', 'literacy_share_age15plus_1970',
'gaez_cof_high_mean_0_100',
'gaez_cot_high_mean_0_100',
'gaez_mze_high_mean_0_100',
'gaez_phb_high_mean_0_100',
'gaez_rcd_high_mean_0_100',
'gaez_soy_high_mean_0_100',
'gaez_suc_high_mean_0_100',
'gaez_whe_high_mean_0_100',
'gaez_core8_high_mean_0_100',
'MG', 'GO', 'MS', 'MT', 'TO'        
    ]

results = {}
for name, treatment in possible_treat_assigns.items():
    results[treatment] = {}
    treated_amcs = treat_assign[['amc_code', treatment]]

    control_treatment = pd.merge(
        treated_amcs, controls_1970, on=['amc_code']
        )
    ps, smd, analytics = generate_psw_weights(
        df=control_treatment, id_column='amc_code', 
        treatment=treatment, covariates=covariates, trim=0.01
        )
    ps_density, ps_density_ax = plot_propensity_diagnostics(
        ps, treatment, title=f'{name} treatment assignment PS overlap'
        )

    results[treatment]['ps'] = (ps, smd, analytics)
    results[treatment]['ps_density'] = ps_density
    # regressions
    for outcome in outcomes:
        if outcome in ['amc_code', 'year', 'gdp_growth_year_gap',
        'va_agriculture_growth_year_gap', 'population_growth_year_gap',
        'has_gdp_outcome', 'has_population_outcome', 'has_migration_outcome',
        'has_mapbiomas_outcome']:
            continue
        results[treatment][outcome] = {}

        outcome_df = outcomes[['amc_code', 'year', outcome]].dropna()
        treatment_outcome_df = pd.merge(
            outcome_df, treated_amcs, on=['amc_code']
            )
        treatment_outcome_df[treatment] = treatment_outcome_df[treatment].apply(
            lambda x: 1 if x else 0
            )
        # Generating year dummies
        # We exclude the one that's nearest to 1975
        min_diff = (treatment_outcome_df['year']-1970).abs().min()
        reference_year = 1970+min_diff
        results[treatment][outcome]['ref_year'] = reference_year
        
        years_dummies = pd.get_dummies(treatment_outcome_df['year'])
        for year in years_dummies.columns:
            if year == reference_year:
                continue
            treatment_outcome_df[f'treated_{year}'] = years_dummies[year] * \
                treatment_outcome_df[treatment]
        event_dummies = [x for x in treatment_outcome_df if 'treated' in x]
        event_dummies_str = '+ '.join(event_dummies)
        

        if outcome == 'weighted_interstate_migrants':
            # In this case its a persons count
            # so PPML is the chosen method
            # Otherwise we're good
            spec = pf.fepois
        else:
            spec = pf.feols
        # Regression formulae
        year_reg = f'{outcome} ~ {event_dummies_str} | year'
        fe_reg = f'{outcome} ~ {event_dummies_str}'\
            ' | year + amc_code'
        fml_dict = {'Year FEs': year_reg, 'Year+AMC FEs': fe_reg}
        # Adding ps weights
        reg_df = treatment_outcome_df.merge(
            ps[['amc_code', 'psw_weight']], on=['amc_code'])
        reg_df['treated'] = reg_df[event_dummies].sum(axis=1)
        
# =============================================================================
#         # Trying to look at redistribution losses?
#         atnt_df = reg_df[reg_df['treated']==0].dropna()
#         atnt_df['treated_candidates'] = atnt_df['psw_weight']>0.2499
#         for year in atnt_df['year'].unique():
#             if year == reference_year:
#                 continue
#             atnt_df[f'treated_{year}'] = atnt_df['treated_candidates']*atnt_df['year']==year
# =============================================================================

        results[treatment][outcome]['regs'] = {}
        for key, fml in fml_dict.items():
            reg = spec(
                fml, data=reg_df,
                vcov={"CRV1": "amc_code"})
            ps_reg = spec(
                fml, data=reg_df,
                vcov={"CRV1": "amc_code"},
                weights = 'psw_weight')
            results[treatment][outcome]['regs'][key] = reg
            if key=='Year+AMC FEs':
                results[treatment][outcome]['regs'][f'{key} +PS Weights'] = ps_reg
        # Synthesis output: a plot of all four coeffs with 1sd and 2sd bands
        coef_comp, coef_comp_ax = compare_regression_coefficients(
            results[treatment][outcome]['regs'], coefficients=event_dummies,
            x_labels=[int(x[-4:]) for x in event_dummies],
            title=f'Polocentro effect on {outcome.replace("_", " ")}\n'\
                f'Reference year: {reference_year}; Treatment assignment: {name}'
            )

        coef_comp_simple, coef_comp_ax_simple = compare_regression_coefficients(
            {1: results[treatment][outcome]['regs'][f'{key} +PS Weights']},
            coefficients=event_dummies,
            x_labels=[int(x[-4:]) for x in event_dummies],
            title=f'Polocentro effect on {outcome.replace("_", " ")}\n'\
                f'Reference year: {reference_year}'
            )
        results[treatment][outcome]['main_plot'] = (coef_comp_simple, coef_comp_ax_simple)



#%% Some light edits to the main plot and saving
save_dir = r'C:\Users\josez\OneDrive\Desktop\EPGE\climate_migration\reports\Apts\polocentro_event_study_assets\single_spec_assets'
for name, treatement in possible_treat_assigns.items():
    if name == 'lax':
        title = 'Any overlap'
    elif name == 'rigid':
        title = 'Majority area'
    else:
        title = 'Core'
    for outcome in outcomes:
        if outcome in ['amc_code', 'year', 'gdp_growth_year_gap',
        'va_agriculture_growth_year_gap', 'population_growth_year_gap',
        'has_gdp_outcome', 'has_population_outcome', 'has_migration_outcome',
        'has_mapbiomas_outcome']:
            continue
        data = results[treatement][outcome]
    
        figure = data["main_plot"][0]
        axis = figure.axes[0]
        figure.set_size_inches(5.0, 4.2)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Event year", fontsize=9)
        axis.set_ylabel("Estimate", fontsize=9)
        axis.tick_params(axis="both", labelsize=7)
        tick_labels = axis.get_xticklabels()
        if len(tick_labels) > 15:
            for index, tick_label in enumerate(tick_labels):
                tick_label.set_visible(index % 5 == 0 or index == len(tick_labels) - 1)
        axis.get_legend().set_visible(False)

    
        figure.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.22)
        figure.savefig(
            f'{save_dir}/{name}_{outcome}.pdf',
            bbox_inches="tight",
        )
    

# The best claim here is that the policy
# likely had positive effects on:
    # Level GDP and agricultural level GDP
    # Population, partially via migration influx
    # Maybe soybeans, and maybe vegetation loss
# And no effect on agri area, pasture are!

# Taking the widest view of treatment (any overlap),
# we are able to detect some effect on pastures
# and vegetation loss

# Taking the most restrictive, that considers only
# majority area, we lose the effects on overall GDP,
# pasture, vegetation loss
# And retain under lower confidence effects on agriculture,
# soybeans, migration and population









