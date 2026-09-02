# Fixed-land Polocentro model

This directory contains a deliberately small spatial-equilibrium model for
studying whether Polocentro created national production or displaced activity
from untreated regions.

## Economic environment

A measure of potential firms begins at an inactive outside option `O`. Every
firm observes a location-specific Fréchet productivity match. Conditional on
choosing location `l`, it rents capital and land and produces

```text
Y_il = A_l * eta_il * K_il**alpha * L_il**beta,
```

with `alpha + beta < 1`. The residual share is the return to a fixed
entrepreneurial input owned by the firm. A firm moves only if operating surplus
exceeds the location's moving cost.

Every productive location has a fixed land endowment. Regional land rents clear
the land markets. The national interest rate clears aggregate capital demand
against

```text
S_K(R) = K_ref * (R / R_ref)**capital_supply_elasticity.
```

Polocentro is represented by a proportional discount on the capital user cost
paid by firms in treated locations. Capital suppliers still receive the national
rate, so the difference is a subsidy and the code reports its fiscal bill.

## Files

- `model.py`: calibration, simulated data, treatment assignment, equilibrium
  solver, counterfactual comparisons, and output plots.
- `run_simulation.py`: editable analysis script.
- `test_model.py`: numerical and accounting regression tests.

## Running the example

From the repository root:

```powershell
& '.\.venv\Scripts\python.exe' 'code\polocentro_model\run_simulation.py'
```

Outputs are written to `outputs/polocentro_model_simulation`:

- national counterfactual table;
- location-level counterfactual table;
- production creation/displacement decomposition;
- firm transition table;
- equilibrium-level and policy-impact plots.

Run tests with:

```powershell
$env:PYTHONPATH = 'code'
& '.\.venv\Scripts\python.exe' -m unittest polocentro_model.test_model
```

## Main calibration surface

Edit the `CALIBRATION` block in `run_simulation.py`. The central parameters are:

- `alpha`, `beta`: capital and land output elasticities;
- `frechet_shape`, `frechet_scale`: firm-location match distribution;
- `capital_supply_elasticity`: national capital-supply response;
- `program_interest_rate_discount`: treated-region policy wedge;
- treatment assignment weights and random-effect dispersion;
- simulation sizes and random seeds.

The Fréchet shape must exceed `1 / (1 - alpha - beta)`. This guarantees that
mean output is finite in the underlying continuum economy.

The simulated assignment treats approximately `treatment_share` of locations
with the highest latent scores. Distance and productivity raise the systematic
score, while a location random effect ensures that assignment is not a
deterministic function of observables.

For a capital-supply sensitivity grid, use:

```python
from polocentro_model import capital_supply_elasticity_sweep

sweep = capital_supply_elasticity_sweep(
    economy,
    calibration,
    elasticities=[0.0, 0.5, 1.0, 2.0, 5.0],
)
```

## Replacing simulated locations with data

Construct a pandas DataFrame with one row per location and these columns:

| Column | Meaning |
|---|---|
| `location_id` | Unique identifier |
| `distance` | Distance or road travel cost from `O` |
| `moving_cost` | Monetary equivalent of the location cost |
| `productivity` | Location productivity `A_l`, e.g. crop suitability |
| `land_supply` | Fixed productive land endowment |
| `treated` | Boolean treatment assignment |

Then construct:

```python
economy = EconomyData(
    locations=locations,
    match_productivity=draw_match_productivity(calibration, len(locations)),
)
```

The solver and analysis code require no other changes.

## Numerical interpretation

The continuum of firms is approximated with Monte Carlo draws. Firms choose
locations deterministically. A large `n_firms` therefore makes aggregate demand
smoother and improves market clearing. Always inspect
`max_abs_market_residual`; it is the largest absolute log demand-supply ratio
across the national capital market and all local land markets.
