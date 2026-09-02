"""Fixed-land spatial equilibrium model for Polocentro counterfactuals.

The economy contains a finite Monte Carlo approximation to a measure of
potential firms. Each firm knows its Fréchet-distributed match productivity in
every productive location. Conditional on locating in location ``l``, firm
``i`` solves

    max_{K, L} A_l * eta_il * K**alpha * L**beta
               - q_l * K - rho_l * L - moving_cost_l,

where ``alpha + beta < 1``. The residual share can be interpreted as the return
to the firm's fixed entrepreneurial/managerial input.

Land is fixed in every location. Its rental price ``rho_l`` adjusts until all
land is rented. A national interest rate ``R`` adjusts until total capital
demand equals an isoelastic aggregate capital supply curve. Under the policy,
firms in treated locations pay a fraction of ``R``; capital suppliers still
receive ``R``, so the wedge is interpreted as a government-financed subsidy.

The code deliberately keeps the economic structure small. Location data are a
plain pandas DataFrame, which makes replacing the simulated locations with road
distance and crop-suitability data straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
    raise ImportError(
        "polocentro_model requires scipy. Install numpy, pandas, scipy, and "
        "matplotlib in the active Python environment."
    ) from exc


REQUIRED_LOCATION_COLUMNS = {
    "location_id",
    "distance",
    "moving_cost",
    "productivity",
    "land_supply",
    "treated",
}


@dataclass(frozen=True)
class Calibration:
    """Transparent collection of structural and simulation parameters.

    Parameters most likely to be swept or calibrated later are grouped first.
    All prices and quantities are normalized; only relative magnitudes matter.

    ``capital_supply_at_reference`` is total capital supplied when the national
    interest rate equals ``reference_interest_rate``. Capital supply is

        S_K(R) = K_ref * (R / R_ref)**capital_supply_elasticity.

    ``capital_supply_elasticity=0`` gives a fixed national capital stock.

    The Fréchet draw has CDF

        P(eta <= x) = exp(-(frechet_scale / x)**frechet_shape).

    To make population moments of output finite in the continuum model, the
    Fréchet shape should exceed ``1 / (1 - alpha - beta)``. The code enforces
    this condition because otherwise Monte Carlo results are dominated by the
    single largest draw.
    """

    # Production and match heterogeneity.
    alpha: float = 0.25
    beta: float = 0.20
    frechet_shape: float = 5.0
    frechet_scale: float = 1.0

    # National capital supply.
    capital_supply_elasticity: float = 1.0
    capital_supply_at_reference: float = 10.0
    reference_interest_rate: float = 1.0

    # Program: treated firms pay R * (1 - program_interest_rate_discount).
    program_interest_rate_discount: float = 0.30

    # Size of the Monte Carlo economy and simulated geography.
    n_firms: int = 30_000
    firm_mass: float = 1.0
    n_locations: int = 15
    land_per_location: float = 1.0

    # Simulated location fundamentals. These are ignored once real data enter.
    location_productivity_log_sd: float = 0.30
    location_productivity_distance_gradient: float = 0.15
    moving_cost_scale: float = 0.25
    moving_cost_curvature: float = 1.15

    # Simulated treatment assignment.
    treatment_share: float = 0.35
    treatment_distance_weight: float = 1.25
    treatment_productivity_weight: float = 1.25
    treatment_random_effect_sd: float = 1.25

    # Reproducibility.
    location_seed: int = 19_750
    treatment_seed: int = 19_751
    match_seed: int = 19_752

    # Numerical solver controls.
    solver_tolerance: float = 2.0e-7
    solver_max_evaluations: int = 600
    log_price_lower_bound: float = -12.0
    log_price_upper_bound: float = 12.0
    equilibrium_residual_warning: float = 5.0e-3

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie strictly between zero and one.")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie strictly between zero and one.")
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be strictly below one.")
        if self.frechet_shape <= 0.0 or self.frechet_scale <= 0.0:
            raise ValueError("Fréchet shape and scale must be positive.")

        minimum_shape = 1.0 / (1.0 - self.alpha - self.beta)
        if self.frechet_shape <= minimum_shape:
            raise ValueError(
                "frechet_shape must exceed 1 / (1 - alpha - beta) so mean "
                f"output is finite. It must exceed {minimum_shape:.3f}."
            )
        if self.capital_supply_elasticity < 0.0:
            raise ValueError("capital_supply_elasticity cannot be negative.")
        if self.capital_supply_at_reference <= 0.0:
            raise ValueError("capital_supply_at_reference must be positive.")
        if self.reference_interest_rate <= 0.0:
            raise ValueError("reference_interest_rate must be positive.")
        if not (0.0 <= self.program_interest_rate_discount < 1.0):
            raise ValueError(
                "program_interest_rate_discount must lie in [0, 1)."
            )
        if self.n_firms < 2:
            raise ValueError("n_firms must be at least two.")
        if self.n_locations < 2:
            raise ValueError("n_locations must be at least two.")
        if self.firm_mass <= 0.0 or self.land_per_location <= 0.0:
            raise ValueError("firm_mass and land_per_location must be positive.")
        if not (0.0 < self.treatment_share < 1.0):
            raise ValueError("treatment_share must lie strictly between 0 and 1.")
        if self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be positive.")

    @property
    def variable_input_share(self) -> float:
        """Return alpha + beta."""

        return self.alpha + self.beta

    @property
    def entrepreneur_share(self) -> float:
        """Return the residual share paid to the fixed entrepreneurial input."""

        return 1.0 - self.alpha - self.beta


@dataclass(frozen=True)
class EconomyData:
    """Exogenous location fundamentals and firm-location match draws."""

    locations: pd.DataFrame
    match_productivity: np.ndarray

    def __post_init__(self) -> None:
        locations = self.locations
        missing = REQUIRED_LOCATION_COLUMNS.difference(locations.columns)
        if missing:
            raise ValueError(f"locations is missing required columns: {sorted(missing)}")
        if locations["location_id"].duplicated().any():
            raise ValueError("location_id must uniquely identify locations.")
        if (locations["productivity"] <= 0.0).any():
            raise ValueError("All location productivities must be positive.")
        if (locations["land_supply"] <= 0.0).any():
            raise ValueError("All fixed land supplies must be positive.")
        if (locations["moving_cost"] < 0.0).any():
            raise ValueError("Moving costs cannot be negative.")

        matches = np.asarray(self.match_productivity)
        expected_shape = (matches.shape[0], len(locations))
        if matches.ndim != 2 or matches.shape != expected_shape:
            raise ValueError(
                "match_productivity must have shape (n_firms, n_locations); "
                f"received {matches.shape}."
            )
        if not np.isfinite(matches).all() or (matches <= 0.0).any():
            raise ValueError("All match-productivity draws must be finite and positive.")

    @property
    def n_firms(self) -> int:
        return int(self.match_productivity.shape[0])

    @property
    def n_locations(self) -> int:
        return int(self.match_productivity.shape[1])


@dataclass
class EquilibriumResult:
    """Prices, allocations, and numerical diagnostics from one equilibrium."""

    policy_active: bool
    national_interest_rate: float
    location_results: pd.DataFrame
    aggregate_results: pd.Series
    firm_results: pd.DataFrame
    firm_choices: np.ndarray
    log_prices: np.ndarray
    residuals: np.ndarray
    solver_success: bool
    solver_message: str
    solver_evaluations: int

    @property
    def max_abs_residual(self) -> float:
        """Largest absolute log market-clearing residual."""

        return float(np.max(np.abs(self.residuals)))


def _standardize(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    std = float(array.std(ddof=0))
    if std <= 1.0e-14:
        return np.zeros_like(array)
    return (array - float(array.mean())) / std


def _logistic_intercept(scores: np.ndarray, target_mean: float) -> float:
    """Find an intercept so mean logistic probability equals target_mean."""

    lower, upper = -40.0, 40.0
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        probabilities = 1.0 / (1.0 + np.exp(-(midpoint + scores)))
        if float(probabilities.mean()) < target_mean:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def generate_simulated_locations(calibration: Calibration) -> pd.DataFrame:
    """Generate placeholder geography, productivity, and moving costs.

    Distances span the unit interval. Productivity includes a modest positive
    distance gradient plus an idiosyncratic component, so far-away locations can
    be highly suitable despite initially high access costs.
    """

    rng = np.random.default_rng(calibration.location_seed)
    n_locations = calibration.n_locations

    distance = np.sort(rng.uniform(0.04, 1.0, size=n_locations))
    productivity_shock = rng.normal(
        loc=0.0,
        scale=calibration.location_productivity_log_sd,
        size=n_locations,
    )
    log_productivity = (
        calibration.location_productivity_distance_gradient * distance
        + productivity_shock
    )
    productivity = np.exp(log_productivity)
    productivity /= np.exp(np.mean(np.log(productivity)))

    moving_cost = calibration.moving_cost_scale * np.power(
        distance,
        calibration.moving_cost_curvature,
    )

    locations = pd.DataFrame(
        {
            "location_id": np.arange(n_locations, dtype=int),
            "distance": distance,
            "moving_cost": moving_cost,
            "productivity": productivity,
            "land_supply": calibration.land_per_location,
        }
    )
    return assign_treatment(locations, calibration)


def assign_treatment(
    locations: pd.DataFrame,
    calibration: Calibration,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Assign treatment using distance, productivity, and a random effect.

    The deterministic propensity is increasing in both distance and potential
    productivity. A location-specific normal random effect enters the latent
    assignment index before Bernoulli treatment is drawn. Consequently, some
    apparently attractive candidates remain untreated and some less obvious
    candidates receive treatment.

    The function records both the ex-ante probability (before the random effect)
    and the realized latent score. It treats the requested number of regions
    with the highest noisy latent scores. This keeps the treatment share close
    to its calibration target while allowing high-propensity regions with bad
    random draws to remain untreated.
    """

    required = {"distance", "productivity"}
    missing = required.difference(locations.columns)
    if missing:
        raise ValueError(f"Treatment assignment needs columns: {sorted(missing)}")

    result = locations.copy().reset_index(drop=True)
    rng = np.random.default_rng(calibration.treatment_seed if seed is None else seed)

    systematic_score = (
        calibration.treatment_distance_weight * _standardize(result["distance"])
        + calibration.treatment_productivity_weight
        * _standardize(np.log(result["productivity"]))
    )
    intercept = _logistic_intercept(systematic_score, calibration.treatment_share)
    assignment_probability = 1.0 / (
        1.0 + np.exp(-(intercept + systematic_score))
    )

    random_effect = rng.normal(
        0.0,
        calibration.treatment_random_effect_sd,
        size=len(result),
    )
    latent_score = intercept + systematic_score + random_effect
    n_treated = int(np.clip(round(calibration.treatment_share * len(result)), 1, len(result) - 1))
    treated = np.zeros(len(result), dtype=bool)
    treated[np.argsort(latent_score)[-n_treated:]] = True

    result["treatment_probability"] = assignment_probability
    result["treatment_random_effect"] = random_effect
    result["treatment_latent_score"] = latent_score
    result["treated"] = treated.astype(bool)
    return result


def draw_match_productivity(
    calibration: Calibration,
    n_locations: int | None = None,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Draw firm-location productivity matches from a Fréchet distribution."""

    locations = calibration.n_locations if n_locations is None else int(n_locations)
    if locations < 1:
        raise ValueError("n_locations must be positive.")

    rng = np.random.default_rng(calibration.match_seed if seed is None else seed)
    uniforms = rng.uniform(
        np.finfo(float).eps,
        1.0 - np.finfo(float).eps,
        size=(calibration.n_firms, locations),
    )
    matches = calibration.frechet_scale * np.power(
        -np.log(uniforms),
        -1.0 / calibration.frechet_shape,
    )
    return matches


def prepare_simulated_economy(calibration: Calibration) -> EconomyData:
    """Construct reproducible simulated locations and firm-match draws."""

    locations = generate_simulated_locations(calibration)
    matches = draw_match_productivity(calibration, len(locations))
    return EconomyData(locations=locations, match_productivity=matches)


def capital_supply(interest_rate: float, calibration: Calibration) -> float:
    """Evaluate the isoelastic national capital supply curve."""

    return calibration.capital_supply_at_reference * (
        interest_rate / calibration.reference_interest_rate
    ) ** calibration.capital_supply_elasticity


def _allocation_at_prices(
    log_prices: np.ndarray,
    economy: EconomyData,
    calibration: Calibration,
    policy_active: bool,
    *,
    return_full_matrix: bool = False,
) -> dict[str, Any]:
    """Compute firm choices and aggregate demands at candidate prices."""

    n_locations = economy.n_locations
    if len(log_prices) != n_locations + 1:
        raise ValueError("log_prices must contain R followed by one rent per location.")

    interest_rate = float(np.exp(log_prices[0]))
    land_rents = np.exp(log_prices[1:])
    treated = economy.locations["treated"].to_numpy(dtype=bool)

    discount = calibration.program_interest_rate_discount if policy_active else 0.0
    capital_user_cost = interest_rate * (1.0 - discount * treated.astype(float))

    productivity = economy.locations["productivity"].to_numpy(dtype=float)
    moving_cost = economy.locations["moving_cost"].to_numpy(dtype=float)
    match = economy.match_productivity

    alpha = calibration.alpha
    beta = calibration.beta
    residual_share = calibration.entrepreneur_share

    # Work in logs for stability. The expression is the closed-form solution to
    # the firm's conditional K-L problem.
    log_output = (
        np.log(productivity)[None, :]
        + np.log(match)
        + alpha * np.log(alpha)
        + beta * np.log(beta)
        - alpha * np.log(capital_user_cost)[None, :]
        - beta * np.log(land_rents)[None, :]
    ) / residual_share
    output = np.exp(np.clip(log_output, -700.0, 700.0))
    capital = alpha * output / capital_user_cost[None, :]
    land = beta * output / land_rents[None, :]
    net_profit = residual_share * output - moving_cost[None, :]

    best_location = np.argmax(net_profit, axis=1)
    row_index = np.arange(economy.n_firms)
    best_profit = net_profit[row_index, best_location]
    active = best_profit > 0.0

    # Choice -1 denotes the nonproductive outside option O.
    choices = np.where(active, best_location, -1)
    chosen_output = np.where(active, output[row_index, best_location], 0.0)
    chosen_capital = np.where(active, capital[row_index, best_location], 0.0)
    chosen_land = np.where(active, land[row_index, best_location], 0.0)
    chosen_profit = np.where(active, best_profit, 0.0)
    chosen_match = np.where(active, match[row_index, best_location], np.nan)

    firm_weight = calibration.firm_mass / economy.n_firms
    active_locations = choices[active]

    def aggregate(values: np.ndarray) -> np.ndarray:
        return np.bincount(
            active_locations,
            weights=values[active] * firm_weight,
            minlength=n_locations,
        )

    firm_mass_by_location = np.bincount(
        active_locations,
        weights=np.full(active.sum(), firm_weight),
        minlength=n_locations,
    )
    location_output = aggregate(chosen_output)
    location_capital = aggregate(chosen_capital)
    location_land = aggregate(chosen_land)
    location_profit = aggregate(chosen_profit)

    result: dict[str, Any] = {
        "interest_rate": interest_rate,
        "land_rents": land_rents,
        "capital_user_cost": capital_user_cost,
        "choices": choices,
        "active": active,
        "chosen_output": chosen_output,
        "chosen_capital": chosen_capital,
        "chosen_land": chosen_land,
        "chosen_profit": chosen_profit,
        "chosen_match": chosen_match,
        "firm_mass_by_location": firm_mass_by_location,
        "location_output": location_output,
        "location_capital": location_capital,
        "location_land": location_land,
        "location_profit": location_profit,
        "total_capital_demand": float(location_capital.sum()),
        "total_capital_supply": capital_supply(interest_rate, calibration),
        "firm_weight": firm_weight,
    }
    if return_full_matrix:
        result.update(
            {
                "output_matrix": output,
                "capital_matrix": capital,
                "land_matrix": land,
                "net_profit_matrix": net_profit,
            }
        )
    return result


def _market_clearing_residuals(
    log_prices: np.ndarray,
    economy: EconomyData,
    calibration: Calibration,
    policy_active: bool,
) -> np.ndarray:
    allocation = _allocation_at_prices(
        log_prices,
        economy,
        calibration,
        policy_active,
    )
    land_supply = economy.locations["land_supply"].to_numpy(dtype=float)
    tiny = 1.0e-300

    capital_residual = np.log(
        max(allocation["total_capital_demand"], tiny)
        / max(allocation["total_capital_supply"], tiny)
    )
    land_residuals = np.log(
        np.maximum(allocation["location_land"], tiny) / land_supply
    )
    return np.concatenate(([capital_residual], land_residuals))


def _default_initial_log_prices(
    economy: EconomyData,
    calibration: Calibration,
) -> np.ndarray:
    interest_rate = calibration.reference_interest_rate
    # A common rent near beta times average revenue is a neutral scale guess.
    # The least-squares solver works in log prices and quickly adjusts this.
    initial_rent = max(calibration.beta, 0.05)
    return np.log(
        np.concatenate(
            ([interest_rate], np.full(economy.n_locations, initial_rent))
        )
    )


def solve_equilibrium(
    economy: EconomyData,
    calibration: Calibration,
    *,
    policy_active: bool,
    initial_log_prices: np.ndarray | None = None,
) -> EquilibriumResult:
    """Solve jointly for the national interest rate and regional land rents.

    Prices are represented in logs, guaranteeing positivity. The objective is
    the vector of log demand-supply ratios in the national capital market and
    each local land market. Firm choices are deterministic. With a finite Monte
    Carlo sample the location-choice margins are very slightly discontinuous;
    ``n_firms`` should therefore be reasonably large for precise clearing.
    """

    if economy.n_firms != calibration.n_firms:
        raise ValueError(
            "EconomyData and Calibration disagree about n_firms: "
            f"{economy.n_firms} versus {calibration.n_firms}."
        )

    x0 = (
        _default_initial_log_prices(economy, calibration)
        if initial_log_prices is None
        else np.asarray(initial_log_prices, dtype=float)
    )
    if x0.shape != (economy.n_locations + 1,):
        raise ValueError("initial_log_prices has the wrong length.")

    lower = np.full_like(x0, calibration.log_price_lower_bound)
    upper = np.full_like(x0, calibration.log_price_upper_bound)

    optimizer = least_squares(
        _market_clearing_residuals,
        x0=x0,
        bounds=(lower, upper),
        args=(economy, calibration, policy_active),
        xtol=calibration.solver_tolerance,
        ftol=calibration.solver_tolerance,
        gtol=calibration.solver_tolerance,
        max_nfev=calibration.solver_max_evaluations,
        diff_step=2.0e-4,
        x_scale="jac",
        verbose=0,
    )

    residuals = _market_clearing_residuals(
        optimizer.x,
        economy,
        calibration,
        policy_active,
    )
    allocation = _allocation_at_prices(
        optimizer.x,
        economy,
        calibration,
        policy_active,
    )

    max_residual = float(np.max(np.abs(residuals)))
    if max_residual > calibration.equilibrium_residual_warning:
        warnings.warn(
            "Equilibrium markets did not clear to the requested reporting "
            f"threshold. max |log(D/S)| = {max_residual:.4g}. Increase "
            "n_firms, change the initial prices, or inspect solver diagnostics.",
            RuntimeWarning,
            stacklevel=2,
        )

    locations = economy.locations.copy().reset_index(drop=True)
    active_mass = float(allocation["firm_mass_by_location"].sum())
    total_firm_mass = calibration.firm_mass
    land_supply = locations["land_supply"].to_numpy(dtype=float)

    locations["capital_user_cost"] = allocation["capital_user_cost"]
    locations["land_rent"] = allocation["land_rents"]
    locations["firm_mass"] = allocation["firm_mass_by_location"]
    locations["firm_share_total"] = (
        allocation["firm_mass_by_location"] / total_firm_mass
    )
    locations["firm_share_active"] = np.divide(
        allocation["firm_mass_by_location"],
        active_mass,
        out=np.zeros_like(allocation["firm_mass_by_location"]),
        where=active_mass > 0.0,
    )
    locations["production"] = allocation["location_output"]
    locations["capital"] = allocation["location_capital"]
    locations["land_demand"] = allocation["location_land"]
    locations["operating_profit_net_moving_cost"] = allocation["location_profit"]
    locations["land_market_log_residual"] = np.log(
        np.maximum(allocation["location_land"], 1.0e-300) / land_supply
    )
    locations["production_share"] = locations["production"] / max(
        float(locations["production"].sum()),
        1.0e-300,
    )

    choices = allocation["choices"]
    choice_labels = np.full(economy.n_firms, "outside", dtype=object)
    active = allocation["active"]
    location_labels = locations["location_id"].astype(str).to_numpy()
    choice_labels[active] = location_labels[choices[active]]

    firm_results = pd.DataFrame(
        {
            "firm_id": np.arange(economy.n_firms, dtype=int),
            "choice_index": choices,
            "choice_location_id": choice_labels,
            "active": active,
            "output": allocation["chosen_output"],
            "capital": allocation["chosen_capital"],
            "land": allocation["chosen_land"],
            "profit_net_moving_cost": allocation["chosen_profit"],
            "chosen_match_productivity": allocation["chosen_match"],
        }
    )

    treated_mask = locations["treated"].to_numpy(dtype=bool)
    aggregate_results = pd.Series(
        {
            "national_interest_rate": allocation["interest_rate"],
            "total_production": float(locations["production"].sum()),
            "treated_production": float(
                locations.loc[treated_mask, "production"].sum()
            ),
            "untreated_production": float(
                locations.loc[~treated_mask, "production"].sum()
            ),
            "total_capital_demand": allocation["total_capital_demand"],
            "total_capital_supply": allocation["total_capital_supply"],
            "total_land_demand": float(locations["land_demand"].sum()),
            "total_land_supply": float(locations["land_supply"].sum()),
            "active_firm_mass": active_mass,
            "inactive_firm_mass": total_firm_mass - active_mass,
            "active_firm_share": active_mass / total_firm_mass,
            "treated_firm_mass": float(locations.loc[treated_mask, "firm_mass"].sum()),
            "untreated_firm_mass": float(
                locations.loc[~treated_mask, "firm_mass"].sum()
            ),
            "subsidy_bill": float(
                (
                    calibration.program_interest_rate_discount
                    * allocation["interest_rate"]
                    * locations.loc[treated_mask, "capital"]
                ).sum()
                if policy_active
                else 0.0
            ),
            "max_abs_market_residual": max_residual,
        },
        name="policy" if policy_active else "no_policy",
    )

    return EquilibriumResult(
        policy_active=policy_active,
        national_interest_rate=allocation["interest_rate"],
        location_results=locations,
        aggregate_results=aggregate_results,
        firm_results=firm_results,
        firm_choices=choices.copy(),
        log_prices=optimizer.x.copy(),
        residuals=residuals,
        solver_success=bool(optimizer.success),
        solver_message=str(optimizer.message),
        solver_evaluations=int(optimizer.nfev),
    )


def solve_counterfactuals(
    economy: EconomyData,
    calibration: Calibration,
) -> tuple[EquilibriumResult, EquilibriumResult]:
    """Solve no-policy and policy equilibria using identical fundamentals."""

    no_policy = solve_equilibrium(
        economy,
        calibration,
        policy_active=False,
    )
    policy = solve_equilibrium(
        economy,
        calibration,
        policy_active=True,
        initial_log_prices=no_policy.log_prices,
    )
    return no_policy, policy


def _firm_group(choices: np.ndarray, treated: np.ndarray) -> np.ndarray:
    group = np.full(len(choices), "outside", dtype=object)
    active = choices >= 0
    group[active] = np.where(treated[choices[active]], "treated", "untreated")
    return group


def compare_equilibria(
    no_policy: EquilibriumResult,
    policy: EquilibriumResult,
    economy: EconomyData,
    calibration: Calibration,
) -> dict[str, pd.DataFrame]:
    """Construct national, location, decomposition, and transition tables."""

    if no_policy.policy_active or not policy.policy_active:
        raise ValueError("Pass the no-policy result first and policy result second.")

    location_keys = ["location_id"]
    outcomes = [
        "capital_user_cost",
        "land_rent",
        "firm_mass",
        "firm_share_total",
        "firm_share_active",
        "production",
        "production_share",
        "capital",
        "land_demand",
        "operating_profit_net_moving_cost",
    ]
    baseline = no_policy.location_results[location_keys + outcomes].copy()
    treated_columns = [
        "location_id",
        "distance",
        "moving_cost",
        "productivity",
        "land_supply",
        "treated",
    ]
    for optional_column in (
        "treatment_probability",
        "treatment_random_effect",
        "treatment_latent_score",
    ):
        if optional_column in policy.location_results.columns:
            treated_columns.append(optional_column)
    policy_locations = policy.location_results[treated_columns + outcomes].copy()

    comparison = policy_locations.merge(
        baseline,
        on="location_id",
        suffixes=("_policy", "_no_policy"),
        validate="one_to_one",
    )
    for outcome in outcomes:
        comparison[f"delta_{outcome}"] = (
            comparison[f"{outcome}_policy"]
            - comparison[f"{outcome}_no_policy"]
        )
        denominator = comparison[f"{outcome}_no_policy"].abs()
        comparison[f"pct_delta_{outcome}"] = np.where(
            denominator > 1.0e-12,
            100.0 * comparison[f"delta_{outcome}"] / denominator,
            np.nan,
        )

    national = pd.concat(
        [no_policy.aggregate_results, policy.aggregate_results],
        axis=1,
    )
    national["change"] = national["policy"] - national["no_policy"]
    national["percent_change"] = np.where(
        national["no_policy"].abs() > 1.0e-12,
        100.0 * national["change"] / national["no_policy"].abs(),
        np.nan,
    )
    national = national.reset_index(names="outcome")

    treated_gain = float(
        policy.aggregate_results["treated_production"]
        - no_policy.aggregate_results["treated_production"]
    )
    untreated_loss = float(
        no_policy.aggregate_results["untreated_production"]
        - policy.aggregate_results["untreated_production"]
    )
    national_creation = float(
        policy.aggregate_results["total_production"]
        - no_policy.aggregate_results["total_production"]
    )
    decomposition = pd.DataFrame(
        {
            "component": [
                "Gross treated-region production gain",
                "National production creation",
                "Production displaced from untreated regions",
                "Identity check: creation + displacement - treated gain",
            ],
            "value": [
                treated_gain,
                national_creation,
                untreated_loss,
                national_creation + untreated_loss - treated_gain,
            ],
        }
    )
    if abs(treated_gain) > 1.0e-12:
        decomposition["share_of_treated_gain"] = (
            decomposition["value"] / treated_gain
        )
    else:
        decomposition["share_of_treated_gain"] = np.nan

    treated = economy.locations["treated"].to_numpy(dtype=bool)
    origin_group = _firm_group(no_policy.firm_choices, treated)
    destination_group = _firm_group(policy.firm_choices, treated)
    firm_weight = calibration.firm_mass / economy.n_firms
    transitions = (
        pd.DataFrame(
            {
                "no_policy_group": origin_group,
                "policy_group": destination_group,
                "firm_mass": firm_weight,
            }
        )
        .groupby(["no_policy_group", "policy_group"], as_index=False)["firm_mass"]
        .sum()
        .sort_values(["no_policy_group", "policy_group"])
        .reset_index(drop=True)
    )
    transitions["share_of_all_firms"] = transitions["firm_mass"] / calibration.firm_mass

    return {
        "national": national,
        "locations": comparison,
        "decomposition": decomposition,
        "transitions": transitions,
    }


def capital_supply_elasticity_sweep(
    economy: EconomyData,
    calibration: Calibration,
    elasticities: list[float] | tuple[float, ...] | np.ndarray,
) -> pd.DataFrame:
    """Solve the counterfactual at several capital-supply elasticities.

    The same location fundamentals and firm-match draws are used at every value,
    so differences across rows come only from the capital-market closure. This
    function intentionally returns a compact national table; location-level
    results remain available by calling ``solve_counterfactuals`` directly.
    """

    records: list[dict[str, float]] = []
    for elasticity in elasticities:
        elasticity = float(elasticity)
        if elasticity < 0.0 or not np.isfinite(elasticity):
            raise ValueError("Sweep elasticities must be finite and nonnegative.")

        sweep_calibration = replace(
            calibration,
            capital_supply_elasticity=elasticity,
        )
        no_policy, policy = solve_counterfactuals(economy, sweep_calibration)

        treated_gain = float(
            policy.aggregate_results["treated_production"]
            - no_policy.aggregate_results["treated_production"]
        )
        national_creation = float(
            policy.aggregate_results["total_production"]
            - no_policy.aggregate_results["total_production"]
        )
        untreated_displacement = float(
            no_policy.aggregate_results["untreated_production"]
            - policy.aggregate_results["untreated_production"]
        )
        records.append(
            {
                "capital_supply_elasticity": elasticity,
                "interest_rate_no_policy": no_policy.national_interest_rate,
                "interest_rate_policy": policy.national_interest_rate,
                "treated_production_gain": treated_gain,
                "national_production_creation": national_creation,
                "untreated_production_displacement": untreated_displacement,
                "creation_share_of_treated_gain": (
                    national_creation / treated_gain
                    if abs(treated_gain) > 1.0e-12
                    else np.nan
                ),
                "displacement_share_of_treated_gain": (
                    untreated_displacement / treated_gain
                    if abs(treated_gain) > 1.0e-12
                    else np.nan
                ),
                "capital_change": float(
                    policy.aggregate_results["total_capital_demand"]
                    - no_policy.aggregate_results["total_capital_demand"]
                ),
                "subsidy_bill": float(policy.aggregate_results["subsidy_bill"]),
                "max_market_residual": max(
                    no_policy.max_abs_residual,
                    policy.max_abs_residual,
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def save_analysis(
    tables: dict[str, pd.DataFrame],
    no_policy: EquilibriumResult,
    policy: EquilibriumResult,
    output_directory: str | Path,
) -> list[Path]:
    """Save comparison tables and a compact set of policy-impact plots.

    Matplotlib is imported only here, so solving the model does not require a
    plotting backend.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("save_analysis requires matplotlib.") from exc

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for name, table in tables.items():
        csv_path = output_path / f"{name}.csv"
        table.to_csv(csv_path, index=False)
        created.append(csv_path)

    locations = tables["locations"].sort_values("distance").reset_index(drop=True)
    colors = np.where(locations["treated"], "#b2182b", "#2166ac")

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    plot_specs = [
        ("production", "Production"),
        ("land_rent", "Land rent"),
        ("firm_share_total", "Share of all potential firms"),
    ]
    for ax, (variable, label) in zip(axes, plot_specs, strict=True):
        ax.plot(
            locations["distance"],
            locations[f"{variable}_no_policy"],
            color="#555555",
            marker="o",
            label="No policy",
        )
        ax.plot(
            locations["distance"],
            locations[f"{variable}_policy"],
            color="#b2182b",
            marker="s",
            label="Policy",
        )
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Distance from origin O")
    fig.suptitle("Equilibrium outcomes with and without Polocentro")
    fig.tight_layout()
    path = output_path / "equilibrium_levels.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    impact_specs = [
        ("delta_production", "Change in production"),
        ("delta_land_rent", "Change in land rent"),
        ("delta_firm_share_total", "Change in firm share"),
    ]
    for ax, (variable, label) in zip(axes, impact_specs, strict=True):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.scatter(
            locations["distance"],
            locations[variable],
            c=colors,
            s=55,
            edgecolor="white",
            linewidth=0.7,
        )
        ax.set_xlabel("Distance from origin O")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    fig.suptitle("Policy impacts by location (red = treated, blue = untreated)")
    fig.tight_layout()
    path = output_path / "policy_impacts_by_distance.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    decomposition = tables["decomposition"].iloc[:3].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bar_colors = ["#7f0000", "#238b45", "#6a51a3"]
    ax.barh(decomposition["component"], decomposition["value"], color=bar_colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in production")
    ax.set_title("Treated gain = national creation + untreated displacement")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = output_path / "production_decomposition.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    return created
