r"""Run the simulated Polocentro economy and save counterfactual analytics.

Usage from the repository root:

    .venv\Scripts\python.exe code\polocentro_model\run_simulation.py

Edit ``CALIBRATION`` below to sweep parameters. Later, replace
``prepare_simulated_economy`` with an ``EconomyData`` object constructed from
the roads/crop-suitability treatment data while retaining the same solver.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY.parent) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY.parent))

from polocentro_model import (  # noqa: E402
    Calibration,
    compare_equilibria,
    prepare_simulated_economy,
    save_analysis,
    solve_counterfactuals,
)


# ---------------------------------------------------------------------------
# Main editing surface
# ---------------------------------------------------------------------------
CALIBRATION = Calibration(
    alpha=0.25,
    beta=0.20,
    frechet_shape=5.0,
    frechet_scale=1.0,
    capital_supply_elasticity=1.0,
    capital_supply_at_reference=10.0,
    reference_interest_rate=1.0,
    program_interest_rate_discount=0.30,
    n_firms=30_000,
    n_locations=15,
    treatment_share=0.35,
)


def main() -> None:
    repository_root = MODULE_DIRECTORY.parents[1]
    output_directory = repository_root / "outputs" / "polocentro_model_simulation"

    economy = prepare_simulated_economy(CALIBRATION)
    no_policy, policy = solve_counterfactuals(economy, CALIBRATION)
    tables = compare_equilibria(no_policy, policy, economy, CALIBRATION)
    created = save_analysis(tables, no_policy, policy, output_directory)

    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda value: f"{value:,.5f}")

    print("\nNATIONAL COUNTERFACTUAL SUMMARY")
    print(tables["national"].to_string(index=False))

    print("\nPRODUCTION CREATION / DISPLACEMENT")
    print(tables["decomposition"].to_string(index=False))

    print("\nFIRM TRANSITIONS")
    print(tables["transitions"].to_string(index=False))

    print("\nSOLVER DIAGNOSTICS")
    print(
        f"No policy: success={no_policy.solver_success}, "
        f"max residual={no_policy.max_abs_residual:.3e}, "
        f"evaluations={no_policy.solver_evaluations}"
    )
    print(
        f"Policy:    success={policy.solver_success}, "
        f"max residual={policy.max_abs_residual:.3e}, "
        f"evaluations={policy.solver_evaluations}"
    )

    print(f"\nSaved {len(created)} files to {output_directory}")

    # Optional capital-supply sweep. It is left off because it solves two new
    # equilibria per grid value. Uncomment when running sensitivity analysis:
    #
    # from polocentro_model import capital_supply_elasticity_sweep
    # sweep = capital_supply_elasticity_sweep(
    #     economy,
    #     CALIBRATION,
    #     elasticities=[0.0, 0.5, 1.0, 2.0, 5.0],
    # )
    # sweep.to_csv(output_directory / "capital_supply_sweep.csv", index=False)
    # print("\nCAPITAL-SUPPLY ELASTICITY SWEEP")
    # print(sweep.to_string(index=False))


if __name__ == "__main__":
    main()
