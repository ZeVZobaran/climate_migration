"""Lightweight regression tests for the fixed-land Polocentro model."""

from __future__ import annotations

import unittest

import numpy as np

from polocentro_model import (
    Calibration,
    compare_equilibria,
    prepare_simulated_economy,
    solve_counterfactuals,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.calibration = Calibration(
            n_firms=8_000,
            n_locations=8,
            capital_supply_at_reference=6.0,
            treatment_share=0.375,
            solver_max_evaluations=500,
            equilibrium_residual_warning=2.0e-2,
        )
        cls.economy = prepare_simulated_economy(cls.calibration)
        cls.no_policy, cls.policy = solve_counterfactuals(
            cls.economy,
            cls.calibration,
        )

    def test_reproducible_simulation(self) -> None:
        second = prepare_simulated_economy(self.calibration)
        np.testing.assert_allclose(
            self.economy.match_productivity,
            second.match_productivity,
        )
        self.assertTrue(self.economy.locations.equals(second.locations))

    def test_both_treatment_states_exist(self) -> None:
        treated = self.economy.locations["treated"]
        self.assertTrue(treated.any())
        self.assertTrue((~treated).any())

    def test_market_clearing(self) -> None:
        self.assertLess(self.no_policy.max_abs_residual, 2.0e-2)
        self.assertLess(self.policy.max_abs_residual, 2.0e-2)

    def test_firm_shares_add_up(self) -> None:
        for result in (self.no_policy, self.policy):
            active_share = result.aggregate_results["active_firm_share"]
            location_share = result.location_results["firm_share_total"].sum()
            self.assertAlmostEqual(active_share, location_share, places=10)

    def test_policy_reduces_treated_user_cost(self) -> None:
        treated = self.economy.locations["treated"].to_numpy(dtype=bool)
        expected = (
            self.policy.national_interest_rate
            * (1.0 - self.calibration.program_interest_rate_discount)
        )
        np.testing.assert_allclose(
            self.policy.location_results.loc[treated, "capital_user_cost"],
            expected,
        )

    def test_decomposition_identity(self) -> None:
        tables = compare_equilibria(
            self.no_policy,
            self.policy,
            self.economy,
            self.calibration,
        )
        identity_error = tables["decomposition"].loc[
            tables["decomposition"]["component"].str.startswith("Identity"),
            "value",
        ].iloc[0]
        self.assertAlmostEqual(identity_error, 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
