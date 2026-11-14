"""Parity and SOD-expansion tests for python/pricing + projects/pricing."""

import datetime as dt
import os
import time
import unittest
from pathlib import Path

from pricing.rack import (
    FraSpec,
    SwapSpec,
    build_curve_scenarios,
    compute_bump_dv01_python,
    expand_fra_to_cashflows,
    expand_swap_to_cashflows,
    fit_zero_curve,
    price_cashflows_python,
    run_pure_pricing_duckdb,
)


def _find_pricing_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "projects" / "pricing"
        if (cand / "model.pure").is_file():
            return cand
    # Bazel runfiles fallback
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        cand = Path(srcdir) / "_main" / "projects" / "pricing"
        if (cand / "model.pure").is_file():
            return cand
    raise FileNotFoundError("Cannot locate projects/pricing")


class PricingRackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.val_date = dt.date(2026, 1, 2)
        self.base_holidays = {dt.date(2026, 1, 1), dt.date(2026, 12, 25)}
        self.benchmarks = [
            SwapSpec("BM_6M", "OIS", "USD", "USD_OIS", "USD_OIS", self.val_date, 6, 6, 1_000_000.0, 0.0410),
            SwapSpec("BM_1Y", "OIS", "USD", "USD_OIS", "USD_OIS", self.val_date, 12, 6, 1_000_000.0, 0.0430),
            SwapSpec("BM_2Y", "OIS", "USD", "USD_OIS", "USD_OIS", self.val_date, 24, 6, 1_000_000.0, 0.0460),
            SwapSpec("BM_5Y", "OIS", "USD", "USD_OIS", "USD_OIS", self.val_date, 60, 12, 1_000_000.0, 0.0490),
        ]

    def test_sod_holiday_change_updates_cashflow_rack_and_pv(self) -> None:
        knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")
        swap = SwapSpec("SW_TEST", "IRS", "USD", "USD_OIS", "USD_OIS", self.val_date, 12, 6, 10_000_000.0, 0.0430)

        cfs_no_extra = expand_swap_to_cashflows(swap, self.val_date, self.base_holidays)
        # Adding 2026-07-02 (6M coupon date) as a dynamic holiday shifts pay_t and dcf
        extra_holidays = self.base_holidays | {dt.date(2026, 7, 2)}
        cfs_with_holiday = expand_swap_to_cashflows(swap, self.val_date, extra_holidays)

        self.assertNotEqual(cfs_no_extra[0].pay_t, cfs_with_holiday[0].pay_t)
        pv_a = price_cashflows_python(cfs_no_extra, knots)[("BASE", "SW_TEST")]
        pv_b = price_cashflows_python(cfs_with_holiday, knots)[("BASE", "SW_TEST")]
        self.assertNotAlmostEqual(pv_a, pv_b, places=2)

    def test_external_curve_fit_prices_benchmarks_to_par(self) -> None:
        for method in ("LOG_LINEAR_DF", "LINEAR_ZERO"):
            knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", method)
            bm_cfs = []
            for bm in self.benchmarks:
                bm_cfs.extend(expand_swap_to_cashflows(bm, self.val_date, self.base_holidays))
            npvs = price_cashflows_python(bm_cfs, knots)
            for bm in self.benchmarks:
                self.assertAlmostEqual(npvs[("BASE", bm.trade_id)], 0.0, delta=1e-6, msg=f"{method} {bm.trade_id}")

    def test_pure_pricer_duckdb_parity_and_bump_dv01_across_interpolations(self) -> None:
        pricing_dir = _find_pricing_dir()

        # Build a 100-trade portfolio: 50 IRS + 30 OIS + 20 FRAs
        cashflows = []
        for i in range(50):
            sw = SwapSpec(
                f"IRS_{i:03d}", "IRS", "USD", "USD_OIS", "USD_OIS",
                self.val_date, 12 * ((i % 5) + 1), 6,
                1_000_000.0 * ((i % 4) + 1),
                0.040 + 0.0002 * (i % 15),
                pay_fixed=(i % 2 == 0),
            )
            cashflows.extend(expand_swap_to_cashflows(sw, self.val_date, self.base_holidays))
        for i in range(30):
            sw = SwapSpec(
                f"OIS_{i:03d}", "OIS", "USD", "USD_OIS", "USD_OIS",
                self.val_date, 12 * ((i % 4) + 1), 12,
                2_000_000.0,
                0.042 + 0.0003 * (i % 10),
                pay_fixed=(i % 2 == 1),
            )
            cashflows.extend(expand_swap_to_cashflows(sw, self.val_date, self.base_holidays))
        for i in range(20):
            s_d = dt.date(2026, 4, 2) + dt.timedelta(days=30 * (i % 6))
            e_d = s_d + dt.timedelta(days=180)
            fra = FraSpec(
                f"FRA_{i:03d}", "USD", "USD_OIS", "USD_OIS",
                s_d, e_d, 5_000_000.0, 0.043 + 0.0002 * i,
                pay_fixed=(i % 2 == 0),
            )
            cashflows.extend(expand_fra_to_cashflows(fra, self.val_date, self.base_holidays))

        for method in ("LOG_LINEAR_DF", "LINEAR_ZERO"):
            base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", method)
            scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)

            py_npvs = price_cashflows_python(cashflows, scen_knots)
            t0 = time.perf_counter()
            sql_npvs = run_pure_pricing_duckdb(scen_knots, cashflows, projects_dir=pricing_dir)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            self.assertEqual(set(py_npvs.keys()), set(sql_npvs.keys()))
            self.assertEqual(len(sql_npvs), 900)  # 100 trades x (1 BASE + 8 bumps)

            max_npv_diff = max(abs(py_npvs[k] - sql_npvs[k]) for k in py_npvs)
            self.assertLess(max_npv_diff, 1e-6, f"{method} max NPV diff={max_npv_diff}")

            py_dv01 = compute_bump_dv01_python(py_npvs, base_knots)
            sql_dv01 = compute_bump_dv01_python(sql_npvs, base_knots)
            max_dv01_diff = max(abs(py_dv01[k] - sql_dv01[k]) for k in py_dv01)
            self.assertLess(max_dv01_diff, 1e-7, f"{method} max DV01 diff={max_dv01_diff}")
            print(
                f"[pricing_test] method={method} trades=100 scenarios=9 "
                f"rows={len(sql_npvs)} duckdb_ms={elapsed_ms:.1f} "
                f"max_npv_diff={max_npv_diff:.2e} max_dv01_diff={max_dv01_diff:.2e}"
            )


if __name__ == "__main__":
    unittest.main()
