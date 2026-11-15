"""Parity and SOD-expansion tests for python/pricing + projects/pricing."""

import dataclasses
import datetime as dt
import os
import time
import unittest
from pathlib import Path

import duckdb
import legend_lite as ll

from pricing.rack import (
    FraSpec,
    SwapSpec,
    SwaptionSpec,
    VolPoint,
    XccyBasisSwapSpec,
    build_curve_scenarios,
    compute_bump_dv01_python,
    expand_fra_to_cashflows,
    expand_swap_to_cashflows,
    expand_swaption_to_rows,
    expand_xccy_basis_swap_to_cashflows,
    fit_curve_from_cashflows,
    fit_zero_curve,
    next_imm_dates,
    price_cashflows_python,
    price_swaptions_python,
    run_pure_pricing_duckdb,
    run_pure_swaption_pricing_duckdb,
    run_pure_dv01_duckdb,
    run_pure_vega_duckdb,
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

    def test_pure_swaption_pricer_duckdb_parity_and_scaling(self) -> None:
        pricing_dir = _find_pricing_dir()
        base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")
        scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)  # 9 scenarios

        # 2x3 vol surface grid in (expiryT, swapTenorT)
        vol_points = [
            VolPoint("BASE", "USD_SWP_VOL", 0, 0.5, 1.0, 0.0096, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 1, 0.5, 2.0, 0.0088, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 2, 0.5, 5.0, 0.0080, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 3, 2.0, 1.0, 0.0084, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 4, 2.0, 2.0, 0.0076, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 5, 2.0, 5.0, 0.0072, beta=0.0),
        ]

        # Build 500 swaptions (covering in-between expiries/tenors so bilinear vol interp is exercised)
        swaptions = []
        cashflows = []
        for i in range(500):
            exp_m = [6, 9, 12, 18, 24][i % 5]
            ten_m = [12, 24, 36, 48, 60][i % 5]
            swp_spec = SwapSpec(f"DUMMY_{i}", "IRS", "USD", "USD_OIS", "USD_OIS", self.val_date, 12, 6, 1.0, 0.04)
            _ = swp_spec
            spec = SwaptionSpec(
                trade_id=f"SWP_{i:03d}",
                ccy="USD",
                discount_curve_id="USD_OIS",
                proj_curve_id="USD_OIS",
                vol_surface_id="USD_SWP_VOL",
                expiry_months=exp_m,
                swap_tenor_months=ten_m,
                freq_months=6,
                notional=1_000_000.0 * ((i % 3) + 1),
                strike=0.038 + 0.001 * (i % 15),
                is_payer=(i % 2 == 0),
                pay_receive=1.0 if (i % 3 != 0) else -1.0,
                settlement_type="CASH" if (i % 2 == 1) else "PHYSICAL",
            )
            sw_row, cfs = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(sw_row)
            cashflows.extend(cfs)

        t_py = time.perf_counter()
        py_pvs = price_swaptions_python(swaptions, cashflows, scen_knots, vol_points)
        py_ms = (time.perf_counter() - t_py) * 1000.0

        t_fn = time.perf_counter()
        fn_pvs = run_pure_swaption_pricing_duckdb(
            scen_knots, cashflows, swaptions, vol_points, projects_dir=pricing_dir, mode="function"
        )
        fn_ms = (time.perf_counter() - t_fn) * 1000.0

        t_rs = time.perf_counter()
        rs_pvs = run_pure_swaption_pricing_duckdb(
            scen_knots, cashflows, swaptions, vol_points, projects_dir=pricing_dir, mode="row_sum"
        )
        rs_ms = (time.perf_counter() - t_rs) * 1000.0

        self.assertEqual(set(py_pvs.keys()), set(fn_pvs.keys()))
        self.assertEqual(set(py_pvs.keys()), set(rs_pvs.keys()))
        self.assertEqual(len(fn_pvs), 4500)  # 500 swaptions x 9 scenarios

        max_fn_diff = max(abs(py_pvs[k] - fn_pvs[k]) for k in py_pvs)
        max_rs_diff = max(abs(py_pvs[k] - rs_pvs[k]) for k in py_pvs)
        max_dual_diff = max(abs(fn_pvs[k] - rs_pvs[k]) for k in fn_pvs)
        self.assertLess(max_fn_diff, 1e-6, f"Closed-form max PV diff={max_fn_diff}")
        self.assertLess(max_rs_diff, 1e-6, f"Row-sum max PV diff={max_rs_diff}")
        self.assertLess(max_dual_diff, 1e-8, f"Function vs row-sum diff={max_dual_diff}")
        print(
            f"[swaption_test] swaptions=500 (250 PHYS + 250 CASH) scenarios=9 rows={len(fn_pvs)} "
            f"py_ms={py_ms:.1f} fn_total_ms={fn_ms:.1f} row_sum_total_ms={rs_ms:.1f} "
            f"max_fn_diff={max_fn_diff:.2e} max_rs_diff={max_rs_diff:.2e} dual_diff={max_dual_diff:.2e}"
        )

    def test_pure_ad_dv01_parity_and_speed(self) -> None:
        pricing_dir = _find_pricing_dir()

        # 1. 100 linear swaps/FRAs across both interpolation methods
        cashflows = []
        for i in range(80):
            swap = SwapSpec(
                f"IRS_{i:03d}", "IRS", "USD", "USD_OIS", "USD_OIS",
                self.val_date, 12 * (1 + (i % 10)), 6, 1_000_000.0 * (1 + (i % 5)),
                0.04 + 0.0005 * (i % 10), pay_fixed=(i % 2 == 0),
            )
            cashflows.extend(expand_swap_to_cashflows(swap, self.val_date, self.base_holidays))
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
            py_bump_dv01 = compute_bump_dv01_python(py_npvs, base_knots)

            t0 = time.perf_counter()
            ad_dv01 = run_pure_dv01_duckdb(base_knots, cashflows, projects_dir=pricing_dir, kind="cashflow")
            ad_ms = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            bump_npvs = run_pure_pricing_duckdb(scen_knots, cashflows, projects_dir=pricing_dir)
            bump_ms = (time.perf_counter() - t0) * 1000.0
            _ = bump_npvs

            max_diff = max(
                abs(ad_dv01.get((t_id, c_id, k_idx), 0.0) - val)
                for (t_id, c_id, k_idx), val in py_bump_dv01.items()
            )
            self.assertLess(max_diff, 5e-2, f"{method} linear AD vs bump max_diff={max_diff}")
            print(
                f"[ad_dv01_linear] method={method} trades=100 ad_ms={ad_ms:.1f} "
                f"bump_9scen_ms={bump_ms:.1f} max_diff_vs_1bp_bump={max_diff:.2e}"
            )

        # 2. 500 swaptions (250 PHYSICAL + 250 CASH)
        base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")
        scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)
        vol_points = [
            VolPoint("BASE", "USD_SWP_VOL", 0, 0.5, 1.0, 0.0096, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 1, 0.5, 2.0, 0.0088, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 2, 0.5, 5.0, 0.0080, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 3, 2.0, 1.0, 0.0084, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 4, 2.0, 2.0, 0.0076, beta=0.0),
            VolPoint("BASE", "USD_SWP_VOL", 5, 2.0, 5.0, 0.0072, beta=0.0),
        ]
        swaptions = []
        swp_cfs = []
        for i in range(500):
            exp_m = [6, 9, 12, 18, 24][i % 5]
            ten_m = [12, 24, 36, 48, 60][i % 5]
            spec = SwaptionSpec(
                trade_id=f"SWP_{i:03d}",
                ccy="USD",
                discount_curve_id="USD_OIS",
                proj_curve_id="USD_OIS",
                vol_surface_id="USD_SWP_VOL",
                expiry_months=exp_m,
                swap_tenor_months=ten_m,
                freq_months=6,
                notional=1_000_000.0 * ((i % 3) + 1),
                strike=0.038 + 0.001 * (i % 15),
                is_payer=(i % 2 == 0),
                pay_receive=1.0 if (i % 3 != 0) else -1.0,
                settlement_type="PHYSICAL" if (i % 2 == 0) else "CASH",
            )
            swp_row, cfs = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(swp_row)
            swp_cfs.extend(cfs)

        py_swp_pvs = price_swaptions_python(swaptions, swp_cfs, scen_knots, vol_points)
        py_swp_bump_dv01 = compute_bump_dv01_python(py_swp_pvs, base_knots)

        t0 = time.perf_counter()
        ad_swp_dv01 = run_pure_dv01_duckdb(
            base_knots, swp_cfs, swaptions=swaptions, vol_points=vol_points,
            projects_dir=pricing_dir, kind="swaption",
        )
        ad_swp_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        _ = run_pure_swaption_pricing_duckdb(
            scen_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, mode="function"
        )
        bump_swp_ms = (time.perf_counter() - t0) * 1000.0

        max_swp_diff = max(
            abs(ad_swp_dv01.get((t_id, c_id, k_idx), 0.0) - val)
            for (t_id, c_id, k_idx), val in py_swp_bump_dv01.items()
        )
        self.assertLess(max_swp_diff, 5e-2, f"swaption AD vs bump max_diff={max_swp_diff}")
        print(
            f"[ad_dv01_swaption] swaptions=500 ad_ms={ad_swp_ms:.1f} "
            f"bump_9scen_ms={bump_swp_ms:.1f} max_diff_vs_1bp_bump={max_swp_diff:.2e}"
        )

    def test_pure_ad_vega_10x10_grid_parity_and_sparsity(self) -> None:
        pricing_dir = _find_pricing_dir()
        base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")

        expiries = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
        tenors = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        vol_points = []
        idx = 0
        for ie, e in enumerate(expiries):
            for it, t in enumerate(tenors):
                vol_points.append(
                    VolPoint("BASE", "USD_SWP_VOL", idx, e, t, 0.0096 - 0.0002 * ie - 0.0001 * it, beta=0.0)
                )
                idx += 1

        swaptions = []
        swp_cfs = []
        for i in range(200):
            exp_m = [6, 9, 15, 24, 42][i % 5]
            ten_m = [12, 18, 36, 54, 72][i % 5]
            spec = SwaptionSpec(
                trade_id=f"SWPV_{i:03d}",
                ccy="USD",
                discount_curve_id="USD_OIS",
                proj_curve_id="USD_OIS",
                vol_surface_id="USD_SWP_VOL",
                expiry_months=exp_m,
                swap_tenor_months=ten_m,
                freq_months=6,
                notional=1_000_000.0 * ((i % 3) + 1),
                strike=0.038 + 0.001 * (i % 10),
                is_payer=(i % 2 == 0),
                pay_receive=1.0 if (i % 3 != 0) else -1.0,
                settlement_type="PHYSICAL" if (i % 2 == 0) else "CASH",
            )
            swp_row, cfs = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(swp_row)
            swp_cfs.extend(cfs)

        t0 = time.perf_counter()
        ad_vega = run_pure_vega_duckdb(base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir)
        ad_ms = (time.perf_counter() - t0) * 1000.0

        # Sparsity: 200 swaptions x <= 4 corner vol points (vs 200 x 100 = 20,000 dense)
        self.assertGreater(len(ad_vega), 0)
        self.assertLessEqual(len(ad_vega), 200 * 4)

        # Central 1bp vol bump across all 100 grid points in Python
        bump = 1e-4
        scen_knots = []
        scen_vols = list(vol_points)
        for p in vol_points:
            for tag, delta in ((f"VUP_{p.point_idx}", bump), (f"VDN_{p.point_idx}", -bump)):
                for k in base_knots:
                    scen_knots.append(dataclasses.replace(k, scenario_id=tag))
                for vp in vol_points:
                    v_val = vp.vol + (delta if vp.point_idx == p.point_idx else 0.0)
                    scen_vols.append(
                        VolPoint(tag, vp.vol_surface_id, vp.point_idx, vp.expiry_t, vp.swap_tenor_t, v_val, beta=vp.beta)
                    )

        py_pvs = price_swaptions_python(swaptions, swp_cfs, scen_knots, scen_vols)
        max_diff = 0.0
        for s in swaptions:
            for p in vol_points:
                bump_vega = 0.5 * (py_pvs[(f"VUP_{p.point_idx}", s.trade_id)] - py_pvs[(f"VDN_{p.point_idx}", s.trade_id)])
                ad_val = ad_vega.get((s.trade_id, "USD_SWP_VOL", p.point_idx), 0.0)
                max_diff = max(max_diff, abs(ad_val - bump_vega))

        self.assertLess(max_diff, 1e-2, f"10x10 vol AD vega vs central 1bp bump max_diff={max_diff}")
        print(
            f"[ad_vega_10x10] swaptions=200 grid=10x10 (100 pts) sparse_rows={len(ad_vega)}/20000 "
            f"ad_ms={ad_ms:.1f} max_diff_vs_1bp_bump={max_diff:.2e}"
        )

    def test_sabr_vol_smile_pv_ad_dv01_and_alpha_vega(self) -> None:
        pricing_dir = _find_pricing_dir()
        base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")
        scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)

        # 3x3 SABR surface: (alpha, beta, rho, nu) per grid knot
        expiries = [0.5, 2.0, 5.0]
        tenors = [1.0, 3.0, 5.0]
        vol_points = []
        idx = 0
        for ie, e in enumerate(expiries):
            for it, t in enumerate(tenors):
                vol_points.append(VolPoint(
                    "BASE", "USD_SWP_VOL", idx, e, t,
                    vol=0.044 - 0.002 * ie - 0.001 * it,  # alpha (CEV backbone)
                    beta=0.5 + 0.05 * it,
                    rho=-0.35 + 0.05 * ie,
                    nu=0.30 + 0.05 * ie + 0.02 * it,
                ))
                idx += 1

        swaptions = []
        swp_cfs = []
        for i in range(200):
            exp_m = [6, 9, 15, 24, 42][i % 5]
            ten_m = [12, 24, 36, 48, 60][i % 5]
            spec = SwaptionSpec(
                trade_id=f"SABR_{i:03d}",
                ccy="USD",
                discount_curve_id="USD_OIS",
                proj_curve_id="USD_OIS",
                vol_surface_id="USD_SWP_VOL",
                expiry_months=exp_m,
                swap_tenor_months=ten_m,
                freq_months=6,
                notional=1_000_000.0 * ((i % 3) + 1),
                strike=0.028 + 0.002 * (i % 15),
                is_payer=(i % 2 == 0),
                pay_receive=1.0 if (i % 3 != 0) else -1.0,
                settlement_type="PHYSICAL" if (i % 2 == 0) else "CASH",
            )
            swp_row, cfs = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(swp_row)
            swp_cfs.extend(cfs)

        # 1. PV parity (Pure function + row_sum vs Python SABR)
        py_pvs = price_swaptions_python(swaptions, swp_cfs, scen_knots, vol_points)
        sql_fn_pvs = run_pure_swaption_pricing_duckdb(
            scen_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, mode="function"
        )
        sql_rs_pvs = run_pure_swaption_pricing_duckdb(
            scen_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, mode="row_sum"
        )
        max_pv_diff = max(abs(py_pvs[k] - sql_fn_pvs[k]) for k in py_pvs)
        max_rs_diff = max(abs(py_pvs[k] - sql_rs_pvs[k]) for k in py_pvs)
        self.assertLess(max_pv_diff, 1e-8, f"SABR PV fn diff={max_pv_diff}")
        self.assertLess(max_rs_diff, 1e-8, f"SABR PV row_sum diff={max_rs_diff}")

        # 2. SABR AD DV01 (includes Vega * d(sigma_SABR)/dF smile delta via :diff!)
        py_bump_dv01 = compute_bump_dv01_python(py_pvs, base_knots)
        t0 = time.perf_counter()
        ad_dv01 = run_pure_dv01_duckdb(
            base_knots, swp_cfs, swaptions=swaptions, vol_points=vol_points,
            projects_dir=pricing_dir, kind="swaption",
        )
        dv01_ms = (time.perf_counter() - t0) * 1000.0
        max_dv01_diff = max(
            abs(ad_dv01.get((t_id, c_id, k_idx), 0.0) - val)
            for (t_id, c_id, k_idx), val in py_bump_dv01.items()
        )
        self.assertLess(max_dv01_diff, 5e-2, f"SABR AD DV01 vs 1bp bump max_diff={max_dv01_diff}")

        # 3. SABR AD alpha-vega vs central 1bp alpha bump
        t0 = time.perf_counter()
        ad_alpha_vega = run_pure_vega_duckdb(base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir)
        vega_ms = (time.perf_counter() - t0) * 1000.0

        bump = 1e-4
        v_scen_knots = []
        v_scen_vols = list(vol_points)
        for p in vol_points:
            for tag, delta in ((f"AUP_{p.point_idx}", bump), (f"ADN_{p.point_idx}", -bump)):
                for k in base_knots:
                    v_scen_knots.append(dataclasses.replace(k, scenario_id=tag))
                for vp in vol_points:
                    v_scen_vols.append(dataclasses.replace(
                        vp, scenario_id=tag, vol=vp.vol + (delta if vp.point_idx == p.point_idx else 0.0)
                    ))
        py_v_pvs = price_swaptions_python(swaptions, swp_cfs, v_scen_knots, v_scen_vols)
        max_vega_diff = 0.0
        for s in swaptions:
            for p in vol_points:
                bump_v = 0.5 * (py_v_pvs[(f"AUP_{p.point_idx}", s.trade_id)] - py_v_pvs[(f"ADN_{p.point_idx}", s.trade_id)])
                ad_v = ad_alpha_vega.get((s.trade_id, "USD_SWP_VOL", p.point_idx), 0.0)
                max_vega_diff = max(max_vega_diff, abs(ad_v - bump_v))
        self.assertLess(max_vega_diff, 1e-3, f"SABR AD alpha-vega max_diff={max_vega_diff}")
        print(
            f"[sabr_test] swaptions=200 (100 PHYS + 100 CASH) pv_diff={max_pv_diff:.2e} "
            f"dv01_diff={max_dv01_diff:.2e} ({dv01_ms:.1f}ms) "
            f"alpha_vega_diff={max_vega_diff:.2e} ({vega_ms:.1f}ms)"
        )

    def test_pluggable_vol_model_switching_flat_normal_vs_sabr(self) -> None:
        pricing_dir = _find_pricing_dir()
        base_knots = fit_zero_curve(self.benchmarks, self.val_date, self.base_holidays, "USD_OIS", "LOG_LINEAR_DF")
        scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)

        # Same PR_VOL_POINT table carries both normal vol (alpha) and SABR smile (beta, rho, nu)
        vol_points = [
            VolPoint("BASE", "USD_SWP_VOL", 0, 0.5, 1.0, vol=0.0088, beta=0.85, rho=-0.30, nu=0.35),
            VolPoint("BASE", "USD_SWP_VOL", 1, 0.5, 5.0, vol=0.0080, beta=0.85, rho=-0.30, nu=0.35),
            VolPoint("BASE", "USD_SWP_VOL", 2, 2.0, 1.0, vol=0.0084, beta=0.85, rho=-0.30, nu=0.35),
            VolPoint("BASE", "USD_SWP_VOL", 3, 2.0, 5.0, vol=0.0076, beta=0.85, rho=-0.30, nu=0.35),
        ]
        swaptions = []
        swp_cfs = []
        for i in range(40):
            spec = SwaptionSpec(
                trade_id=f"SWP_MOD_{i:02d}",
                ccy="USD",
                discount_curve_id="USD_OIS",
                proj_curve_id="USD_OIS",
                vol_surface_id="USD_SWP_VOL",
                expiry_months=[6, 12, 18, 24][i % 4],
                swap_tenor_months=[12, 24, 36, 60][i % 4],
                freq_months=6,
                notional=1_000_000.0,
                strike=0.035 + 0.002 * (i % 6),
                is_payer=(i % 2 == 0),
                pay_receive=1.0,
                settlement_type="PHYSICAL" if (i % 2 == 0) else "CASH",
            )
            r, c = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(r)
            swp_cfs.extend(c)

        # 1. Switch to flatNormalVolInterpolate / flatNormalVolKernel on the same table
        flat_pvs = run_pure_swaption_pricing_duckdb(
            base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, vol_model="flat_normal"
        )
        sabr_pvs = run_pure_swaption_pricing_duckdb(
            base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, vol_model="sabr"
        )
        # Flat-normal ignores beta/rho/nu so PVs differ materially from SABR
        max_model_gap = max(abs(flat_pvs[k] - sabr_pvs[k]) for k in flat_pvs)
        self.assertGreater(max_model_gap, 100.0)

        # Verify flat_normal AD DV01 & vega match flat-normal Python oracle (beta=0, rho=0, nu=0)
        flat_vols = [dataclasses.replace(vp, beta=0.0, rho=0.0, nu=0.0) for vp in vol_points]
        py_flat_pvs = price_swaptions_python(swaptions, swp_cfs, scen_knots, flat_vols)
        py_flat_dv01 = compute_bump_dv01_python(py_flat_pvs, base_knots)

        t0 = time.perf_counter()
        ad_flat_dv01 = run_pure_dv01_duckdb(
            base_knots, swp_cfs, swaptions=swaptions, vol_points=vol_points,
            projects_dir=pricing_dir, kind="swaption", vol_model="flat_normal",
        )
        flat_dv01_ms = (time.perf_counter() - t0) * 1000.0

        max_flat_dv01_diff = max(
            abs(ad_flat_dv01.get((t_id, c_id, k_idx), 0.0) - val)
            for (t_id, c_id, k_idx), val in py_flat_dv01.items()
        )
        self.assertLess(max_flat_dv01_diff, 5e-2, f"flat_normal AD DV01 diff={max_flat_dv01_diff}")

        ad_flat_vega = run_pure_vega_duckdb(
            base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, vol_model="flat_normal"
        )
        self.assertGreater(len(ad_flat_vega), 0)
        print(
            f"[pluggable_vol_model] swaptions=40 flat_vs_sabr_pv_gap={max_model_gap:.1f} "
            f"flat_dv01_diff={max_flat_dv01_diff:.2e} ({flat_dv01_ms:.1f}ms)"
        )

    def test_negative_rate_swaption_bachelier_and_normal_sabr(self) -> None:
        pricing_dir = _find_pricing_dir()
        neg_benchmarks = [
            SwapSpec("BM_6M", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 6, 6, 1_000_000.0, -0.015),
            SwapSpec("BM_1Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 12, 6, 1_000_000.0, -0.012),
            SwapSpec("BM_2Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 24, 6, 1_000_000.0, -0.008),
            SwapSpec("BM_3Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 36, 6, 1_000_000.0, -0.004),
            SwapSpec("BM_5Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 60, 6, 1_000_000.0, 0.002),
            SwapSpec("BM_7Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 84, 6, 1_000_000.0, 0.006),
            SwapSpec("BM_10Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 120, 6, 1_000_000.0, 0.010),
        ]
        base_knots = fit_zero_curve(neg_benchmarks, self.val_date, self.base_holidays, "EUR_OIS", "LOG_LINEAR_DF")
        scen_knots = build_curve_scenarios(base_knots, bump_bp=1.0)

        # Normal SABR (beta=0.0) supports negative forwards & strikes cleanly
        vol_points = [
            VolPoint("BASE", "EUR_SWP_VOL", 0, 0.5, 1.0, vol=0.0075, beta=0.0, rho=-0.20, nu=0.25),
            VolPoint("BASE", "EUR_SWP_VOL", 1, 0.5, 5.0, vol=0.0070, beta=0.0, rho=-0.20, nu=0.25),
            VolPoint("BASE", "EUR_SWP_VOL", 2, 2.0, 1.0, vol=0.0072, beta=0.0, rho=-0.20, nu=0.25),
            VolPoint("BASE", "EUR_SWP_VOL", 3, 2.0, 5.0, vol=0.0068, beta=0.0, rho=-0.20, nu=0.25),
        ]
        swaptions = []
        swp_cfs = []
        for i in range(20):
            spec = SwaptionSpec(
                trade_id=f"NEG_SWP_{i:02d}",
                ccy="EUR",
                discount_curve_id="EUR_OIS",
                proj_curve_id="EUR_OIS",
                vol_surface_id="EUR_SWP_VOL",
                expiry_months=[6, 12, 18, 24][i % 4],
                swap_tenor_months=[12, 24, 36, 60][i % 4],
                freq_months=6,
                notional=1_000_000.0,
                strike=-0.012 + 0.003 * (i % 6),  # negative to positive strikes
                is_payer=(i % 2 == 0),
                pay_receive=1.0 if (i % 3 != 0) else -1.0,
                settlement_type="PHYSICAL" if (i % 2 == 0) else "CASH",
            )
            r, c = expand_swaption_to_rows(spec, self.val_date, self.base_holidays)
            swaptions.append(r)
            swp_cfs.extend(c)

        py_pvs = price_swaptions_python(swaptions, swp_cfs, scen_knots, vol_points)
        fn_pvs = run_pure_swaption_pricing_duckdb(
            scen_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir, mode="function"
        )
        max_pv_diff = max(abs(py_pvs[k] - fn_pvs[k]) for k in py_pvs)
        self.assertLess(max_pv_diff, 1e-8, f"Negative-rate Bachelier PV diff={max_pv_diff}")

        py_dv01 = compute_bump_dv01_python(py_pvs, base_knots)
        ad_dv01 = run_pure_dv01_duckdb(
            base_knots, swp_cfs, swaptions=swaptions, vol_points=vol_points,
            projects_dir=pricing_dir, kind="swaption",
        )
        max_dv01_diff = max(
            abs(ad_dv01.get((t_id, c_id, k_idx), 0.0) - val)
            for (t_id, c_id, k_idx), val in py_dv01.items()
        )
        self.assertLess(max_dv01_diff, 5e-2, f"Negative-rate AD DV01 diff={max_dv01_diff}")

        ad_vega = run_pure_vega_duckdb(base_knots, swp_cfs, swaptions, vol_points, projects_dir=pricing_dir)
        self.assertGreater(len(ad_vega), 0)
        print(
            f"[neg_rate_bachelier] swaptions=20 pv_diff={max_pv_diff:.2e} dv01_diff={max_dv01_diff:.2e}"
        )

    def test_imm_ois_and_xccy_multi_curve_skinny_table_and_ad_dv01(self) -> None:
        pricing_dir = _find_pricing_dir()
        imms = next_imm_dates(self.val_date, 5)  # 4 quarterly IMM 3M contracts

        # 1. USD_OIS fit from IMM 3M FRAs + OIS swaps
        usd_bm_cfs = [
            expand_fra_to_cashflows(
                FraSpec(f"USD_IMM_{i}", "USD", "USD_OIS", "USD_OIS", imms[i], imms[i + 1], 1_000_000.0, 0.041 + 0.0005 * i),
                self.val_date, self.base_holidays,
            )
            for i in range(4)
        ] + [
            expand_swap_to_cashflows(
                SwapSpec(f"USD_OIS_{m}M", "OIS", "USD", "USD_OIS", "USD_OIS", self.val_date, m, 6, 1_000_000.0, r),
                self.val_date, self.base_holidays,
            )
            for m, r in [(24, 0.0430), (36, 0.0445), (60, 0.0465), (120, 0.0485)]
        ]
        usd_knots = fit_curve_from_cashflows(usd_bm_cfs, "USD_OIS", "LOG_LINEAR_DF")

        # 2. EUR_OIS fit from IMM 3M FRAs + EUR OIS swaps
        eur_bm_cfs = [
            expand_fra_to_cashflows(
                FraSpec(f"EUR_IMM_{i}", "EUR", "EUR_OIS", "EUR_OIS", imms[i], imms[i + 1], 1_000_000.0, 0.025 + 0.0005 * i),
                self.val_date, self.base_holidays,
            )
            for i in range(4)
        ] + [
            expand_swap_to_cashflows(
                SwapSpec(f"EUR_OIS_{m}M", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, m, 6, 1_000_000.0, r),
                self.val_date, self.base_holidays,
            )
            for m, r in [(24, 0.0270), (36, 0.0285), (60, 0.0305), (120, 0.0325)]
        ]
        eur_knots = fit_curve_from_cashflows(eur_bm_cfs, "EUR_OIS", "LOG_LINEAR_DF")

        # 3. EUR_DISC_USD fit from EUR/USD XCCY basis swaps (holding USD_OIS & EUR_OIS fixed)
        fx_spot = 1.10
        xccy_bm_cfs = [
            expand_xccy_basis_swap_to_cashflows(
                XccyBasisSwapSpec(
                    f"XCCY_BM_{m}M", "EUR", "USD",
                    "EUR_DISC_USD", "EUR_OIS", "USD_OIS", "USD_OIS",
                    self.val_date, m, 6, 1_000_000.0, fx_spot, spread,
                ),
                self.val_date, self.base_holidays,
            )
            for m, spread in [(12, -0.0012), (24, -0.0015), (60, -0.0018), (120, -0.0020)]
        ]
        eur_disc_usd_knots = fit_curve_from_cashflows(
            xccy_bm_cfs, "EUR_DISC_USD", "LOG_LINEAR_DF",
            fixed_knots=usd_knots + eur_knots, initial_rate=0.026,
        )

        all_base_knots = usd_knots + eur_knots + eur_disc_usd_knots

        # Verify every benchmark (IMM FRA, OIS, XCCY) prices to par
        all_bms = usd_bm_cfs + eur_bm_cfs + xccy_bm_cfs
        bm_flat = [cf for cfs in all_bms for cf in cfs]
        bm_pvs = price_cashflows_python(bm_flat, all_base_knots)
        max_bm_res = max(abs(bm_pvs[("BASE", cfs[0].trade_id)]) / cfs[0].notional for cfs in all_bms)
        self.assertLess(max_bm_res, 1e-10, f"Multi-curve benchmark par residual={max_bm_res}")

        # Build mixed portfolio (IMM FRAs + OIS swaps + XCCY basis swaps) in single skinny PR_CASHFLOW
        portfolio_cfs = []
        for i in range(4):
            portfolio_cfs.extend(expand_fra_to_cashflows(
                FraSpec(f"PF_IMM_{i}", "USD", "USD_OIS", "USD_OIS", imms[i], imms[i + 1], 5_000_000.0, 0.039 + 0.001 * i),
                self.val_date, self.base_holidays,
            ))
        portfolio_cfs.extend(expand_swap_to_cashflows(
            SwapSpec("PF_EUR_OIS_5Y", "OIS", "EUR", "EUR_OIS", "EUR_OIS", self.val_date, 60, 6, 3_000_000.0, 0.032),
            self.val_date, self.base_holidays,
        ))
        portfolio_cfs.extend(expand_xccy_basis_swap_to_cashflows(
            XccyBasisSwapSpec(
                "PF_XCCY_5Y", "EUR", "USD",
                "EUR_DISC_USD", "EUR_OIS", "USD_OIS", "USD_OIS",
                self.val_date, 60, 6, 2_000_000.0, fx_spot, -0.0010,
            ),
            self.val_date, self.base_holidays,
        ))

        scen_knots = build_curve_scenarios(all_base_knots, bump_bp=1.0)
        py_pvs = price_cashflows_python(portfolio_cfs, scen_knots)
        sql_pvs = run_pure_pricing_duckdb(scen_knots, portfolio_cfs, projects_dir=pricing_dir)
        max_pv_diff = max(abs(py_pvs[k] - sql_pvs[k]) for k in py_pvs)
        self.assertLess(max_pv_diff, 1e-8, f"IMM+OIS+XCCY PV diff={max_pv_diff}")

        py_dv01 = compute_bump_dv01_python(py_pvs, all_base_knots)
        ad_dv01 = run_pure_dv01_duckdb(all_base_knots, portfolio_cfs, projects_dir=pricing_dir, kind="cashflow")
        max_dv01_diff = max(
            abs(ad_dv01.get((t_id, c_id, k_idx), 0.0) - val)
            for (t_id, c_id, k_idx), val in py_dv01.items()
        )
        self.assertLess(max_dv01_diff, 5e-3, f"IMM+OIS+XCCY AD DV01 diff={max_dv01_diff}")
        print(
            f"[imm_ois_xccy] curves=3 (USD_OIS, EUR_OIS, EUR_DISC_USD) trades=6 "
            f"bm_par_res={max_bm_res:.2e} pv_diff={max_pv_diff:.2e} ad_dv01_diff={max_dv01_diff:.2e}"
        )

    def test_pure_fold_bootstrap_and_multi_param_sweep_duckdb(self) -> None:
        from pricing.rack import _load_pricing_pure_sources
        pricing_dir = _find_pricing_dir()
        model_text = _load_pricing_pure_sources(pricing_dir)
        con = duckdb.connect(":memory:")
        try:
            frames = ll.Frames(con)
            with ll.HeldModel(model_text) as held:
                par_rates = [0.040, 0.042, 0.045, 0.047, 0.050]
                plan_boot = held.plan(
                    ll.parse(f"pricing::foldBootstrapOis([{', '.join(str(r) for r in par_rates)}])"),
                    "pricing::DuckDbRuntime",
                )
                dfs = [row["value"] for row in frames.run(plan_boot).to_pylist()]

                # Python reference sequential bootstrap
                ref_dfs: list[float] = []
                for r in par_rates:
                    ref_dfs.append((1.0 - r * sum(ref_dfs)) / (1.0 + r))
                self.assertEqual(len(dfs), len(ref_dfs))
                max_boot_diff = max(abs(a - b) for a, b in zip(dfs, ref_dfs))
                self.assertLess(max_boot_diff, 1e-12)

                # Multi-parameter list accumulator sweep updating [r0, r1] simultaneously
                plan_sweep = held.plan(
                    ll.parse("pricing::foldMultiParamSweep(0.04, 0.042)"),
                    "pricing::DuckDbRuntime",
                )
                r0, r1 = [row["value"] for row in frames.run(plan_sweep).to_pylist()]
                import math
                ref_df0 = 1.0 / (1.0 + 0.04)
                ref_df1 = (1.0 - 0.042 * ref_df0) / (1.0 + 0.042)
                self.assertLess(abs(r0 - (-math.log(ref_df0))), 1e-12)
                self.assertLess(abs(r1 - (-math.log(ref_df1) / 2.0)), 1e-12)
                print(f"[fold_duckdb] boot_diff={max_boot_diff:.2e} sweep_r=({r0:.6f}, {r1:.6f})")
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
