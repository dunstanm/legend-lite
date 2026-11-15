"""SOD schedule + holiday expansion, external curve fitting, and Pure pricing bridge.

Keeps schedule/holiday rules and multivariable curve calibration in Python, feeding two
independent domain tables (PR_CURVE_KNOT and PR_CASHFLOW) into projects/pricing.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pyarrow as pa
import legend_lite as ll


@dataclass(frozen=True)
class CurveKnot:
    scenario_id: str
    curve_id: str
    knot_idx: int
    knot_t: float
    zero_rate: float
    interp_method: str


@dataclass(frozen=True)
class CashflowRow:
    trade_id: str
    leg_id: str
    cf_idx: int
    instrument_type: str
    ccy: str
    discount_curve_id: str
    proj_curve_id: str
    accrual_start_t: float
    accrual_end_t: float
    pay_t: float
    dcf: float
    notional: float
    fixed_rate: float
    spread: float
    is_float: bool
    pay_receive: float


@dataclass(frozen=True)
class SwapSpec:
    trade_id: str
    instrument_type: str  # "IRS" or "OIS"
    ccy: str
    discount_curve_id: str
    proj_curve_id: str
    start_date: dt.date
    maturity_months: int
    freq_months: int
    notional: float
    fixed_rate: float
    spread: float = 0.0
    pay_fixed: bool = True
    day_count: str = "ACT/365F"


@dataclass(frozen=True)
class FraSpec:
    trade_id: str
    ccy: str
    discount_curve_id: str
    proj_curve_id: str
    start_date: dt.date
    end_date: dt.date
    notional: float
    fixed_rate: float
    pay_fixed: bool = True
    day_count: str = "ACT/360"


@dataclass(frozen=True)
class XccyBasisSwapSpec:
    """Float-for-float cross-currency basis swap with initial/final notional exchanges.

    Leg 1 (foreign, e.g. EUR) pays/receives float(leg1_proj) + spread, discounted on
    leg1_disc (e.g. EUR_DISC_USD), scaled by fx_spot (USD per 1 unit of ccy1) into
    reporting currency so all cashflows share the single skinny PR_CASHFLOW table.
    Leg 2 (base, e.g. USD) pays/receives flat float(leg2_proj), discounted on leg2_disc.
    """

    trade_id: str
    ccy1: str
    ccy2: str
    leg1_discount_curve_id: str
    leg1_proj_curve_id: str
    leg2_discount_curve_id: str
    leg2_proj_curve_id: str
    start_date: dt.date
    maturity_months: int
    freq_months: int
    notional_ccy1: float
    fx_spot_ccy2_per_ccy1: float
    leg1_spread: float
    pay_leg1: bool = True
    day_count1: str = "ACT/360"
    day_count2: str = "ACT/360"


@dataclass(frozen=True)
class SwaptionSpec:
    trade_id: str
    ccy: str
    discount_curve_id: str
    proj_curve_id: str
    vol_surface_id: str
    expiry_months: int
    swap_tenor_months: int
    freq_months: int
    notional: float
    strike: float
    is_payer: bool = True
    pay_receive: float = 1.0
    day_count: str = "ACT/365F"
    settlement_type: str = "PHYSICAL"


@dataclass(frozen=True)
class SwaptionRow:
    trade_id: str
    ccy: str
    discount_curve_id: str
    proj_curve_id: str
    vol_surface_id: str
    expiry_t: float
    settle_t: float
    swap_tenor_t: float
    cash_freq: float
    n_periods: float
    settlement_type: str
    strike: float
    notional: float
    is_payer: bool
    pay_receive: float


@dataclass(frozen=True)
class VolPoint:
    scenario_id: str
    vol_surface_id: str
    point_idx: int
    expiry_t: float
    swap_tenor_t: float
    vol: float
    beta: float = 1.0
    rho: float = 0.0
    nu: float = 0.0


def _add_months(d: dt.date, months: int) -> dt.date:
    month0 = d.month - 1 + months
    year = d.year + month0 // 12
    month = month0 % 12 + 1
    # clamp day to month end
    for day in (d.day, 30, 29, 28):
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    raise ValueError(f"Invalid date {year}-{month}-{d.day}")


def is_business_day(d: dt.date, holidays: set[dt.date]) -> bool:
    return d.weekday() < 5 and d not in holidays


def adjust_date(
    d: dt.date,
    holidays: set[dt.date],
    convention: str = "MOD_FOLLOWING",
) -> dt.date:
    """Adjust a schedule date for weekends and dynamic holiday calendars."""
    if is_business_day(d, holidays):
        return d
    if convention == "FOLLOWING":
        cur = d
        while not is_business_day(cur, holidays):
            cur += dt.timedelta(days=1)
        return cur
    if convention == "MOD_FOLLOWING":
        cur = d
        while not is_business_day(cur, holidays):
            cur += dt.timedelta(days=1)
        if cur.month == d.month:
            return cur
        cur = d
        while not is_business_day(cur, holidays):
            cur -= dt.timedelta(days=1)
        return cur
    raise ValueError(f"Unsupported convention: {convention}")


def imm_date(year: int, month: int) -> dt.date:
    """Return the 3rd Wednesday (IMM date) of the given year and month."""
    first = dt.date(year, month, 1)
    # Wednesday is weekday() == 2
    offset = (2 - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 14)


def next_imm_dates(valuation_date: dt.date, count: int) -> list[dt.date]:
    """Return the next `count` quarterly IMM dates (Mar/Jun/Sep/Dec) strictly after valuation_date."""
    out: list[dt.date] = []
    y, m = valuation_date.year, valuation_date.month
    q_months = (3, 6, 9, 12)
    while len(out) < count:
        for qm in q_months:
            if (y, qm) >= (valuation_date.year, valuation_date.month):
                d = imm_date(y, qm)
                if d > valuation_date:
                    out.append(d)
                    if len(out) == count:
                        break
        y += 1
    return out


def year_fraction(d1: dt.date, d2: dt.date, basis: str = "ACT/365F") -> float:
    days = (d2 - d1).days
    if basis == "ACT/365F":
        return days / 365.0
    if basis == "ACT/360":
        return days / 360.0
    if basis == "30/360":
        d1_d = min(d1.day, 30)
        d2_d = 30 if (d2.day == 31 and d1_d == 30) else d2.day
        return ((d2.year - d1.year) * 360 + (d2.month - d1.month) * 30 + (d2_d - d1_d)) / 360.0
    raise ValueError(f"Unsupported day count basis: {basis}")


def expand_swap_to_cashflows(
    spec: SwapSpec,
    valuation_date: dt.date,
    holidays: set[dt.date],
) -> list[CashflowRow]:
    """Expand a swap at SOD with live holiday adjustments into normalized cashflow rows."""
    n_periods = spec.maturity_months // spec.freq_months
    unadj_dates = [_add_months(spec.start_date, i * spec.freq_months) for i in range(n_periods + 1)]
    adj_dates = [adjust_date(d, holidays, "MOD_FOLLOWING") for d in unadj_dates]

    fixed_sign = -1.0 if spec.pay_fixed else 1.0
    float_sign = -fixed_sign
    rows: list[CashflowRow] = []

    for i in range(n_periods):
        s_d, e_d = adj_dates[i], adj_dates[i + 1]
        t_start = year_fraction(valuation_date, s_d, "ACT/365F")
        t_end = year_fraction(valuation_date, e_d, "ACT/365F")
        dcf = year_fraction(s_d, e_d, spec.day_count)
        rows.append(CashflowRow(
            trade_id=spec.trade_id,
            leg_id="FIXED",
            cf_idx=i,
            instrument_type=spec.instrument_type,
            ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id,
            proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start,
            accrual_end_t=t_end,
            pay_t=t_end,
            dcf=dcf,
            notional=spec.notional,
            fixed_rate=spec.fixed_rate,
            spread=0.0,
            is_float=False,
            pay_receive=fixed_sign,
        ))
        rows.append(CashflowRow(
            trade_id=spec.trade_id,
            leg_id="FLOAT",
            cf_idx=i,
            instrument_type=spec.instrument_type,
            ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id,
            proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start,
            accrual_end_t=t_end,
            pay_t=t_end,
            dcf=dcf,
            notional=spec.notional,
            fixed_rate=0.0,
            spread=spec.spread,
            is_float=True,
            pay_receive=float_sign,
        ))
    return rows


def expand_fra_to_cashflows(
    spec: FraSpec,
    valuation_date: dt.date,
    holidays: set[dt.date],
) -> list[CashflowRow]:
    """Expand a FRA into fixed + floating cashflow rows with live holiday adjustments."""
    s_d = adjust_date(spec.start_date, holidays, "MOD_FOLLOWING")
    e_d = adjust_date(spec.end_date, holidays, "MOD_FOLLOWING")
    t_start = year_fraction(valuation_date, s_d, "ACT/365F")
    t_end = year_fraction(valuation_date, e_d, "ACT/365F")
    dcf = year_fraction(s_d, e_d, spec.day_count)
    fixed_sign = -1.0 if spec.pay_fixed else 1.0
    float_sign = -fixed_sign
    return [
        CashflowRow(
            trade_id=spec.trade_id,
            leg_id="FIXED",
            cf_idx=0,
            instrument_type="FRA",
            ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id,
            proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start,
            accrual_end_t=t_end,
            pay_t=t_end,
            dcf=dcf,
            notional=spec.notional,
            fixed_rate=spec.fixed_rate,
            spread=0.0,
            is_float=False,
            pay_receive=fixed_sign,
        ),
        CashflowRow(
            trade_id=spec.trade_id,
            leg_id="FLOAT",
            cf_idx=0,
            instrument_type="FRA",
            ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id,
            proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start,
            accrual_end_t=t_end,
            pay_t=t_end,
            dcf=dcf,
            notional=spec.notional,
            fixed_rate=0.0,
            spread=0.0,
            is_float=True,
            pay_receive=float_sign,
        ),
    ]


def expand_xccy_basis_swap_to_cashflows(
    spec: XccyBasisSwapSpec,
    valuation_date: dt.date,
    holidays: set[dt.date],
) -> list[CashflowRow]:
    """Expand a cross-currency basis swap (coupons + initial/final notional exchanges)
    into the single skinny PR_CASHFLOW schema, expressed in ccy2 (reporting currency).
    """
    n_periods = spec.maturity_months // spec.freq_months
    unadj_dates = [_add_months(spec.start_date, i * spec.freq_months) for i in range(n_periods + 1)]
    adj_dates = [adjust_date(d, holidays, "MOD_FOLLOWING") for d in unadj_dates]

    notional_rep = spec.notional_ccy1 * spec.fx_spot_ccy2_per_ccy1
    leg1_sign = -1.0 if spec.pay_leg1 else 1.0
    leg2_sign = -leg1_sign
    rows: list[CashflowRow] = []

    # Initial notional exchange at start_date (t_0): receive leg1 notional, pay leg2 notional (or vice versa)
    t0 = year_fraction(valuation_date, adj_dates[0], "ACT/365F")
    t_mat = year_fraction(valuation_date, adj_dates[-1], "ACT/365F")
    rows.append(CashflowRow(
        trade_id=spec.trade_id, leg_id="LEG1_NX", cf_idx=0,
        instrument_type="XCCY", ccy=spec.ccy1,
        discount_curve_id=spec.leg1_discount_curve_id, proj_curve_id=spec.leg1_proj_curve_id,
        accrual_start_t=t0, accrual_end_t=t0, pay_t=t0, dcf=1.0,
        notional=notional_rep, fixed_rate=1.0, spread=0.0, is_float=False,
        pay_receive=-leg1_sign,
    ))
    rows.append(CashflowRow(
        trade_id=spec.trade_id, leg_id="LEG2_NX", cf_idx=0,
        instrument_type="XCCY", ccy=spec.ccy2,
        discount_curve_id=spec.leg2_discount_curve_id, proj_curve_id=spec.leg2_proj_curve_id,
        accrual_start_t=t0, accrual_end_t=t0, pay_t=t0, dcf=1.0,
        notional=notional_rep, fixed_rate=1.0, spread=0.0, is_float=False,
        pay_receive=-leg2_sign,
    ))

    for i in range(n_periods):
        s_d, e_d = adj_dates[i], adj_dates[i + 1]
        t_start = year_fraction(valuation_date, s_d, "ACT/365F")
        t_end = year_fraction(valuation_date, e_d, "ACT/365F")
        dcf1 = year_fraction(s_d, e_d, spec.day_count1)
        dcf2 = year_fraction(s_d, e_d, spec.day_count2)
        rows.append(CashflowRow(
            trade_id=spec.trade_id, leg_id="LEG1", cf_idx=i,
            instrument_type="XCCY", ccy=spec.ccy1,
            discount_curve_id=spec.leg1_discount_curve_id, proj_curve_id=spec.leg1_proj_curve_id,
            accrual_start_t=t_start, accrual_end_t=t_end, pay_t=t_end, dcf=dcf1,
            notional=notional_rep, fixed_rate=0.0, spread=spec.leg1_spread, is_float=True,
            pay_receive=leg1_sign,
        ))
        rows.append(CashflowRow(
            trade_id=spec.trade_id, leg_id="LEG2", cf_idx=i,
            instrument_type="XCCY", ccy=spec.ccy2,
            discount_curve_id=spec.leg2_discount_curve_id, proj_curve_id=spec.leg2_proj_curve_id,
            accrual_start_t=t_start, accrual_end_t=t_end, pay_t=t_end, dcf=dcf2,
            notional=notional_rep, fixed_rate=0.0, spread=0.0, is_float=True,
            pay_receive=leg2_sign,
        ))

    # Final notional exchange at maturity
    rows.append(CashflowRow(
        trade_id=spec.trade_id, leg_id="LEG1_NX", cf_idx=1,
        instrument_type="XCCY", ccy=spec.ccy1,
        discount_curve_id=spec.leg1_discount_curve_id, proj_curve_id=spec.leg1_proj_curve_id,
        accrual_start_t=t_mat, accrual_end_t=t_mat, pay_t=t_mat, dcf=1.0,
        notional=notional_rep, fixed_rate=1.0, spread=0.0, is_float=False,
        pay_receive=leg1_sign,
    ))
    rows.append(CashflowRow(
        trade_id=spec.trade_id, leg_id="LEG2_NX", cf_idx=1,
        instrument_type="XCCY", ccy=spec.ccy2,
        discount_curve_id=spec.leg2_discount_curve_id, proj_curve_id=spec.leg2_proj_curve_id,
        accrual_start_t=t_mat, accrual_end_t=t_mat, pay_t=t_mat, dcf=1.0,
        notional=notional_rep, fixed_rate=1.0, spread=0.0, is_float=False,
        pay_receive=leg2_sign,
    ))
    return rows


def expand_swaption_to_rows(
    spec: SwaptionSpec,
    valuation_date: dt.date,
    holidays: set[dt.date],
) -> tuple[SwaptionRow, list[CashflowRow]]:
    """Expand a European swaption into its PR_SWAPTION row + underlying PR_CASHFLOW legs."""
    expiry_date = adjust_date(_add_months(valuation_date, spec.expiry_months), holidays, "MOD_FOLLOWING")
    expiry_t = year_fraction(valuation_date, expiry_date, "ACT/365F")
    swap_tenor_t = spec.swap_tenor_months / 12.0
    n_periods = spec.swap_tenor_months // spec.freq_months
    cash_freq = 12.0 / spec.freq_months
    sw_row = SwaptionRow(
        trade_id=spec.trade_id,
        ccy=spec.ccy,
        discount_curve_id=spec.discount_curve_id,
        proj_curve_id=spec.proj_curve_id,
        vol_surface_id=spec.vol_surface_id,
        expiry_t=expiry_t,
        settle_t=expiry_t,
        swap_tenor_t=swap_tenor_t,
        cash_freq=cash_freq,
        n_periods=float(n_periods),
        settlement_type=spec.settlement_type,
        strike=spec.strike,
        notional=spec.notional,
        is_payer=spec.is_payer,
        pay_receive=spec.pay_receive,
    )
    unadj_dates = [_add_months(expiry_date, i * spec.freq_months) for i in range(n_periods + 1)]
    adj_dates = [adjust_date(d, holidays, "MOD_FOLLOWING") for d in unadj_dates]
    cfs: list[CashflowRow] = []
    for i in range(n_periods):
        s_d, e_d = adj_dates[i], adj_dates[i + 1]
        t_start = year_fraction(valuation_date, s_d, "ACT/365F")
        t_end = year_fraction(valuation_date, e_d, "ACT/365F")
        dcf = year_fraction(s_d, e_d, spec.day_count)
        cfs.append(CashflowRow(
            trade_id=spec.trade_id, leg_id="FIXED", cf_idx=i,
            instrument_type="SWAPTION_UNDERLYING", ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id, proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start, accrual_end_t=t_end, pay_t=t_end,
            dcf=dcf, notional=spec.notional, fixed_rate=spec.strike,
            spread=0.0, is_float=False, pay_receive=1.0,
        ))
        cfs.append(CashflowRow(
            trade_id=spec.trade_id, leg_id="FLOAT", cf_idx=i,
            instrument_type="SWAPTION_UNDERLYING", ccy=spec.ccy,
            discount_curve_id=spec.discount_curve_id, proj_curve_id=spec.proj_curve_id,
            accrual_start_t=t_start, accrual_end_t=t_end, pay_t=t_end,
            dcf=dcf, notional=spec.notional, fixed_rate=0.0,
            spread=0.0, is_float=True, pay_receive=1.0,
        ))
    return sw_row, cfs


def _ref_df(knots: Sequence[tuple[float, float]], method: str, t: float) -> float:
    if t <= 0.0:
        return 1.0
    if t <= knots[0][0]:
        return math.exp(-knots[0][1] * t)
    if t >= knots[-1][0]:
        return math.exp(-knots[-1][1] * t)
    for i in range(len(knots) - 1):
        t0, r0 = knots[i]
        t1, r1 = knots[i + 1]
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            if method in ("LOG_LINEAR_DF", "FLAT_FORWARD"):
                return math.exp(-((1.0 - w) * (r0 * t0) + w * (r1 * t1)))
            r = (1.0 - w) * r0 + w * r1
            return math.exp(-r * t)
    raise RuntimeError(f"Unbracketed t={t}")


def price_cashflows_python(
    cashflows: Sequence[CashflowRow],
    knots: Sequence[CurveKnot],
) -> dict[tuple[str, str], float]:
    """Python float reference oracle: returns {(scenario_id, trade_id): npv}."""
    by_scen_curve: dict[tuple[str, str], tuple[str, list[tuple[float, float]]]] = {}
    scenarios: set[str] = set()
    for k in sorted(knots, key=lambda x: (x.scenario_id, x.curve_id, x.knot_t)):
        scenarios.add(k.scenario_id)
        key = (k.scenario_id, k.curve_id)
        if key not in by_scen_curve:
            by_scen_curve[key] = (k.interp_method, [])
        by_scen_curve[key][1].append((k.knot_t, k.zero_rate))

    totals: dict[tuple[str, str], float] = {}
    for scen in scenarios:
        for cf in cashflows:
            d_method, d_pts = by_scen_curve[(scen, cf.discount_curve_id)]
            df_pay = _ref_df(d_pts, d_method, cf.pay_t)
            if cf.is_float:
                p_method, p_pts = by_scen_curve[(scen, cf.proj_curve_id)]
                df_s = _ref_df(p_pts, p_method, cf.accrual_start_t)
                df_e = _ref_df(p_pts, p_method, cf.accrual_end_t)
                rate = ((df_s / df_e) - 1.0) / cf.dcf + cf.spread
            else:
                rate = cf.fixed_rate
            pv = cf.pay_receive * cf.notional * cf.dcf * rate * df_pay
            tkey = (scen, cf.trade_id)
            totals[tkey] = totals.get(tkey, 0.0) + pv
    return totals


def fit_zero_curve(
    benchmark_swaps: Sequence[SwapSpec],
    valuation_date: dt.date,
    holidays: set[dt.date],
    curve_id: str,
    interp_method: str = "LOG_LINEAR_DF",
    max_iter: int = 20,
    tol: float = 1e-12,
) -> list[CurveKnot]:
    """External simultaneous multivariable Gauss-Newton curve fitter in Python floats."""
    cfs_by_swap = [expand_swap_to_cashflows(sw, valuation_date, holidays) for sw in benchmark_swaps]
    knot_ts = [max(cf.pay_t for cf in cfs) for cfs in cfs_by_swap]
    rates = [sw.fixed_rate for sw in benchmark_swaps]
    n = len(knot_ts)

    def residuals(r_vec: list[float]) -> list[float]:
        knots = [
            CurveKnot("BASE", curve_id, idx, knot_ts[idx], r_vec[idx], interp_method)
            for idx in range(n)
        ]
        res = []
        for idx, cfs in enumerate(cfs_by_swap):
            npv = price_cashflows_python(cfs, knots)[("BASE", benchmark_swaps[idx].trade_id)]
            res.append(npv / benchmark_swaps[idx].notional)
        return res

    eps = 1e-7
    for _ in range(max_iter):
        f0 = residuals(rates)
        if max(abs(v) for v in f0) < tol:
            break
        jac = [[0.0] * n for _ in range(n)]
        for j in range(n):
            r_bump = list(rates)
            r_bump[j] += eps
            f1 = residuals(r_bump)
            for i in range(n):
                jac[i][j] = (f1[i] - f0[i]) / eps
        # Gaussian elimination with partial pivoting for J * delta = -f0
        aug = [jac[i][:] + [-f0[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            pv = aug[col][col]
            for r in range(col + 1, n):
                factor = aug[r][col] / pv
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
        delta = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = aug[i][n] - sum(aug[i][c] * delta[c] for c in range(i + 1, n))
            delta[i] = s / aug[i][i]
        rates = [rates[i] + delta[i] for i in range(n)]

    return [
        CurveKnot("BASE", curve_id, idx, knot_ts[idx], rates[idx], interp_method)
        for idx in range(n)
    ]


def fit_curve_from_cashflows(
    benchmark_cashflows: Sequence[Sequence[CashflowRow]],
    curve_id: str,
    interp_method: str = "LOG_LINEAR_DF",
    fixed_knots: Sequence[CurveKnot] = (),
    initial_rate: float = 0.03,
    max_iter: int = 25,
    tol: float = 1e-12,
) -> list[CurveKnot]:
    """Fit `curve_id` from arbitrary pre-expanded benchmark instruments (IMM FRAs, OIS swaps,
    or XCCY basis swaps) holding any upstream `fixed_knots` constant.
    """
    knot_ts = sorted({max(cf.pay_t for cf in cfs) for cfs in benchmark_cashflows})
    if len(knot_ts) != len(benchmark_cashflows):
        # Fall back to per-instrument max pay_t in given order if two pillars share a date
        knot_ts = [max(cf.pay_t for cf in cfs) for cfs in benchmark_cashflows]
    n = len(benchmark_cashflows)
    rates = [initial_rate] * n
    fixed_list = list(fixed_knots)
    trade_ids = [cfs[0].trade_id for cfs in benchmark_cashflows]
    scales = [cfs[0].notional for cfs in benchmark_cashflows]

    def residuals(r_vec: list[float]) -> list[float]:
        knots = fixed_list + [
            CurveKnot("BASE", curve_id, idx, knot_ts[idx], r_vec[idx], interp_method)
            for idx in range(n)
        ]
        res = []
        for idx, cfs in enumerate(benchmark_cashflows):
            npv = price_cashflows_python(cfs, knots)[("BASE", trade_ids[idx])]
            res.append(npv / scales[idx])
        return res

    eps = 1e-7
    for _ in range(max_iter):
        f0 = residuals(rates)
        if max(abs(v) for v in f0) < tol:
            break
        jac = [[0.0] * n for _ in range(n)]
        for j in range(n):
            r_bump = list(rates)
            r_bump[j] += eps
            f1 = residuals(r_bump)
            for i in range(n):
                jac[i][j] = (f1[i] - f0[i]) / eps
        aug = [jac[i][:] + [-f0[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            pv = aug[col][col]
            for r in range(col + 1, n):
                factor = aug[r][col] / pv
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
        delta = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = aug[i][n] - sum(aug[i][c] * delta[c] for c in range(i + 1, n))
            delta[i] = s / aug[i][i]
        rates = [rates[i] + delta[i] for i in range(n)]

    return [
        CurveKnot("BASE", curve_id, idx, knot_ts[idx], rates[idx], interp_method)
        for idx in range(n)
    ]


def build_curve_scenarios(
    base_knots: Sequence[CurveKnot],
    bump_bp: float = 1.0,
) -> list[CurveKnot]:
    """Generate BASE + central +/- bump_bp scenarios for each pillar knot."""
    bump = bump_bp * 1e-4
    out = list(base_knots)
    for k in base_knots:
        for sign, tag in ((1.0, "UP"), (-1.0, "DN")):
            scen_id = f"BUMP_{k.curve_id}_{k.knot_idx}_{tag}"
            for row in base_knots:
                rate = row.zero_rate + (sign * bump if row.knot_idx == k.knot_idx and row.curve_id == k.curve_id else 0.0)
                out.append(CurveKnot(scen_id, row.curve_id, row.knot_idx, row.knot_t, rate, row.interp_method))
    return out


def compute_bump_dv01_python(
    npvs: dict[tuple[str, str], float],
    base_knots: Sequence[CurveKnot],
) -> dict[tuple[str, str, int], float]:
    """Compute central 1bp DV01 per (trade_id, curve_id, knot_idx) from scenario NPVs."""
    trades = sorted({t for _, t in npvs.keys()})
    dv01s: dict[tuple[str, str, int], float] = {}
    for trade in trades:
        for k in base_knots:
            up = npvs[(f"BUMP_{k.curve_id}_{k.knot_idx}_UP", trade)]
            dn = npvs[(f"BUMP_{k.curve_id}_{k.knot_idx}_DN", trade)]
            dv01s[(trade, k.curve_id, k.knot_idx)] = 0.5 * (up - dn)
    return dv01s


def _tables_to_arrow(
    knots: Sequence[CurveKnot],
    cashflows: Sequence[CashflowRow],
) -> tuple[pa.Table, pa.Table]:
    knot_tbl = pa.table({
        "SCENARIO_ID": pa.array([k.scenario_id for k in knots], type=pa.string()),
        "CURVE_ID": pa.array([k.curve_id for k in knots], type=pa.string()),
        "KNOT_IDX": pa.array([k.knot_idx for k in knots], type=pa.int16()),
        "KNOT_T": pa.array([k.knot_t for k in knots], type=pa.float64()),
        "ZERO_RATE": pa.array([k.zero_rate for k in knots], type=pa.float64()),
        "INTERP_METHOD": pa.array([k.interp_method for k in knots], type=pa.string()),
    })
    cf_tbl = pa.table({
        "TRADE_ID": pa.array([c.trade_id for c in cashflows], type=pa.string()),
        "LEG_ID": pa.array([c.leg_id for c in cashflows], type=pa.string()),
        "CF_IDX": pa.array([c.cf_idx for c in cashflows], type=pa.int16()),
        "INSTRUMENT_TYPE": pa.array([c.instrument_type for c in cashflows], type=pa.string()),
        "CCY": pa.array([c.ccy for c in cashflows], type=pa.string()),
        "DISCOUNT_CURVE_ID": pa.array([c.discount_curve_id for c in cashflows], type=pa.string()),
        "PROJ_CURVE_ID": pa.array([c.proj_curve_id for c in cashflows], type=pa.string()),
        "ACCRUAL_START_T": pa.array([c.accrual_start_t for c in cashflows], type=pa.float64()),
        "ACCRUAL_END_T": pa.array([c.accrual_end_t for c in cashflows], type=pa.float64()),
        "PAY_T": pa.array([c.pay_t for c in cashflows], type=pa.float64()),
        "DCF": pa.array([c.dcf for c in cashflows], type=pa.float64()),
        "NOTIONAL": pa.array([c.notional for c in cashflows], type=pa.float64()),
        "FIXED_RATE": pa.array([c.fixed_rate for c in cashflows], type=pa.float64()),
        "SPREAD": pa.array([c.spread for c in cashflows], type=pa.float64()),
        "IS_FLOAT": pa.array([c.is_float for c in cashflows], type=pa.bool_()),
        "PAY_RECEIVE": pa.array([c.pay_receive for c in cashflows], type=pa.float64()),
    })
    return knot_tbl, cf_tbl


def _load_pricing_pure_sources(projects_dir: Path) -> str:
    model = (projects_dir / "model.pure").read_text(encoding="utf-8")
    store = (projects_dir / "store.pure").read_text(encoding="utf-8")
    mapping = (projects_dir / "mapping.pure").read_text(encoding="utf-8")
    runtime = """
###Connection
import pricing::*;
RelationalDatabaseConnection pricing::DuckDbConn
{
   type: DuckDB;
   specification: DuckDB { };
   auth: Test;
}
###Runtime
import pricing::*;
Runtime pricing::DuckDbRuntime
{
   mappings: [ pricing::Mapping ];
   connections: [ pricing::Store: [ environment: pricing::DuckDbConn ] ];
}
"""
    return "\n".join((model, store, mapping, runtime))


def run_pure_pricing_duckdb(
    knots: Sequence[CurveKnot],
    cashflows: Sequence[CashflowRow],
    projects_dir: Path | None = None,
) -> dict[tuple[str, str], float]:
    """Execute projects/pricing::cashflowPvs() on DuckDB via legend_lite."""
    if projects_dir is None:
        projects_dir = Path(__file__).resolve().parents[2] / "projects" / "pricing"
    knot_tbl, cf_tbl = _tables_to_arrow(knots, cashflows)

    conn = duckdb.connect(":memory:")
    try:
        conn.register("_arrow_knots", knot_tbl)
        conn.register("_arrow_cfs", cf_tbl)
        conn.execute("CREATE TABLE PR_CURVE_KNOT AS SELECT * FROM _arrow_knots")
        conn.execute("CREATE TABLE PR_CASHFLOW AS SELECT * FROM _arrow_cfs")

        frames = ll.Frames(conn)
        model_text = _load_pricing_pure_sources(projects_dir)
        with ll.HeldModel(model_text) as held:
            query = (
                "pricing::cashflowPvs()"
                "->groupBy(~[scenarioId, tradeId], ~[npv: x | $x.pv: y | $y->sum()])"
                "->sort([~scenarioId->ascending(), ~tradeId->ascending()])"
            )
            plan = held.plan(ll.parse(query), "pricing::DuckDbRuntime")
            out = frames.run(plan)
    finally:
        conn.close()

    return {
        (row["scenarioId"], row["tradeId"]): float(row["npv"])
        for row in out.to_pylist()
    }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sabr_normal_vol(
    fwd_rate: float,
    strike: float,
    expiry_t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Hagan (2002) Normal SABR implied volatility."""
    f_mid = 0.5 * (fwd_rate + strike)
    if beta == 0.0:
        c_mid = 1.0
        g1 = 0.0
        g2 = 0.0
    else:
        c_mid = f_mid ** beta
        g1 = beta / f_mid
        g2 = beta * (beta - 1.0) / (f_mid * f_mid)
    dfk = fwd_rate - strike
    z = (nu / alpha) * (dfk / c_mid)
    if abs(z) < 1.0e-7:
        z_over_x = 1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho * rho) / 12.0) * z * z
    else:
        z_over_x = z / math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    smile = 1.0 + (g2 / 24.0) * dfk * dfk
    corr = 1.0 + expiry_t * (
        ((2.0 * g2 - g1 * g1) / 24.0) * (alpha * alpha) * (c_mid * c_mid)
        + 0.25 * rho * nu * alpha * g1 * c_mid
        + ((2.0 - 3.0 * rho * rho) / 24.0) * nu * nu
    )
    return alpha * c_mid * smile * z_over_x * corr


def _interp_vol_py(
    vol_pts: Sequence[VolPoint],
    scen: str,
    surf: str,
    exp_t: float,
    ten_t: float,
    fwd_rate: float | None = None,
    strike: float | None = None,
) -> float:
    pts = [v for v in vol_pts if v.scenario_id == scen and v.vol_surface_id == surf]
    if not pts:
        pts = [v for v in vol_pts if v.scenario_id == "BASE" and v.vol_surface_id == surf]
    exps = sorted({v.expiry_t for v in pts})
    tens = sorted({v.swap_tenor_t for v in pts})
    e0 = max([e for e in exps if e <= exp_t], default=exps[0])
    e1 = min([e for e in exps if e >= exp_t], default=exps[-1])
    t0 = max([t for t in tens if t <= ten_t], default=tens[0])
    t1 = min([t for t in tens if t >= ten_t], default=tens[-1])
    grid = {(v.expiry_t, v.swap_tenor_t): v for v in pts}
    we = 0.0 if e1 <= e0 else min(1.0, max(0.0, (exp_t - e0) / (e1 - e0)))
    wt = 0.0 if t1 <= t0 else min(1.0, max(0.0, (ten_t - t0) / (t1 - t0)))
    w00 = (1.0 - we) * (1.0 - wt)
    w01 = (1.0 - we) * wt
    w10 = we * (1.0 - wt)
    w11 = we * wt
    p00, p01, p10, p11 = grid[(e0, t0)], grid[(e0, t1)], grid[(e1, t0)], grid[(e1, t1)]
    alpha = w00 * p00.vol + w01 * p01.vol + w10 * p10.vol + w11 * p11.vol
    beta = w00 * p00.beta + w01 * p01.beta + w10 * p10.beta + w11 * p11.beta
    rho = w00 * p00.rho + w01 * p01.rho + w10 * p10.rho + w11 * p11.rho
    nu = w00 * p00.nu + w01 * p01.nu + w10 * p10.nu + w11 * p11.nu
    if fwd_rate is None or strike is None:
        return alpha
    return sabr_normal_vol(fwd_rate, strike, exp_t, alpha, beta, rho, nu)


def price_swaptions_python(
    swaptions: Sequence[SwaptionRow],
    cashflows: Sequence[CashflowRow],
    knots: Sequence[CurveKnot],
    vol_points: Sequence[VolPoint],
) -> dict[tuple[str, str], float]:
    """Python float Black-76 swaption reference oracle: returns {(scenario_id, trade_id): pv}."""
    by_scen_curve: dict[tuple[str, str], tuple[str, list[tuple[float, float]]]] = {}
    scenarios: set[str] = set()
    for k in sorted(knots, key=lambda x: (x.scenario_id, x.curve_id, x.knot_t)):
        scenarios.add(k.scenario_id)
        key = (k.scenario_id, k.curve_id)
        if key not in by_scen_curve:
            by_scen_curve[key] = (k.interp_method, [])
        by_scen_curve[key][1].append((k.knot_t, k.zero_rate))

    cfs_by_trade: dict[str, list[CashflowRow]] = {}
    for cf in cashflows:
        if cf.instrument_type == "SWAPTION_UNDERLYING":
            cfs_by_trade.setdefault(cf.trade_id, []).append(cf)

    out: dict[tuple[str, str], float] = {}
    for scen in sorted(scenarios):
        for sw in swaptions:
            ann = 0.0
            flt_pv = 0.0
            d_method, d_pts = by_scen_curve[(scen, sw.discount_curve_id)]
            p_method, p_pts = by_scen_curve[(scen, sw.proj_curve_id)]
            for cf in cfs_by_trade[sw.trade_id]:
                df_pay = _ref_df(d_pts, d_method, cf.pay_t)
                if cf.is_float:
                    df_s = _ref_df(p_pts, p_method, cf.accrual_start_t)
                    df_e = _ref_df(p_pts, p_method, cf.accrual_end_t)
                    fwd = ((df_s / df_e) - 1.0) / cf.dcf + cf.spread
                    flt_pv += cf.dcf * fwd * df_pay
                else:
                    ann += cf.dcf * df_pay
            fwd_rate = flt_pv / ann
            if sw.settlement_type == "CASH":
                df_settle = _ref_df(d_pts, d_method, sw.settle_t)
                if abs(fwd_rate) < 1e-8:
                    c_ann = (sw.n_periods / sw.cash_freq) - (
                        sw.n_periods * (sw.n_periods + 1.0) / (2.0 * sw.cash_freq * sw.cash_freq)
                    ) * fwd_rate
                else:
                    c_ann = (1.0 - (1.0 + fwd_rate / sw.cash_freq) ** (-sw.n_periods)) / fwd_rate
                eff_ann = df_settle * c_ann
            else:
                eff_ann = ann
            vol = _interp_vol_py(
                vol_points, scen, sw.vol_surface_id, sw.expiry_t, sw.swap_tenor_t,
                fwd_rate=fwd_rate, strike=sw.strike,
            )
            std = vol * math.sqrt(sw.expiry_t)
            if std <= 1e-12:
                opt = max(fwd_rate - sw.strike, 0.0) if sw.is_payer else max(sw.strike - fwd_rate, 0.0)
            else:
                d = (fwd_rate - sw.strike) / std
                phi_d = 0.3989422804014327 * math.exp(-0.5 * d * d)
                if sw.is_payer:
                    opt = (fwd_rate - sw.strike) * _norm_cdf(d) + std * phi_d
                else:
                    opt = (sw.strike - fwd_rate) * _norm_cdf(-d) + std * phi_d
            out[(scen, sw.trade_id)] = sw.pay_receive * sw.notional * eff_ann * opt
    return out


def _swaption_tables_to_arrow(
    swaptions: Sequence[SwaptionRow],
    vol_points: Sequence[VolPoint],
) -> tuple[pa.Table, pa.Table]:
    swp_tbl = pa.table({
        "TRADE_ID": pa.array([s.trade_id for s in swaptions], type=pa.string()),
        "CCY": pa.array([s.ccy for s in swaptions], type=pa.string()),
        "DISCOUNT_CURVE_ID": pa.array([s.discount_curve_id for s in swaptions], type=pa.string()),
        "PROJ_CURVE_ID": pa.array([s.proj_curve_id for s in swaptions], type=pa.string()),
        "VOL_SURFACE_ID": pa.array([s.vol_surface_id for s in swaptions], type=pa.string()),
        "EXPIRY_T": pa.array([s.expiry_t for s in swaptions], type=pa.float64()),
        "SETTLE_T": pa.array([s.settle_t for s in swaptions], type=pa.float64()),
        "SWAP_TENOR_T": pa.array([s.swap_tenor_t for s in swaptions], type=pa.float64()),
        "CASH_FREQ": pa.array([s.cash_freq for s in swaptions], type=pa.float64()),
        "N_PERIODS": pa.array([s.n_periods for s in swaptions], type=pa.float64()),
        "SETTLEMENT_TYPE": pa.array([s.settlement_type for s in swaptions], type=pa.string()),
        "STRIKE": pa.array([s.strike for s in swaptions], type=pa.float64()),
        "NOTIONAL": pa.array([s.notional for s in swaptions], type=pa.float64()),
        "IS_PAYER": pa.array([s.is_payer for s in swaptions], type=pa.bool_()),
        "PAY_RECEIVE": pa.array([s.pay_receive for s in swaptions], type=pa.float64()),
    })
    vol_tbl = pa.table({
        "SCENARIO_ID": pa.array([v.scenario_id for v in vol_points], type=pa.string()),
        "VOL_SURFACE_ID": pa.array([v.vol_surface_id for v in vol_points], type=pa.string()),
        "POINT_IDX": pa.array([v.point_idx for v in vol_points], type=pa.int16()),
        "EXPIRY_T": pa.array([v.expiry_t for v in vol_points], type=pa.float64()),
        "SWAP_TENOR_T": pa.array([v.swap_tenor_t for v in vol_points], type=pa.float64()),
        "VOL": pa.array([v.vol for v in vol_points], type=pa.float64()),
        "BETA": pa.array([v.beta for v in vol_points], type=pa.float64()),
        "RHO": pa.array([v.rho for v in vol_points], type=pa.float64()),
        "NU": pa.array([v.nu for v in vol_points], type=pa.float64()),
    })
    return swp_tbl, vol_tbl


def run_pure_swaption_pricing_duckdb(
    knots: Sequence[CurveKnot],
    cashflows: Sequence[CashflowRow],
    swaptions: Sequence[SwaptionRow],
    vol_points: Sequence[VolPoint],
    projects_dir: Path | None = None,
    mode: str = "function",
    vol_model: str = "sabr",
) -> dict[tuple[str, str], float]:
    """Execute projects/pricing swaption pricer ('function' or 'row_sum') on DuckDB."""
    if projects_dir is None:
        projects_dir = Path(__file__).resolve().parents[2] / "projects" / "pricing"
    if vol_model not in ("sabr", "flat_normal", "flat_black"):
        raise ValueError(f"Unknown vol_model: {vol_model!r}")
    knot_tbl, cf_tbl = _tables_to_arrow(knots, cashflows)
    swp_tbl, vol_tbl = _swaption_tables_to_arrow(swaptions, vol_points)

    conn = duckdb.connect(":memory:")
    try:
        conn.register("_arrow_knots", knot_tbl)
        conn.register("_arrow_cfs", cf_tbl)
        conn.register("_arrow_swps", swp_tbl)
        conn.register("_arrow_vols", vol_tbl)
        conn.execute("CREATE TABLE PR_CURVE_KNOT AS SELECT * FROM _arrow_knots")
        conn.execute("CREATE TABLE PR_CASHFLOW AS SELECT * FROM _arrow_cfs")
        conn.execute("CREATE TABLE PR_SWAPTION AS SELECT * FROM _arrow_swps")
        conn.execute("CREATE TABLE PR_VOL_POINT AS SELECT * FROM _arrow_vols")

        frames = ll.Frames(conn)
        model_text = _load_pricing_pure_sources(projects_dir)
        vol_fn = "pricing::sabrNormalVolInterpolate" if vol_model == "sabr" else "pricing::flatNormalVolInterpolate"
        base_fn = "pricing::swaptionPvs" if mode == "function" else "pricing::swaptionPvsRowSum"
        fn = f"{base_fn}({{t | pricing::curveDiscountFactors($t)}}, {{v | {vol_fn}($v)}})"
        with ll.HeldModel(model_text) as held:
            query = f"{fn}->sort([~scenarioId->ascending(), ~tradeId->ascending()])"
            plan = held.plan(ll.parse(query), "pricing::DuckDbRuntime")
            out = frames.run(plan)
    finally:
        conn.close()

    return {
        (row["scenarioId"], row["tradeId"]): float(row["pv"])
        for row in out.to_pylist()
    }


def run_pure_dv01_duckdb(
    knots: Sequence[CurveKnot],
    cashflows: Sequence[CashflowRow],
    swaptions: Sequence[SwaptionRow] = (),
    vol_points: Sequence[VolPoint] = (),
    projects_dir: Path | None = None,
    kind: str = "cashflow",
    vol_model: str = "sabr",
) -> dict[tuple[str, str, int], float]:
    """Execute AD pillar DV01s ('cashflow' or 'swaption') on DuckDB for 'BASE' knots."""
    if projects_dir is None:
        projects_dir = Path(__file__).resolve().parents[2] / "projects" / "pricing"
    if kind not in ("cashflow", "swaption"):
        raise ValueError(f"Unknown DV01 kind: {kind!r}")
    if vol_model not in ("sabr", "flat_normal", "flat_black"):
        raise ValueError(f"Unknown vol_model: {vol_model!r}")
    knot_tbl, cf_tbl = _tables_to_arrow(knots, cashflows)
    swp_tbl, vol_tbl = _swaption_tables_to_arrow(swaptions, vol_points)

    conn = duckdb.connect(":memory:")
    try:
        conn.register("_arrow_knots", knot_tbl)
        conn.register("_arrow_cfs", cf_tbl)
        conn.register("_arrow_swps", swp_tbl)
        conn.register("_arrow_vols", vol_tbl)
        conn.execute("CREATE TABLE PR_CURVE_KNOT AS SELECT * FROM _arrow_knots")
        conn.execute("CREATE TABLE PR_CASHFLOW AS SELECT * FROM _arrow_cfs")
        conn.execute("CREATE TABLE PR_SWAPTION AS SELECT * FROM _arrow_swps")
        conn.execute("CREATE TABLE PR_VOL_POINT AS SELECT * FROM _arrow_vols")

        frames = ll.Frames(conn)
        model_text = _load_pricing_pure_sources(projects_dir)
        if kind == "cashflow":
            fn = "pricing::cashflowDv01s()"
        else:
            vs = "pricing::sabrNormalVolWithDerivs" if vol_model == "sabr" else "pricing::flatNormalVolWithDerivs"
            fn = (
                "pricing::swaptionDv01s("
                "{t | pricing::curveDfsWithDeriv($t)}, "
                "{v | pricing::volInterpolateWithWeights($v)}, "
                f"{{s | {vs}($s)}})"
            )
        with ll.HeldModel(model_text) as held:
            query = (
                f"{fn}->sort([~tradeId->ascending(), ~curveId->ascending(), "
                "~knotIdx->ascending()])"
            )
            plan = held.plan(ll.parse(query), "pricing::DuckDbRuntime")
            out = frames.run(plan)
    finally:
        conn.close()

    return {
        (row["tradeId"], row["curveId"], int(row["knotIdx"])): float(row["dv01"])
        for row in out.to_pylist()
    }


def run_pure_vega_duckdb(
    knots: Sequence[CurveKnot],
    cashflows: Sequence[CashflowRow],
    swaptions: Sequence[SwaptionRow],
    vol_points: Sequence[VolPoint],
    projects_dir: Path | None = None,
    vol_model: str = "sabr",
) -> dict[tuple[str, str, int], float]:
    """Execute 1-pass AD surface vegas (1bp vol) on DuckDB for 'BASE'."""
    if projects_dir is None:
        projects_dir = Path(__file__).resolve().parents[2] / "projects" / "pricing"
    if vol_model not in ("sabr", "flat_normal", "flat_black"):
        raise ValueError(f"Unknown vol_model: {vol_model!r}")
    knot_tbl, cf_tbl = _tables_to_arrow(knots, cashflows)
    swp_tbl, vol_tbl = _swaption_tables_to_arrow(swaptions, vol_points)

    conn = duckdb.connect(":memory:")
    try:
        conn.register("_arrow_knots", knot_tbl)
        conn.register("_arrow_cfs", cf_tbl)
        conn.register("_arrow_swps", swp_tbl)
        conn.register("_arrow_vols", vol_tbl)
        conn.execute("CREATE TABLE PR_CURVE_KNOT AS SELECT * FROM _arrow_knots")
        conn.execute("CREATE TABLE PR_CASHFLOW AS SELECT * FROM _arrow_cfs")
        conn.execute("CREATE TABLE PR_SWAPTION AS SELECT * FROM _arrow_swps")
        conn.execute("CREATE TABLE PR_VOL_POINT AS SELECT * FROM _arrow_vols")

        frames = ll.Frames(conn)
        model_text = _load_pricing_pure_sources(projects_dir)
        vs = "pricing::sabrNormalVolWithDerivs" if vol_model == "sabr" else "pricing::flatNormalVolWithDerivs"
        with ll.HeldModel(model_text) as held:
            query = (
                "pricing::swaptionVegas("
                "{t | pricing::curveDiscountFactors($t)}, "
                "{v | pricing::volInterpolateWithWeights($v)}, "
                f"{{s | {vs}($s)}})"
                "->sort([~tradeId->ascending(), ~volSurfaceId->ascending(), ~pointIdx->ascending()])"
            )
            plan = held.plan(ll.parse(query), "pricing::DuckDbRuntime")
            out = frames.run(plan)
    finally:
        conn.close()

    return {
        (row["tradeId"], row["volSurfaceId"], int(row["pointIdx"])): float(row["vega"])
        for row in out.to_pylist()
    }
