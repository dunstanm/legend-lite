# Multi-Curve Fitting, IMM/XCCY Skinny Table, and `fold` in SQL

## 1. Post-LIBOR Multi-Curve Structure

In post-LIBOR OIS markets, each currency's local OIS swaps use the same curve for discounting and projection (`USD_OIS`, `EUR_OIS`), while cross-currency basis swaps link collateralized discounting to foreign OIS. The forward fit is therefore **triangular (a DAG)** rather than circular:
1. Fit `USD_OIS` from USD IMM 3M FRAs + USD OIS swaps.
2. Fit `EUR_OIS` from EUR IMM 3M FRAs + EUR OIS swaps.
3. Fit `EUR_DISC_USD` from EUR/USD XCCY basis swaps, holding `USD_OIS` and `EUR_OIS` fixed.

Even with overlapping IMM dates and OIS tenors, each curve fits independently in topological order (`rack.py`: `fit_curve_from_cashflows`).

## 2. Single Skinny `PR_CASHFLOW` Table for All Linear Instruments

All linear instruments—OIS swaps, IMM-date 3M FRAs, and XCCY basis swaps—expand at SOD/booking time into the same `PR_CASHFLOW` schema (`projects/pricing/store.pure`):
- **IMM 3M FRAs** (`FraSpec` + `imm_date` / `next_imm_dates`): 1 fixed + 1 float row.
- **OIS / IRS swaps** (`SwapSpec`): $N$ fixed + $N$ float coupon rows, with separate `DISCOUNT_CURVE_ID` and `PROJ_CURVE_ID` for dual-curve support.
- **XCCY basis swaps** (`XccyBasisSwapSpec`): float-for-float coupons plus initial ($t_0$) and final ($t_{\text{mat}}$) notional exchanges represented as fixed rows with `DCF = 1.0`, `FIXED_RATE = 1.0`, `IS_FLOAT = false`, scaled to reporting currency via `fx_spot`.

Dates and holiday calendars stay in the Python SOD/booking layer (`rack.py`), while Pure pricing operates cleanly on year fractions (`t`, `dcf`).

## 3. Pure `fold` in SQL (`FoldToListReduce` / `FoldsAsRecursiveCtes`)

In Pure, `xs->fold({x, acc | body}, init)` is a fixed-step sequential reduction over an ordered list `xs`:
- **DuckDB** (`FoldToListReduce.java`): compiles to `list_reduce(...)`.
- **Recursive-CTE dialects such as BigQuery** (`FoldsAsRecursiveCtes.java`): compiles closed folds to a top-level `WITH RECURSIVE` CTE over `0 .. length(xs)`.

**Key insight:** `fold` semantics do **not** restrict `acc` to a single scalar. `acc` can be a `List<Float>` holding all solved discount factors or an entire multi-parameter knot vector `[r0, r1, ...]` updated on every step:
- `pricing::foldBootstrapOis`: sequential OIS par-swap bootstrap using a growing `List<Float>` accumulator.
- `pricing::foldMultiParamSweep`: fixed-sweep Gauss-Seidel/Newton update over a multi-parameter vector `[r0, r1]`.

Both compile and execute cleanly on DuckDB (`test_pure_fold_bootstrap_and_multi_param_sweep_duckdb`). Coupled global splines / $N \times N$ Jacobian solves remain best suited to the external Python fitter (`fit_curve_from_cashflows`) writing `PR_CURVE_KNOT` rows, while `fold` works well for triangular in-DB bootstraps and fixed-sweep updates.
