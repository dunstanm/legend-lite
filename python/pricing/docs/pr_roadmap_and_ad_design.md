# Layered PR Plan & AD Formalization

## Layered PR Sequence

1. **PR 1 — `erf` native (`marrid_erf` rebased on `main`)**
   - `erf` register row, Postgres 16+/ClickHouse native, Cody formula elsewhere, PCT cases, dialect refusals, and `//gates:local`.
2. **PR 2 — `projects/pricing` v1: Linear Rates & Scenario Bump Risk**
   - `PrCurveKnot`, `PrCashflow` (dual-curve `discountCurveId` / `projCurveId`, IMM FRAs, OIS, XCCY notional exchanges), pluggable `curveModel` (`evalSegmentDf`), `cashflowPvs`, and scenario-join bump DV01. Zero core compiler changes.
3. **PR 3 — Swaptions & Pluggable Vol Models**
   - `normCdf` (built on `erf`), `PrSwaption`, `PrVolPoint`, `bachelierSwaptionPv` (Normal model supporting negative rates), `cashSettledAnnuity`, `sabrNormalVol` (Hagan Normal SABR, `beta=1.0` default, `beta=0.0` for negative rates), `flatNormalVolInterpolate`, and pluggable `curveModel` / `volModel` `Function` parameters.
4. **PR 4a — General Compiler `CommonSubexpressions`**
   - Extend `CommonSubexpressions` beyond SQLite/BigQuery so repeated subexpressions shrink generated SQL across all dialects.
5. **PR 4b — `:diff` Design Doc, Ruling & Language PR**
   - Dedicated compiler design & ruling for `:diff`, followed by `cashflowDv01s`, `swaptionDv01s`, and `swaptionVegas` (demonstrating sparse 10×10 surface vega and SABR smile-delta vs. 200 bump scenarios).
6. **PR 5 — In-DB `fold` Calibration & Thin Typed Python Wrapper**
   - `fold`-based bootstrap / fixed-sweep calibration and generated typed Python wrappers over `projects/pricing` (keeping the Python float oracle test-only).

## Formalizing Backprop Across Pipeline Frames (TensorFlow / JAX Analogy)

Scalar `:diff` differentiates a single Pure expression (`Float -> Float`). Extending it to a full `Market Quotes → Curve/Vol Fit → Portfolio PV` pipeline maps cleanly to Vector-Jacobian Products (VJPs) over relational stages:

| Forward Stage | ML / TensorFlow Analogue | Relational Adjoint (VJP) |
|---|---|---|
| `extend` (scalar pricing / vol formula) | Elementwise `map` | Row-wise Jacobian-transpose multiply via scalar `:diff` |
| `groupBy` + `sum` (cashflows → trade → portfolio) | `reduce_sum` | Broadcast / `join` upstream adjoint back to input rows |
| `join` / lookup (`PrCashflow` → `PrCurveKnot`, `PrSwaption` → `PrVolPoint`) | `gather` | `groupBy` + `sum` (`scatter_add`) onto market keys (yields sparse vega) |
| Calibration / root-solve (`fold`, Newton polish) | `custom_vjp` / `custom_root` | Implicit Function Theorem on the zero-residual identity ($J^T \bar{q} = \bar{r}$) rather than unrolling solver iterations |
