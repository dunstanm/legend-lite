# pricing

Layer 0, no dependencies. Rates curve knot data, normalized swap/FRA cashflow rows, and
interpolation-independent Pure pricing functions (`discountFactor`, `forwardRate`, `cashflowPv`).

## Exports

| element | kind | note |
| --- | --- | --- |
| `pricing::PrCurveKnot` | class | one row per curve knot `(scenarioId, curveId, knotIdx)` |
| `pricing::PrCashflow` | class | one row per schedule-adjusted cashflow `(tradeId, legId, cfIdx)` |
| `pricing::PrSwaption` | class | one row per European swaption `(tradeId)` |
| `pricing::PrVolPoint` | class | one row per vol surface grid point `(scenarioId, volSurfaceId, pointIdx)` |
| `pricing::evalSegmentDf` | function | `(t, t0, r0, t1, r1, method): Float[1]` — interpolates `LINEAR_ZERO`, `LOG_LINEAR_DF`, or `FLAT_FORWARD` |
| `pricing::curveDiscountFactors` | function | `(targets: Relation<(scenarioId, curveId, t)>[1]): Relation<(scenarioId, curveId, t, df)>[1]` — interpolates from `PrCurveKnot` |
| `pricing::cashflowPvs` | function | `(): Relation<(scenarioId, tradeId, legId, cfIdx, instrumentType, currency, pv)>[1]` — prices linear `PrCashflow` rows across scenarios |
| `pricing::normCdf` | function | `(x: Float[1]): Float[1]` — standard normal CDF via `erf` |
| `pricing::black76SwaptionPv` | function | `(annuity, fwdRate, strike, expiryT, vol, notional, isPayer, payReceive): Float[1]` |
| `pricing::volInterpolate` | function | `(targets): Relation<(scenarioId, volSurfaceId, expiryT, swapTenorT, vol)>[1]` — bilinear interpolation on `PrVolPoint` |
| `pricing::cashSettledAnnuity` | function | `(rate, nPeriods, freq): Float[1]` — closed-form geometric series for cash-settled annuity |
| `pricing::swaptionPvs` | function | `(): Relation<...>[1]` — Pure-as-function (`cashSettledAnnuity` for `CASH`, physical leg sum for `PHYSICAL`) |
| `pricing::swaptionPvsRowSum` | function | `(): Relation<...>[1]` — Pure-as-compilation (`groupBy` row sum over `PR_CASHFLOW` for `CASH` and `PHYSICAL`) |
| `pricing::Store` | store | tables `PR_CURVE_KNOT`, `PR_CASHFLOW`, `PR_SWAPTION`, `PR_VOL_POINT` |
| `pricing::Mapping` | mapping | sets `prCurveKnot`, `prCashflow`, `prSwaption`, `prVolPoint` |
