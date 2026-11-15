# pricing

Layer 0, no dependencies. Rates curve knot data, normalized swap/FRA cashflow rows, and
interpolation-independent Pure pricing functions (`discountFactor`, `forwardRate`, `cashflowPv`).

## Exports

| element | kind | note |
| --- | --- | --- |
| `pricing::PrCurveKnot` | class | one row per curve knot `(scenarioId, curveId, knotIdx)` |
| `pricing::PrCashflow` | class | one row per schedule-adjusted cashflow `(tradeId, legId, cfIdx)` |
| `pricing::PrSwaption` | class | one row per European swaption `(tradeId)` |
| `pricing::PrVolPoint` | class | one row per SABR vol surface grid point `(scenarioId, volSurfaceId, pointIdx)` (`vol`=alpha, `beta`, `rho`, `nu`) |
| `pricing::evalSegmentDf` | function | `(t, t0, r0, t1, r1, method): Float[1]` — interpolates `LINEAR_ZERO`, `LOG_LINEAR_DF`, or `FLAT_FORWARD` |
| `pricing::curveDiscountFactors` | function | `(targets: Relation<(scenarioId, curveId, t)>[1]): Relation<(scenarioId, curveId, t, df)>[1]` — interpolates from `PrCurveKnot` |
| `pricing::cashflowPvs` | function | `(curveModel): Relation<...>[1]` (+ 0-arg default) — prices linear `PrCashflow` rows across scenarios |
| `pricing::normCdf` | function | `(x: Float[1]): Float[1]` — standard normal CDF via `erf` |
| `pricing::bachelierSwaptionPv` | function | `(annuity, fwdRate, strike, expiryT, vol, notional, isPayer, payReceive): Float[1]` — Normal (Bachelier) swaption PV |
| `pricing::volInterpolate` | function | `(targets): Relation<(scenarioId, volSurfaceId, expiryT, swapTenorT, vol)>[1]` — bilinear interpolation on `PrVolPoint` |
| `pricing::flatNormalVolKernel` | function | `(fwdRate, strike, expiryT, alpha, beta, rho, nu): Float[1]` — flat Normal kernel (returns `alpha`) |
| `pricing::flatNormalVolInterpolate` | function | `(targets): Relation<...>[1]` — pluggable flat Normal vol surface evaluator |
| `pricing::flatNormalVolWithDerivs` | function | `(targets): Relation<(tradeId, vol, dVolDfwd, dVolDalpha)>[1]` — flat Normal + `:diff` derivatives |
| `pricing::sabrNormalVol` | function | `(fwdRate, strike, expiryT, alpha, beta, rho, nu): Float[1]` — Hagan (2002) Normal SABR implied volatility (supports negative rates at `beta=0`, lognormal backbone at `beta=1`) |
| `pricing::sabrNormalVolInterpolate` | function | `(targets): Relation<...>[1]` — pluggable Normal SABR vol surface evaluator |
| `pricing::sabrNormalVolWithDerivs` | function | `(targets): Relation<(tradeId, vol, dVolDfwd, dVolDalpha)>[1]` — Normal SABR + `:diff` derivatives |
| `pricing::cashSettledAnnuity` | function | `(rate, nPeriods, freq): Float[1]` — closed-form geometric series for cash-settled annuity |
| `pricing::swaptionPvs` | function | `(curveModel, volModel): Relation<...>[1]` (+ 0-arg default) — Pure-as-function (`cashSettledAnnuity` for `CASH`) |
| `pricing::swaptionPvsRowSum` | function | `(curveModel, volModel): Relation<...>[1]` (+ 0-arg default) — Pure-as-compilation (`groupBy` row sum) |
| `pricing::evalSegmentDfDr0` | function | `(t, t0, r0, t1, r1, method): Float[1]` — scalar derivative `d(df)/d(r0)` |
| `pricing::evalSegmentDfDr1` | function | `(t, t0, r0, t1, r1, method): Float[1]` — scalar derivative `d(df)/d(r1)` |
| `pricing::curveDfsWithDeriv` | function | `(targets): Relation<(scenarioId, curveId, t, df, k0Idx, dDfDr0, k1Idx, dDfDr1)>[1]` — DF + knot sensitivities |
| `pricing::cashflowDv01s` | function | `(): Relation<(tradeId, curveId, knotIdx, dv01)>[1]` — 1-pass AD pillar DV01s for linear `PrCashflow`s on `'BASE'` |
| `pricing::cashSettledAnnuityDr` | function | `(rate, nPeriods, freq): Float[1]` — scalar derivative `d(annuity)/d(rate)` |
| `pricing::swaptionDv01s` | function | `(curveDerivModel, volParamModel, volSmileModel): Relation<...>[1]` (+ 0-arg default) — 1-pass AD pillar DV01s |
| `pricing::volInterpolateWithWeights` | function | `(targets): Relation<...>[1]` — bilinear vol interpolation + 4 corner `pointIdx`s & weights |
| `pricing::bachelierSwaptionVega` | function | `(annuity, fwdRate, strike, expiryT, vol, notional, payReceive): Float[1]` — scalar `dPV/d(normalVol)` |
| `pricing::swaptionVegas` | function | `(curveModel, volParamModel, volSmileModel): Relation<...>[1]` (+ 0-arg default) — sparse 1-pass AD surface vegas |
| `pricing::Store` | store | tables `PR_CURVE_KNOT`, `PR_CASHFLOW`, `PR_SWAPTION`, `PR_VOL_POINT` |
| `pricing::Mapping` | mapping | sets `prCurveKnot`, `prCashflow`, `prSwaption`, `prVolPoint` |
