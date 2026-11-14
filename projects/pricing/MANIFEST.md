# pricing

Layer 0, no dependencies. Rates curve knot data, normalized swap/FRA cashflow rows, and
interpolation-independent Pure pricing functions (`discountFactor`, `forwardRate`, `cashflowPv`).

## Exports

| element | kind | note |
| --- | --- | --- |
| `pricing::PrCurveKnot` | class | one row per curve knot `(scenarioId, curveId, knotIdx)` |
| `pricing::PrCashflow` | class | one row per schedule-adjusted cashflow `(tradeId, legId, cfIdx)` |
| `pricing::evalSegmentDf` | function | `(t, t0, r0, t1, r1, method): Float[1]` — interpolates `LINEAR_ZERO`, `LOG_LINEAR_DF`, or `FLAT_FORWARD` |
| `pricing::curveDiscountFactors` | function | `(targets: Relation<(scenarioId, curveId, t)>[1]): Relation<(scenarioId, curveId, t, df)>[1]` — interpolates from `PrCurveKnot` |
| `pricing::cashflowPvs` | function | `(): Relation<(scenarioId, tradeId, legId, cfIdx, instrumentType, currency, pv)>[1]` — prices `PrCashflow` across scenarios |
| `pricing::Store` | store | tables `PR_CURVE_KNOT`, `PR_CASHFLOW` |
| `pricing::Mapping` | mapping | sets `prCurveKnot`, `prCashflow` |
