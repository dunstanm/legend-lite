package com.legend.integration;

import com.legend.testing.LaneFloor;
import com.legend.testing.Lean;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end execution test for projects/pricing:
 *  - Independent PR_CURVE_KNOT (row per knot) and PR_CASHFLOW (swap/FRA rack) tables
 *  - Interpolation-independent pricing::cashflowPv calling pricing::discountFactor & forwardRate
 *  - Verifies LINEAR_ZERO, LOG_LINEAR_DF, and FLAT_FORWARD against exact analytical values on DuckDB.
 */
@org.junit.jupiter.api.extension.ExtendWith(LaneFloor.class)
public class PricingIntegrationTest {

    private static String loadProjectPure() throws Exception {
        Path root = Path.of("").toAbsolutePath();
        // Locate projects/pricing relative to Bazel runfiles or workspace root
        Path pricingDir = root.resolve("projects/pricing");
        if (!Files.isDirectory(pricingDir)) {
            pricingDir = Path.of("/home/dunst/wk/legend-lite/projects/pricing");
        }
        String model = Files.readString(pricingDir.resolve("model.pure"));
        String store = Files.readString(pricingDir.resolve("store.pure"));
        String mapping = Files.readString(pricingDir.resolve("mapping.pure"));
        String runtime = """
                ###Connection
                import pricing::*;
                RelationalDatabaseConnection pricing::TestConn
                {
                   type: DuckDB;
                   specification: DuckDB { };
                   auth: Test;
                }
                ###Runtime
                import pricing::*;
                Runtime pricing::TestRuntime
                {
                   mappings: [ pricing::Mapping ];
                   connections: [ pricing::Store: [ environment: pricing::TestConn ] ];
                }
                """;
        return model + "\n" + store + "\n" + mapping + "\n" + runtime;
    }

    @Test
    void testSwapAndFraPricingAcrossInterpolationMethods() throws Exception {
        String fullModel = loadProjectPure();
        try (Connection conn = DriverManager.getConnection("jdbc:duckdb:");
             Statement st = conn.createStatement()) {

            st.execute("""
                    CREATE TABLE PR_CURVE_KNOT (
                        SCENARIO_ID VARCHAR(40),
                        CURVE_ID VARCHAR(40),
                        KNOT_IDX SMALLINT,
                        KNOT_T DOUBLE,
                        ZERO_RATE DOUBLE,
                        INTERP_METHOD VARCHAR(20)
                    )
                    """);
            st.execute("""
                    CREATE TABLE PR_CASHFLOW (
                        TRADE_ID VARCHAR(40),
                        LEG_ID VARCHAR(20),
                        CF_IDX SMALLINT,
                        INSTRUMENT_TYPE VARCHAR(20),
                        CCY CHAR(3),
                        DISCOUNT_CURVE_ID VARCHAR(40),
                        PROJ_CURVE_ID VARCHAR(40),
                        ACCRUAL_START_T DOUBLE,
                        ACCRUAL_END_T DOUBLE,
                        PAY_T DOUBLE,
                        DCF DOUBLE,
                        NOTIONAL DOUBLE,
                        FIXED_RATE DOUBLE,
                        SPREAD DOUBLE,
                        IS_FLOAT BOOLEAN,
                        PAY_RECEIVE DOUBLE
                    )
                    """);

            // Insert 3 scenarios for the same 3-knot curve, each using a different interpolation method
            for (String[] scen : new String[][]{
                    {"BASE_LIN", "LINEAR_ZERO"},
                    {"BASE_LOGDF", "LOG_LINEAR_DF"},
                    {"BASE_FLATFWD", "FLAT_FORWARD"}
            }) {
                String sid = scen[0];
                String method = scen[1];
                st.execute(String.format(
                        "INSERT INTO PR_CURVE_KNOT VALUES " +
                                "('%s', 'USD_OIS', 0, 0.5, 0.040, '%s')," +
                                "('%s', 'USD_OIS', 1, 1.0, 0.045, '%s')," +
                                "('%s', 'USD_OIS', 2, 2.0, 0.050, '%s')",
                        sid, method, sid, method, sid, method));
            }

            // Insert a 1.5Y payer IRS (receive float, pay fixed) + a 0.75x1.25 FRA
            // Pay/accrual dates (0.75, 1.25, 1.5) intentionally fall BETWEEN knots (0.5, 1.0, 2.0)
            st.execute("""
                    INSERT INTO PR_CASHFLOW VALUES
                    ('IRS_1', 'FIXED', 0, 'IRS', 'USD', 'USD_OIS', 'USD_OIS', 0.0, 0.75, 0.75, 0.75, 1000000.0, 0.045, 0.0, false, -1.0),
                    ('IRS_1', 'FIXED', 1, 'IRS', 'USD', 'USD_OIS', 'USD_OIS', 0.75, 1.50, 1.50, 0.75, 1000000.0, 0.045, 0.0, false, -1.0),
                    ('IRS_1', 'FLOAT', 0, 'IRS', 'USD', 'USD_OIS', 'USD_OIS', 0.0, 0.75, 0.75, 0.75, 1000000.0, 0.0, 0.0, true, 1.0),
                    ('IRS_1', 'FLOAT', 1, 'IRS', 'USD', 'USD_OIS', 'USD_OIS', 0.75, 1.50, 1.50, 0.75, 1000000.0, 0.0, 0.0, true, 1.0),
                    ('FRA_1', 'SINGLE', 0, 'FRA', 'USD', 'USD_OIS', 'USD_OIS', 0.75, 1.25, 1.25, 0.50, 2000000.0, 0.0, 0.0, true, 1.0)
                    """);

            String query = """
                    pricing::cashflowPvs()
                      ->groupBy(~[scenarioId, tradeId], ~[npv: x | $x.pv: y | $y->sum()])
                      ->sort([~scenarioId->ascending(), ~tradeId->ascending()])
                    """;

            var result = Lean.run(fullModel, query, "pricing::TestRuntime", conn);
            assertEquals(6, result.rows().size(), "Expected 3 scenarios x 2 trades = 6 rows");

            // Exact analytical reference checks
            double[][] knots = {{0.5, 0.040}, {1.0, 0.045}, {2.0, 0.050}};
            for (int i = 0; i < result.rows().size(); i++) {
                var row = result.rows().get(i);
                String scen = (String) row.get(0);
                String trade = (String) row.get(1);
                double actual = ((Number) row.get(2)).doubleValue();
                String method = switch (scen) {
                    case "BASE_LIN" -> "LINEAR_ZERO";
                    case "BASE_LOGDF" -> "LOG_LINEAR_DF";
                    default -> "FLAT_FORWARD";
                };
                double expected = referenceTradeNpv(knots, method, trade);
                assertEquals(expected, actual, 1e-6, scen + " / " + trade);
            }
        }
    }

    private static double refDf(double[][] knots, String method, double t) {
        if (t <= 0.0) return 1.0;
        if (t <= knots[0][0]) return Math.exp(-knots[0][1] * t);
        if (t >= knots[knots.length - 1][0]) return Math.exp(-knots[knots.length - 1][1] * t);
        for (int i = 0; i < knots.length - 1; i++) {
            double t0 = knots[i][0], r0 = knots[i][1];
            double t1 = knots[i + 1][0], r1 = knots[i + 1][1];
            if (t >= t0 && t <= t1) {
                double w = (t - t0) / (t1 - t0);
                if ("LOG_LINEAR_DF".equals(method) || "FLAT_FORWARD".equals(method)) {
                    return Math.exp(-((1.0 - w) * (r0 * t0) + w * (r1 * t1)));
                }
                double r = (1.0 - w) * r0 + w * r1;
                return Math.exp(-r * t);
            }
        }
        throw new IllegalStateException();
    }

    private static double referenceTradeNpv(double[][] knots, String method, String trade) {
        if ("IRS_1".equals(trade)) {
            double pv1 = 1_000_000.0 * 0.75 * (((refDf(knots, method, 0.0) / refDf(knots, method, 0.75)) - 1.0) / 0.75 - 0.045) * refDf(knots, method, 0.75);
            double pv2 = 1_000_000.0 * 0.75 * (((refDf(knots, method, 0.75) / refDf(knots, method, 1.50)) - 1.0) / 0.75 - 0.045) * refDf(knots, method, 1.50);
            return pv1 + pv2;
        }
        // FRA_1
        double fwd = ((refDf(knots, method, 0.75) / refDf(knots, method, 1.25)) - 1.0) / 0.50;
        return 2_000_000.0 * 0.50 * fwd * refDf(knots, method, 1.25);
    }
}
