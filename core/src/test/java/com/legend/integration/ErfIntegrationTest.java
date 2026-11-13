// Copyright 2026 Legend Contributors
// SPDX-License-Identifier: Apache-2.0

package com.legend.integration;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.sql.SQLException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * End-to-end execution tests for {@code meta::legend::lite::erf} across DuckDB, H2, and SQLite.
 */
public class ErfIntegrationTest extends AbstractDatabaseTest {

    @BeforeEach
    void setUp() throws SQLException {
        connection = backend().open();
        setupDatabase();
    }

    @AfterEach
    void tearDown() throws SQLException {
        if (connection != null) {
            connection.close();
        }
    }

    @Test
    void testErfScalarLiteralsAndIntegerInput() throws SQLException {
        assertEquals(0.0, ((Number) executeRelation("|erf(0.0)").rows().get(0).get(0)).doubleValue(), 1e-15);
        assertEquals(0.5204998778130465, ((Number) executeRelation("|erf(0.5)").rows().get(0).get(0)).doubleValue(), 1e-14);
        assertEquals(-0.5204998778130465, ((Number) executeRelation("|erf(-0.5)").rows().get(0).get(0)).doubleValue(), 1e-14);
        assertEquals(0.8427007929497149, ((Number) executeRelation("|erf(1)").rows().get(0).get(0)).doubleValue(), 1e-14);
        assertEquals(0.999593047982555, ((Number) executeRelation("|erf(2.5)").rows().get(0).get(0)).doubleValue(), 1e-14);
        assertEquals(1.0, ((Number) executeRelation("|erf(10.0)").rows().get(0).get(0)).doubleValue(), 1e-15);
        assertEquals(-1.0, ((Number) executeRelation("|erf(-30.0)").rows().get(0).get(0)).doubleValue(), 1e-15);
    }

    @Test
    void testErfOverTableColumnsAndCompoundExpressions() throws SQLException {
        var result = executeRelation(
                "|#>{store::PersonDatabase.T_PERSON}#->extend(~[x: r | ($r.AGE_VAL - 30) / 10.0, e: r | erf(($r.AGE_VAL - 30) / 10.0)])->select(~[ID, x, e])->sort(ascending(~ID))");
        assertTrue(result.rows().size() > 0);
        for (var row : result.rows()) {
            double x = ((Number) row.get(1)).doubleValue();
            double actual = ((Number) row.get(2)).doubleValue();
            double expected = erfRef(x);
            assertEquals(expected, actual, 1e-13, "erf(" + x + ")");
        }
    }

    /** Reference erf(x) via Abramowitz & Stegun / numerical check only for small test points, or exact known values. */
    private static double erfRef(double x) {
        // Composite Simpson's rule over [0, x] for reference verification in test
        int n = 10000;
        double h = x / n;
        double sum = 1.0 + Math.exp(-x * x);
        for (int i = 1; i < n; i++) {
            double t = i * h;
            sum += (i % 2 == 0 ? 2.0 : 4.0) * Math.exp(-t * t);
        }
        return (2.0 / Math.sqrt(Math.PI)) * (h / 3.0) * sum;
    }
}
