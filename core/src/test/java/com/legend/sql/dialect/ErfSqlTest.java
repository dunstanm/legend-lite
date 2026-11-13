// Copyright 2026 Legend Contributors
// SPDX-License-Identifier: Apache-2.0

package com.legend.sql.dialect;

import com.legend.sql.OutputCol;
import com.legend.sql.SqlExpr;
import com.legend.sql.SqlFn;
import com.legend.sql.SqlSelect;
import com.legend.sql.SqlSource;
import com.legend.sql.SqlType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies SQL rendering of {@link SqlFn#ERF} across dialects:
 * native {@code erf(...)} on Postgres and ClickHouse, IR {@code list_transform}/{@code list_extract}
 * binding on DuckDB, scalar-subquery binding on SQLite, and Cody expansion on other dialects.
 */
class ErfSqlTest {

    private static String renderSelect(AnsiSqlRenderer dialect, SqlExpr expr) {
        SqlSelect select = new SqlSelect(
                List.of(new SqlSelect.Projection(expr, "v", new OutputCol("v", SqlType.Scalar.DOUBLE, true))),
                false,
                new SqlSource.Dual(),
                null, List.of(), null, null, List.of(), null, null, List.of());
        return dialect.render(select);
    }

    private static int countOccurrences(String haystack, String needle) {
        int count = 0;
        int idx = 0;
        while ((idx = haystack.indexOf(needle, idx)) != -1) {
            count++;
            idx += needle.length();
        }
        return count;
    }

    @Test
    void postgresUsesNativeErf() {
        SqlExpr arg = SqlExpr.Call.of(SqlFn.PLUS,
                SqlExpr.Column.of("t", "x", SqlType.Scalar.DOUBLE, true, OutputCol.Origin.DERIVED),
                new SqlExpr.FloatLit(0.5));
        String sql = renderSelect(new Postgres(), SqlExpr.Call.of(SqlFn.ERF, arg));
        assertTrue(sql.contains("erf("), "Expected native erf() in Postgres SQL: " + sql);
        assertFalse(sql.contains("0.46875"), "Postgres should not expand Cody polynomial: " + sql);
    }

    @Test
    void clickHouseUsesNativeErf() {
        SqlExpr arg = SqlExpr.Column.of("t", "x", SqlType.Scalar.DOUBLE, true, OutputCol.Origin.DERIVED);
        String sql = renderSelect(new ClickHouse(), SqlExpr.Call.of(SqlFn.ERF, arg));
        assertTrue(sql.contains("erf("), "Expected native erf() in ClickHouse SQL: " + sql);
        assertFalse(sql.contains("0.46875"), "ClickHouse should not expand Cody polynomial: " + sql);
    }

    @Test
    void duckDbBindsCompoundArgViaListLambdaIr() {
        SqlExpr compound = SqlExpr.Call.of(SqlFn.PLUS,
                SqlExpr.Column.of("t", "x", SqlType.Scalar.DOUBLE, true, OutputCol.Origin.DERIVED),
                new SqlExpr.FloatLit(12.345));
        String sql = renderSelect(new DuckDb(), SqlExpr.Call.of(SqlFn.ERF, compound));
        assertTrue(sql.contains("list_transform("), "Expected list_transform IR in DuckDB SQL: " + sql);
        assertTrue(sql.contains("list_extract("), "Expected list_extract IR in DuckDB SQL: " + sql);
        assertTrue(sql.contains("__erf_x ->"), "Expected lambda param in DuckDB SQL: " + sql);
        assertEquals(1, countOccurrences(sql, "12.345"),
                "Compound argument should be evaluated once inside list_transform: " + sql);
    }

    @Test
    void duckDbExpandsSimpleArgDirectly() {
        SqlExpr simple = new SqlExpr.FloatLit(0.5);
        String sql = renderSelect(new DuckDb(), SqlExpr.Call.of(SqlFn.ERF, simple));
        assertFalse(sql.contains("list_transform("), "Simple literal needs no list_transform: " + sql);
        assertTrue(sql.contains("0.46875"), "Expected Cody expansion in DuckDB SQL: " + sql);
    }

    @Test
    void sqliteBindsCompoundArgViaScalarSubquery() {
        SqlExpr compound = SqlExpr.Call.of(SqlFn.PLUS,
                SqlExpr.Column.of("t", "x", SqlType.Scalar.DOUBLE, true, OutputCol.Origin.DERIVED),
                new SqlExpr.FloatLit(12.345));
        String sql = renderSelect(new SQLite(), SqlExpr.Call.of(SqlFn.ERF, compound));
        assertEquals(1, countOccurrences(sql, "12.345"),
                "Compound argument should be evaluated once in SQLite scalar subquery: " + sql);
    }

    @Test
    void databricksUsesCodyExpansionNotNativeErf() {
        SqlExpr arg = new SqlExpr.FloatLit(0.5);
        String sql = renderSelect(new Databricks(), SqlExpr.Call.of(SqlFn.ERF, arg));
        assertTrue(sql.contains("0.46875"), "Databricks should use Cody expansion: " + sql);
        assertFalse(sql.contains("erf("), "Databricks should not emit unverified native erf(): " + sql);
    }
}
