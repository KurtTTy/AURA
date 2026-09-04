import pytest
from app.analyst.tools import _is_safe_select


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "SELECT 1; DROP TABLE sales",
    "WITH x AS (SELECT 1) DELETE FROM sales",
    "INSERT INTO sales VALUES (1)",
    "ATTACH 'evil.db' AS e",
    "COPY sales TO 'out.csv'",
    "",
    "   ",
])
def test_rejects_dangerous_sql(sql):
    allowed, reason = _is_safe_select(sql)
    assert not allowed
    assert reason            # must explain itself to the model


@pytest.mark.parametrize("sql", [
    "SELECT * FROM sales",
    "select region, avg(price) from sales group by region",
    "WITH t AS (SELECT * FROM sales) SELECT COUNT(*) FROM t",
    "SELECT last_update FROM sales",      # 'update' inside a column name
])
def test_allows_real_queries(sql):
    allowed, reason = _is_safe_select(sql)
    assert allowed, reason
