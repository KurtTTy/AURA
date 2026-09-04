from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

from .loader import DataSession

logger = logging.getLogger(__name__)

ALLOWED_PREFIXES = ("select", "with")
FORBIDDEN = frozenset({
    "drop", "delete", "insert", "update", "alter", "attach", "copy",
    "install", "load", "pragma", "export",
})
MAX_ROWS = 50
CHART_KINDS = frozenset({"bar", "line", "scatter"})


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    display: str | None = None
    artifact: Path | None = None

def _output_dir() -> Path:
    """Charts and exports. Read from settings rather than passed in: TOOLS
    has a fixed calling convention and the model must never choose a path."""
    destination = get_settings().processed_dir
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _slug(text: str, fallback: str = "result") -> str:
    """Filename-safe stem. Same job as table_name_from_path, different rules."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (cleaned[:60] or fallback)


def _write_xlsx(session: DataSession, sql: str) -> Path | None:
    """Save the query's FULL result for the user - re-run with no row cap.

    Returns None on failure; a missing spreadsheet must not turn a working
    answer into an error.
    """
    if not get_settings().analyst_autosave_results:
        return None
    try:
        import pandas as pd

        result = session.sql(sql)
        columns = [column[0] for column in result.description]
        frame = pd.DataFrame(result.fetchall(), columns=columns)
        # Hash the SQL so re-running the same query overwrites instead of
        # littering the folder with near-identical files.
        stem = f"{_slug('-'.join(columns), 'result')}-{abs(hash(sql)) % 10**8:08d}"
        path = _output_dir() / f"{stem}.xlsx"
        frame.to_excel(path, sheet_name="results", index=False)
        return path
    except Exception:
        logger.warning("Could not write .xlsx export", exc_info=True)
        return None


def _markdown_table(columns: list[str], rows: list[tuple]) -> list[str]:
    """Rows as a markdown table, one string per line.

    Markdown because models read it far more reliably than tuples. A "|"
    inside a value would shift every cell after it, so it is escaped.
    """

    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
    ]


def _is_safe_select(query: str) -> tuple[bool, str]:
    """Decide whether this SQL may run. Returns (allowed, reason).

    The reason goes back to the model, so it can correct itself.

    Keyword matching is on WHOLE WORDS: a substring check would reject a
    column named 'last_update' for containing 'update'.
    """
    sql = query.strip()
    if not sql:
        return False, "Query is empty."

    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    if ";" in sql:
        return False, "Multiple SQL statements are not allowed."

    first_word = re.match(r"\w+", sql)
    if not first_word or first_word.group(0).lower() not in ALLOWED_PREFIXES:
        return False, "Only SELECT or WITH queries are allowed."

    forbidden = set(re.findall(r"\b\w+\b", sql.lower())) & FORBIDDEN
    if forbidden:
        return False, f"Forbidden SQL keyword: {sorted(forbidden)[0]}."

    return True, ""
    


def run_sql(session: DataSession, query: str, max_rows: int = MAX_ROWS) -> ToolResult:
    """Execute model-written SQL, safely.

    NEVER RAISES. A failure returns ok=False carrying the error, because
    the model has to see it to fix its query.

    Wrapping as SELECT * FROM (<query>) LIMIT n survives queries that
    already contain their own LIMIT.
    """
    allowed, reason = _is_safe_select(query)
    if not allowed:
        return ToolResult(ok=False, content=reason)

    limit = max(0, min(int(max_rows), MAX_ROWS))
    sql = query.strip().rstrip(";").rstrip()
    try:
        result = session.sql(
            f"SELECT * FROM ({sql}) AS _query_result LIMIT {limit + 1}"
        )
        columns = [column[0] for column in result.description]
        rows = result.fetchall()
        truncated = len(rows) > limit
        rows = rows[:limit]
    except Exception as exc:
        # warning, not exception: a rejected query is EXPECTED here. The model
        # reads the error and retries, so a full traceback in the user's
        # terminal would look like a crash that has not happened.
        logger.warning("SQL query failed: %s", exc)
        return ToolResult(ok=False, content=str(exc))

    if not columns:
        return ToolResult(ok=True, content="Query completed with no columns.")

    lines = _markdown_table(columns, rows)

    export: Path | None = None
    if truncated:
        # Plain prose, AFTER the table, phrased as an instruction. A row of
        # dots or dashes reads as data to a model; this tells it what to do
        # instead of counting rows it cannot see.
        lines.append(
            f"({len(rows)} rows shown, more exist - use COUNT(*) or an "
            "aggregate for totals; do not count the rows above)"
        )
        # The model's view is capped for context budget; the human's is not.
        # Truncation is exactly when a full spreadsheet earns its place.
        export = _write_xlsx(session, sql)
        if export is not None:
            lines.append(f"(the full result was saved for the user as {export.name})")

    return ToolResult(ok=True, content="\n".join(lines), artifact=export)


def describe_table(session: DataSession, table: str) -> ToolResult:
    """Columns, types, row count, and sample rows.

    Sample rows matter: knowing a column is VARCHAR is less useful than
    seeing it holds 'North', 'South', 'West'.
    """
    info = next((item for item in session.table() if item.name == table), None)
    if info is None:
        available = ", ".join(item.name for item in session.table()) or "none"
        return ToolResult(ok=False, content=f"Unknown table {table!r}. Available tables: {available}.")

    try:
        result = session.sql(
            f'SELECT * FROM "{table.replace(chr(34), chr(34) * 2)}" LIMIT 5'
        )
        rows = result.fetchall()
    except Exception as exc:
        logger.warning("Table description failed: %s", exc)
        return ToolResult(ok=False, content=str(exc))

    lines = [info.describe(), "", "Sample rows:"]
    if not rows:
        lines.append("(table is empty)")
    else:
        lines.extend(_markdown_table([name for name, _ in info.columns], rows))
    return ToolResult(ok=True, content="\n".join(lines))


def plot_chart(session: DataSession, query: str, kind: str, x: str, y: str, title: str | None = None) -> ToolResult:
    """Run a query and write a chart image to settings.processed_dir.

    Returns the path, not the picture - the model cannot see it.
    """
    chart_kind = kind.lower()
    if chart_kind not in CHART_KINDS:
        return ToolResult(
            ok=False,
            content=f"Unsupported chart kind {kind!r}. Choose one of: {', '.join(sorted(CHART_KINDS))}.",
        )

    # The safety GATE only - not run_sql(), which would execute the query
    # here and again below. On a large aggregate that is twice the work
    # for one chart.
    allowed, reason = _is_safe_select(query)
    if not allowed:
        return ToolResult(ok=False, content=reason)

    sql = query.strip().rstrip(";").rstrip()
    try:
        result = session.sql(
            f"SELECT * FROM ({sql}) AS _chart_result LIMIT {MAX_ROWS}"
        )
        columns = [column[0] for column in result.description]
        rows = result.fetchall()
    except Exception as exc:
        logger.warning("Chart query failed: %s", exc)
        return ToolResult(ok=False, content=str(exc))

    missing = [column for column in (x, y) if column not in columns]
    if missing:
        return ToolResult(
            ok=False,
            content=f"Unknown chart column(s): {', '.join(missing)}. Available columns: {', '.join(columns)}.",
        )

    try:
        import matplotlib

        # MUST come before pyplot is imported. "Agg" is the file-only
        # renderer; without it matplotlib looks for a GUI, which either
        # pops a window nobody asked for or fails outright headless.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        settings = get_settings()

        # matplotlib wants two parallel sequences, not a list of dicts.
        xi, yi = columns.index(x), columns.index(y)
        xs = [row[xi] for row in rows]
        ys = [row[yi] for row in rows]

        figure, axes = plt.subplots(figsize=(8, 4.5))
        if chart_kind == "bar":
            axes.bar(xs, ys, color="#4C78A8")
        elif chart_kind == "line":
            axes.plot(xs, ys, color="#4C78A8", marker="o")
        else:
            axes.scatter(xs, ys, color="#4C78A8")

        axes.set_xlabel(x)
        axes.set_ylabel(y)
        if title:
            axes.set_title(title)
        axes.spines[["top", "right"]].set_visible(False)
        # Category labels overlap badly once there are more than a few.
        if len(xs) > 6:
            plt.setp(axes.get_xticklabels(), rotation=45, ha="right")

        stem = _slug(title or f"{y}-by-{x}", chart_kind)
        path = _output_dir() / f"{stem}-{time.time_ns()}.{settings.analyst_chart_format}"
        # bbox_inches="tight" stops rotated labels being cropped off.
        figure.savefig(path, dpi=settings.analyst_chart_dpi, bbox_inches="tight")
        # Without this, every figure stays alive in matplotlib's global
        # registry for the life of the process.
        plt.close(figure)
    except Exception as exc:
        logger.warning("Chart generation failed: %s", exc)
        return ToolResult(ok=False, content=str(exc))

    return ToolResult(ok=True, content=f"wrote {path.name}", artifact=path)


TOOLS = {
    "run_sql": run_sql,
    "describe_table": describe_table,
    "plot_chart": plot_chart,
}
