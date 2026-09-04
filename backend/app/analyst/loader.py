from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import duckdb

# .xls (the pre-2007 binary format) is deliberately absent: it needs xlrd,
# a separate dependency, for a format Excel has not written by default since
# 2007. Save as .xlsx instead.
SUPPORTED_EXTENSIONS = frozenset(
    {".txt", ".csv", ".tsv", ".json", ".jsonl", ".parquet", ".xlsx"}
)
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9]")

@dataclass(frozen=True, slots=True)
class TableInfo:
    """One loaded table: its SQL-safe name, its columns, and its size."""

    name: str
    columns: list[tuple[str, str]]
    row_count: int

    def describe(self) -> str:
        """One table's shape: header line plus one line per column.

        Same format as DataSession.schema_summary() on purpose - the model
        sees this string, and two different layouts for the same thing
        would be noise it has to decode.
        """
        lines = [f"{self.name} ({self.row_count} rows)"]
        for col_name, col_type in self.columns:
            lines.append(f'  - "{col_name}": {col_type}')
        return "\n".join(lines)

def table_name_from_path(path: Path) -> str:
    """Derive a table name from a file path.

    Non-alphanumeric characters are replaced with underscores to ensure
    the name is safe for use in SQL queries and as an identifier.

    Args:
        path: The file path from which to derive the table name.

    Returns:
        The derived table name.
    """
    name = path.stem
    safe_name = _SAFE_NAME.sub("_", name.lower())

    # Empty check first: safe_name[0] below would raise IndexError on "".
    # Hard to trigger (punctuation becomes underscores, so "---.csv" gives
    # "___"), but a path with an empty stem reaches here - and there is no
    # sensible default, since inventing one would collide with the next file.
    if not safe_name:
        raise ValueError(
            f"Cannot derive a table name from {path.name!r}. "
            "Pass one explicitly: load(path, table_name='...')."
        )

    # SQL identifiers cannot start with a digit, so "2024sales" is illegal.
    # This is pure string work - no table exists yet at this point.
    if safe_name[0].isdigit():
        safe_name = f"t_{safe_name}"

    return safe_name
    


class DataSession:
    
    def __init__(self, db_path: str | None = None) -> None:
        self.__conn = duckdb.connect(database=db_path or ":memory:", read_only=False)
        self.__tables: dict[str, TableInfo] = {}

    def load(self, path: Path, table_name: str | None = None) -> TableInfo:
        """Read a file into a DuckDB table and remember its shape.

        DuckDB reads csv, tsv, json, and parquet directly. Excel goes
        through pandas first, since DuckDB has no native reader for it.

        Raises:
            FileNotFoundError: the path does not exist.
            ValueError: unsupported extension, or a name that cannot be
                turned into a legal SQL identifier.
        """
        table_name = table_name or table_name_from_path(path)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
        suffix = path.suffix.lower()
        supported = SUPPORTED_EXTENSIONS

        if suffix not in supported:
            raise ValueError(
                f"Unsupported file extension '{suffix}'. Supported extensions are: {', '.join(supported)}"
            )

       
        if suffix in {".csv", ".tsv", ".txt"}:
            self.__conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto(?);", (str(path),))

        elif suffix == ".parquet":
            self.__conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM parquet_scan(?);", (str(path),))

        elif suffix in {".json", ".jsonl"}:
            self.__conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_json(?);", (str(path),))

        elif suffix == ".xlsx":
            # DuckDB cannot read Excel natively, so pandas does the parsing
            # and DuckDB reads the resulting DataFrame.
            import pandas as pd

            frame = pd.read_excel(path)
            # Register under a DIFFERENT name than the table being created.
            # Reusing table_name for both would make the statement read
            # "CREATE TABLE sales AS SELECT * FROM sales", which is
            # self-referential and does not do what it looks like.
            view = f"_import_{table_name}"
            self.__conn.register(view, frame)
            try:
                self.__conn.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {view};"
                )
            finally:
                self.__conn.unregister(view)


        row_count = self.__conn.execute(f"SELECT COUNT(*) FROM {table_name};").fetchone()[0]
        columns = [(r[0], r[1]) for r in self.__conn.execute(f"DESCRIBE {table_name};").fetchall()]


        info = TableInfo(name=table_name, columns=columns, row_count=row_count)
        self.__tables[table_name] = info

        return info
        

    def sql(self, query: str, params: tuple | None = None):
        """Execute a statement and return the DuckDB result.

        The only supported way for other modules to reach the connection.
        Without it callers write `session._DataSession__conn`, defeating
        the name mangling from outside the class and breaking silently
        if the attribute is ever renamed.

        No safety checks here on purpose - this is the session's own
        connection, and everything the MODEL writes goes through
        tools._is_safe_select() before it arrives.
        """
        return self.__conn.execute(query, params) if params else self.__conn.execute(query)

    def table(self) -> list[TableInfo]:
        return list(self.__tables.values())


    def schema_summary(self) -> str:
        lines = []
        for info in self.table():
            lines.append(f"{info.name} ({info.row_count} rows)")
            for col_name, col_type in info.columns:
                lines.append(f'  - "{col_name}": {col_type}')
        return "\n".join(lines)


    def close(self) -> None:
        """Close the DuckDB connection."""
        self.__conn.close()

