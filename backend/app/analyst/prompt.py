from __future__ import annotations

import json
import re
from dataclasses import dataclass

from app.llm_providers import Message

from .tools import TOOLS, ToolResult

MAX_HISTORY = 6

#: Non-greedy stops at the FIRST closing fence; DOTALL lets JSON span lines.
_TOOL_BLOCK = re.compile(r"```tool\s*\n(.*?)\n?```", re.DOTALL)

#: Models reach for ```json out of habit; accepting it saves a turn.
_JSON_BLOCK = re.compile(r"```json\s*\n(.*?)\n?```", re.DOTALL)

TOOL_SPEC = """Available tools:

- `run_sql`: Run a read-only SQL query against the loaded data. Arguments: `query` (string, required), `max_rows` (integer, optional; maximum 50). Example:
  ```tool
  {"tool": "run_sql", "args": {"query": "SELECT category, COUNT(*) AS total FROM sales GROUP BY category ORDER BY total DESC", "max_rows": 10}}
  ```

- `describe_table`: Inspect a table's columns, types, row count, and sample rows. Arguments: `table` (string, required). Example:
  ```tool
  {"tool": "describe_table", "args": {"table": "sales"}}
  ```

- `plot_chart`: Run a read-only SQL query and save a chart. Arguments: `query` (string, required), `kind` (`bar`, `line`, or `scatter`, required), `x` (string column name, required), `y` (string column name, required), `title` (string, optional). Example:
  ```tool
  {"tool": "plot_chart", "args": {"query": "SELECT month, revenue FROM sales ORDER BY month", "kind": "line", "x": "month", "y": "revenue", "title": "Monthly revenue"}}
  ```
"""

ANALYST_SYSTEM_PROMPT = f"""You are a data analyst. Answer questions by inspecting and querying the loaded data; never guess values, columns, or results.

{TOOL_SPEC}

To call a tool, reply with exactly one fenced JSON block and no other tool calls. For example:
```tool
{{"tool": "describe_table", "args": {{"table": "sales"}}}}
```

Use `describe_table` before querying whenever the table or columns are uncertain. Use only columns shown in the schema or a tool result. If the available data cannot answer the question, say so plainly and do not invent data.

SQL rules for this database (DuckDB):
- Wrap every column name in DOUBLE QUOTES, exactly as the schema shows it: `SELECT "AI Adoption (%)" FROM t`. Backticks are MySQL syntax and are a syntax error here.
- Column names may contain spaces, percent signs, and parentheses. Copy them character for character from the schema; do not rename or abbreviate them.
- For `plot_chart`, the `x` and `y` arguments are plain column names with NO quotes and NO backticks - they are looked up as text, not parsed as SQL. Alias awkward columns in the query and use the alias: `SELECT "Year" AS year, "AI Adoption (%)" AS adoption FROM t` with `x: "year"`, `y: "adoption"`.
- If a query fails, read the error and change the query. Do not send the same SQL twice.

After you have enough evidence, answer in concise prose with the numbers you found. Do not include a tool block in a final answer.
"""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


def build_analyst_messages(
    question: str,
    schema_summary: str,
    history: list[Message] | None = None,
) -> list[Message]:
    """[system] + history[-MAX_HISTORY:] + [user].

    The system message carries the schema - the model cannot write SQL
    against columns it has never seen.
    """
    system_content = f"{ANALYST_SYSTEM_PROMPT}\n\nSCHEMA:\n{schema_summary}"
    messages = [Message(role="system", content=system_content)]

    if history:
        messages.extend(history[-MAX_HISTORY:])
    messages.append(Message(role="user", content=question))

    return messages


def parse_tool_call(text: str) -> ToolCall | None:
    """Pull a tool call out of the model's reply, or None if there isn't one.

    None is not an error - it is the loop's "model is finished" signal.
    """
    match = _TOOL_BLOCK.search(text) or _JSON_BLOCK.search(text)
    if not match:
        return None

    raw = match.group(1)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    name = payload.get("tool")
    args = payload.get("args", {})

    if not isinstance(name, str) or not isinstance(args, dict):
        return None

    if name not in TOOLS:
        return None

    return ToolCall(name=name, args=args)


def format_observation(result: ToolResult) -> str:
    """Turn a ToolResult into the message the model reads next.

    Goes back with role "user": the model's own tool call is the assistant
    message. Both as assistant reads as the model talking to itself.
    """
    if not result.ok:
        # Phrased as an instruction - the point of returning errors rather
        # than raising is that the model acts on them.
        return f"OBSERVATION (error - fix and retry):\n{result.content}"

    if result.artifact is not None:
        # The model cannot see the picture, and will invent a description
        # of it if not told otherwise.
        return (
            f"OBSERVATION:\n{result.content}\n"
            "The chart file was written and the user can open it. "
            "Do not describe its appearance."
        )

    return f"OBSERVATION:\n{result.content}"