from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from app.llm_providers import LLMProvider, Message

from .loader import DataSession
from .prompt import build_analyst_messages, format_observation, parse_tool_call
from .tools import TOOLS, ToolResult

logger = logging.getLogger(__name__)

MAX_TURNS = 5


@dataclass
class AnalystResult:
    
    answer: str
    sql: list[str] = field(default_factory=list)
    chart_path: Path | None = None
    #: .xlsx exports. Separate from chart_path: different viewers.
    exports: list[Path] = field(default_factory=list)
    turns: int = 0
    hit_limit: bool = False


#: Suffixes that make an artifact a picture rather than data.
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".svg", ".pdf"})

async def run_analyst(question: str, session: DataSession, provider: LLMProvider, *, model: str | None = None, history: list[Message] | None = None, max_turns: int = MAX_TURNS) -> AnalystResult:

    messages = build_analyst_messages(question, session.schema_summary(), history)
    executed: list[str] = []
    chart: Path | None = None
    exports: list[Path] = []

    for turn in range(1, max_turns + 1):
        reply = await provider.chat(messages, model=model)
        call = parse_tool_call(reply.text)

        if call is None:
            return AnalystResult(
                answer=reply.text, sql=executed, chart_path=chart,
                exports=exports, turns=turn,
            )

        result = _dispatch(call, session)

        if call.name == "run_sql":
            query = call.args.get("query")
            if query:
                executed.append(query)
        if result.artifact is not None:
            if result.artifact.suffix.lower() in _IMAGE_SUFFIXES:
                chart = result.artifact
            elif result.artifact not in exports:
                exports.append(result.artifact)

        messages.append(Message(role="assistant", content=reply.text))
        messages.append(Message(role="user", content=format_observation(result)))

    messages.append(Message(role="user", content="You have reached the maximum number of turns. Please provide a final answer based on what you have found."))
    reply = await provider.chat(messages, model=model)

    return AnalystResult(
        answer=reply.text, sql=executed, chart_path=chart,
        exports=exports, turns=max_turns, hit_limit=True,
    )

def _dispatch(call, session: DataSession) -> ToolResult:
    """Look up the tool by name and call it with its arguments.

    Guard clauses:
      1. name not in TOOLS -> ToolResult(ok=False, ...) naming what IS valid
      2. wrong/missing arguments (TypeError) -> ToolResult(ok=False, str(exc))

    Same rule as §2: never raise. Every failure goes back to the model as
    something it can read and correct.
    """
    
    tool = TOOLS.get(call.name)
    if tool is None:
        # Unreachable today - parse_tool_call already rejects unknown names -
        # but kept so this function is safe to call from anywhere.
        return ToolResult(ok=False, content=f"Unknown tool {call.name!r}. Valid tools: {', '.join(TOOLS)}.",)

    try:
        return tool(session, **call.args)
    except Exception as exc:
        # Broad on purpose: this function must never raise. One escaping
        # exception ends the conversation.
        logger.warning("Tool %s failed: %s", call.name, exc)
        return ToolResult(ok=False, content=f"{type(exc).__name__}: {exc}")
    
