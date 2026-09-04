from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Windows consoles default to cp1252 and choke on the box characters below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from app.config import get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.llm_providers import Message, ProviderError, ProviderRegistry  # noqa: E402
from app.rag import VectorStore, ingest_paths, retrieve  # noqa: E402
from app.rag.prompt import NO_CONTEXT_ANSWER, build_rag_messages  # noqa: E402
from app.analyst import DataSession, run_analyst  # noqa: E402
from app.routers.chat import DEFAULT_SYSTEM_PROMPT  # noqa: E402

#: Raw string - the art is mostly backslashes, and a normal string would
#: eat them as escape sequences.
BANNER = r"""
 ________  ___  ___  ________  ________
|\   __  \|\  \|\  \|\   __  \|\   __  \
\ \  \|\  \ \  \\\  \ \  \|\  \ \  \|\  \
 \ \   __  \ \  \\\  \ \   _  _\ \   __  \
  \ \  \ \  \ \  \\\  \ \  \\  \\ \  \ \  \
   \ \__\ \__\ \_______\ \__\\ _\\ \__\ \__\
    \|__|\|__|\|_______|\|__|\|__|\|__|\|__|
"""

TAGLINE = "Adaptive Unified Retrieval Assistant"


HELP = """
  Two ways to answer a question, each with its own commands.
  ────────────────────────────────────────────────────────────

  DOCUMENTS  — pdf, docx, md, txt.  Finds the passage and quotes it.
    /ingest <path>   add a file or folder
    /docs            list what you have added
    /mode rag        answer from those documents          (default)
    /reset           erase everything ingested (asks first)

  DATA  — csv, tsv, xlsx, json, parquet.  Writes SQL and computes.
    /load <path>     add a spreadsheet or dataset
    /tables          list what you have loaded
    /mode analyst    answer by querying those tables

  CHAT
    /mode chat       plain conversation, no documents, no data

  ANY MODE
    /provider <name> switch LLM: ollama | gemini | anthropic | openai
                     (cloud ones need their API key in .env)
    /model <name>    switch model within the current provider
    /models          list what the current provider can serve
    /search on|off   Google Search grounding — gemini only.
                     The ONLY way anything here reaches the live web.
    /status          provider, model, mode, docs, tables, output folder
    /clear           forget the conversation and clear the screen
                     (documents and tables are kept)
    /help            this
    /exit            quit  (Ctrl+C also works)

  Anything not starting with / is treated as a question.

  /ingest and /load are NOT interchangeable. A PDF goes to documents;
  a spreadsheet goes to data. They use different engines and answer
  different kinds of question.

  Note: embeddings ALWAYS run on local Ollama, whichever provider you
  pick. So Ollama must stay running for /mode rag even on Gemini.
"""


class Session:
    """Holds everything that survives between questions."""

    def __init__(self, settings, registry, store):
        self.settings = settings
        self.registry = registry
        self.store = store
        self.mode = "rag"
        self.history: list[Message] = []

        # One DuckDB session for the whole CLI run. Built lazily-ish
        # here rather than per question, so tables you /load stay
        # loaded across questions.
        self.data = DataSession(db_path=settings.analyst_db_path)

        # Provider is held by NAME, not as an object, so it can change
        # mid-conversation. Resolved fresh on each use via the property
        # below - the registry already caches the instances.
        self.provider_name = settings.default_provider
        self.model = registry.default_model_for(self.provider_name) or ""

    @property
    def provider(self):
        return self.registry.get(self.provider_name)

    def switch_provider(self, name: str) -> str:
        """Point at a different backend, and move the model with it.

        Switching the model too is the important part: model names don't
        transfer between providers. Leaving 'qwen2.5:7b' set while
        talking to Gemini would fail with a confusing 404.
        """
        provider = self.registry.get(name)  # raises if unknown/unconfigured
        self.provider_name = provider.name
        self.model = self.registry.default_model_for(provider.name) or ""
        return provider.name


async def generate(session: Session, messages: list[Message]) -> tuple[str, list[str]]:
    """Produce the answer. Returns (text, web_sources).

    Streams token by token normally. But when the provider is searching
    the web server-side we use the NON-streaming call instead: grounding
    sources arrive in trailing chunks and an async generator has nowhere
    to hand them back. Citations are worth more than live typing here -
    an unsourced web claim is exactly what this project exists to avoid.
    """
    searching = getattr(session.provider, "search_enabled", False)

    if searching:
        print("  (searching the web…)", flush=True)
        try:
            result = await session.provider.chat(messages, model=session.model)
        except ProviderError as exc:
            print(f"  [error] {exc}")
            return "", []
        print(f"\n{result.text.strip()}")
        return result.text, result.sources

    parts: list[str] = []
    try:
        async for fragment in session.provider.chat_stream(messages, model=session.model):
            print(fragment, end="", flush=True)
            parts.append(fragment)
    except ProviderError as exc:
        print(f"\n  [error] {exc}")
        return "", []
    print()
    return "".join(parts), []


def print_web_sources(sources: list[str]) -> None:
    """Show what Google actually consulted, labelled as web - not as your
    documents. Keeping the two visually distinct matters: one you curated,
    one came off the open internet."""
    if not sources:
        return
    print("\n  web sources")
    for i, s in enumerate(sources, start=1):
        print(f"    ({i}) {s}")


async def answer_with_documents(session: Session, question: str) -> None:
    """RAG mode: retrieve, prompt, stream, then show the sources."""
    if session.store.count() == 0:
        print("\n  Nothing indexed yet. Add a file or folder first:")
        print("    /ingest <path to a pdf, docx, md or txt>\n")
        return

    chunks = await retrieve(question, session.store, session.registry,
                            top_k=session.settings.rag_top_k)
    if not chunks:
        print(f"\n  {NO_CONTEXT_ANSWER}\n")
        return

    messages = build_rag_messages(question, chunks, session.history[-6:])
    print()
    answer, web = await generate(session, messages)

    if answer:
        print("\n  sources")
        for i, c in enumerate(chunks, start=1):
            score = f"{c.score:.3f}" if c.score is not None else "n/a"
            print(f"    [{i}] {c.source}  (chunk {c.chunk_index}, score {score})")
        print_web_sources(web)
        print()
        session.history += [
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ]


async def answer_without_documents(session: Session, question: str) -> None:
    """Chat mode: no retrieval, just conversation."""
    messages = [Message(role="system", content=DEFAULT_SYSTEM_PROMPT)]
    messages += session.history[-6:]
    messages.append(Message(role="user", content=question))

    print()
    answer, web = await generate(session, messages)
    print_web_sources(web)
    print()
    if answer:
        session.history += [
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ]


def print_header(session: Session) -> None:
    """Banner plus one line of state. Shown at startup and after /clear."""
    print(BANNER)
    print(f"  {TAGLINE}")
    print(f"  {'─' * len(TAGLINE)}")
    print(f"  provider: {session.provider_name}    model: {session.model}")
    print(f"  mode: {session.mode}    indexed: {session.store.count()} chunks"
          f"    tables: {len(session.data.table())}")


def clear_screen() -> None:
    """ANSI clear + cursor home.

    Escape codes rather than `cls`/`clear`, because spawning a shell just
    to blank the screen is slower and flashes a console window on Windows.
    Windows 10+ terminals handle these natively - and the banner already
    needs a terminal that does, since it is box-drawing characters.
    """
    print("\033[2J\033[H", end="", flush=True)


def open_file(path: Path) -> None:
    """Open a file with the OS default application.

    Three platforms, three mechanisms - os.startfile exists only on
    Windows, so this cannot be one call. Failure is deliberately silent:
    the path is already printed, and a headless or locked-down machine
    should not turn a good answer into a traceback.
    """
    try:
        if sys.platform == "win32":
            os.startfile(path)                      # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])   # noqa: S603,S607
        else:
            subprocess.Popen(["xdg-open", str(path)])  # noqa: S603,S607
    except Exception:
        pass


async def answer_from_data(session: Session, question: str) -> None:
    """Analyst mode: the model writes SQL, we run it, it reads the result.

    Deliberately NOT streamed. The loop makes several model calls and only
    the last one is the answer - streaming would spray tool-call JSON across
    the terminal. Same reasoning as web-search grounding in generate():
    when the useful output arrives at the end, wait for it.
    """
    if not session.data.table():
        print("\n  No data loaded. Try:")
        print("    /load ../data/raw/yourfile.csv\n")
        return

    print("\n  (analysing...)", flush=True)
    try:
        result = await run_analyst(
            question,
            session.data,
            session.provider,
            model=session.model,
            history=session.history[-6:],
            max_turns=session.settings.analyst_max_turns,
        )
    except ProviderError as exc:
        print(f"  [error] {exc}\n")
        return

    # Queries BEFORE the answer, on purpose. The whole point is that you
    # can check the work - an analyst answer you cannot verify is a guess
    # with better formatting.
    if result.sql:
        print("\n  queries")
        for i, query in enumerate(result.sql, start=1):
            print(f"    [{i}] {query}")

    print(f"\n{result.answer.strip()}")

    if result.chart_path or result.exports:
        print("\n  files")
        if result.chart_path:
            print(f"    chart  {result.chart_path}")
        for export in result.exports:
            print(f"    data   {export}")

    # Open the chart only. A chart was explicitly asked for, so showing it
    # is the point; a spreadsheet written as a side effect is not, and
    # launching Excel uninvited is obnoxious.
    if result.chart_path:
        open_file(result.chart_path)

    if result.hit_limit:
        print(f"\n  (stopped after {result.turns} turns - answer may be incomplete)")

    print()
    session.history += [
        Message(role="user", content=question),
        Message(role="assistant", content=result.answer),
    ]


async def do_load(session: Session, raw: str) -> None:
    """Resolve a path the way /ingest does, then load it into DuckDB."""
    path = Path(raw.strip().strip('"').strip("'"))
    if not path.is_absolute():
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else session.settings.raw_dir / path

    try:
        info = session.data.load(path)
    except (FileNotFoundError, ValueError) as exc:
        # Both mean the user's input was wrong (missing file, unusable
        # name), so print it rather than throwing a traceback at them.
        print(f"  {exc}\n")
        return

    print(f"  Loaded '{info.name}' - {info.row_count} rows, {len(info.columns)} columns")
    for col_name, col_type in info.columns:
        print(f"    {col_name}: {col_type}")
    print()


async def do_ingest(session: Session, raw: str) -> None:
    path = Path(raw.strip().strip('"').strip("'"))
    if not path.is_absolute():
        # try as given, then relative to data/raw - whichever exists
        candidate = Path.cwd() / path
        path = candidate if candidate.exists() else session.settings.raw_dir / path

    if not path.exists():
        print(f"  Not found: {path}\n")
        return

    print(f"  Reading {path} ...")
    try:
        report = await ingest_paths(
            [path], session.store, session.registry,
            chunk_size=session.settings.chunk_size,
            chunk_overlap=session.settings.chunk_overlap,
        )
    except ProviderError as exc:
        print(f"  [error] {exc}\n")
        return

    for doc in report.documents:
        note = f"  (skipped: {doc.skipped_reason})" if doc.skipped_reason else ""
        print(f"    {doc.source}: {doc.chunks} chunks{note}")
    print(f"  Store now holds {session.store.count()} chunks.\n")


async def handle_command(session: Session, line: str) -> bool:
    """Returns False when the user wants to quit."""
    cmd, _, arg = line[1:].partition(" ")
    cmd, arg = cmd.lower(), arg.strip()

    if cmd in ("exit", "quit", "q"):
        return False

    if cmd == "help":
        print(HELP)

    elif cmd == "ingest":
        if not arg:
            print("  Usage: /ingest <file or folder>\n")
        else:
            await do_ingest(session, arg)

    elif cmd == "docs":
        sources = session.store.sources()
        if not sources:
            print("  Nothing indexed yet.\n")
        else:
            print(f"  {session.store.count()} chunks from {len(sources)} document(s):")
            for s in sources:
                print(f"    {s}")
            print()

    elif cmd == "clear":
        # Conversation only. Wiping ingested documents or loaded tables here
        # would be a nasty surprise - /reset exists for documents, and
        # tables go when the session does.
        turns = len(session.history)
        session.history.clear()
        clear_screen()
        print_header(session)
        print(f"\n  Conversation cleared ({turns} message(s) forgotten).")
        print("  Documents and loaded tables are untouched.\n")

    elif cmd == "load":
        if not arg:
            print("  Usage: /load <file.csv>\n")
        else:
            await do_load(session, arg)

    elif cmd == "tables":
        tables = session.data.table()
        if not tables:
            print("  No tables loaded. Use /load <file.csv>\n")
        else:
            print(f"  {len(tables)} table(s) loaded:")
            for info in tables:
                print(f"    {info.name} ({info.row_count} rows, {len(info.columns)} cols)")
            print()

    elif cmd == "mode":
        if arg in ("rag", "chat", "analyst"):
            session.mode = arg
            what = {
                "rag": "your documents",
                "chat": "the model's own knowledge",
                "analyst": "SQL over your loaded tables",
            }[arg]
            print(f"  Mode: {arg} — answering from {what}.\n")
        else:
            print(f"  Mode is '{session.mode}'. Use /mode rag, /mode chat or /mode analyst\n")

    elif cmd == "provider":
        if not arg:
            print(f"  Provider is '{session.provider_name}'.")
            print(f"  Configured: {', '.join(session.registry.available)}")
            print("  Cloud providers need their API key in .env\n")
        else:
            try:
                name = session.switch_provider(arg)
            except ProviderError as exc:
                print(f"  {exc}\n")
            else:
                ok = await session.provider.health()
                state = "reachable" if ok else "NOT REACHABLE — check the key in .env"
                print(f"  Provider: {name}   model: {session.model}   ({state})\n")

    elif cmd == "model":
        if not arg:
            print(f"  Model is '{session.model}' on provider '{session.provider_name}'\n")
        else:
            available = await session.provider.list_models()
            session.model = arg
            print(f"  Model: {arg}  (provider: {session.provider_name})")
            if available and arg not in available:
                print(f"  Warning: '{arg}' is not in that provider's list. /models to see it.")
            print()

    elif cmd == "models":
        models = await session.provider.list_models()
        if not models:
            print(f"  '{session.provider_name}' reported no models "
                  "(unreachable, or the key was rejected).\n")
        else:
            print(f"  {session.provider_name} can serve {len(models)} model(s):")
            for m in models:
                mark = " *" if m == session.model or m.removesuffix(":latest") == session.model else "  "
                print(f"   {mark} {m}")
            print("\n  * = current\n")

    elif cmd == "search":
        provider = session.provider
        if not hasattr(provider, "search_enabled"):
            print(f"  '{session.provider_name}' cannot search the web.")
            print("  Only gemini can today — /provider gemini\n")
        elif arg in ("on", "off"):
            provider.search_enabled = arg == "on"
            if arg == "on":
                print("  Web search: ON — Google searches, then answers from what it finds.")
                print("  Your questions now go to Google's search infrastructure too.\n")
            else:
                print("  Web search: OFF — answers come from training data only.\n")
        else:
            state = "on" if provider.search_enabled else "off"
            print(f"  Web search is {state}. Use /search on or /search off\n")

    elif cmd == "status":
        provider = session.provider
        search = getattr(provider, "search_enabled", None)
        search_text = "n/a (provider cannot search)" if search is None else ("on" if search else "off")
        print(f"  provider : {session.provider_name}")
        print(f"  model    : {session.model}")
        print(f"  mode     : {session.mode}")
        print(f"  web search: {search_text}")
        print(f"  indexed  : {session.store.count()} chunks")
        print(f"  tables   : {len(session.data.table())} loaded")
        print(f"  output   : {session.settings.processed_dir}")
        print(f"  history  : {len(session.history)} messages\n")

    elif cmd == "reset":
        confirm = await asyncio.to_thread(input, "  Delete every indexed chunk? [y/N] ")
        if confirm.strip().lower() == "y":
            session.store.reset()
            print("  Vector store wiped.\n")
        else:
            print("  Cancelled.\n")

    else:
        print(f"  Unknown command '/{cmd}'. Try /help\n")

    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description="Talk to your documents.")
    parser.add_argument("question", nargs="*", help="ask once and exit")
    parser.add_argument("--ingest", metavar="PATH", help="ingest, then exit")
    parser.add_argument("--mode", choices=("rag", "chat", "analyst"), default="rag")
    parser.add_argument("--provider", help="backend to start on, e.g. gemini")
    parser.add_argument("--model", help="model to start on")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    registry = ProviderRegistry(settings)
    store = VectorStore(persist_dir=settings.vectorstore_dir,
                        collection_name=settings.chroma_collection)
    session = Session(settings, registry, store)
    session.mode = args.mode

    if args.provider:
        try:
            session.switch_provider(args.provider)
        except ProviderError as exc:
            print(f"\n  {exc}\n")
            return 1
    if args.model:
        session.model = args.model

    try:
        # Ollama is checked regardless of provider: embeddings are always
        # local, so rag mode needs it even when Gemini writes the answer.
        if not await registry.get("ollama").health():
            print("\n  Ollama is not reachable, or the models are not pulled.")
            print("  Embeddings are always local, so rag mode needs it even")
            print("  when another provider generates the answer.")
            print("  Start it, then try again:")
            print('    Start-Process "$env:LOCALAPPDATA\\Programs\\Ollama\\ollama app.exe"\n')
            return 1

        if args.ingest:
            await do_ingest(session, args.ingest)
            return 0

        if args.question:
            question = " ".join(args.question)
            handler = {
                "rag": answer_with_documents,
                "analyst": answer_from_data,
            }.get(session.mode, answer_without_documents)
            await handler(session, question)
            return 0

        # ── interactive ──────────────────────────────────────────
        print_header(session)
        others = [p for p in registry.available if p != session.provider_name]
        if others:
            print(f"  also configured: {', '.join(others)}  (/provider <name> to switch)")
        print("  /help for commands, /exit to quit")
        if store.count() == 0:
            print("\n  Nothing loaded yet — try:")
            print("    /ingest <a document>    or    /load <a .csv>")
        print()

        while True:
            try:
                line = await asyncio.to_thread(input, "> ")
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if not await handle_command(session, line):
                    break
                continue

            if session.mode == "rag":
                await answer_with_documents(session, line)
            elif session.mode == "analyst":
                await answer_from_data(session, line)
            else:
                await answer_without_documents(session, line)

        print("\n  Bye.\n")
        return 0
    finally:
        await registry.aclose()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n  Bye.\n")
        sys.exit(130)
