from __future__ import annotations

import asyncio
import shutil
import tempfile
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

# Windows consoles default to cp1252, which cannot encode the box-drawing
# characters this script prints - and redirecting output to a file makes
# Python pick the locale encoding even when the console itself would cope.
# Force UTF-8 so `> out.txt` never crashes the run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


from app.config import get_settings  # noqa: E402
from app.db import init_db  # noqa: E402
from app.llm_providers import ProviderError, ProviderRegistry  # noqa: E402
from app.models import Mode, QueryRequest  # noqa: E402
from app.rag import VectorStore, chunk_text, ingest_paths, load_document, retrieve  # noqa: E402
from app.rag.prompt import build_rag_messages  # noqa: E402
from app.routers.rag import answer_question  # noqa: E402

CHECK_COLLECTION = "healthcheck"

# The fixture is generated into a temp file rather than shipped, so the repo
# carries no sample data. Fictional, with facts the checks below assert on.
SAMPLE_TEXT = """# Northwind Robotics - Employee Handbook (FICTIONAL)

## Leave and Time Off

Employees are entitled to **25 days of paid annual leave** per calendar year,
in addition to public holidays. Annual leave accrues monthly and unused days
may be carried over to a maximum of 5 days into the following year.

New parents are entitled to **90 days of paid parental leave**. Parental leave
must be requested at least **30 days in advance** through the HR portal, except
in cases of medical emergency.

Sick leave is uncapped but requires a doctor's note after **5 consecutive days**
of absence.

## Remote Work

Employees may work remotely up to **3 days per week**. Fully remote
arrangements require director approval and a review every 6 months.

## Equipment

Each employee receives a laptop refreshed every **3 years**, and a one-off
**home-office allowance of 800 EUR**.
"""

# A question the sample document DOES answer (the handbook says 90 days).
GROUNDED_Q = "How many days of paid parental leave do employees get?"
GROUNDED_EXPECT = "90"

# A question the sample document does NOT answer. The system must refuse
# rather than answer from the model's own training.
UNGROUNDED_Q = "What is the capital city of Mongolia?"
REFUSAL_TRIPWIRE = "ulaanbaatar"

results: list[tuple[str, str]] = []


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n  {text}\n{'=' * 70}")


def record(stage: str, status: str, detail: str = "") -> None:
    results.append((stage, status))
    print(f"\n  [{status}] {stage}" + (f" — {detail}" if detail else ""))


def preview(text: str, limit: int = 400) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"… (+{len(text) - limit} more chars)"


async def main() -> int:
    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    banner("STAGE 0 — Environment")

    registry = ProviderRegistry(settings)
    ollama_ok = await registry.get("ollama").health()
    print(f"  chat model : {settings.ollama_chat_model}")
    print(f"  embed model: {settings.ollama_embed_model}")
    print(f"  providers  : {registry.available}")

    if not ollama_ok:
        record("Ollama reachable + models pulled", "FAIL",
               f"run: ollama pull {settings.ollama_chat_model}")
        print("\n  Cannot continue without Ollama. Fix the above and re-run.")
        return 1
    record("Ollama reachable + models pulled", "PASS")

    sample_dir = Path(tempfile.mkdtemp(prefix="aura-healthcheck-"))
    SAMPLE = sample_dir / "sample-handbook.md"
    SAMPLE.write_text(SAMPLE_TEXT, encoding="utf-8")

    store = VectorStore(persist_dir=settings.vectorstore_dir, collection_name=CHECK_COLLECTION)
    store.reset()  # start clean every run

    # ── STAGE 1: chunking ───────────────────────────────────────────
    banner("STAGE 1 — chunk_text()")

    text = load_document(SAMPLE)
    print(f"  Loaded {SAMPLE.name}: {len(text)} characters")

    try:
        chunks = chunk_text(
            text,
            source=SAMPLE.name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            extra_metadata={"file_type": "md"},
        )
    except NotImplementedError:
        record("chunk_text() implemented", "TODO",
               "not implemented: backend/app/rag/chunking.py")
        summary()
        return 1
    except Exception as exc:
        record("chunk_text() implemented", "FAIL", f"{type(exc).__name__}: {exc}")
        summary()
        return 1

    if not chunks:
        record("chunk_text() produces chunks", "FAIL", "returned an empty list")
        summary()
        return 1

    sizes = [len(c.text) for c in chunks]
    print(f"  Produced {len(chunks)} chunks "
          f"(min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)} chars)")
    print(f"\n  ─── chunk 0 ───\n  {preview(chunks[0].text, 300)}")
    if len(chunks) > 1:
        print(f"\n  ─── chunk 1 ───\n  {preview(chunks[1].text, 300)}")
    record("chunk_text() implemented", "PASS", f"{len(chunks)} chunks")

    # ── STAGE 2: ingestion (embed + store) ──────────────────────────
    banner("STAGE 2 — Ingestion: embed + store")

    try:
        report = await ingest_paths(
            [SAMPLE], store, registry,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except ProviderError as exc:
        record("Ingestion", "FAIL", str(exc))
        summary()
        return 1

    for doc in report.documents:
        note = f" (skipped: {doc.skipped_reason})" if doc.skipped_reason else ""
        print(f"  {doc.source}: {doc.chunks} chunks, {doc.characters} chars{note}")
    print(f"  Vector store now holds {store.count()} chunks")

    if store.count() == 0:
        record("Ingestion", "FAIL", "nothing was indexed")
        summary()
        return 1
    record("Ingestion", "PASS", f"{store.count()} chunks indexed")

    # ── STAGE 3: retrieval ──────────────────────────────────────────
    banner("STAGE 3 — Retrieval")

    print(f"  Query: {GROUNDED_Q!r}\n")
    hits = await retrieve(GROUNDED_Q, store, registry, top_k=settings.rag_top_k)

    if not hits:
        record("Retrieval returns chunks", "FAIL", "no hits for a question the doc answers")
        summary()
        return 1

    for rank, hit in enumerate(hits, start=1):
        score = f"{hit.score:.3f}" if hit.score is not None else "n/a"
        print(f"  [{rank}] score={score}  {hit.source} (chunk {hit.chunk_index})")
        print(f"      {preview(hit.text, 160)}\n")

    top_has_answer = GROUNDED_EXPECT in hits[0].text
    record(
        "Retrieval returns chunks", "PASS",
        f"top score {hits[0].score:.3f}" if hits[0].score is not None else "",
    )
    record(
        "Top chunk actually contains the answer",
        "PASS" if top_has_answer else "WARN",
        "expected '90' in the top chunk — tune chunk size if missing" if not top_has_answer else "",
    )

    # ── STAGE 4: prompt construction ────────────────────────────────
    banner("STAGE 4 — build_rag_messages()")

    try:
        messages = build_rag_messages(GROUNDED_Q, hits, [])
    except NotImplementedError:
        record("build_rag_messages() implemented", "TODO",
               "not implemented: backend/app/rag/prompt.py")
        summary()
        return 1
    except Exception as exc:
        record("build_rag_messages() implemented", "FAIL", f"{type(exc).__name__}: {exc}")
        summary()
        return 1

    print(f"  Built {len(messages)} messages: {[m.role for m in messages]}\n")
    for message in messages:
        print(f"  ─── {message.role.upper()} ───")
        print(f"  {preview(message.content, 700)}\n")

    problems = []
    if not messages or messages[0].role != "system":
        problems.append("first message must be role='system'")
    if sum(1 for m in messages if m.role == "system") != 1:
        problems.append("exactly one system message required")
    if not any(GROUNDED_Q in m.content for m in messages):
        problems.append("the question never reaches the model")
    if not any(hits[0].text[:40] in m.content for m in messages):
        problems.append("the retrieved context never reaches the model")

    if problems:
        record("build_rag_messages() implemented", "FAIL", "; ".join(problems))
        summary()
        return 1
    record("build_rag_messages() implemented", "PASS", f"{len(messages)} messages")

    # ── STAGE 5: grounded answer ────────────────────────────────────
    banner("STAGE 5 — answer_question()")

    print(f"  Question: {GROUNDED_Q!r}")
    print("  (generating — a 7B model takes a few seconds)\n")

    try:
        response = await answer_question(
            QueryRequest(question=GROUNDED_Q, mode=Mode.RAG), store, registry, settings
        )
    except NotImplementedError:
        record("answer_question() implemented", "TODO",
               "not implemented: backend/app/routers/rag.py")
        summary()
        return 1
    except Exception as exc:
        record("answer_question() implemented", "FAIL", f"{type(exc).__name__}: {exc}")
        summary()
        return 1

    print(f"  ANSWER ({response.provider}/{response.model}):")
    print(f"  {preview(response.answer, 600)}\n")
    print(f"  SOURCES: {len(response.sources)}")
    for source in response.sources:
        score = f"{source.score:.3f}" if source.score is not None else "n/a"
        print(f"    - {source.source} (chunk {source.chunk_index}, score {score})")

    record("answer_question() implemented", "PASS")
    record(
        "ACCEPTANCE 1 — answer is grounded and correct",
        "PASS" if GROUNDED_EXPECT in response.answer else "FAIL",
        "" if GROUNDED_EXPECT in response.answer
        else f"expected '{GROUNDED_EXPECT}' in the answer — check your system prompt",
    )
    record(
        "ACCEPTANCE 3 — sources returned (traceability)",
        "PASS" if response.sources else "FAIL",
        "" if response.sources else "sources list is empty — a grounded answer must cite",
    )

    # ── STAGE 6: refusal ────────────────────────────────────────────
    banner("STAGE 6 — Refusal (the check people skip)")

    print(f"  Question: {UNGROUNDED_Q!r}")
    print("  The handbook says nothing about geography. It MUST decline.\n")

    refusal = await answer_question(
        QueryRequest(question=UNGROUNDED_Q, mode=Mode.RAG), store, registry, settings
    )
    print(f"  ANSWER:\n  {preview(refusal.answer, 400)}\n")

    leaked = REFUSAL_TRIPWIRE in refusal.answer.lower()
    record(
        "ACCEPTANCE 2 — refuses ungrounded questions",
        "FAIL" if leaked else "PASS",
        "answered from the model's own knowledge — strengthen SYSTEM_PROMPT"
        if leaked else "",
    )

    store.reset()
    return summary()


def summary() -> int:
    banner("SUMMARY")
    width = max(len(stage) for stage, _ in results) + 2
    for stage, status in results:
        print(f"  {stage:<{width}} {status}")

    failed = [s for s, st in results if st == "FAIL"]
    todo = [s for s, st in results if st == "TODO"]

    print()
    if todo:
        print(f"  NEXT UP: {todo[0]}")
        print("  Some stages are not implemented.")
        return 1
    if failed:
        print(f"  {len(failed)} check(s) failed.")
        return 1

    print("  ALL CHECKS PASSED — the document pipeline is healthy.")
    return 0


async def _run() -> int:
    """Wrapper so the temp fixture is cleaned up on every exit path."""
    try:
        return await main()
    finally:
        for leftover in Path(tempfile.gettempdir()).glob("aura-healthcheck-*"):
            shutil.rmtree(leftover, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
