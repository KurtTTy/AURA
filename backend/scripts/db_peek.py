from __future__ import annotations

import argparse
import sqlite3
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


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def show_sqlite(db_path: Path, full: bool) -> None:
    rule(f"SQLITE  —  {db_path}")

    if not db_path.exists():
        print("  No database yet. It's created on first API startup or ingest.")
        return

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )]

    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"\n  {table}  ({count} rows)")
        for col in con.execute(f"PRAGMA table_info({table})"):
            flags = " PK" if col["pk"] else ""
            flags += "" if col["notnull"] else " NULL"
            print(f"    {col['name']:<16} {col['type']:<14}{flags}")

        if full and count:
            print("    ── rows ──")
            for row in con.execute(f"SELECT * FROM {table} LIMIT 20"):
                values = {
                    k: (str(row[k])[:60] + "…" if row[k] and len(str(row[k])) > 60 else row[k])
                    for k in row.keys()
                }
                print(f"    {values}")
    con.close()


def show_chroma(persist_dir: Path, full: bool) -> None:
    rule(f"CHROMA  —  {persist_dir}")

    if not persist_dir.exists():
        print("  No vector store yet. Ingest a document first.")
        return

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    # On-disk layout, so the UUID folders aren't a mystery.
    print("  on-disk files:")
    for path in sorted(persist_dir.iterdir()):
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"    [index segment] {path.name}  ({size:,} B of HNSW data)")
        elif path.name != ".gitkeep":
            print(f"    {path.name:<30} {path.stat().st_size:,} B")

    client = chromadb.PersistentClient(
        path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )

    collections = client.list_collections()
    if not collections:
        print("\n  No collections.")
        return

    print("\n  collections:")
    for entry in collections:
        name = entry if isinstance(entry, str) else entry.name
        collection = client.get_collection(name)
        count = collection.count()
        print(f"\n    {name}: {count} chunks")

        if not count:
            continue

        got = collection.get(
            limit=(20 if full else 3), include=["documents", "metadatas", "embeddings"]
        )
        embeddings = got.get("embeddings")
        if embeddings is not None and len(embeddings):
            print(f"      embedding dimensions: {len(embeddings[0])}")

        sources = {str((m or {}).get("source", "?")) for m in (got.get("metadatas") or [])}
        print(f"      sources in sample: {', '.join(sorted(sources))}")

        for i, chunk_id in enumerate(got["ids"]):
            text = (got["documents"] or [])[i] if got.get("documents") else ""
            metadata = (got["metadatas"] or [{}])[i]
            preview = text[: (300 if full else 100)].replace("\n", " ")
            print(f"\n      id       : {chunk_id}")
            print(f"      metadata : {metadata}")
            print(f"      text     : {preview}…")


def run_sql(db_path: Path, query: str) -> None:
    rule(f"SQL  —  {query}")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(query).fetchall()
    except sqlite3.Error as exc:
        print(f"  SQL error: {exc}")
        return
    finally:
        con.close()

    if not rows:
        print("  (no rows)")
        return
    print(f"  {len(rows)} row(s)\n")
    for row in rows:
        print(f"  {dict(row)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the project's databases.")
    parser.add_argument("--full", action="store_true", help="show row and chunk contents")
    parser.add_argument("--sql", metavar="QUERY", help="run a read-only query against app.db")
    args = parser.parse_args()

    settings = get_settings()
    app_db = settings.data_dir / "app.db"

    if args.sql:
        if not app_db.exists():
            print(f"No database at {app_db}")
            return 1
        run_sql(app_db, args.sql)
        return 0

    show_sqlite(app_db, args.full)
    show_chroma(settings.vectorstore_dir, args.full)

    print("\n  Tip: --full for contents, --sql \"SELECT ...\" to query app.db directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
