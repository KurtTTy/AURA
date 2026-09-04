from __future__ import annotations

import pytest

from app.rag.chunking import Chunk, chunk_text, make_chunk_id

# A document with clear paragraph boundaries, long enough to force
# several chunks at small chunk_size values.
SAMPLE = "\n\n".join(
    f"Paragraph {i}. " + "This sentence exists to add length to the paragraph. " * 4
    for i in range(1, 11)
)


class TestBasics:
    def test_returns_chunks(self):
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=300, chunk_overlap=50)
        assert chunks, "Expected at least one chunk"
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_short_text_is_one_chunk(self):
        """Text shorter than chunk_size shouldn't be split."""
        chunks = chunk_text("Just a short note.", source="note.txt", chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "Just a short note."

    def test_empty_text_returns_empty_list(self):
        assert chunk_text("", source="empty.txt") == []
        assert chunk_text("   \n\n  \t ", source="blank.txt") == []


class TestRequirements:
    """The five requirements listed in chunking.py's docstring."""

    def test_no_empty_chunks(self):
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=200, chunk_overlap=40)
        assert all(c.text.strip() for c in chunks), "A chunk was empty or whitespace-only"

    def test_every_chunk_has_source_metadata(self):
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=200, chunk_overlap=40)
        assert all(c.metadata.get("source") == "sample.txt" for c in chunks)

    def test_indices_are_sequential_from_zero(self):
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=200, chunk_overlap=40)
        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_no_content_is_silently_dropped(self):
        """Every paragraph marker must survive into some chunk.

        The classic chunking bug is an off-by-one in the stride that
        skips a slice of the document. Nothing errors; text just
        vanishes and can never be retrieved.
        """
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=250, chunk_overlap=50)
        combined = " ".join(c.text for c in chunks)
        for i in range(1, 11):
            assert f"Paragraph {i}." in combined, f"Paragraph {i} was dropped"

    def test_overlap_not_smaller_than_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text(SAMPLE, source="s.txt", chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError):
            chunk_text(SAMPLE, source="s.txt", chunk_size=100, chunk_overlap=250)


class TestSizeAndOverlap:
    def test_chunks_respect_size_ceiling(self):
        """Allows modest overshoot for boundary-aware splitting.

        A smarter chunker may extend slightly past chunk_size to finish
        a sentence. 1.5x is generous; far beyond that means the size
        parameter isn't really being honoured.
        """
        size = 300
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=size, chunk_overlap=50)
        oversized = [c for c in chunks if len(c.text) > size * 1.5]
        assert not oversized, f"{len(oversized)} chunk(s) far exceeded chunk_size={size}"

    def test_consecutive_chunks_actually_overlap(self):
        """The point of overlap: adjacent chunks must share text.

        Checks that the tail of chunk N reappears in chunk N+1. Without
        this, a fact split across a boundary is unrecoverable.
        """
        overlap = 80
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=400, chunk_overlap=overlap)
        if len(chunks) < 2:
            pytest.skip("Need at least two chunks to compare")

        # Use a short probe from the tail so boundary-snapping
        # implementations still pass.
        probe_len = 20
        shared = 0
        for current, following in zip(chunks, chunks[1:]):
            probe = current.text[-probe_len:].strip()
            if probe and probe in following.text:
                shared += 1
        assert shared > 0, "No overlap detected between any consecutive chunks"

    def test_zero_overlap_is_allowed(self):
        chunks = chunk_text(SAMPLE, source="sample.txt", chunk_size=300, chunk_overlap=0)
        assert len(chunks) > 1


class TestMetadata:
    def test_extra_metadata_is_merged(self):
        chunks = chunk_text(
            SAMPLE,
            source="sample.txt",
            chunk_size=300,
            extra_metadata={"file_type": "pdf", "author": "test"},
        )
        assert all(c.metadata.get("file_type") == "pdf" for c in chunks)
        assert all(c.metadata.get("author") == "test" for c in chunks)

    def test_extra_metadata_does_not_clobber_source(self):
        chunks = chunk_text(
            SAMPLE, source="real.txt", chunk_size=300, extra_metadata={"file_type": "txt"}
        )
        assert all(c.metadata["source"] == "real.txt" for c in chunks)


class TestChunkIds:
    """make_chunk_id() is provided, not yours - these just document it."""

    def test_ids_are_stable(self):
        assert make_chunk_id("a.txt", 0, "hello") == make_chunk_id("a.txt", 0, "hello")

    def test_ids_differ_when_content_changes(self):
        assert make_chunk_id("a.txt", 0, "hello") != make_chunk_id("a.txt", 0, "goodbye")

    def test_ids_differ_across_documents_and_positions(self):
        assert make_chunk_id("a.txt", 0, "x") != make_chunk_id("b.txt", 0, "x")
        assert make_chunk_id("a.txt", 0, "x") != make_chunk_id("a.txt", 1, "x")


def _chunk_worker(queue, text: str, size: int, overlap: int) -> None:
    """Run chunk_text in a child process and report a small summary.

    Module-level (not a closure) because Windows uses the "spawn" start
    method, which pickles the target by reference and re-imports this
    module in the child.

    Only counts go on the queue, never chunk text — a large payload on a
    multiprocessing queue can deadlock the join.
    """
    from app.rag.chunking import chunk_text as ct

    try:
        chunks = ct(text, source="s.txt", chunk_size=size, chunk_overlap=overlap)
        queue.put(("ok", len(chunks), len({c.text for c in chunks})))
    except ValueError as exc:
        queue.put(("valueerror", str(exc), 0))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}", 0))


class TestForwardProgress:
    """Guards against the chunking loop never terminating.

    Boundary-aware chunking can end a chunk EARLY (at a paragraph or
    sentence break) rather than at exactly start+chunk_size. If the next
    window then rewinds by the full chunk_overlap, the cursor can land
    at — or before — where it already was, and the loop stops making
    forward progress. It never raises; the process just hangs.

    This is pitfall #3 in docs/guidecode.md §1.5.

    Runs in a CHILD PROCESS, not a thread: a runaway loop can be killed
    with terminate(). A stuck thread cannot be killed, and one spinning
    at 100% CPU starves the rest of the suite through the GIL.
    """

    TIMEOUT = 10.0

    @staticmethod
    def _run_isolated(size: int, overlap: int):
        """Returns (hung: bool, outcome: tuple | None)."""
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        process = ctx.Process(
            target=_chunk_worker, args=(queue, SAMPLE, size, overlap), daemon=True
        )
        process.start()
        process.join(timeout=TestForwardProgress.TIMEOUT)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            return True, None

        outcome = queue.get(timeout=5) if not queue.empty() else None
        return False, outcome

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [
            (1000, 150),  # configured default — 15%
            (400, 80),    # 20%
            (300, 200),   # 67%
            (200, 150),   # 75%
            (100, 80),    # 80%
            (100, 99),    # pathological, but legal per the ValueError rule
        ],
    )
    def test_terminates_for_any_legal_overlap(self, size, overlap):
        """Every overlap < chunk_size must terminate.

        `chunk_overlap < chunk_size` is the documented contract, so any
        such pair has to finish — either producing chunks or rejecting
        the arguments with ValueError. Hanging is never acceptable.
        """
        hung, outcome = self._run_isolated(size, overlap)

        assert not hung, (
            f"chunk_text() did not terminate with chunk_size={size}, "
            f"chunk_overlap={overlap} ({overlap / size:.0%} of chunk_size).\n"
            "The cursor stopped advancing. Either clamp the next start so it "
            "always moves forward, or raise the boundary-search floor above "
            "chunk_overlap — see docs/guidecode.md §1.5 pitfall #3."
        )
        assert outcome is not None, "Child process exited without reporting"

        kind, first, unique = outcome
        assert kind in {"ok", "valueerror"}, f"Unexpected failure: {first}"
        if kind == "ok":
            assert first > 0, "Terminated but produced no chunks"
            assert first == unique, (
                f"{first - unique} duplicate chunk(s) — the cursor is "
                "revisiting text it already emitted."
            )
