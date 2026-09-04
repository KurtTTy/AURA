from __future__ import annotations

import re

import pytest

from app.llm_providers import Message
from app.rag.prompt import NO_CONTEXT_ANSWER, build_rag_messages, format_context
from app.rag.retrieve import RetrievedChunk

QUESTION = "How many days of paid parental leave do employees get?"


def make_chunks() -> list[RetrievedChunk]:
    """Two chunks, most relevant first - the shape retrieve() returns."""
    return [
        RetrievedChunk(
            text="Employees are entitled to 90 days of paid parental leave.",
            source="handbook.md",
            chunk_index=0,
            score=0.746,
            metadata={"source": "handbook.md", "chunk_index": 0},
        ),
        RetrievedChunk(
            text="Hotel accommodation is reimbursed up to 75 per night.",
            source="expenses.md",
            chunk_index=2,
            score=0.577,
            metadata={"source": "expenses.md", "chunk_index": 2},
        ),
    ]


# ═══════════════════════════════════════════════════════════════════
#  §2a - format_context()
# ═══════════════════════════════════════════════════════════════════


class TestFormatContext:
    def test_returns_a_string(self):
        assert isinstance(format_context(make_chunks()), str)

    def test_empty_chunks_returns_empty_string(self):
        """No chunks is a normal case, not an error - don't crash."""
        assert format_context([]) == ""

    def test_every_chunk_text_appears(self):
        """The whole point: if a chunk's text isn't here, the model never
        sees it and the answer can't be grounded in it."""
        chunks = make_chunks()
        output = format_context(chunks)
        for chunk in chunks:
            assert chunk.text in output, f"chunk {chunk.chunk_index} text missing"

    def test_every_source_appears(self):
        """Citations need the filename, or the reader can't check the claim."""
        output = format_context(make_chunks())
        assert "handbook.md" in output
        assert "expenses.md" in output

    def test_order_is_preserved(self):
        """Most relevant first. Models weight earlier context more heavily,
        so re-sorting throws away retrieval's ranking."""
        chunks = make_chunks()
        output = format_context(chunks)
        assert output.index(chunks[0].text) < output.index(chunks[1].text)

    def test_chunks_are_separated(self):
        """Something must sit between chunks - a label, a rule, a blank line.
        Run them together and the model reads it as one document, which
        muddles the citations."""
        chunks = make_chunks()
        output = format_context(chunks)
        raw_length = sum(len(c.text) for c in chunks)
        assert len(output) > raw_length + 10, (
            "Output is barely longer than the raw text - are the chunks "
            "labelled and separated at all?"
        )

    def test_each_chunk_has_a_referable_label(self):
        """The model needs a handle to cite, e.g. "according to [1]".

        This assumes the [N] convention recommended in guidecode.md §2a.
        If you deliberately chose a different labelling scheme, change this
        test to match it - just make sure SOME per-chunk handle exists.
        """
        output = format_context(make_chunks())
        labels = re.findall(r"\[\d+\]", output)
        assert len(labels) >= 2, (
            f"Expected a [N] label per chunk, found {labels}. "
            "See guidecode.md §2a for the recommended layout."
        )

    def test_labels_start_at_one_not_zero(self):
        """[1] reads naturally to a model and a human; [0] invites confusion."""
        output = format_context(make_chunks())
        assert "[1]" in output, "Number the labels from 1, not 0"


# ═══════════════════════════════════════════════════════════════════
#  §2b - build_rag_messages()
# ═══════════════════════════════════════════════════════════════════


class TestBuildRagMessages:
    def test_returns_messages(self):
        messages = build_rag_messages(QUESTION, make_chunks(), [])
        assert isinstance(messages, list)
        assert messages, "Returned an empty list"
        assert all(isinstance(m, Message) for m in messages)

    def test_exactly_one_system_message_and_it_is_first(self):
        messages = build_rag_messages(QUESTION, make_chunks(), [])
        systems = [m for m in messages if m.role == "system"]
        assert len(systems) == 1, f"Expected 1 system message, got {len(systems)}"
        assert messages[0].role == "system", "The system message must come first"

    def test_system_prompt_is_not_empty(self):
        """It carries the grounding and refusal instructions - the whole
        reason this file exists."""
        messages = build_rag_messages(QUESTION, make_chunks(), [])
        assert len(messages[0].content.strip()) > 50, (
            "System prompt looks too short to carry grounding + refusal "
            "instructions - see the checklist in guidecode.md §2b"
        )

    def test_question_reaches_the_model(self):
        messages = build_rag_messages(QUESTION, make_chunks(), [])
        assert any(QUESTION in m.content for m in messages), "The question is missing"

    def test_context_reaches_the_model(self):
        chunks = make_chunks()
        messages = build_rag_messages(QUESTION, chunks, [])
        combined = "\n".join(m.content for m in messages)
        for chunk in chunks:
            assert chunk.text in combined, (
                f"chunk {chunk.chunk_index} never reaches the model - "
                "did you call format_context()?"
            )

    def test_context_goes_in_the_user_message_not_the_system_message(self):
        """A documented design decision, worth enforcing.

        The system prompt is stable instructions; context changes every
        request. Mixing them makes small models lose track of which text is
        instruction and which is evidence.
        """
        chunks = make_chunks()
        messages = build_rag_messages(QUESTION, chunks, [])
        assert chunks[0].text not in messages[0].content, (
            "Retrieved context is in the SYSTEM message. Put it in the user "
            "message instead - see guidecode.md §2b."
        )

    def test_last_message_is_the_user_turn(self):
        """The model answers the last thing it read."""
        messages = build_rag_messages(QUESTION, make_chunks(), [])
        assert messages[-1].role == "user"

    def test_history_is_included(self):
        history = [
            Message(role="user", content="What is the leave policy called?"),
            Message(role="assistant", content="It is the parental leave entitlement."),
        ]
        messages = build_rag_messages(QUESTION, make_chunks(), history)
        combined = "\n".join(m.content for m in messages)
        assert "parental leave entitlement" in combined, "History was dropped"

    def test_history_none_is_accepted(self):
        """`history` is optional - None must not crash."""
        messages = build_rag_messages(QUESTION, make_chunks(), None)
        assert messages

    def test_empty_chunks_does_not_crash(self):
        """Return a valid message list even with no context.

        Deciding what to do about an empty result is answer_question()'s job
        (§3) - this function's only duty is not to blow up.
        """
        messages = build_rag_messages(QUESTION, [], [])
        assert messages
        assert messages[0].role == "system"


class TestConstants:
    """NO_CONTEXT_ANSWER is provided, not yours - documented here so you
    know it exists and what §3 uses it for."""

    def test_no_context_answer_is_usable(self):
        assert isinstance(NO_CONTEXT_ANSWER, str)
        assert len(NO_CONTEXT_ANSWER) > 20
