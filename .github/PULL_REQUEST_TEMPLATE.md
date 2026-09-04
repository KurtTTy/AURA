## Description

<!-- What this PR does and why. -->



**Related issue:** Fixes #(number) — or "none"

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (API shape, config keys, or CLI commands)
- [ ] Documentation
- [ ] Refactoring
- [ ] Tests or CI

## Areas affected

- [ ] Retrieval — `app/rag/`
- [ ] Analyst — `app/analyst/`
- [ ] Providers — `app/llm_providers/`
- [ ] API — `app/routers/`, `app/main.py`, `app/models/`
- [ ] Terminal client — `scripts/ask.py`
- [ ] Configuration — `app/config.py`, `.env.example`
- [ ] Tests / CI
- [ ] Documentation

## How this was tested

<!-- Commands you ran and what came back. Paste the output. -->

```
cd backend
..\.venv\Scripts\python.exe -m pytest -m "not integration"

```

## Checklist

- [ ] Offline tests pass (86; the other 6 need a live Ollama)
- [ ] New behaviour has a test, or a note on why it can't have one
- [ ] `.env.example` updated if a setting was added or renamed
- [ ] README or `docs/info.md` updated if behaviour or commands changed
- [ ] No documents, datasets, charts, spreadsheets, or `.env` committed
- [ ] Ready for review

---

## Extra checks by area

Only fill in the section you touched. These cover the changes that can pass
every test and still be wrong.

<details>
<summary><b>Providers</b> — the contract in <code>base.py</code></summary>

- [ ] `health()` never raises — one that throws can't decide fallback
- [ ] `list_models()` never raises — `[]` means "I don't know"
- [ ] Cloud `embed()` still refuses, keeping one coordinate space in the store
- [ ] Concatenated `chat_stream()` fragments equal what `chat()` returns
- [ ] The provider and model that actually ran are reported, not the requested ones
- [ ] Ran the full `pytest` against a live Ollama, not just the offline subset

</details>

<details>
<summary><b>Analyst tools</b> — the SQL safety gate</summary>

`_is_safe_select()` is the only thing between a language model and the database.

- [ ] Non-`SELECT` statements rejected
- [ ] `SELECT 1; DROP TABLE x` rejected
- [ ] Forbidden keywords rejected
- [ ] A column named `last_update` is still **allowed** — matching is on word
      boundaries, and a substring check gets this wrong
- [ ] Tool failures return `ToolResult(ok=False)` and never raise
- [ ] The turn cap in `agent.py` still holds

</details>

<details>
<summary><b>Prompts</b> — no test catches a worse prompt</summary>

- Model tested against:
- What you asked, and what came back:
- [ ] Refusal still works — a question the sources can't answer is refused
      rather than invented

</details>
