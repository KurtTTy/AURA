# Security Policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting instead: go to the **Security** tab
of this repository and choose **Report a vulnerability**. That opens a private
thread visible only to the maintainers.

This is a personal project maintained in spare time, so responses are best
effort — expect a few days rather than a few hours. You will get an
acknowledgement either way, including if the report turns out to be out of
scope.

## Supported versions

The latest commit on `main`. There are no long-lived release branches and no
backports; fixes land on `main`.

## Where the real risk is

AURA runs locally and holds credentials, so the interesting surfaces are worth
naming rather than leaving to a generic template.

**Model-written SQL.** `run_sql()` executes SQL produced by a language model.
It is gated by an allowlist in `backend/app/analyst/tools.py`: `SELECT` and
`WITH` only, single statement, forbidden keywords matched on word boundaries,
and a row cap. A way to get a statement past that gate — anything that writes,
attaches a database, reads the filesystem, or escapes the read path — is a
genuine vulnerability and worth reporting.

**Prompt injection through ingested content.** Retrieved chunks are placed in
the model's context and it is instructed to trust them. A document, spreadsheet
cell, or web page that manipulates the model into ignoring its instructions is
a real concern. It is also partly inherent to retrieval-augmented systems, so
concrete examples help far more than the general observation.

**Credential handling.** API keys live in `.env`, which is gitignored. Keys must
never reach logs, error messages, or terminal output. The Gemini provider passes
its key as a URL parameter and scrubs it from exception text for exactly this
reason — if you find a path where any key leaks, that is a bug.

**Path handling.** `/ingest` and `/load` take paths from the user, and
`table_name_from_path()` sanitises names before they reach a SQL identifier.
Path traversal or an unsanitised identifier reaching a query is in scope.

## What is not a vulnerability

- **A wrong or invented answer from the model.** Bad output is a quality issue,
  not a security one. Open a normal issue for it.
- **Vulnerabilities in Ollama, Chroma, DuckDB, or another dependency.** Report
  those upstream; if AURA's usage makes an upstream flaw materially worse, that
  part is in scope here.
- **Anything requiring an attacker to already have your machine or your
  `.env`.** At that point the credentials are already lost.
- **Sending your data to a cloud provider when you have configured one.** That
  is the documented behaviour of setting an API key. Embeddings always stay
  local; if you find generation or embedding leaving the machine *without* a
  configured provider, that is very much a bug.
