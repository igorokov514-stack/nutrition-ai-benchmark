# Nutrition AI Benchmark

A local Streamlit dashboard for comparing OpenAI and Anthropic models on a 600-question clinical nutrition benchmark. It runs direct and retrieval-augmented model calls, scores structured answers, optionally grades open-ended responses with an LLM judge, and provides downloadable results.

> [!IMPORTANT]
> The included question set is an **unreviewed pilot draft**. It has not received named registered-dietitian sign-off and must not be treated as clinical guidance, a validated assessment, or an official Commission on Dietetic Registration exam blueprint.

## Features

- Runs a pilot subset, the complete dataset, or filters by difficulty and question type
- Compares OpenAI RAG, OpenAI direct, and Anthropic direct configurations
- Scores multiple-choice and numeric-tolerance answers automatically
- Supports optional rubric-based LLM grading for open-ended answers
- Tracks latency and token usage in a local SQLite database
- Exports response-level results as JSON or CSV

## Quick start

Requires Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run nutrition_benchmark_local_dashboard.py
```

Open the local URL printed by Streamlit, then enter credentials in the sidebar. You can also provide them as environment variables:

```bash
cp .env.example .env
```

The app reads `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, and `OPENAI_VECTOR_STORE_ID`. It does not load `.env` automatically; export those variables in your shell or enter credentials in the UI. Never commit real credentials.

## Data and local state

The bundled dataset lives at [`data/questions.json`](data/questions.json). It contains answer keys and grading rubrics, so it is suitable for evaluation tooling but not for blind test distribution.

Runs are stored locally in `data/benchmark.sqlite3`. The database, its journal files, result exports, virtual environments, and Streamlit secrets are ignored by Git.

Model calls can incur API charges. Start with **Pilot: first 20**, use the preflight check, and confirm model names and token limits available to your accounts before starting a full run.

## Security and privacy

API credentials are held in process memory for calls and are not written to the benchmark database. Model responses and provider payloads are stored locally in SQLite; review them before sharing because they may contain retrieved text or other sensitive content.

## Project status

This repository is an experimental local benchmarking tool. Results depend on model versions, prompts, provider settings, judge configuration, and the unresolved quality limitations of the pilot dataset.
