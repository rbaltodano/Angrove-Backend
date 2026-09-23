# Aquinas Backend

[![Tests](https://github.com/rbaltodano/Aquinas-Backend/actions/workflows/tests.yml/badge.svg)](https://github.com/rbaltodano/Aquinas-Backend/actions/workflows/tests.yml)

The local FastAPI and MLX service used by [Aquinas](https://github.com/rbaltodano/Aquinas-iOS)
during development. It provides model generation, structured response validation, semantic
retrieval, Insight Tree operations, and conversation-scoped SQLite persistence.

> **Development topology only.** This Mac-hosted service supports integration, evaluation, and
> advanced tree features while the app is being built. It is not a hosted production service; the
> product boundary remains private, local-first, on-device operation.

## Main areas

| Path | Responsibility |
| --- | --- |
| `server.py` | FastAPI routes, schemas, startup, and HTTP boundaries |
| `main.py` | MLX model loading and serialized generation |
| `structured_generation.py` | Prompt contracts, JSON validation, and filtered streaming |
| `generation_coordinator.py` | Foreground priority and background-task preemption |
| `grounding_retrieval.py` | Corpus retrieval and ranked evidence |
| `relatedness.py` | MiniLM embeddings and similarity |
| `insight_tree.py` / `tree_store.py` | Tree decisions and SQLite persistence |
| `tests/` | Focused contract and regression tests |
| `evaluation/` | Prompt-quality and retrieval evaluation data |
| `scripts/` | Corpus, conversion, export, and benchmarking tools |

The Home dashboard also reads optional development-time sections from this service: Loose Thread,
Terms You Glossed Over, Today in History, and Your Quote. The tree-analysis route returns before
the quote-notability check finishes; that best-effort check runs in the background and never
changes a completed tree-analysis response.

Read [`CLAUDE.md`](CLAUDE.md) for the current model checkpoint, API contract, and safety rules.
Read [`MODEL-INTEGRATION.md`](https://github.com/rbaltodano/Aquinas-Foundations/blob/main/MODEL-INTEGRATION.md)
in Aquinas Foundations before changing a client-facing contract or model behavior.

## Local setup

Requires an Apple silicon Mac (MLX) and Python 3.14. Model weights are not included in this
repository; see [`CLAUDE.md`](CLAUDE.md) for the expected checkpoint layout.

```sh
python3 -m venv aquinas_env
source aquinas_env/bin/activate
pip install -r requirements.txt
uvicorn server:app --reload
```

Run focused tests and contract validation with:

```sh
python -m unittest discover -s tests
python scripts/evaluate_prompt_quality.py --validate-only
```

Large model weights, generated corpora, databases, and evaluation outputs are local artifacts and
are intentionally excluded from source control.

## License

Copyright © 2026 Ryan Baltodano. All rights reserved. The source is public for reference and
review; see [`LICENSE`](LICENSE) for details.
