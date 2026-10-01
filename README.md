<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/angrove-app-icon-dark.png">
    <img src="docs/brand/angrove-app-icon-light.png" alt="Angrove app icon" width="128">
  </picture>
</p>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/angrove-logo-light-text.png">
    <img src="docs/brand/angrove-logo-dark-text.png" alt="Angrove" width="360">
  </picture>
</p>

# Angrove Backend

The local FastAPI and MLX development service behind Angrove.

Angrove is a private, local-first study and conversation environment for serious
questions about philosophy, theology, Scripture, meaning, and human flourishing.
This repository supports the iOS client with model generation, source-grounded
retrieval, Insight Tree operations, and conversation-scoped persistence.

## Main areas

| Path | Responsibility |
| --- | --- |
| `server.py` | FastAPI routes, schemas, startup, and HTTP boundaries |
| `main.py` | MLX model loading and serialized generation |
| `structured_generation.py` | Prompt contracts, JSON validation, and filtered streaming |
| `generation_coordinator.py` | Foreground priority and background-task preemption |
| `grounding_retrieval.py` | Corpus retrieval and ranked evidence |
| `relatedness.py` | Local MiniLM embeddings, vector comparison, centroids, and semantic relatedness |
| `insight_tree.py` / `tree_store.py` | Tree decisions and SQLite persistence |
| `tests/` | Focused contract and regression tests |
| `evaluation/` | Prompt-quality and retrieval evaluation data |
| `scripts/` | Corpus, conversion, export, and benchmarking tools |

The Home dashboard also reads optional development-time sections from this service: Loose Thread,
Terms You Glossed Over, Today in History, and Your Quote. The tree-analysis route returns before
the quote-notability check finishes; that best-effort check runs in the background and never
changes a completed tree-analysis response.

Read [`CLAUDE.md`](CLAUDE.md) for the current model checkpoint, API contract, and safety rules.
Read [`Angrove Foundations — MODEL-INTEGRATION.md`](https://github.com/rbaltodano/Aquinas-Foundations/blob/main/MODEL-INTEGRATION.md)
before changing a client-facing contract or model behavior.

## Local setup

```sh
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

## Related repositories

- [Angrove iOS](https://github.com/rbaltodano/Aquinas-iOS) — SwiftUI client and local-first study experience.
- [Angrove Foundations](https://github.com/rbaltodano/Aquinas-Foundations) — shared product, design, and architecture contracts.

## Project status

This is an active development and research repository, not a hosted production
API. The backend is primarily used for local integration, model experiments,
retrieval evaluation, and validating client-facing contracts.

Because the service depends on local model checkpoints and generated corpus data,
the commands above assume those development assets have already been provisioned.

## Semantic exploration

The backend provides the semantic layer behind the Insight Tree. A local
embedding model turns concise concept descriptions into normalized vectors;
cosine similarity compares those vectors, and centroids represent higher-level
nodes built from related Insights. The resulting relatedness signals drive
retrieval, grouping, graph topology, and the visual distances between ideas.

This is one of the central capabilities of Angrove: the app can help a person
explore how ideas interact in semantic space, not only generate a response to
the latest question. The language model proposes and explains concepts, while
the relatedness provider performs the numeric comparison and graph decisions.
