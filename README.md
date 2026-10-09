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

Offline tooling behind [Angrove](https://github.com/rbaltodano/Angrove-iOS), a private iOS study
app whose language model, retrieval, and Insight Tree all run on the iPhone.

The app has no network backend. This repository builds the assets the app ships with and measures
the model and retrieval behavior it depends on. Nothing here runs in production.

## What it produces

| Area | Entry points | Output used by the app |
| --- | --- | --- |
| Source corpus | `corpus/sources.yaml`, `ingest_corpus.py` | Chunked, embedded primary sources: Scripture, the Church Fathers, Thomas Aquinas, councils, catechisms, and classical philosophy |
| On-device retrieval | `scripts/export_on_device_grounding.py` | `embeddings.bin` and `passages.json` for the app's `LocalGrounding` store |
| On-device embeddings | `scripts/export_minilm_coreml.py` | `all-MiniLM-L6-v2` as a Core ML model, shared by retrieval and the Insight Tree |
| Language model packages | `scripts/export_litert_aquinas.py`, `scripts/export_litert_aquinas_stage.py` | Gemma 4 LiteRT-LM packages for the on-device runtime |
| Fine-tuning and quantization research | `scripts/train_voice_lora.py`, `scripts/rebuild_training_data.py`, `scripts/dwq_*.py`, `scripts/*quantized_candidate*.py` | Candidate checkpoints; the launch build uses the stock Gemma 4 E4B package |
| Home content | `today_in_history.py` | The hand-curated Today in History dataset |

## Evaluation

| Suite | Entry point | Measures |
| --- | --- | --- |
| Retrieval | `evaluation/evaluate_retrieval.py`, `evaluation/retrieval_cases.json` | Whether on-device grounding retrieves passages that actually contain the answer |
| Prompt quality | `scripts/evaluate_prompt_quality.py`, `evaluation/prompt_quality_*.json` | End-to-end answer contracts and reviewed quality cases |
| Latency and quality | `scripts/benchmark_latency_quality.py` | One checkpoint against the reviewed latency/quality set |
| Evidence ablation | `evaluation/evidence_ablation/` | How retrieved evidence changes answers |

Results and the decisions they informed are summarized in the app's
[evaluation report](https://github.com/rbaltodano/Angrove-iOS/blob/main/Documentation/Evaluation.md)
and [case study](https://github.com/rbaltodano/Angrove-iOS/blob/main/Documentation/Case-Study.md).

## Python reference implementation

Before the app moved fully on device (September 26, 2026), a local FastAPI and MLX service served
it during development. `server.py`, `main.py`, `structured_generation.py`,
`generation_coordinator.py`, `grounding_retrieval.py`, `relatedness.py`, `insight_tree.py`, and
`tree_store.py` remain as the tested reference for behavior the app now implements in Swift:
structured generation, foreground/background preemption, ranked grounding, MiniLM relatedness,
and Insight Tree decisions.

## Setup

Requires an Apple silicon Mac and Python 3. Model weights, generated corpora, and evaluation
outputs are local artifacts, intentionally excluded from source control; see
[`CLAUDE.md`](CLAUDE.md) for the expected layout.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/evaluate_prompt_quality.py --validate-only
```

## Related repositories

- [Angrove iOS](https://github.com/rbaltodano/Angrove-iOS): the SwiftUI app and on-device runtime.
- [Angrove Foundations](https://github.com/rbaltodano/Angrove-Foundations): product, design, and
  model-integration documentation.
- [angrove.app](https://angrove.app): the product site, User Guide, and case studies.

## Naming

Angrove was originally named Aquinas. Script names, model directories, and some identifiers keep
that name so existing artifacts and tooling continue to work; references to Thomas Aquinas are to
the theologian.

## License

Copyright © 2026 Ryan Baltodano (Sine Viridian). All rights reserved. The source is published for
reference and review; see [`LICENSE`](LICENSE). Third-party models and libraries keep their own
licenses, and the public-domain source texts remain in the public domain.
