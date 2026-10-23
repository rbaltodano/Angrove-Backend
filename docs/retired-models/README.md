# Retired E2B model artifacts (2026-09-29)

The iOS app now ships Google's Gemma 4 E4B LiteRT-LM package (Aquinas-iOS PR #8). The Angrove
fine-tune of Gemma 4 **E2B** is retired, and its large local weights were deleted to free disk.
This folder keeps each artifact's small metadata, so the record and the rebuild path survive.

## What was deleted

| Artifact | Size | What it was | Metadata kept here |
| --- | --- | --- | --- |
| `models/Aquinas-Final` | 9.3 GB | MLX fused text checkpoint (base E2B + `aquinas_adapters`) | `Aquinas-Final/` (config, tokenizer config, chat template, tensor index) |
| `models/Aquinas-Final-HF` | 10.3 GB | Canonical Hugging Face Gemma 4 checkpoint built from the fused weights (600 language tensors) | `Aquinas-Final-HF/` (incl. `aquinas_conversion_manifest.json`) |
| `models/Aquinas-Final-LiteRT` | 2.7 GB | Custom LiteRT-LM export, SHA-256 `5cb26c8e…5569`. Failed on the phone (see `CLAUDE.md`) | — (SHA and history are in `CLAUDE.md`) |
| `runs/dwq-controlled` | 1.7 GB | DWQ distillation-quantization experiment (2026-08-13) | `dwq-controlled/` (configs, monitors, logs, statuses) |
| LM Studio `mlx-community/aquinas-gemma-4-mlx` | 9.3 GB | Copy of `Aquinas-Final` with a text-only (`Gemma4ForCausalLM`) config for LM Studio | `lmstudio-aquinas-gemma-4-mlx/` |
| iOS `gemma-4-E2B-it.litertlm` ("B0") | 3.9 GB | The app's previous bundled model, SHA-256 `9a6345f1…65282`, 3,862,121,696 bytes | — (recorded in the iOS E4B progress ledger) |
| LM Studio `gemma-4-E2B-it-MLX-4bit` | 4.3 GB | Stock E2B, re-downloadable | — |

## What was kept, and how to rebuild

- **`models/aquinas_adapters/` (195 MB) is the fine-tune itself.** Adapters, plus checkpoints every
  100 iterations.
  - `adapters.safetensors`: SHA-256 `52b77a6587a62bf9ddbbf98afadcbd78696918241f8cb1ef76bd23b8073c541b`.
  - `adapter_config.json` records the hyperparameters actually used: rank 8, scale 20, 600
    iterations, learning rate 1e-5, 16 layers. These differ from `lora_config_v4.yaml` (rank 32,
    5,000 iterations), which was never the shipped run.
- **Training data:** `data/training_data/` (train 2,553, valid 280, test 280 examples) and
  `data/raw_data/aquinas_train.jsonl` (6,129). `data/` is gitignored, so **this Mac holds the
  only copy. Back it up.**
- **Rebuilding `models/Aquinas-Final`:** run `scripts/fuse_aquinas.py`. It fuses `aquinas_adapters`
  into `google/gemma-4-E2B-it`, the `MODEL_BASE_ID` in `model_identity.py`.
- **Rebuilding `models/Aquinas-Final-HF`:** run `scripts/build_hf_aquinas_checkpoint.py` (see
  `CLAUDE.md`).
- `model_identity.MODEL_PATH` still defaults to `models/Aquinas-Final`. Evaluation and quantization
  scripts that read it need that fuse step first.
