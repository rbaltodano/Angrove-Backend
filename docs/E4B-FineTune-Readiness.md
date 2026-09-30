# Fine-tuning Gemma 4 E4B: readiness and gaps (2026-09-29)

The app now ships Google's stock Gemma 4 E4B LiteRT-LM package (`gemma-4-E4B-it.litertlm`,
3,659,530,240 bytes, SHA-256 `0b2a8980…`). It runs with F32 GPU activations; see the iOS repo's
`Documentation/Gemma4-E4B-Post-C9-Diagnostics.md`. A fine-tune is a **post-launch** step (iOS
`Documentation/Model-Strategy-Plan.md`, Phase 3). None of the retired E2B outputs are needed for
it (see `docs/retired-models/README.md`).

## What we already have

- Training data: `data/training_data/` and `data/raw_data/`. It's local only, so back it up.
- The MLX LoRA workflow and `scripts/fuse_aquinas.py`.
- The iOS evaluation harness: the frozen held-out set, `score_objective.py`, and the blind rubric
  review and phone runners in `Aquinas-iOS/LocalModels/e4b-eval/`. This is how a fine-tune gets
  judged against stock E4B.

## Gaps to close before training

1. **Configs and scripts are E2B-specific (small).**
   - `lora_config_v4.yaml` and `model_identity.MODEL_BASE_ID` name `google/gemma-4-E2B-it`.
   - `scripts/build_hf_aquinas_checkpoint.py` expects E2B's 600 language tensors. E4B has more
     layers, so the mapping and count checks need updating.
   - `docs/retired-models/Aquinas-Final/model.safetensors.index.json` shows the E2B layout to
     compare against.
2. **Export to the phone format (large; the real blocker).**
   - Google's E4B package is a QAT mixed-precision build: INT2 embeddings and per-layer
     embeddings, a mostly-INT4 decoder, and INT8 parts.
   - `scripts/export_litert_aquinas.py` (`litert_torch`) can't emit INT2 (iOS plan D7). A
     fine-tuned E4B exported that way will be larger, lose the QAT quality benefit, or both.
   - The last custom export, E2B `Aquinas-Final-LiteRT`, also failed to initialize on the phone.
   - **Options to investigate, in order:**
     1. Google's QAT checkpoint (`google/gemma-4-E4B-it-qat-mobile-transformers`) as the
        training base, with a QAT-preserving export if Google publishes the recipe.
     2. An INT4-only export, sized and tested on the phone first.
     3. A different runtime (MLX Swift) for the fine-tuned model.
   - Plain QLoRA on a QAT checkpoint discards its calibration, so plan a distillation-aware
     re-quantization.
3. **The training data doesn't match the current app voice (medium).**
   - `train.jsonl` is mostly Summa-style disputation text ("Objection 1: It would seem…") and has
     no `{{term}}` Insight markers.
   - The app prompt asks for warm, modern prose and says not to imitate archaic source prose.
   - Build a new dataset in the current voice and output contract, using the existing data as
     source material. Keep fine-tuning behavior- and voice-focused; facts come from retrieval.

## Order of work

1. Back up `data/`.
2. Settle the export path (gap 2) with a no-training smoke test: export stock E4B yourself and run
   it on the phone.
3. Build the new dataset (gap 3).
4. Update the configs and scripts (gap 1).
5. Train, export, then evaluate against stock E4B on the phone (F32 GPU). Ship only on a clear win.
