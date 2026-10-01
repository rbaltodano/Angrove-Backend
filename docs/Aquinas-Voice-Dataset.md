# Angrove voice dataset v1 (2026-09-30)

The fine-tuning dataset for Gemma 4 E4B: 213 hand-written examples in a voice the owner chose
after two pilots. It lives in `data/aquinas_voice_v1/`. `data/` is gitignored, so this document
records how the set was made and how to rebuild the training files.

## Voice

Athanasian in manner, scholarly and a little ornate, in modern English (owner feedback on pilot v2:
"more scholarly, don't be afraid to use some flowery language"). Four habits:
1. State the problem plainly.
2. Reason from God's goodness to what was fitting.
3. Use one vivid image.
4. Land a short, confident conclusion.

Answers use real distinctions and precise references, run 150–200 words (definitions 65–90), and
carry the app's `{{term}}` Insight markers.

**Every attribution must be real.** An image or claim is credited to Aquinas, Athanasius, or anyone
else only where they actually make it, with the locus cited where known. One fabricated
attribution in the pilot was caught and removed.

## Composition

| Kind | Count | Purpose |
| --- | --- | --- |
| core | 93 | Questions answered in the voice without passages |
| grounded | 43 | Answers that use only the passages supplied. The passages are formatted exactly as the app delivers them: Summa articles as "Question / Aquinas's conclusion / Aquinas's own answer" (`summa_evidence.py`), *On the Incarnation*, Augustine, Anselm, Justin, Scripture, and the Reformation confessions presented on their own terms. About a third carry an irrelevant extra passage, which the answer ignores. |
| definition | 27 | Short definitions |
| followup | 24 | Pronoun and item references, "say more" and "in one sentence" requests, topic changes that drop old context, valid pushback accepted, and mistaken pushback declined |
| honesty | 16 | False premises, anachronisms, unknowable details, and a passage that doesn't answer the question |
| offtopic | 10 | Brief, plain answers outside the corpus |

## Kept apart from evaluation

`check_overlap.py` rejects questions close to any dev, held-out, sealed-1, or sealed-2 question,
and any question on a sealed-2 topic. It lists answer mentions of those topics for review.
Reviewed false positives are listed in the script. Examples dropped or rewritten for leakage
include humility, fasting, grace perfecting nature, the resurrection of the body, and a mention of
the *Divine Comedy*. `data/training_data/train.jsonl` (2,553 rows, all one prompt, each answer a
full Summa article) is **not** part of this set.

## Rendering for training

`python3 render.py` writes `train.jsonl` (203) and `valid.jsonl` (10), a stable split by id. Each
example uses the app's exact conversation prompt, taken from a recorded generation
(`prompt-samples.json`, run `Q3-held-sim-1`): references in the app's format, the app's
latest-turn wrapper, and the pronoun note for follow-ups. The median rendered example is about
1,600 tokens.

**Personality alignment (owner decision needed).** The app's default *balanced* personality asks
for relaxed, casual language, which conflicts with this voice. Training renders under the app's
*scholarly* personality instead, which matches it. For the fine-tune to behave as trained, the app
should either make *scholarly* the default or rewrite *balanced* to describe this voice.

## Next

1. Back up `data/`: it exists only on the owner's Mac.
2. Train LoRA on a rented GPU. Start from `google/gemma-4-E4B-it`; the export pipeline was
   validated on RunPod on 2026-09-30.
3. Export with `scripts/export_litert_aquinas.py --skip-vision --quantization-recipe
   dynamic_wi4_afp32`. The stock export was 4.12 GB, so it needs the Apple-hosted asset pack.
4. On the phone, compare against stock E4B on sealed set 2, and read the answers side by side.

## Training results — 2026-09-30 (RunPod A100 80 GB, LoRA r16 on language layers)

| Run | Data | Settings | Valid loss | Outcome |
| --- | --- | --- | --- | --- |
| v1 | examples.jsonl | 3 epochs, lr 1e-4 | 2.90 → 1.55 | Voice achieved. **Fabricated citations**: a Catechism "lost toy" analogy, "Summa I–II q.16 a.1" on selfishness, a "seven loves" list |
| v2 | examples-v2 (locators kept only in grounded answers) | 3 epochs, lr 1e-4 | 2.88 → 1.57 | Fewer locators, but fluent confabulation: Gregory of Nyssa as "the soul is a part of God", a wrong Summa location, an invented Augustine analogy |
| v3 | examples-v2 | 1 epoch, lr 5e-5 | 2.88 → 1.77 | Voice mostly lost; still a wrong locator and an invented Gregory metaphor |
| stock + voice prompt | none | stock E4B | — | Plain prose with an image; no invented locators; Gregory broadly correct. Chosen for launch |

The stock model without the voice prompt was accurate but verbose, and it used markdown, LaTeX,
and "student". The voice prompt now lives in the app's scholarly personality (Aquinas-iOS
`feature/scholarly-default-voice`, `55f2a37`). The stock model still misplaces some Summa
locations on its own, prompt or no prompt.

**Lesson:** about 200 hand-written examples in a confident, reference-rich voice teach the style
of authority without the knowledge behind it. A later attempt needs thousands of examples whose
content comes from the model itself or from supplied passages, plus a fabrication check before
anything ships.

Tooling fixes found on the pod: install PyTorch built for CUDA 12.8 (`torch 2.11.0+cu128`) when the
driver reports CUDA 12.8; `apply_chat_template(..., return_dict=True)` for generation in
transformers 5.x.
