# Voice self-distillation tooling (tracked copies)

Working copies run from `data/aquinas_voice_distill/` and `data/aquinas_voice_v1/` (gitignored;
the scripts resolve data files relative to their own folder). Copy them back there to run.

1. `build_seeds.py`: about 2,800 seeds (Summa articles with and without the article as a reference,
   passages from the Fathers and catechisms, Scripture chapters, topics, short questions), with
   evaluation questions and sealed-2 topics excluded through `check_overlap.py`.
2. Precompute prompts (`seeds-with-prompts.jsonl`, `templates.json`) with `render.py`: the voice
   prompt for generation, the app's short scholarly personality for training.
3. On the pod: `distill_generate.py` writes questions, answers them under the app's prompt (greedy),
   and adds about 300 follow-ups.
4. Locally: `filter_generated.py` rejects markdown, leaked prompt text, forms of address, loops,
   bad lengths, evaluation overlap, and any locator not present in the example's references.
5. Train with `scripts/train_voice_lora.py`.
