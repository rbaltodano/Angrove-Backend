"""Runs on the GPU pod. Stage 1 writes a question per seed, stage 2 answers it under the app's
prompt with the voice instruction, stage 3 adds follow-up turns. Greedy decoding for answers.
Usage: python distill_generate.py --model /root/models/gemma-4-E4B-it-base --dir /root/distill"""
import argparse, json, random, re, pathlib, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--dir", required=True)
ap.add_argument("--batch", type=int, default=24); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--voice", default="voice"); ap.add_argument("--answer-temp", type=float, default=0.0)
a = ap.parse_args(); D = pathlib.Path(a.dir)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
T = json.load(open(D / "templates.json"))
PRON = {"that","this","these","those","it","its","they","them","their","he","him","his","she","her"}

def generate(convs, max_new, sample, temp=0.8):
    outs = []
    for i in range(0, len(convs), a.batch):
        chunk = convs[i:i + a.batch]
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new, do_sample=sample, temperature=temp if sample else None, top_p=0.95 if sample else None)
        outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        print(f"  {min(i + a.batch, len(convs))}/{len(convs)}", flush=True)
    return outs

def user_turn(q, note=""):
    return T["wrapper"] + "\n" + note + "\nUser question:\n" + q

seeds = [json.loads(l) for l in open(D / "seeds-with-prompts.jsonl")]
if a.limit: seeds = seeds[: a.limit]
# Stage 1: questions
qs = generate([[{"role": "user", "content": s["instruction"]}] for s in seeds], 80, True)
keep = []
for s, q in zip(seeds, qs):
    q = q.strip().strip('"').split("\n")[0].strip()
    if q.endswith("?") and 12 <= len(q) <= 220: s["question"] = q; keep.append(s)
print("questions kept", len(keep), "of", len(seeds), flush=True)
# Stage 2: answers
VOICE = T[a.voice]
ans = generate([[{"role": "system", "content": s["system_train"].replace(T["scholarly"], VOICE)}, {"role": "user", "content": user_turn(s["question"])}] for s in keep], 450, a.answer_temp > 0, a.answer_temp)
rows = []
for s, x in zip(keep, ans):
    rows.append({"id": s["id"], "kind": s["kind"], "references": s["references"], "system_train": s["system_train"],
                 "history": [], "question": s["question"], "answer": x})
# Stage 3: follow-ups on a sample of ungrounded pairs
random.seed(7)
base = [r for r in rows if not r["references"] and r["kind"] in ("topic", "summa")]
random.shuffle(base); base = base[:320]
fq = generate([[{"role": "user", "content": f"A person asked: {r['question']}\nThey got this answer: {r['answer'][:700]}\n\nWrite their natural follow-up question. Use a pronoun such as it, that, or he to refer back, or ask for an example, or ask for a shorter version. Output only the question."}] for r in base], 60, True)
fu = []
for r, q in zip(base, fq):
    q = q.strip().strip('"').split("\n")[0].strip()
    if not (q.endswith("?") and 6 <= len(q) <= 200): continue
    note = ""
    if set(re.findall(r"[a-z]+", q.lower())) & PRON:
        note = f"\nThis question continues the conversation. The previous question was: “{r['question']}” Read a pronoun in the new question as pointing to what that question asked about.\n"
    fu.append((r, q, note))
fa = generate([[{"role": "system", "content": r["system_train"].replace(T["scholarly"], VOICE)},
                {"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]},
                {"role": "user", "content": user_turn(q, note)}] for r, q, note in fu], 450, a.answer_temp > 0, a.answer_temp)
for (r, q, note), x in zip(fu, fa):
    rows.append({"id": r["id"] + "-f", "kind": "followup", "references": [], "system_train": r["system_train"],
                 "history": [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                 "question": q, "note": note, "answer": x})
with open(D / "generated.jsonl", "w") as f:
    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
print("generated", len(rows), flush=True)
