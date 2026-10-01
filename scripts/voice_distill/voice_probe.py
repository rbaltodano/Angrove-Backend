"""Answer the probe questions under each voice variant. Runs on the pod."""
import json, sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
import os
M = os.environ.get("MODEL", "/root/models/gemma-4-E4B-it-base"); D = "/root/distill/"
tok = AutoTokenizer.from_pretrained(M); tok.padding_side = "left"
m = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()
T = json.load(open(D + "templates.json")); P = json.load(open(D + "voice_probe.json"))
TEMP = float(__import__("os").environ.get("TEMP", "0"))
variants = sys.argv[1:] or ["voice", "friend_a", "friend_b"]
convs, keys = [], []
for v in variants:
    for r in P:
        sys_ = r["system_train"].replace(T["scholarly"], T[v])
        convs.append([{"role": "system", "content": sys_}, {"role": "user", "content": T["wrapper"] + "\n\nUser question:\n" + r["question"]}])
        keys.append((v, r["question"]))
texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]
enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
with torch.no_grad(): g = m.generate(**enc, max_new_tokens=450, do_sample=TEMP>0, temperature=TEMP or None, top_p=0.9 if TEMP else None)
out = [{"variant": v, "question": q, "answer": tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()} for (v, q), x in zip(keys, g)]
json.dump(out, open(D + os.environ.get("OUT", "voice_probe_out.json"), "w"), ensure_ascii=False, indent=1)
