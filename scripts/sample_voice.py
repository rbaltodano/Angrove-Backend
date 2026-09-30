"""Generates answers from a merged checkpoint for a quick voice check (greedy).
Usage: python sample_voice.py --model /root/models/aquinas-e4b-voice-v1 [--base ...]"""
import argparse, json, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
QS = ["Why is hope different from wishful thinking?", "What did Augustine mean by the order of love?",
      "Is it selfish to want to be happy?", "What's 7 times 8?"]
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--system", default=None)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model)
m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
sysmsg = open(a.system).read() if a.system else None
for q in QS:
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": q}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    ids = enc["input_ids"]
    out = m.generate(**enc, max_new_tokens=400, do_sample=False)
    print("=== " + q + "\n" + tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True) + "\n")
