"""LoRA fine-tune of Gemma 4 E4B on the Angrove voice dataset, then merge for LiteRT export.

Run on a GPU pod (validated target: 1x A100 80 GB, 2026-09-30):
  python train_voice_lora.py --base /root/models/gemma-4-E4B-it-base \
      --data /root/data --output /root/models/aquinas-e4b-voice-v1

Trains only the language model's linear layers, computes loss on assistant turns only,
evaluates on valid.jsonl each epoch, then merges the adapter into a full
Gemma4ForConditionalGeneration checkpoint that scripts/export_litert_aquinas.py accepts.
See docs/Aquinas-Voice-Dataset.md.
"""
import argparse, json, math, pathlib, random, shutil, time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"

class Chats(Dataset):
    def __init__(self, path, tok, max_len):
        self.items = []
        for line in open(path):
            msgs = json.loads(line)["messages"]
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            full = tok.apply_chat_template(msgs, tokenize=False)
            assert full.startswith(prompt), "chat template prefix mismatch"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * min(len(p_ids), len(f_ids)) + f_ids[len(p_ids):]
            self.items.append((f_ids, labels))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    ids, labels = batch[0]
    return torch.tensor([ids]), torch.tensor([labels])

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); total, n = 0.0, 0
    for ids, labels in loader:
        out = model(input_ids=ids.to(device), labels=labels.to(device))
        total += out.loss.item(); n += 1
    model.train(); return total / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16); ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    data = pathlib.Path(a.data)
    train = Chats(data / "train.jsonl", tok, a.max_len); valid = Chats(data / "valid.jsonl", tok, a.max_len)
    print(f"train {len(train)} valid {len(valid)}; max tokens {max(len(x[0]) for x in train.items)}")
    tl = DataLoader(train, batch_size=1, shuffle=True, collate_fn=collate)
    vl = DataLoader(valid, batch_size=1, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    steps = math.ceil(len(train) / a.accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, steps // 10), steps)
    print(f"epoch 0 valid loss {evaluate(model, vl, device):.4f}")
    best = float("inf"); adapter_dir = pathlib.Path(a.output + "-adapter")
    for epoch in range(1, a.epochs + 1):
        t0 = time.time(); running = 0.0
        for i, (ids, labels) in enumerate(tl, 1):
            loss = model(input_ids=ids.to(device), labels=labels.to(device)).loss / a.accum
            loss.backward(); running += loss.item()
            if i % a.accum == 0 or i == len(tl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        v = evaluate(model, vl, device)
        print(f"epoch {epoch} train loss {running * a.accum / len(tl):.4f} valid loss {v:.4f} ({time.time()-t0:.0f}s)")
        if v < best:
            best = v; model.save_pretrained(adapter_dir); print(f"  saved adapter (best valid {v:.4f})")
    # Merge the best adapter into a full checkpoint for export.
    base = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16)
    from peft import PeftModel
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = pathlib.Path(a.output); merged.save_pretrained(out, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
              "generation_config.json"):
        if (pathlib.Path(a.base) / f).exists(): shutil.copy(pathlib.Path(a.base) / f, out / f)
    cfg = json.load(open(out / "config.json"))
    print("merged:", out, cfg.get("architectures"), f"best valid loss {best:.4f}")

if __name__ == "__main__":
    main()
