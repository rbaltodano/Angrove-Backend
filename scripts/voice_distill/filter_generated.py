"""Filters generated.jsonl into training files. Writes train.jsonl, valid.jsonl, and
rejected.jsonl (with reasons). A locator (question, article, part, chapter, book) is allowed only
when it appears in that example's own reference passages."""
import json, re, sys, random, hashlib, pathlib, collections
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(pathlib.Path.home() / "Developer/Aquinas_Backend/data/aquinas_voice_v1"))
sys.argv = sys.argv[:1]
import check_overlap as co
T = json.load(open(HERE / "templates.json"))

MARKDOWN = re.compile(r"\*\*|^#+\s|^\s*[-*•]\s|\$\\|\\text|```|^\s*\d+\.\s", re.M)
LEAK = re.compile(r"User question|Reference passages|retrieved|internal notes|TASK:|<turn|\[.*? — .*?\]", re.I)
ADDRESS = re.compile(r"\b(student|my dear|dear friend|my friend)\b", re.I)
LOCATOR = re.compile(r"\b(?:q\.\s?\d+|Q\.\s?\d+|Question \d+|Article \d+|(?:First|Second|Third) Part|Prima Pars|Secunda Pars|Tertia Pars|I–II|II–II|chapter \d+|[IVX]+\.\d+|book [IVX]+)\b", re.I)

def words(t): return len(re.findall(r"\w+", t))
def loops(t):
    s = [x.strip().lower() for x in re.split(r"[.!?\n]", t) if len(x.strip()) > 30]
    return len(s) != len(set(s))

def blocked(q):
    if any(t in q.lower() for t in co.SEALED2_TOPICS): return True
    w = co.words(q)
    return any(w and ew and (len(w & ew) / len(w | ew) >= 0.5 or (len(ew) <= 2 and ew <= w)) for _, _, ew in co.evals)

def reason(r):
    a, q = r["answer"], r["question"]
    if not a: return "empty"
    if blocked(q): return "eval-overlap"
    if MARKDOWN.search(a): return "markdown"
    if LEAK.search(a): return "leak"
    if ADDRESS.search(a): return "address"
    if loops(a): return "loop"
    n = words(a)
    if r["kind"] == "short" and n > 60: return "too-long-short"
    if r["kind"] != "short" and r["kind"] != "followup" and not 60 <= n <= 320: return "length"
    if r["kind"] == "followup" and not 15 <= n <= 320: return "length"
    refs = " ".join(x["text"] for x in r["references"]).lower()
    for m in LOCATOR.finditer(a):
        if m.group(0).lower() not in refs: return "unsupported-locator"
    return None

rows = [json.loads(l) for l in open(HERE / "generated.jsonl")]
kept, rejected = [], []
for r in rows:
    why = reason(r)
    (rejected if why else kept).append({**r, "reason": why} if why else r)
def to_train(r):
    msgs = [{"role": "system", "content": r["system_train"]}] + r["history"]
    msgs.append({"role": "user", "content": T["wrapper"] + "\n" + r.get("note", "") + "\nUser question:\n" + r["question"]})
    msgs.append({"role": "assistant", "content": r["answer"]})
    return {"id": r["id"], "kind": r["kind"], "messages": msgs}
train, valid = [], []
for r in kept:
    (valid if int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 20 == 0 else train).append(to_train(r))
for name, data in (("train.jsonl", train), ("valid.jsonl", valid), ("rejected.jsonl", rejected)):
    with open(HERE / name, "w") as f: f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in data)
print(f"generated {len(rows)} kept {len(kept)} (train {len(train)}, valid {len(valid)}) rejected {len(rejected)}")
print("reject reasons:", collections.Counter(r["reason"] for r in rejected).most_common())
print("kept by kind:", collections.Counter(r["kind"] for r in kept).most_common())
