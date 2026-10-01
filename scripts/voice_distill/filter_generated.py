"""Filters generated.jsonl into training files. Writes train.jsonl, valid.jsonl, and
rejected.jsonl (with reasons). A locator (question, article, part, chapter, book) is allowed only
when it appears in that example's own reference passages."""
import json, re, sys, random, hashlib, pathlib, collections
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(pathlib.Path.home() / "Developer/Aquinas_Backend/data/aquinas_voice_v1"))
sys.argv = sys.argv[:1]
import check_overlap as co
T = json.load(open(HERE / "templates.json"))

MARKDOWN = re.compile(r"\*\*|(?<![*\w])\*[^*\n]+\*(?![*\w])|(?<!\w)_[^_\n]+_(?!\w)|^#+\s|^\s*[-*•]\s|\$\\|\\text|```|^\s*\d+\.\s", re.M)
MARKER = re.compile(r"\{\{([^{}]+)\}\}")
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

ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])")
DELIGHT = re.compile(r"\b(so |truly |quite |most |deeply |rather |really )?beautiful\b(?= (?:how|that|here|about|because|to see|to consider|to me|in this|in the way|is the way|thing|point|tension|thought|idea|way|distinction))", re.I)
DELIGHT_WORDS = ["striking", "lovely", "remarkable", "wonderful", "illuminating", "satisfying",
                 "fascinating", "elegant", "moving", "delightful"]
def vary_delight(a, key):
    # Stock E4B reaches for "beautiful" in nearly every aside; vary it so the tune doesn't learn a tic.
    n = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    def sub(m):
        nonlocal n
        word = DELIGHT_WORDS[n % len(DELIGHT_WORDS)]; n //= len(DELIGHT_WORDS)
        out = (m.group(1) or "") + word
        return out[0].upper() + out[1:] if m.group(0)[0].isupper() else out
    a = DELIGHT.sub(sub, a)
    return re.sub(r"\b([Aa]) (illuminating|elegant)\b", r"\1n \2", a)

def normalize(a):
    # Unwrap single-asterisk italics; keep only the first marker for each term.
    a = ITALIC.sub(r"\1", a)
    seen = set()
    def one(m):
        k = m.group(1).strip().lower()
        if k in seen: return m.group(1)
        seen.add(k); return m.group(0)
    return MARKER.sub(one, a)

def reason(r):
    r["answer"] = vary_delight(normalize(r["answer"]), r["id"])
    a, q = r["answer"], r["question"]
    if not a: return "empty"
    if blocked(q): return "eval-overlap"
    if MARKDOWN.search(a): return "markdown"
    if LEAK.search(a): return "leak"
    if ADDRESS.search(a): return "address"
    if loops(a): return "loop"
    marks = MARKER.findall(a)
    if len(marks) > 8 or any(len(m.split()) > 4 for m in marks): return "markers"
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

# Stock phrases: no single one may appear in more than 6% of kept answers, so the tuned model
# doesn't learn a verbal tic. Praise of the question itself is always rejected.
PRAISE = re.compile(r"^(that is|that's|what) an? (\w+ )?(question|thought)|^(what a|such a) ", re.I)
STOCK = ["think of it like this", "think of it this way", "what is beautiful", "what strikes me",
         "quietly brilliant", "it is so beautiful", "what i find", "the crucial distinction here",
         "the governing reason", "it is truly striking", "it is a profound mystery", "imagine a boundless ocean",
         "boundless ocean"]
cap = max(1, int(0.06 * len(kept))); used = collections.Counter(); final = []
for r in sorted(kept, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
    low = r["answer"].lower()
    if PRAISE.search(r["answer"]): rejected.append({**r, "reason": "praises-question"}); continue
    hits = [p for p in STOCK if p in low]
    over = [p for p in hits if used[p] >= cap]
    if over: rejected.append({**r, "reason": "stock-phrase:" + over[0]}); continue
    used.update(hits); final.append(r)
kept = final
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
print("stock phrase use:", used.most_common())
print("kept by kind:", collections.Counter(r["kind"] for r in kept).most_common())
