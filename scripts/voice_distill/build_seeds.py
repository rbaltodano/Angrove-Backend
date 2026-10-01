"""Builds generation seeds for the self-distilled voice dataset (see docs/Aquinas-Voice-Dataset.md).
Each seed asks the stock model to write one natural user question; answers are generated later
under the app's prompt. Seeds touching any evaluation question or sealed-2 topic are dropped."""
import json, random, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path.home() / "Developer/Aquinas_Backend/data/aquinas_voice_v1"))
from summa_evidence import article_reference, S, STARTS, clean, _ps
import check_overlap as co   # reuses the eval sets and the sealed-2 topic list
random.seed(20260930)

def blocked(text):
    t = text.lower()
    if any(topic in t for topic in co.SEALED2_TOPICS): return True
    w = co.words(text)
    for _, _, ew in co.evals:
        if w and ew and (len(w & ew) / len(w | ew) >= 0.5 or (len(ew) <= 2 and ew <= w)): return True
    return False

seeds = []
# 1. Summa articles: half answered with the article as a reference, half without.
titles = []
for s in STARTS:
    q = clean(S[s]); titles.append(q[: q.index("?") + 1])
random.shuffle(titles)
n = 0
for t in titles:
    if n >= 1400: break
    if blocked(t) or len(t) > 220: continue
    grounded = n % 2 == 0
    ref = article_reference(t, budget=2600) if grounded else None
    if grounded and not ref: continue
    seeds.append({"kind": "summa-grounded" if grounded else "summa", "seed": t,
                  "references": [ref] if ref else [],
                  "instruction": "Rewrite this scholastic question as one natural question a thoughtful modern reader might ask. Mention Aquinas only if the question needs it. Output only the question.\n\n" + t})
    n += 1
# 2. Passages from the Fathers and other works: a question the passage answers, grounded.
SOURCES = {"athanasius-on-incarnation": 120, "augustine-city-of-god": 120, "irenaeus-against-heresies": 80,
           "justin-martyr-first-apology": 50, "anselm-proslogion": 30, "augustine-confessions": 60,
           "roman-catechism-donovan": 80, "baltimore-catechism-3": 60, "westminster-confession": 30,
           "heidelberg-catechism": 30, "ecumenical-creeds-schaff": 20}
for src, k in SOURCES.items():
    ps = [p for p in _ps if p["sourceId"] == src and 500 <= len(p["text"]) <= 2400]
    random.shuffle(ps); took = 0
    for p in ps:
        if took >= k: break
        if blocked(p["text"][:400]): continue
        seeds.append({"kind": "passage-grounded", "seed": p["title"],
                      "references": [{"title": p["title"], "sourceName": p["title"], "text": p["text"][:2600]}],
                      "instruction": f"Here is a passage from {p['title']}:\n\n{p['text'][:1800]}\n\nWrite one natural question a curious reader might ask that this passage answers well. Output only the question."})
        took += 1
# 3. Scripture chapters (first chunk as reference).
bible = [p for p in _ps if p["sourceId"] == "web-bible" and re.match(r"\s*﻿?\[[A-Z0-9]{3}\d+\]", p["text"])]
random.shuffle(bible)
for p in bible[:220]:
    tag = re.search(r"\[([A-Z0-9]{3})(\d+)\]", p["text"])
    if blocked(p["text"][:300]): continue
    seeds.append({"kind": "scripture-grounded", "seed": tag.group(0),
                  "references": [{"title": "World English Bible", "sourceName": "World English Bible", "text": p["text"][:2600]}],
                  "instruction": f"Here is the start of a Bible chapter:\n\n{p['text'][:1500]}\n\nWrite one natural question someone might ask about what this chapter says or means. Output only the question."})
# 4. Open topics, no references.
TOPICS = open(pathlib.Path(__file__).with_name("topics.txt")).read().split("\n")
for t in [x.strip() for x in TOPICS if x.strip()]:
    if blocked(t): continue
    for style in ("a beginner", "a thoughtful skeptic", "someone praying through a hard season"):
        seeds.append({"kind": "topic", "seed": t, "references": [],
                      "instruction": f"Write one natural question that {style} might ask about this subject: {t}. Output only the question."})
# 5. Short everyday and factual questions, answered briefly.
for i in range(150):
    seeds.append({"kind": "short", "seed": str(i), "references": [],
                  "instruction": "Write one short, everyday factual question (arithmetic, geography, science, history, or a practical matter) that has a brief, certain answer. Make it different from common examples. Output only the question."})
# 6. Follow-ups: built later from generated pairs (pronoun, 'say more', 'shorter').
print(len(seeds), {k: sum(1 for s in seeds if s['kind'] == k) for k in sorted(set(s['kind'] for s in seeds))})
open(pathlib.Path(__file__).with_name("seeds.jsonl"), "w").write("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds))
