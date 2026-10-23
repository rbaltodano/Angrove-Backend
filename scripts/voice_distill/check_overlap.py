"""Flags training questions too close to any evaluation question.
Usage: python3 check_overlap.py examples.jsonl
Evaluation sets: dev, held-out, sealed-1, sealed-2 (Aquinas-iOS e4b-eval/eval-set)."""
import json, re, sys, pathlib
EVAL = pathlib.Path.home() / "Developer/Aquinas-iOS-e4b-qat/LocalModels/e4b-eval/eval-set"
STOP = set("a an the of to in on and or is are was were be do does did what why how who which when whether for with about according aquinas thomas summa say says teach mean means i you it its that this can could would should must ever".split())
def words(t): return {w for w in re.findall(r"[a-z]+", t.lower()) if w not in STOP and len(w) > 2}
evals = []
for f in ("dev.jsonl", "heldout.jsonl", "sealed-1.jsonl", "sealed-2.jsonl"):
    for l in open(EVAL / f):
        c = json.loads(l); evals.append((c["id"], c["question"], words(c["question"])))
# Topics of the untouched sealed-2 set: no training question or answer may teach them.
SEALED2_TOPICS = ["humility", "fasting", "fortitude", "patience", "magnanimity", "habitus",
    "contemplative life", "active life", "matter and form", "pity", "rash judg", "judge others",
    "oath", "restitution", "love oneself", "self-love", "curiosity", "angels know", "future contingent",
    "governed by", "power of the soul", "soul is a body", "charity is friendship", "friendship with god",
    "break a promise", "barnabas", "council of ephesus came", "virtue and a vice", "boethius",
    "benedict", "bonaventure", "temperance", "natural law be changed", "can the natural law change",
    "council of trent open", "divine comedy", "luther", "how tall", "hours a day", "burned his",
    "1 corinthians 12", "matthew 22:37", "wisdom and knowledge", "habit and"]

def main():
    # Reviewed word-overlap false positives (different question, shared common word).
    OVERLAP_OK = {"followup-003", "followup-008", "honesty-005", "followup-016", "followup-018", "honesty-014", "honesty-016"}
    bad = 0
    for l in open(sys.argv[1]):
        e = json.loads(l); w = words(e["question"])
        if e["id"] in OVERLAP_OK: continue
        for cid, q, ew in evals:
            if not w or not ew: continue
            j = len(w & ew) / len(w | ew)
            if j >= 0.5 or (len(ew) <= 2 and ew <= w):
                print(f"OVERLAP {e['id']}: {e['question']!r} ~ {cid}: {q!r} ({j:.2f})"); bad += 1

    # Answers reviewed by hand: the topic word appears only in passing.
    REVIEWED = {"core-014", "core-020", "core-054", "core-056", "core-061", "core-068", "honesty-003", "followup-014", "followup-015", "followup-024", "grounded-043", "core-089", "core-094"}
    for l in open(sys.argv[1]):
        e = json.loads(l); q = e["question"].lower(); a = e["answer"].lower()
        for t in SEALED2_TOPICS:
            if t in q:
                print(f"SEALED-2 TOPIC {e['id']}: '{t}' in question"); bad += 1
            elif t in a and e["id"] not in REVIEWED:
                print(f"review {e['id']}: '{t}' mentioned in answer")
    print(f"{bad} overlaps")

if __name__ == "__main__":
    main()
