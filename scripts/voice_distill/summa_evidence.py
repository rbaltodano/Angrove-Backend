"""Builds Summa references the way the app does (SummaArticleIndex): the article's own answer,
led by its conclusion. Usage: from summa_evidence import article_reference"""
import json, re, pathlib
CORPUS = pathlib.Path.home() / "Developer/Aquinas-iOS-e4b-qat/Aquinas-iOS/LocalGrounding/passages.json"
_ps = json.load(open(CORPUS))
S = [p["text"] for p in _ps if p["sourceId"] == "summa-theologica"]
HDR = re.compile(r"\b\d{1,5}\s+(?:Article|Question)\.\s*\d+\s*-\s*[^?…]{0,300}[?…]\s*")
BROKEN = re.compile(r"(?<=[a-z])- (?=[a-z])")
def clean(t): return BROKEN.sub("", HDR.sub("", t)).strip()
STARTS = [i for i, t in enumerate(S) if re.match(r"^Whether\b[^?]{5,400}\?\s*Objection 1", clean(t))]
CONCL = re.compile(r"(?:^|[.;:!?][\"”’']?\s+)((?:Hence|Therefore|Wherefore|Consequently|Accordingly|Thus|So then|We must therefore|It follows|It is therefore|It is evident|It is clear|It is manifest)\b[^.]{20,400}\.?)\s*$")
def shorten(a, budget):
    if len(a) <= budget: return a
    h = a[: budget * 2 // 5]; h = h[: max(h.rfind(". "), h.rfind("; ")) + 1] or h
    t = a[-(budget - len(h)):]; k = t.find(". "); t = t[k + 2:] if k >= 0 else t
    return f"{h} […] {t}"
def article_reference(question_phrase, budget=1300):
    for n, s in enumerate(STARTS):
        q = clean(S[s]); q = q[: q.index("?") + 1]
        if question_phrase.lower() in q.lower():
            end = STARTS[n + 1] if n + 1 < len(STARTS) else s + 40
            text = " ".join(clean(S[i]) for i in range(s, end))
            m = text.find("I answer that")
            if m < 0: return None
            ans = text[m:]; r = ans.find("Reply to Objection 1"); ans = (ans[:r] if r >= 0 else ans).strip()
            c = CONCL.search(ans); concl = c.group(1).strip() if c else None
            body = shorten(ans, budget - (len(concl) if concl else 0))
            lines = [f"Question: {q}"] + ([f"Aquinas's conclusion: {concl}"] if concl else []) + [f"Aquinas's own answer: {body}"]
            return {"title": "Summa Theologica", "sourceName": "Summa Theologica", "text": "\n".join(lines)}
    return None
def source_passage(source_id, contains, max_chars=1300):
    for p in _ps:
        if p["sourceId"] == source_id and contains.lower() in p["text"].lower():
            return {"title": p["title"], "sourceName": p["title"], "text": p["text"][:max_chars]}
    return None
