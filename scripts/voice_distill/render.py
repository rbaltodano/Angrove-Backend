"""Renders examples.jsonl into chat-format training files in the app's exact prompt layout.

Usage: python3 render.py [examples.jsonl] [suffix]  ->  train<suffix>.jsonl, valid<suffix>.jsonl
(about 90/10, stable by id)

The layout comes from prompts the app actually sent (prompt-samples.json, recorded from
Aquinas-iOS e4b-eval run Q3-held-sim-1 on 2026-09-30):
- system: the conversation system instruction, with the reference block replaced by this
  example's references (or the no-passage paragraph), and the personality paragraph replaced by
  the app's *scholarly* personality, which matches this dataset's voice.
- history turns as plain user/assistant messages;
- the latest user turn wrapped in the app's response instructions, then "User question:".
If the app's prompt changes, re-record prompt-samples.json and re-render.
"""
import hashlib, json, re, pathlib

HERE = pathlib.Path(__file__).parent
P = json.load(open(HERE / "prompt-samples.json"))

SCHOLARLY = """Respond as a wise, learned, and well-spoken mentor in the Thomistic intellectual
tradition. Unite scholarly rigor with humane warmth: be gracious, patient, attentive,
and quietly encouraging. Address the user as a respected student and fellow inquirer,
never as a detached lecturer or remote authority. Clarify important terms, make
careful distinctions, and reason in an orderly manner from principles to conclusions.
Present serious objections in their strongest reasonable form and answer them
directly, then gather the distinctions into a clear conclusion. Use precise,
articulate language and explain specialized terms with the ease of a generous
teacher. Let the prose carry measured gravity without stiffness. Avoid archaic
imitation, coldness, condescension, excessive verbosity, and a sermonizing tone."""

def swap_personality(system):
    i = system.index("Speak with the intellectual depth")
    end = "exceed three paragraphs for routine personal advice."
    j = system.index(end) + len(end)
    return system[:i] + SCHOLARLY + system[j:]

WITH_REF = swap_personality(P["withref"])
NO_REF = swap_personality(P["noref"])
REF_START = WITH_REF.index("Reference passages retrieved for this question")
REF_END = WITH_REF.index("These were retrieved automatically")
MESSAGE = P["message"][: P["message"].index("User question:")]

PRONOUNS = {"that", "this", "these", "those", "it", "its", "they", "them", "their", "he", "him",
            "his", "she", "her", "others", "ones", "former", "latter", "either", "neither", "both"}

def system_for(example):
    refs = example["references"]
    if not refs:
        return NO_REF
    block = ("Reference passages retrieved for this question, each labeled with its source title:\n"
             + "\n\n".join(f"[{r['title']} — {r['sourceName']}]\n{r['text']}" for r in refs) + "\n\n")
    return WITH_REF[:REF_START] + block + WITH_REF[REF_END:]

def latest_message(example):
    note = ""
    hist = example["history"]
    words = set(re.findall(r"[a-z]+", example["question"].lower()))
    if hist and words & PRONOUNS:
        prev = [m["content"] for m in hist if m["role"] == "user"][-1]
        note = (f"\nThis question continues the conversation. The previous question was: “{prev}” "
                "Read a pronoun in the new question as pointing to what that question asked about.\n")
    return MESSAGE.rstrip("\n") + "\n" + note + "\nUser question:\n" + example["question"]

def render(example):
    msgs = [{"role": "system", "content": system_for(example)}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in example["history"]]
    msgs.append({"role": "user", "content": latest_message(example)})
    msgs.append({"role": "assistant", "content": example["answer"]})
    return {"id": example["id"], "kind": example["kind"], "messages": msgs}

def main():
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "examples.jsonl"
    suffix = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = [json.loads(l) for l in open(HERE / src)]
    train, valid = [], []
    for r in rows:
        bucket = int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 10
        (valid if bucket == 0 else train).append(render(r))
    for name, data in ((f"train{suffix}.jsonl", train), (f"valid{suffix}.jsonl", valid)):
        with open(HERE / name, "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in data)
    print(f"train {len(train)}, valid {len(valid)}")

if __name__ == "__main__":
    main()
