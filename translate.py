"""
WAHM stage 1: dialect translation via few-shot prompting.

Reads   wahm_seed_msa.csv           300 MSA questions + gold
        exemplars_<dialect>.csv      10 human-authored MSA->dialect pairs
Writes  candidates_<dialect>.csv     machine translations + back-translations

Only the QUESTION is translated. The gold answer stays MSA: it is the canonical
fact, and facts (dates, numbers, names) do not change across dialects.

Usage
    export OPENAI_API_KEY=...
    python translate.py gulf --limit 20                          # pilot
    python translate.py gulf --limit 20 --no-coda --suffix _nocoda   # A/B the CODA rule
    python translate.py gulf                                     # full 300
"""

import argparse, csv, os, sys, time
from openai import OpenAI

MODEL = "gpt-4o"
SEED = "wahm_seed_msa.csv"

DIALECT_NAMES = {
    "gulf": "Gulf (Khaleeji) Arabic",
    "egyptian": "Egyptian (Masri) Arabic",
    "levantine": "Levantine (Shami) Arabic",
    "sudanese": "Sudanese Arabic",
}

client = OpenAI()

CODA_RULE = ("4. Follow CODA orthography: spell dialect words like their MSA "
             "cognates where a cognate exists, write dialect-only words "
             "phonetically, use no diacritics, and never use Latin letters or "
             "digits for Arabic sounds.\n")

BASE_RULES = """Rules:
1. Preserve the meaning EXACTLY. Never add, drop, or soften any part of the question. The correct answer to the question must not change.
2. Use {dialect} vocabulary and phrasing, not MSA. This applies especially to question words, negation, and everyday verbs.
3. Keep proper nouns, numbers, and dates exactly as written.
{coda}5. Do NOT answer the question. Translate it only.
6. Output nothing except the translated question. No quotes, no preamble, no explanation."""

SYSTEM = """You are a native speaker of {speaker} and an expert dialect translator.

You translate questions from Modern Standard Arabic into natural, everyday {dialect}, the way a native speaker would actually ask the question out loud.

{rules}"""

BACKTRANS_SYSTEM = ("You translate {dialect} questions into Modern Standard "
                    "Arabic. Preserve the meaning exactly. Output only the MSA "
                    "question, nothing else.")


def load_exemplars(dialect):
    path = f"exemplars_{dialect}.csv"
    pairs, subvariety = [], ""
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            msa = r["msa_question"].strip()
            dia = r.get("dialect_question", "").strip()
            if msa and dia:
                pairs.append((msa, dia))
            if r.get("sub_variety", "").strip() and not subvariety:
                subvariety = r["sub_variety"].strip()
    if not pairs:
        sys.exit(f"ERROR: {path} has no completed dialect_question rows.\n"
                 "This script will not run without human exemplars.")
    if len(pairs) < 5:
        print(f"WARNING: only {len(pairs)} exemplars; few-shot quality will suffer")
    return pairs, subvariety


def build_system(dialect_name, subvariety, use_coda):
    speaker = f"{subvariety} {dialect_name}" if subvariety else dialect_name
    rules = BASE_RULES.format(dialect=dialect_name,
                              coda=CODA_RULE if use_coda else "")
    return SYSTEM.format(speaker=speaker, dialect=dialect_name, rules=rules)


def build_messages(system, pairs, question):
    """Exemplars go in as real prior turns, not pasted into the system message."""
    msgs = [{"role": "system", "content": system}]
    for msa, dia in pairs:
        msgs.append({"role": "user", "content": msa})
        msgs.append({"role": "assistant", "content": dia})
    msgs.append({"role": "user", "content": question})
    return msgs


def call(msgs, temperature, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=msgs,
                temperature=temperature, max_tokens=300)
            return r.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)


def translate(dialect, limit=None, use_coda=True, temperature=0.3,
              backtranslate=True, suffix=""):
    name = DIALECT_NAMES[dialect]
    pairs, subvariety = load_exemplars(dialect)
    system = build_system(name, subvariety, use_coda)

    print(f"{name}")
    print(f"  exemplars   : {len(pairs)}")
    print(f"  sub-variety : {subvariety or 'UNSPECIFIED (ask your validator)'}")
    print(f"  CODA rule   : {'on' if use_coda else 'off'}")
    print(f"  temperature : {temperature}")

    seed = list(csv.DictReader(open(SEED, encoding="utf-8")))
    if limit:
        seed = seed[:limit]

    out = f"candidates_{dialect}{suffix}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "question_msa", "gold_answer", "dialect_candidate",
                    "backtranslation_msa", "sub_variety", "model",
                    "n_exemplars", "coda", "temperature"])
        for i, r in enumerate(seed, 1):
            q = r["question_msa"].strip()
            dia = call(build_messages(system, pairs, q), temperature)
            back = ""
            if backtranslate:
                back = call([
                    {"role": "system",
                     "content": BACKTRANS_SYSTEM.format(dialect=name)},
                    {"role": "user", "content": dia}], 0.0)
            w.writerow([r["qid"], q, r["gold_answer"], dia, back, subvariety,
                        MODEL, len(pairs), int(use_coda), temperature])
            f.flush()
            if i % 10 == 0 or i == len(seed):
                print(f"  {i}/{len(seed)}")

    print(f"\nwrote {out}\nnext: python qc.py {dialect}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dialect", choices=list(DIALECT_NAMES))
    p.add_argument("--limit", type=int, default=None, help="pilot on first N")
    p.add_argument("--no-coda", action="store_true",
                   help="drop the CODA rule (for the A/B test)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--no-backtranslate", action="store_true")
    p.add_argument("--suffix", default="", help="tag output file, e.g. _nocoda")
    a = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY")

    translate(a.dialect, a.limit, not a.no_coda, a.temperature,
              not a.no_backtranslate, a.suffix)
