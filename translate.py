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

MODEL = os.getenv("WAHM_TRANSLATION_MODEL", "gpt-4o")
SEED = "wahm_seed_msa.csv"
OUTPUT_FIELDS = ["qid", "question_msa", "gold_answer", "dialect_candidate",
                 "backtranslation_msa", "sub_variety", "requested_model",
                 "translation_model", "backtranslation_model", "n_exemplars",
                 "coda", "temperature"]

DIALECT_NAMES = {
    "gulf": "Gulf (Khaleeji) Arabic",
    "egyptian": "Egyptian (Masri) Arabic",
    "levantine": "Levantine (Shami) Arabic",
    "sudanese": "Sudanese Arabic",
}

_client = None

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
    try:
        f = open(path, encoding="utf-8", newline="")
    except FileNotFoundError:
        sys.exit(f"ERROR: {path} not found. Complete the native-speaker "
                 "exemplar template before making API calls.")
    with f:
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


def call(msgs, temperature, model=MODEL, retries=3):
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    for attempt in range(retries):
        try:
            r = _client.chat.completions.create(
                model=model, messages=msgs,
                temperature=temperature, max_tokens=300)
            return (r.choices[0].message.content.strip(),
                    getattr(r, "model", model))
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  retry in {wait}s ({e})")
            time.sleep(wait)


def translate(dialect, limit=None, use_coda=True, temperature=0.3,
              backtranslate=True, suffix="", model=MODEL, overwrite=False):
    name = DIALECT_NAMES[dialect]
    pairs, subvariety = load_exemplars(dialect)
    system = build_system(name, subvariety, use_coda)

    print(f"{name}")
    print(f"  exemplars   : {len(pairs)}")
    print(f"  sub-variety : {subvariety or 'UNSPECIFIED (ask your validator)'}")
    print(f"  CODA rule   : {'on' if use_coda else 'off'}")
    print(f"  temperature : {temperature}")
    print(f"  model       : {model}")

    with open(SEED, encoding="utf-8", newline="") as source:
        seed = list(csv.DictReader(source))
    if limit:
        seed = seed[:limit]

    out = f"candidates_{dialect}{suffix}.csv"
    done = set()
    if os.path.exists(out) and not overwrite:
        with open(out, encoding="utf-8", newline="") as existing:
            reader = csv.DictReader(existing)
            if reader.fieldnames != OUTPUT_FIELDS:
                sys.exit(f"ERROR: {out} has an incompatible header. Use a new "
                         "--suffix or explicitly pass --overwrite.")
            existing_rows = list(reader)
        incompatible = [r["qid"] for r in existing_rows
                        if r["requested_model"] != model
                        or r["sub_variety"] != subvariety
                        or r["coda"] != str(int(use_coda))
                        or float(r["temperature"]) != temperature]
        if incompatible:
            sys.exit(f"ERROR: {out} contains a different experiment setup "
                     f"({len(incompatible)} incompatible rows). Use --suffix.")
        done = {r["qid"] for r in existing_rows
                if r["dialect_candidate"].strip()}

    todo = [row for row in seed if row["qid"] not in done]
    mode = "w" if overwrite or not os.path.exists(out) else "a"
    with open(out, mode, encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            w.writeheader()
        for i, r in enumerate(todo, 1):
            q = r["question_msa"].strip()
            dia, translation_model = call(
                build_messages(system, pairs, q), temperature, model)
            back, backtranslation_model = "", ""
            if backtranslate:
                back, backtranslation_model = call([
                    {"role": "system",
                     "content": BACKTRANS_SYSTEM.format(dialect=name)},
                    {"role": "user", "content": dia}], 0.0, model)
            w.writerow(dict(qid=r["qid"], question_msa=q,
                            gold_answer=r["gold_answer"], dialect_candidate=dia,
                            backtranslation_msa=back, sub_variety=subvariety,
                            requested_model=model, translation_model=translation_model,
                            backtranslation_model=backtranslation_model,
                            n_exemplars=len(pairs), coda=int(use_coda),
                            temperature=temperature))
            f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} new ({len(done)} already complete)")

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
    p.add_argument("--model", default=MODEL,
                   help="translation model ID; record a pinned ID when available")
    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing matching output instead of resuming")
    a = p.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("ERROR: set OPENAI_API_KEY")

    translate(a.dialect, a.limit, not a.no_coda, a.temperature,
              not a.no_backtranslate, a.suffix, a.model, a.overwrite)
