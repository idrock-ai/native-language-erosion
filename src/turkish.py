"""TurkishMMLU generalization harness (E7): does the format-tax mechanism found on DTM
travel to a second language? Mirrors src/data.py (loader) and src/run.py (runner) for a
5-option, Turkish-language MCQ benchmark. Self-contained: downloads via urllib, no
`datasets` library dependency.

Native-language subject = Turkish_Language_and_Literature (analogue of DTM's ona_tili);
reasoning = Mathematics, Physics; knowledge = History. Both HF splits (test=100, dev=5)
are pooled per subject (105 items/subject, 420 total) and re-split by this module.

Fetch endpoint (HF datasets-server, no `datasets` lib needed):
  https://datasets-server.huggingface.co/rows?dataset=AYueksel/TurkishMMLU&config=<SUBJECT>&split=<SPLIT>&offset=<N>&length=100
Row schema: {question: str, choices: list[str] (5), answer: str ("A".."E")}.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from .data import cap_per_subject
from .instrument import instrumented_eval
from .program import compliant_metric, metric
from .run import CONDITION_NAMES, build_items, make_lm_factory, score_items

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = str(REPO / "data" / "turkishmmlu.json")

HF_DATASET = "AYueksel/TurkishMMLU"
HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

TURKISH_LETTERS = "ABCDE"
SUBJECTS = ["Turkish_Language_and_Literature", "Mathematics", "Physics", "History"]
SPLITS = ["test", "dev"]
# Explicit (not derived): "Turkish_Language_and_Literature" -> initials of the three
# content words (Turkish, Language, Literature); the single-word subjects use their
# own first three letters. Used to build human-legible qids (tr<initials><n>).
SUBJECT_INITIALS = {
    "Turkish_Language_and_Literature": "TLL",
    "Mathematics": "MAT",
    "Physics": "PHY",
    "History": "HIS",
}


def _fetch_rows(subject: str, split: str) -> list[dict]:
    """Page through the HF datasets-server rows endpoint for one (subject, split)."""
    rows: list[dict] = []
    offset = 0
    while True:
        qs = urllib.parse.urlencode({"dataset": HF_DATASET, "config": subject,
                                     "split": split, "offset": offset, "length": 100})
        with urllib.request.urlopen(f"{HF_ROWS_ENDPOINT}?{qs}", timeout=30) as resp:
            payload = json.load(resp)
        batch = [r["row"] for r in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= payload.get("num_rows_total", offset):
            break
    return rows


def fetch_turkishmmlu(cache_path: str = DEFAULT_CACHE) -> list[dict]:
    """Download the four TurkishMMLU subjects (test+dev splits) via urllib, tag each raw
    row with its subject, cache the combined list to `cache_path`, and return it. Never
    re-downloads when the cache file already exists."""
    path = Path(cache_path)
    if path.exists():
        return json.loads(path.read_text())
    all_rows: list[dict] = []
    for subject in SUBJECTS:
        for split in SPLITS:
            for row in _fetch_rows(subject, split):
                all_rows.append({**row, "subject": subject, "split": split})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(all_rows, ensure_ascii=False))
    return all_rows


def normalize_turkish_row(row: dict, rng: random.Random) -> dict:
    """Normalize one raw TurkishMMLU row into the canonical record shape ({question,
    options(5), answer, subject, qid, usable}), shuffling the five options via an INDEX
    PERMUTATION so the gold answer always tracks its option text (mirrors src/data.py
    normalize_row; never uses text .index()). `qid` is filled in later by the caller,
    since raw TurkishMMLU rows carry no stable id of their own."""
    choices = row.get("choices") or []
    a = str(row.get("answer", "")).strip().upper()
    has_five = len(choices) == 5
    usable = has_five and len(a) == 1 and a in TURKISH_LETTERS
    idx = list(range(5))
    rng.shuffle(idx)                              # consumed even when unusable (alignment)
    options = [choices[i] for i in idx] if has_five else list(choices)
    answer = TURKISH_LETTERS[idx.index(TURKISH_LETTERS.index(a))] if usable else None
    return {"question": row.get("question"), "options": options, "answer": answer,
            "subject": row.get("subject", "unknown"), "qid": None, "usable": usable}


def format_turkish_options(options: list[str]) -> str:
    return "\n".join(f"{TURKISH_LETTERS[i]}) {opt}" for i, opt in enumerate(options))


def to_turkish_example(record: dict) -> dspy.Example:
    return dspy.Example(
        question=record["question"], options=format_turkish_options(record["options"]),
        answer_letter=record["answer"], subject=record["subject"], qid=record.get("qid"),
    ).with_inputs("question", "options")


def load_turkish_splits(seed: int = 42, n_train_per_subject: int = 40,
                        cache_path: str = DEFAULT_CACHE):
    """Load TurkishMMLU (both HF splits pooled per subject), normalize + shuffle options,
    assign qids (tr<subject-initials><n>, e.g. trTLL07), and split each subject
    deterministically into (train, test) dspy Examples. Deterministic under `seed`."""
    raw = fetch_turkishmmlu(cache_path=cache_path)
    by_subject: dict[str, list[dict]] = {}
    for row in raw:
        by_subject.setdefault(row.get("subject", "unknown"), []).append(row)

    norm_rng = random.Random(seed)
    split_rng = random.Random(seed)
    train, test = [], []
    for subject in sorted(by_subject):
        initials = SUBJECT_INITIALS.get(subject, subject[:3].upper())
        records = []
        for n, row in enumerate(by_subject[subject], start=1):
            rec = normalize_turkish_row(row, norm_rng)
            rec["qid"] = f"tr{initials}{n:02d}"
            records.append(rec)
        usable = [r for r in records if r["usable"]]
        split_rng.shuffle(usable)
        n_train = min(n_train_per_subject, len(usable))
        train += usable[:n_train]
        test += usable[n_train:]
    return ([to_turkish_example(r) for r in train],
            [to_turkish_example(r) for r in test])


class TurkishCoT(dspy.Signature):
    """Türkçe test sorusunu cevapla."""
    question: str = dspy.InputField()
    options: str = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Çok kısa gerekçe, en fazla 2 cümle.")
    answer_letter: str = dspy.OutputField(desc="Sadece bir harf: A, B, C, D veya E.")


class TurkishCoTSolver(dspy.Module):
    """Brevity-constrained zero-shot chain of thought for TurkishMMLU (5-option)."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TurkishCoT)

    def forward(self, question, options):
        r = self.predict(question=question, options=options)
        return dspy.Prediction(answer_letter=r.answer_letter,
                               reasoning=getattr(r, "reasoning", ""))


def turkish_metric(example, pred, trace=None) -> bool:
    """metric (correctness), letters-bound to the 5-option TurkishMMLU alphabet."""
    return metric(example, pred, trace, letters=TURKISH_LETTERS)


def turkish_compliant_metric(example, pred, trace=None) -> bool:
    """compliant_metric (correctness AND brevity-compliance), 5-option-bound."""
    return compliant_metric(example, pred, trace, letters=TURKISH_LETTERS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--conditions", default="cot,bootstrap,bootstrap_compliant")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=15)
    ap.add_argument("--out-dir", default="results/e7")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())
    train, test = load_turkish_splits(seed=args.seed)
    train_pool = cap_per_subject(train, args.train_cap)
    random.Random(args.seed).shuffle(train_pool)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    print(f"[{args.model}@{args.engine}] test={len(test)} pool={len(train_pool)} "
          f"max_tokens={args.max_tokens} conditions={conds}")

    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    safe = args.model.replace("/", "_").replace(":", "_")
    report = {"model": args.model, "engine": args.engine, "seed": args.seed,
              "max_tokens": args.max_tokens, "conditions": {}}
    all_items = []

    def record(name, program):
        recs = instrumented_eval(test, program, factory, args.workers, name,
                                 max_tokens=args.max_tokens, letters=TURKISH_LETTERS)
        items = build_items(test, recs, args.model, args.engine, name, args.max_tokens)
        all_items.extend(items)
        report["conditions"][name] = {"deployment": score_items(items, "is_correct"),
                                      "rescue": score_items(items, "rescue_correct")}
        d = report["conditions"][name]["deployment"]["overall"]
        errs = sum(i["parse_error"] for i in items)
        trunc = sum(i["truncated"] for i in items)
        print(f"  {name}: overall={d} parse_errors={errs} truncated={trunc}")

    for cond in conds:
        name = CONDITION_NAMES.get(cond, cond)
        if cond == "cot":
            record(name, TurkishCoTSolver())
        elif cond in ("bootstrap", "bootstrap_compliant"):
            m = turkish_metric if cond == "bootstrap" else turkish_compliant_metric
            compiled = BootstrapFewShot(metric=m, max_bootstrapped_demos=4,
                                        max_labeled_demos=0).compile(TurkishCoTSolver(),
                                                                     trainset=train_pool)
            suffix = "_bootstrap.json" if cond == "bootstrap" else "_bootstrap_compliant.json"
            compiled.save(str(out / f"{safe}{suffix}"))
            record(name, compiled)
        else:
            raise SystemExit(f"unknown condition: {cond}")

    (out / f"{safe}_report.json").write_text(json.dumps(report, indent=1))
    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()
