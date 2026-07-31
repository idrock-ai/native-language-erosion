import json, pytest
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "DTM_benchmark.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="benchmark not present")

def test_recovered_answers_applied():
    from src.data import load_raw, RECOVERED_ANSWERS
    raw = load_raw()
    byid = {r["id"]: r for r in raw}
    assert RECOVERED_ANSWERS == {299: "A", 314: "A"}
    for qid, ans in RECOVERED_ANSWERS.items():
        assert byid[qid]["answer"] == ans

def test_split_sizes_and_usability():
    from src.data import load_splits
    train, dev, test = load_splits()
    assert len(test) == 251                      # matches shipped traces
    assert len(train) == 500 - 5                 # 5 unrecoverable rows dropped from train
    assert len(dev) == 249 - 1                   # 1 dropped from dev
    assert all(e.answer_letter in "ABCD" for e in test)

def test_test_set_matches_shipped_traces():
    from src.data import load_splits
    _, _, test = load_splits()
    rows = [json.loads(l) for l in open("results/main/qwen3.5_9b_traces.jsonl")]
    cot = [r for r in rows if r["condition"] == "cot"]
    assert [e.subject for e in test] == [r["subject"] for r in cot]
    assert [e.answer_letter for e in test] == [r["correct"] for r in cot]

def test_duplicate_option_text_keeps_gold_position():
    # Gold text ("dup") is duplicated on another option: the old text-search
    # mapping mislabeled ~50% of seeds; the index-permutation mapping never does.
    import random
    from src.data import normalize_row
    row = {"question": "q", "option_A": "dup", "option_B": "dup",
           "option_C": "x", "option_D": "y", "answer": "A", "subject": "s", "id": 9}
    for seed in range(50):
        r = normalize_row(row, random.Random(seed))
        idx = list(range(4))
        random.Random(seed).shuffle(idx)          # same rng stream -> same permutation
        assert r["answer"] == "ABCD"[idx.index(0)]  # option_A's original index is 0
