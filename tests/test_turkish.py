"""Fixture-level tests for E7 (TurkishMMLU generalization harness). No network calls:
every test either exercises a pure function or points the loader at a tmp_path fixture
cache file (and asserts the network path is never touched when that cache exists)."""
import json
import random
import re

from src.program import parse_letter


def test_parse_letter_five_letter_support():
    # Standalone "E" at the end, no standalone A-D token anywhere earlier in the text.
    text = "After careful thought, the correct choice is E"
    assert parse_letter(text, letters="ABCDE") == "E"
    assert parse_letter(text) != "E"          # default 4-letter call can't see E


def test_normalize_turkish_row_duplicate_option_text_keeps_gold_position():
    from src.turkish import normalize_turkish_row
    # Gold text ("dup") is duplicated on another option: a text-search ( .index(text) )
    # mapping would mislabel it non-deterministically; the index-permutation mapping
    # (mirrors src/data.py normalize_row) never does, over 20 seeds.
    row = {"question": "q", "choices": ["dup", "dup", "x", "y", "z"],
           "answer": "A", "subject": "Mathematics"}
    for seed in range(20):
        rec = normalize_turkish_row(row, random.Random(seed))
        assert rec["usable"]
        assert len(rec["options"]) == 5
        idx = list(range(5))
        random.Random(seed).shuffle(idx)          # same rng stream -> same permutation
        assert rec["answer"] == "ABCDE"[idx.index(0)]    # choices[0] is the gold option
        assert rec["options"][idx.index(0)] == "dup"


def _write_fixture_cache(path, n_per_subject=10):
    from src.turkish import SUBJECTS
    rows = []
    for subject in SUBJECTS:
        for i in range(n_per_subject):
            letter = "ABCDE"[i % 5]
            choices = [f"{subject}-{i}-opt{j}" for j in range(5)]
            rows.append({"question": f"{subject} q{i}", "choices": choices,
                        "answer": letter, "subject": subject, "split": "test"})
    path.write_text(json.dumps(rows))
    return rows


def test_fetch_turkishmmlu_never_calls_network_when_cache_exists(tmp_path, monkeypatch):
    from src.turkish import fetch_turkishmmlu
    cache = tmp_path / "cache.json"
    rows = _write_fixture_cache(cache, n_per_subject=3)

    def boom(*a, **k):
        raise AssertionError("network should not be called when the cache exists")
    monkeypatch.setattr("urllib.request.urlopen", boom)

    out = fetch_turkishmmlu(cache_path=str(cache))
    assert len(out) == len(rows)


def test_load_turkish_splits_deterministic_and_disjoint(tmp_path):
    from src.turkish import load_turkish_splits, SUBJECTS
    cache = tmp_path / "cache.json"
    _write_fixture_cache(cache, n_per_subject=10)

    train1, test1 = load_turkish_splits(seed=42, n_train_per_subject=4, cache_path=str(cache))
    train2, test2 = load_turkish_splits(seed=42, n_train_per_subject=4, cache_path=str(cache))

    # same seed + same cache -> identical qid order, both splits
    assert [e.qid for e in train1] == [e.qid for e in train2]
    assert [e.qid for e in test1] == [e.qid for e in test2]

    # sizes as configured (10 usable/subject, 4 subjects)
    assert len(train1) == 4 * len(SUBJECTS)
    assert len(test1) == (10 - 4) * len(SUBJECTS)

    train_qids = [e.qid for e in train1]
    test_qids = [e.qid for e in test1]
    assert len(set(train_qids)) == len(train_qids)      # no dupes within train
    assert len(set(test_qids)) == len(test_qids)         # no dupes within test
    assert set(train_qids).isdisjoint(test_qids)         # no overlap train/test
    assert all(re.match(r"^tr[A-Z]+\d+$", q) for q in train_qids + test_qids)
