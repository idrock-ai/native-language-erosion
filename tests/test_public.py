import pytest
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "data" / "DTM2019_public.csv"
pytestmark = pytest.mark.skipif(not CSV.exists(), reason="public csv not present")

def test_load_public_counts_and_subjects():
    from src.data import load_public
    recs = load_public()
    assert 2000 <= len(recs) <= 2066            # 2066 minus ~38 benchmark dups
    assert {r["subject"] for r in recs} == {"matematika", "fizika", "tarix", "ona_tili"}
    assert all(r["answer"] in "ABCD" for r in recs)
    assert all(str(r["qid"]).startswith("pub") for r in recs)

def test_replication_set_frozen_and_deduped():
    from src.data import load_public, replication_onatili
    a = [e.qid for e in replication_onatili()]
    b = [e.qid for e in replication_onatili()]
    assert a == b and len(a) >= 380             # 426 ona_tili minus dups
    import json
    from src.data import DEFAULT_DATASET
    bench_q = {" ".join(r["question"].split()).lower() for r in json.load(open(DEFAULT_DATASET))}
    assert all(" ".join(e.question.split()).lower() not in bench_q for e in replication_onatili())
