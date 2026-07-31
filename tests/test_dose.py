import json
from analysis.dose_response import collect

def _mk(tmp, mt, acc_ona, trunc):
    d = tmp / f"mt{mt}"; d.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(10):
        items.append({"model": "m", "condition": "cot", "subject": "ona_tili", "qid": i,
                      "correct": "A", "is_correct": 1, "parse_error": False,
                      "truncated": False, "predicted": "A", "rescue_correct": 1,
                      "max_tokens": mt})
        items.append({"model": "m", "condition": "dspy_bootstrap", "subject": "ona_tili",
                      "qid": i, "correct": "A", "is_correct": int(i < acc_ona),
                      "parse_error": False, "truncated": i < trunc, "predicted": "A",
                      "rescue_correct": int(i < acc_ona), "max_tokens": mt})
    (d / "m_items.jsonl").write_text("\n".join(json.dumps(x) for x in items))

def test_collect_trend(tmp_path):
    _mk(tmp_path, 256, 4, 6); _mk(tmp_path, 512, 6, 4)
    _mk(tmp_path, 1024, 8, 2); _mk(tmp_path, 2048, 10, 0)
    table = collect(str(tmp_path), budgets=(256, 512, 1024, 2048))
    row = table["m"]
    assert [row[b]["trunc_boot"] for b in (256, 512, 1024, 2048)] == [6, 4, 2, 0]
    assert row["trend_p"] < 0.05
