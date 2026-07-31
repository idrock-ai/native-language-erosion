import json
from analysis.decompose import classify_flip, decompose_dir

def _item(cond, qid, is_correct, parse_error=False, truncated=False,
          rescue_correct=None, predicted="A"):
    return {"model": "m", "condition": cond, "subject": "ona_tili", "qid": qid,
            "correct": "A", "is_correct": is_correct, "parse_error": parse_error,
            "truncated": truncated, "predicted": predicted,
            "rescue_correct": is_correct if rescue_correct is None else rescue_correct}

def test_classify_flip():
    cot = _item("cot", 1, 1)
    assert classify_flip(cot, _item("dspy_bootstrap", 1, 0, truncated=True)) == "truncation"
    assert classify_flip(cot, _item("dspy_bootstrap", 1, 0, parse_error=True)) == "format_drift"
    assert classify_flip(cot, _item("dspy_bootstrap", 1, 0, predicted="B")) == "content"

def test_decompose_dir(tmp_path):
    items = [_item("cot", i, 1) for i in range(4)] + [
        _item("dspy_bootstrap", 0, 0, truncated=True),
        _item("dspy_bootstrap", 1, 0, parse_error=True, rescue_correct=1),
        _item("dspy_bootstrap", 2, 0, predicted="B"),
        _item("dspy_bootstrap", 3, 1)]
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    out = decompose_dir(str(tmp_path))
    f = out["per_model"]["m"]["flips"]
    assert f == {"truncation": 1, "format_drift": 1, "content": 1}
    assert out["pooled"]["b"] == 3 and out["pooled"]["c"] == 0
