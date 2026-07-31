from src.run import build_items, score_items

REC = dict(predicted="B", reasoning="ok", raw_text="RAW", parse_error=False,
           error_type="", finish_reason="stop", prompt_tokens=10,
           completion_tokens=5, truncated=False, latency_ms=9, rescue_predicted="B")

class Ex:
    question, options, subject, answer_letter, qid = "q", "o", "ona_tili", "B", 7

def test_build_items_schema_and_scoring():
    [it] = build_items([Ex()], [dict(REC)], model="m", engine="ollama",
                       condition="cot", max_tokens=512)
    for key in ("model", "engine", "condition", "max_tokens", "subject", "qid",
                "correct", "is_correct", "rescue_correct", "predicted", "raw_text",
                "parse_error", "finish_reason", "completion_tokens", "truncated"):
        assert key in it
    assert it["is_correct"] == 1 and it["rescue_correct"] == 1 and it["correct"] == "B"
    rep = score_items([it], "is_correct")
    assert rep["overall"] == 100.0 and rep["by_subject"]["ona_tili"]["accuracy"] == 100.0
