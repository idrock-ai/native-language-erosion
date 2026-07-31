from types import SimpleNamespace as NS
from src.instrument import instrumented_eval, AdapterParseError
from src.program import ConciseCoT


class FakeLM:
    def __init__(self):
        self.history = []


def make_entry(content, finish="stop", pt=100, ct=50):
    return {"response": {"choices": [{"message": {"content": content},
                                      "finish_reason": finish}]},
            "usage": {"prompt_tokens": pt, "completion_tokens": ct}}


EXAMPLES = [NS(question="q?", options="A) 1\nB) 2\nC) 3\nD) 4")]


def test_success_path():
    lm = FakeLM()
    def program(question, options):
        lm.history.append(make_entry("RAW OK", "stop", 120, 40))
        return NS(answer_letter="Javob: B", reasoning="Qisqa.")
    [r] = instrumented_eval(EXAMPLES, program, lambda: lm, workers=1)
    assert r["predicted"] == "B" and r["raw_text"] == "RAW OK"
    assert r["finish_reason"] == "stop" and not r["truncated"] and not r["parse_error"]
    assert r["rescue_predicted"] == "B" and r["completion_tokens"] == 40

def test_adapter_parse_error_salvage_and_truncation():
    lm = FakeLM()
    raw = "[[ ## reasoning ## ]]\nuzun matn...\n[[ ## answer_letter ## ]]\nC"
    def program(question, options):
        lm.history.append(make_entry(raw, "length", 900, 512))
        e = AdapterParseError(adapter_name="ChatAdapter", signature=ConciseCoT, lm_response=raw)
        raise e
    [r] = instrumented_eval(EXAMPLES, program, lambda: lm, workers=1, max_tokens=512)
    assert r["parse_error"] and r["error_type"] == "AdapterParseError"
    assert r["truncated"] and r["raw_text"] == raw
    assert r["predicted"] == "" and r["rescue_predicted"] == "C"

def test_generic_error():
    lm = FakeLM()
    def program(question, options):
        raise RuntimeError("boom")
    [r] = instrumented_eval(EXAMPLES, program, lambda: lm, workers=1)
    assert r["error_type"] == "RuntimeError" and r["predicted"] == ""
