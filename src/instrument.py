"""Instrumented evaluation loop shared by all experiment runners (E1-E6):
captures raw completions, token usage, finish_reason; salvages AdapterParseError;
computes rescue predictions. Thread-safe via one LM per worker thread."""
from __future__ import annotations
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import dspy
from tqdm import tqdm

from .program import CHOICE_LETTERS, parse_letter, rescue_letter

try:
    from dspy.utils.exceptions import AdapterParseError
except Exception:  # pragma: no cover - dspy relocation guard
    class AdapterParseError(Exception):
        lm_response = ""


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_history_entry(entry):
    """(raw_text, finish_reason, prompt_tokens, completion_tokens) from an LM history entry."""
    raw, fin, pt, ct = "", "", None, None
    resp = _get(entry, "response")
    choices = _get(resp, "choices") or []
    if choices:
        ch = choices[0]
        msg = _get(ch, "message")
        raw = _get(msg, "content") or ""
        fin = _get(ch, "finish_reason") or ""
    usage = _get(entry, "usage") or {}
    pt, ct = _get(usage, "prompt_tokens"), _get(usage, "completion_tokens")
    return (raw or "")[:8000], fin or "", pt, ct


def instrumented_eval(examples, program, lm_factory, workers: int = 8,
                      desc: str = "", max_tokens: int | None = None,
                      letters: str = CHOICE_LETTERS):
    local = threading.local()

    def one(ex):
        if not hasattr(local, "lm"):
            local.lm = lm_factory()
        lm = local.lm
        n0 = len(getattr(lm, "history", []))
        rec = dict(predicted="", reasoning="", raw_text="", parse_error=False,
                   error_type="", finish_reason="", prompt_tokens=None,
                   completion_tokens=None, truncated=False, latency_ms=0,
                   rescue_predicted="")
        t0 = time.time()
        try:
            with dspy.context(lm=lm):
                p = program(question=ex.question, options=ex.options)
            rec["predicted"] = parse_letter(p.answer_letter, letters)
            rec["reasoning"] = (getattr(p, "reasoning", "") or "")[:4000]
        except AdapterParseError as e:
            rec["parse_error"] = True
            rec["error_type"] = "AdapterParseError"
            rec["raw_text"] = (getattr(e, "lm_response", "") or "")[:8000]
        except Exception as e:  # noqa: BLE001 - record and continue
            rec["error_type"] = type(e).__name__
        rec["latency_ms"] = int(1000 * (time.time() - t0))
        hist = getattr(lm, "history", [])
        if len(hist) > n0:
            raw, fin, pt, ct = extract_history_entry(hist[-1])
            rec["raw_text"] = rec["raw_text"] or raw
            rec["finish_reason"], rec["prompt_tokens"], rec["completion_tokens"] = fin, pt, ct
        rec["truncated"] = rec["finish_reason"] == "length" or (
            max_tokens is not None and rec["completion_tokens"] is not None
            and rec["completion_tokens"] >= max_tokens)
        rec["rescue_predicted"] = rec["predicted"] or rescue_letter(rec["raw_text"], letters)
        return rec

    out = [None] * len(examples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, r in tqdm(zip(range(len(examples)), pool.map(one, examples)),
                         total=len(examples), desc=desc, unit="q"):
            out[i] = r
    return out
