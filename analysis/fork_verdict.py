#!/usr/bin/env python3
"""Precision fork verdict for the anchor (Qwen3.5-9B): ona_tili CoT->DSPy delta at
Q4 (Ollama GGUF) vs AWQ-4bit (vLLM) vs bf16 (vLLM). The same-engine AWQ-vs-bf16
contrast is the clean precision test."""
import json, sys

FILES = {
    "Q4  (Ollama GGUF)":  "results/main/qwen3.5_9b_report.json",
    "AWQ (vLLM 4-bit) ":  "results/precision/QuantTrio_Qwen3.5-9B-AWQ_report.json",
    "bf16(vLLM 16-bit)":  "results/precision/Qwen_Qwen3.5-9B_report.json",
}

def row(f):
    c = json.load(open(f))["conditions"]
    out = {}
    for subj in ("ona_tili", "tarix", "matematika", "fizika"):
        try:
            b = c["baseline"]["by_subject"][subj]["accuracy"]
            ct = c["cot"]["by_subject"][subj]["accuracy"]
            d = c["dspy_bootstrap"]["by_subject"][subj]["accuracy"]
            out[subj] = (b, ct, d)
        except Exception:
            out[subj] = None
    return out, c["cot"]["overall"], c["dspy_bootstrap"]["overall"]

print(f"{'precision':18}{'ona:cot':>8}{'ona:dspy':>9}{'ona dCoT->DSPy':>15}{'  |  math dCoT->DSPy':>20}")
print("-" * 72)
ona_deltas = {}
for label, f in FILES.items():
    try:
        r, ov_c, ov_d = row(f)
    except FileNotFoundError:
        print(f"{label:18}   (not ready)")
        continue
    o = r["ona_tili"]; m = r["matematika"]
    if o:
        ona_deltas[label] = o[2] - o[1]
        md = (m[2]-m[1]) if m else float('nan')
        print(f"{label:18}{o[1]:>8.0f}{o[2]:>9.0f}{o[2]-o[1]:>+15.0f}{md:>+20.0f}")

print("-" * 72)
if "AWQ (vLLM 4-bit) " in ona_deltas and "bf16(vLLM 16-bit)" in ona_deltas:
    awq, bf = ona_deltas["AWQ (vLLM 4-bit) "], ona_deltas["bf16(vLLM 16-bit)"]
    print(f"\nSAME-ENGINE (vLLM) precision test on ona_tili CoT->DSPy:")
    print(f"  AWQ 4-bit = {awq:+.0f}   bf16 16-bit = {bf:+.0f}")
    if awq < -1 and bf >= 0:
        print("  => VANISHES at bf16  ->  precision-dependent (quantization x prompt-opt story)")
    elif awq < -1 and bf < -1:
        print("  => PERSISTS at bf16  ->  precision-independent (DSPy demo-skew failure mode)")
    else:
        print("  => inconclusive / weak at 4-bit; inspect full table")
