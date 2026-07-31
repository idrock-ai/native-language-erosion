import json
import sys

from analysis.residual_stats import collect_set, main


def _item(qid, condition, is_correct, reasoning, rescue_correct=None,
          parse_error=False, truncated=False):
    """One build_items-shaped row (see src/run.py build_items + instrument.py record
    fields); only the keys collect_set actually reads, plus the header fields every
    real *_items.jsonl row carries."""
    return {"model": "m", "engine": "ollama", "condition": condition, "max_tokens": 512,
            "subject": "ona_tili", "qid": qid, "correct": "A", "is_correct": is_correct,
            "rescue_correct": is_correct if rescue_correct is None else rescue_correct,
            "parse_error": parse_error, "truncated": truncated, "reasoning": reasoning}


def _write(tmp_path, items, name="m_paper_items.jsonl"):
    (tmp_path / name).write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items))


def _fixture_items():
    """4 qids on the 'paper' set:
    qid=1: clean b-flip (cot right, boot wrong); boot reasoning hedges ("Balki").
    qid=2: clean c-flip (cot wrong, boot right).
    qid=3: boot truncated -> excluded from knowledge (clean) pool but still counted in
           deployment (pooled)/rescue; cot right & boot wrong, so it is ALSO a
           deployment-only b-flip (this is what must make knowledge's b differ from
           deployment's b, not just n).
    qid=4: stable-correct (cot right, boot right), clean.
    """
    return [
        _item(1, "cot", 1, "To'g'ri, chunki qonuniyat shunday."),
        _item(1, "dspy_bootstrap", 0, "Balki bu javob noto'g'ri edi."),
        _item(2, "cot", 0, "Noto'g'ri javob berdim shekilli."),
        _item(2, "dspy_bootstrap", 1, "To'g'ri javob aniq."),
        _item(3, "cot", 1, "cot uchinchi savolga javob."),
        _item(3, "dspy_bootstrap", 0, "kesilgan javob...", truncated=True),
        _item(4, "cot", 1, "cot to'rtinchi javob."),
        _item(4, "dspy_bootstrap", 1, "boot to'rtinchi, barqaror javob."),
    ]


# Hand-computed expectations (see docstring above for the qid-by-qid story):
#   deployment pairs (cot.is_correct, boot.is_correct) over ALL 4 qids:
#     q1=(1,0) q2=(0,1) q3=(1,0) q4=(1,1)
#     b (right->wrong) = q1,q3 = 2; c (wrong->right) = q2 = 1; n = 4
#   knowledge (clean) drops q3 (boot truncated), leaving q1=(1,0) q2=(0,1) q4=(1,1):
#     b = q1 = 1; c = q2 = 1; n = 3
#   mcnemar_exact(2,1): n=3,min=1 -> tail=(C(3,0)+C(3,1))/8=4/8=0.5 -> p=min(1,1.0)=1.0
#   mcnemar_exact(1,1): n=2,min=1 -> tail=(C(2,0)+C(2,1))/4=3/4=0.75 -> p=min(1,1.5)=1.0

def test_collect_set_deployment_and_knowledge_counts(tmp_path):
    _write(tmp_path, _fixture_items())
    stats = collect_set(str(tmp_path), "paper")
    dep, kno = stats["deployment"], stats["knowledge"]
    assert (dep["b"], dep["c"], dep["n"]) == (2, 1, 4)
    assert dep["p"] == 1.0
    assert (kno["b"], kno["c"], kno["n"]) == (1, 1, 3)
    assert kno["p"] == 1.0
    assert kno["n"] < dep["n"]          # truncated q3 excluded from knowledge only
    # rescue_correct == is_correct in this fixture, so rescue must reproduce deployment
    # exactly -- a cheap, free check that the "rescue" pool is wired to the right field.
    rescue = stats["rescue"]
    assert (rescue["b"], rescue["c"], rescue["n"]) == (dep["b"], dep["c"], dep["n"])


def test_collect_set_style_counts_hedge_on_clean_b_flip_only(tmp_path):
    _write(tmp_path, _fixture_items())
    style = collect_set(str(tmp_path), "paper")["style"]
    # Only q1 is a clean b-flip (q3's b-flip is excluded from "clean" by truncation).
    assert style["b_n"] == 1
    assert style["b_hedge"] == 1
    assert style["b_year"] == 0
    assert style["b_quote"] == 0
    assert len(style["b_len"]) == 1
    assert len(style["c_len"]) == 1      # q2
    assert len(style["stable_len"]) == 1  # q4


def test_main_prints_both_sets_and_all_pool_names(tmp_path, monkeypatch, capsys):
    _write(tmp_path, _fixture_items())
    monkeypatch.setattr(sys, "argv", ["residual_stats.py", str(tmp_path)])
    main()
    out = capsys.readouterr().out
    assert "== paper" in out and "== replication" in out
    assert "deployment" in out and "knowledge" in out and "rescue" in out
    assert "style: mean boot-reasoning chars" in out
