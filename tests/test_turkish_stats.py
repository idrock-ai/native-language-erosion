import json
import pytest
from analysis.turkish_stats import (NATIVE, analyse, binom_two_sided, spearman,
                                    spearman_perm_p)


def _item(cond, qid, is_correct, subject=NATIVE, truncated=False, parse_error=False,
          model="m"):
    return {"model": model, "condition": cond, "subject": subject, "qid": qid,
            "correct": "A", "is_correct": is_correct, "rescue_correct": is_correct,
            "predicted": "A", "parse_error": parse_error, "truncated": truncated}


def test_binom_two_sided_matches_hand_computed():
    assert binom_two_sided(3, 3) == pytest.approx(0.25)
    assert binom_two_sided(6, 6) == pytest.approx(0.03125)
    assert binom_two_sided(5, 6) == pytest.approx(0.21875)
    assert binom_two_sided(3, 6) == pytest.approx(1.0)
    assert binom_two_sided(0, 0) == 1.0


def test_spearman_and_exact_permutation_p():
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman([1, 1, 1], [1, 2, 3]) is None          # zero variance
    # a perfect rank correlation is NOT evidence at n=3, but is at n=6
    assert spearman_perm_p([1, 2, 3], [3, 2, 1]) == pytest.approx(1 / 3, abs=1e-3)
    assert spearman_perm_p([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]) == pytest.approx(2 / 720,
                                                                                    abs=1e-4)
    assert spearman_perm_p([1, 2], [2, 1]) is None         # refuses n<3


def test_analyse_accuracy_failures_and_repair(tmp_path):
    # cot: 3/4 native correct; vanilla: 1/4 (two truncation-driven losses);
    # compliant: 3/4 with the truncations gone -> full recovery, no reasoning cost.
    items = (
        [_item("cot", f"n{i}", 1 if i < 3 else 0) for i in range(4)]
        + [_item("dspy_bootstrap", "n0", 0, truncated=True),
           _item("dspy_bootstrap", "n1", 0, parse_error=True),
           _item("dspy_bootstrap", "n2", 1), _item("dspy_bootstrap", "n3", 0)]
        + [_item("dspy_bootstrap_compliant", f"n{i}", 1 if i < 3 else 0)
           for i in range(4)]
        + [_item(c, f"m{i}", 1, subject="Mathematics")
           for c in ("cot", "dspy_bootstrap", "dspy_bootstrap_compliant")
           for i in range(2)]
        + [_item(c, f"p{i}", 1, subject="Physics")
           for c in ("cot", "dspy_bootstrap", "dspy_bootstrap_compliant")
           for i in range(2)]
    )
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    out = analyse(str(tmp_path))

    r = out["per_model"]["m"]
    assert r["accuracy"]["cot"][NATIVE] == 75.0
    assert r["accuracy"]["dspy_bootstrap"][NATIVE] == 25.0
    assert r["failures"]["dspy_bootstrap"] == {"truncated": 1, "parse_error": 1,
                                               "truncated_native": 1,
                                               "parse_error_native": 1}
    assert r["native_mcnemar"] == {"b": 2, "c": 0, "n": 4, "p": 0.5, "p_holm": 0.5}
    assert r["repair"]["native_lost_pp"] == 50.0
    assert r["repair"]["native_recovery_pct"] == 100.0
    assert r["repair"]["reasoning_vanilla"] == r["repair"]["reasoning_compliant"] == 100.0

    assert out["pooled"]["deployment"] == {"b": 2, "c": 0, "n": 4}
    assert out["flips_per_model"]["m"] == {"truncation": 1, "format_drift": 1,
                                           "content": 0}
    assert out["mechanism_link"]["delta_native_acc_pp"] == [-50.0]
    assert out["mechanism_link"]["delta_native_failures"] == [2]
    assert out["failure_pressure"]["models_with_more_truncations_under_vanilla"] == 1


def test_repair_reports_na_when_no_erosion(tmp_path):
    """A model that GAINS under vanilla has nothing to recover -> None, not a bogus %."""
    items = ([_item("cot", "n0", 0), _item("dspy_bootstrap", "n0", 1),
              _item("dspy_bootstrap_compliant", "n0", 1)]
             + [_item(c, "m0", 1, subject="Mathematics")
                for c in ("cot", "dspy_bootstrap", "dspy_bootstrap_compliant")]
             + [_item(c, "p0", 1, subject="Physics")
                for c in ("cot", "dspy_bootstrap", "dspy_bootstrap_compliant")])
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    out = analyse(str(tmp_path))
    assert out["per_model"]["m"]["repair"]["native_recovery_pct"] is None
