import json
import sys

from analysis.demolab_stats import CONTRASTS, collect_stats, main


def _write_items(tmp_path, model, rows):
    """rows: list of (condition, qid, is_correct). Writes a <model>_items.jsonl fixture
    with the fields analysis/demolab_stats.py and src/run.py's build_items both rely on."""
    items = [{"model": model, "engine": "ollama", "condition": cond, "max_tokens": 512,
              "subject": "ona_tili", "qid": qid, "correct": "A", "is_correct": is_correct,
              "rescue_correct": is_correct} for cond, qid, is_correct in rows]
    (tmp_path / f"{model}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items))


def _m1_rows():
    """m1: long clearly worse than short (reason group) -- 6 of 8 flip from right to wrong."""
    rows = []
    for qid in range(8):
        rows.append(("cot", qid, 1))
        rows.append(("reason_short", qid, 1))
        rows.append(("reason_long", qid, 1 if qid < 2 else 0))
    return rows


def _m2_rows():
    """m2: flat -- cot/short/long agree on every qid, so every contrast is discordant-free."""
    rows = []
    for qid in range(8):
        v = 1 if qid % 2 == 0 else 0
        rows += [("cot", qid, v), ("reason_short", qid, v), ("reason_long", qid, v)]
    return rows


def test_six_contrast_keys_present(tmp_path):
    _write_items(tmp_path, "m1", _m1_rows())
    stats = collect_stats(str(tmp_path))
    assert len(CONTRASTS) == 6
    assert set(stats["pooled"]) == set(CONTRASTS)
    assert set(stats["per_model"]["m1"]) == set(CONTRASTS)


def test_per_model_nets_have_expected_signs(tmp_path):
    _write_items(tmp_path, "m1", _m1_rows())
    _write_items(tmp_path, "m2", _m2_rows())
    stats = collect_stats(str(tmp_path))
    m1 = stats["per_model"]["m1"]["long vs short"]
    m2 = stats["per_model"]["m2"]["long vs short"]
    assert m1["net"] < 0    # long demos clearly worse than short for m1
    assert m2["net"] == 0   # flat model: no discordant pairs at all


def test_pooled_long_vs_short_equals_sum_of_per_model(tmp_path):
    _write_items(tmp_path, "m1", _m1_rows())
    _write_items(tmp_path, "m2", _m2_rows())
    stats = collect_stats(str(tmp_path))
    m1 = stats["per_model"]["m1"]["long vs short"]
    m2 = stats["per_model"]["m2"]["long vs short"]
    pooled = stats["pooled"]["long vs short"]
    assert pooled["b"] == m1["b"] + m2["b"]
    assert pooled["c"] == m1["c"] + m2["c"]
    assert pooled["n"] == m1["n"] + m2["n"]


def test_pooled_matches_sum_of_per_model_for_every_contrast(tmp_path):
    """Guards against duplicated/divergent pairing logic across ALL six contrasts, not
    just long-vs-short: pooled (b, c) must equal the sum of every model's own (b, c)."""
    _write_items(tmp_path, "m1", _m1_rows())
    _write_items(tmp_path, "m2", _m2_rows())
    stats = collect_stats(str(tmp_path))
    for contrast in CONTRASTS:
        b_sum = sum(m[contrast]["b"] for m in stats["per_model"].values())
        c_sum = sum(m[contrast]["c"] for m in stats["per_model"].values())
        assert stats["pooled"][contrast]["b"] == b_sum
        assert stats["pooled"][contrast]["c"] == c_sum


def test_main_prints_per_model_sign_test_and_suffixed_pooled_table(tmp_path, monkeypatch,
                                                                    capsys):
    _write_items(tmp_path, "m1", _m1_rows())
    _write_items(tmp_path, "m2", _m2_rows())
    monkeypatch.setattr(sys, "argv", ["demolab_stats.py", str(tmp_path)])
    main()
    out = capsys.readouterr().out
    assert "per-model breakdown" in out
    assert "m1" in out and "m2" in out
    assert "long-vs-short:" in out and "models negative" in out
    for contrast in CONTRASTS:
        assert f"{contrast} (pooled)" in out
