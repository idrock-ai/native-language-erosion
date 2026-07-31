import json
import sys

from analysis.paper_numbers import build, main


def _item(cond, qid, is_correct, parse_error=False, truncated=False,
          rescue_correct=None, predicted="A", max_tokens=512):
    """One build_items-shaped row -- just the keys decompose_dir/collect read
    (see tests/test_decompose.py, tests/test_dose.py for the same convention)."""
    return {"model": "m", "condition": cond, "subject": "ona_tili", "qid": qid,
            "correct": "A", "is_correct": is_correct, "parse_error": parse_error,
            "truncated": truncated, "predicted": predicted, "max_tokens": max_tokens,
            "rescue_correct": is_correct if rescue_correct is None else rescue_correct}


def _write_e1(tmp_path):
    """Minimal E1 fixture: 3 paired qids, exactly one clean content b-flip, rest
    stable-correct -- just enough for decompose_dir (and, via the e1_dir 512
    fallback, dose_response.collect) to produce a well-formed, non-trivial result."""
    e1 = tmp_path / "results" / "e1"
    e1.mkdir(parents=True)
    items = [_item("cot", i, 1) for i in range(3)] + [
        _item("dspy_bootstrap", 0, 0, predicted="B"),
        _item("dspy_bootstrap", 1, 1),
        _item("dspy_bootstrap", 2, 1)]
    (e1 / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    return e1


def test_build_e1_key_present_and_well_formed(tmp_path):
    _write_e1(tmp_path)
    out = build(root=str(tmp_path))
    assert "e1" in out
    assert out["e1"]["per_model"]["m"]["flips"] == {
        "truncation": 0, "format_drift": 0, "content": 1}
    assert out["e1"]["pooled"]["b"] == 1 and out["e1"]["pooled"]["c"] == 0


def test_build_missing_optional_files_no_crash_no_key(tmp_path):
    _write_e1(tmp_path)  # only results/e1 exists -- no e2, e2/dose_response.json, e5
    out = build(root=str(tmp_path))
    assert "e2" not in out
    assert "e2_json" not in out
    assert "e5_fixes" not in out


def test_build_on_totally_empty_root_no_crash(tmp_path):
    # No results/ directory at all -- the pipeline stage "before E1 has run".
    out = build(root=str(tmp_path))
    assert out == {}


def test_build_ignores_e2_json_disk_copy_but_merges_e5_fixes(tmp_path):
    """The old e2/dose_response.json disk-copy merge is gone: recomputed e2 (when
    present) is the only source of truth, so a stray dose_response.json on disk
    must NOT surface as an "e2_json" key. e5's fixes.json is still merged verbatim."""
    _write_e1(tmp_path)
    e2 = tmp_path / "results" / "e2"
    e2.mkdir(parents=True)
    (e2 / "dose_response.json").write_text(json.dumps({"m": {"512": {"trend_p": 1.0}}}))
    e5 = tmp_path / "results" / "e5"
    e5.mkdir(parents=True)
    (e5 / "fixes.json").write_text(json.dumps([{"model": "m", "fixes": {}}]))

    out = build(root=str(tmp_path))
    assert "e2_json" not in out
    assert out["e5_fixes"] == [{"model": "m", "fixes": {}}]


def test_build_e2_key_absent_when_e2_dir_empty(tmp_path):
    """Empty e2 directory (no per-budget subdirs) should NOT produce e2 key."""
    _write_e1(tmp_path)
    (tmp_path / "results" / "e2").mkdir(parents=True)  # exists but empty
    out = build(root=str(tmp_path))
    assert "e2" not in out


def test_build_e2_key_absent_when_budget_subdir_empty(tmp_path):
    """Empty mt2048 subdir (created but unpopulated) should NOT produce e2 key."""
    _write_e1(tmp_path)
    e2 = tmp_path / "results" / "e2"
    e2.mkdir(parents=True)
    (e2 / "mt2048").mkdir(parents=True)  # exists but empty

    out = build(root=str(tmp_path))
    assert "e2" not in out


def test_build_e2_key_computed_when_e2_has_budget_data(tmp_path):
    """e2 key present only when actual per-budget data (mt2048, etc.) exists, and
    the model clears the degenerate-row filter: here the e1-fallback 512 cell plus
    a real 2048 cell gives it 2 distinct budget keys (see the drop-case test below,
    where a model only ever has the borrowed 512 cell)."""
    _write_e1(tmp_path)
    e2 = tmp_path / "results" / "e2"
    e2.mkdir(parents=True)
    mt2048 = e2 / "mt2048"
    mt2048.mkdir(parents=True)
    items = [_item("cot", i, 1, max_tokens=2048) for i in range(3)]
    (mt2048 / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))

    out = build(root=str(tmp_path))
    assert "e2" in out
    assert "m" in out["e2"]


def test_build_malformed_json_degraded_gracefully(tmp_path, capsys):
    """Corrupted JSON files should not crash build(); key absent + warning stderr."""
    _write_e1(tmp_path)
    e5 = tmp_path / "results" / "e5"
    e5.mkdir(parents=True)
    (e5 / "fixes.json").write_text("{not json")  # malformed JSON

    out = build(root=str(tmp_path))
    assert "e5_fixes" not in out
    assert "e1" in out  # other keys still present
    assert "[paper_numbers] WARNING: skipping malformed" in capsys.readouterr().err


def test_main_writes_paper_numbers_json(tmp_path, monkeypatch, capsys):
    _write_e1(tmp_path)
    monkeypatch.setattr(sys, "argv", ["paper_numbers.py", str(tmp_path)])
    main()
    out_path = tmp_path / "results" / "paper_numbers.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert "e1" in written
    assert "wrote" in capsys.readouterr().out


def test_main_succeeds_with_malformed_json(tmp_path, monkeypatch, capsys):
    """main() should still write output even if optional JSON files are corrupted."""
    _write_e1(tmp_path)
    e5 = tmp_path / "results" / "e5"
    e5.mkdir(parents=True)
    (e5 / "fixes.json").write_text("{broken json]")

    monkeypatch.setattr(sys, "argv", ["paper_numbers.py", str(tmp_path)])
    main()
    out_path = tmp_path / "results" / "paper_numbers.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert "e1" in written
    assert "e5_fixes" not in written
    assert "wrote" in capsys.readouterr().out


def test_build_e2_drops_degenerate_e1_fallback_only_models(tmp_path):
    """A model with a real e2 budget sweep (mt2048) is kept; a model that only ever
    appears via e1's borrowed 512 cell -- no real per-budget data anywhere in e2 --
    is dropped from e2 entirely (the bug this fix closes: pre-fix, EVERY e1 model
    showed up in e2 with a single-budget, statistically-meaningless row)."""
    e1 = tmp_path / "results" / "e1"
    e1.mkdir(parents=True)
    rows = [_item("cot", i, 1) for i in range(3)] + [_item("dspy_bootstrap", i, 1)
                                                       for i in range(3)]
    for model in ("swept", "fallback_only"):
        (e1 / f"{model}_items.jsonl").write_text(
            "\n".join(json.dumps({**r, "model": model}) for r in rows))

    e2 = tmp_path / "results" / "e2"
    mt2048 = e2 / "mt2048"
    mt2048.mkdir(parents=True)
    swept = [_item("cot", i, 1, max_tokens=2048) for i in range(3)]
    (mt2048 / "swept_items.jsonl").write_text(
        "\n".join(json.dumps({**r, "model": "swept"}) for r in swept))

    out = build(root=str(tmp_path))
    assert "swept" in out["e2"]
    assert "fallback_only" not in out["e2"]


def test_build_e3_key_present_when_populated_absent_when_missing(tmp_path):
    _write_e1(tmp_path)
    out = build(root=str(tmp_path))
    assert "e3" not in out  # no results/e3 directory at all yet

    e3 = tmp_path / "results" / "e3"
    e3.mkdir(parents=True)
    rows = ([_item("cot", i, 1) for i in range(4)] +
            [_item("reason_short", i, 1 if i < 3 else 0) for i in range(4)])
    (e3 / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in rows))

    out = build(root=str(tmp_path))
    assert "e3" in out
    assert "m" in out["e3"]["per_model"]


def test_build_e4_key_present_when_populated_absent_when_missing(tmp_path):
    _write_e1(tmp_path)
    out = build(root=str(tmp_path))
    assert "e4" not in out  # no results/e4 directory at all yet

    e4 = tmp_path / "results" / "e4"
    e4.mkdir(parents=True)
    rows = [{**_item(cond, i, 1), "reasoning": "javob shu."}
            for cond in ("cot", "dspy_bootstrap") for i in range(3)]
    (e4 / "m_paper_items.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    out = build(root=str(tmp_path))
    assert set(out["e4"]) == {"paper", "replication"}
    assert out["e4"]["paper"]["deployment"]["n"] == 3
    assert out["e4"]["replication"]["deployment"]["n"] == 0
