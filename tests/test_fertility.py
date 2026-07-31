"""Fixture test for fertility.py empirical token analysis."""
import json
import tempfile
from pathlib import Path

import pytest

from analysis.fertility import collect_cells


@pytest.fixture
def tmp_items_dir(tmp_path):
    """Create a temporary directory with one items.jsonl containing two rows with known token/char ratios."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    # Create items.jsonl with two rows: different subjects, known tok/char ratios
    items_file = results_dir / "test_items.jsonl"

    # Row 1: subject=ona_tili, model=test_model, condition=cot
    # raw_text with 100 chars, completion_tokens=10
    row1 = {
        "raw_text": "x" * 100,
        "completion_tokens": 10,
        "model": "test_model",
        "condition": "cot",
        "subject": "ona_tili"
    }

    # Row 2: subject=kamba, model=test_model, condition=cot
    # raw_text with 50 chars, completion_tokens=5
    row2 = {
        "raw_text": "y" * 50,
        "completion_tokens": 5,
        "model": "test_model",
        "condition": "cot",
        "subject": "kamba"
    }

    with open(items_file, "w") as f:
        f.write(json.dumps(row1) + "\n")
        f.write(json.dumps(row2) + "\n")

    return str(results_dir)


def test_fertility_collect_cells(tmp_items_dir):
    """Test collect_cells aggregates token/char and token/item metrics correctly."""
    cells = collect_cells([tmp_items_dir])

    # Check that we have exactly 2 cells (one per subject)
    assert len(cells) == 2

    # Check keys are tuples of (model, condition, subject)
    for key in cells.keys():
        assert isinstance(key, tuple)
        assert len(key) == 3
        model, condition, subject = key
        assert model == "test_model"
        assert condition == "cot"
        assert subject in ["ona_tili", "kamba"]

    # Check ona_tili: 100 chars, 10 tokens, 1 item
    # tok/char = 10/100 = 0.1, tok/item = 10/1 = 10
    key_ona = ("test_model", "cot", "ona_tili")
    assert key_ona in cells
    cell_ona = cells[key_ona]
    assert cell_ona["tok"] == 10
    assert cell_ona["chars"] == 100
    assert cell_ona["n"] == 1
    assert cell_ona["tok"] / cell_ona["chars"] == 0.1
    assert cell_ona["tok"] / cell_ona["n"] == 10

    # Check kamba: 50 chars, 5 tokens, 1 item
    # tok/char = 5/50 = 0.1, tok/item = 5/1 = 5
    key_kamba = ("test_model", "cot", "kamba")
    assert key_kamba in cells
    cell_kamba = cells[key_kamba]
    assert cell_kamba["tok"] == 5
    assert cell_kamba["chars"] == 50
    assert cell_kamba["n"] == 1
    assert cell_kamba["tok"] / cell_kamba["chars"] == 0.1
    assert cell_kamba["tok"] / cell_kamba["n"] == 5
