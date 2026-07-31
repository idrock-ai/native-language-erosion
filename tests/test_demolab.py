from types import SimpleNamespace as NS
from src.demolab import split_pools, select_demos

def _demo(subject, reasoning):
    return NS(subject=subject, reasoning=reasoning, question="q", options="o",
              answer_letter="A", qid=1)

def test_split_pools_by_compliance():
    demos = [_demo("matematika", "Qisqa."),
             _demo("matematika", "Gap bir. Gap ikki. Gap uch. " + "x" * 300),
             _demo("ona_tili", "Qisqa gap."),
             _demo("ona_tili", "Uzun gap bir. Uzun gap ikki. Uzun gap uch. " + "y" * 300)]
    pools = split_pools(demos)
    assert len(pools[("reason", "short")]) == 1
    assert len(pools[("reason", "long")]) == 1
    assert len(pools[("native", "short")]) == 1
    assert len(pools[("native", "long")]) == 1

def test_select_relaxes_when_short_supply(capsys):
    demos = [_demo("matematika", "Qisqa.")] * 2
    picked = select_demos(split_pools(demos), "reason", "short", k=4, seed=1)
    assert len(picked) == 2          # takes what exists, never crashes
