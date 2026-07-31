from src.stats import mcnemar_exact, wilson, holm, cochran_armitage, flips

def test_mcnemar_exact_small():
    # b=9,c=2: two-sided exact = 2*P(X<=2 | n=11) = 2*(1+11+55)/2048
    assert abs(mcnemar_exact(9, 2) - 134 / 2048) < 1e-12
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0

def test_mcnemar_paper_numbers():
    assert 0.008 < mcnemar_exact(75, 46) < 0.014     # paper headline, exact version
    assert 0.065 < mcnemar_exact(66, 46) < 0.080     # clean-item reanalysis

def test_wilson_known():
    pct, lo, hi = wilson(5, 10)
    assert abs(pct - 50.0) < 1e-9 and abs(lo - 23.66) < 0.05 and abs(hi - 76.34) < 0.05

def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]

def test_cochran_armitage_trend_vs_flat():
    z, p = cochran_armitage([10, 20, 30, 40], [50, 50, 50, 50])
    assert p < 1e-3 and z > 0
    _, pflat = cochran_armitage([25, 25, 25, 25], [50, 50, 50, 50])
    assert pflat > 0.9

def test_flips():
    assert flips([(1, 0), (1, 0), (0, 1), (1, 1), (0, 0)]) == (2, 1)
