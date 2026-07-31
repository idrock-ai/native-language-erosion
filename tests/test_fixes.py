from analysis.fixes_table import recovery

def test_recovery_and_retention():
    # cot ona=36, vanilla boot ona=27, fix ona=34 -> recovery 7/9
    assert abs(recovery(cot=36.0, vanilla=27.0, fixed=34.0) - 7 / 9 * 100) < 1e-6
    assert recovery(cot=36.0, vanilla=36.0, fixed=36.0) == 100.0   # nothing to recover
