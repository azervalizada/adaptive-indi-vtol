"""Summary/table logic on synthetic rows (added 28 Oct 2026 after the independent review: findings
C2, M4, m2, m3, m4 were all in code that had no test)."""
import numpy as np

from run_qp_benchmark import gap_closed, by_speed_table, lowest_from_which_all, by_scenario


def _row(c, s, V, tilt, lost=False, rec=True, err=None):
    r = {"controller": c, "scenario": s, "V": V, "tilt_peak_deg": tilt, "lost": lost, "recovered": rec}
    if err:
        r["error"] = err
    return r


def test_by_speed_uses_the_common_survived_set():
    """C2: controller A survives a hard case B loses; A must not look worse because of it."""
    rows = [_row("baseline", "s1", 0.0, 2.0), _row("l1", "s1", 0.0, 1.0),
            _row("baseline", "s2", 0.0, 20.0), _row("l1", "s2", 0.0, None, lost=True, rec=False)]
    line = {l.split()[0]: l for l in by_speed_table(rows, ["baseline", "l1"])[3:]}
    assert "2.00" in line["baseline"] and "1.00" in line["l1"] and "L1" in line["l1"]
    assert "(n=1)" in by_speed_table(rows, ["baseline", "l1"])[2]


def test_gap_closed_does_not_depend_on_other_controllers():
    """m2: adding a variant that loses a scenario must not change another method's gap closed."""
    rows = []
    for i, (b, o, m) in enumerate([(3.0, 1.0, 2.0), (5.0, 1.0, 1.0)]):
        rows += [_row("baseline", f"s{i}", 0, b), _row("oracle", f"s{i}", 0, o), _row("rls", f"s{i}", 0, m)]
    g1 = gap_closed(rows, "rls")[0]
    rows2 = rows + [_row("variant", "s1", 0, None, lost=True, rec=False), _row("variant", "s0", 0, 1.0)]
    assert np.isclose(g1, gap_closed(rows2, "rls")[0]) and np.isclose(g1, 100 * (1 + 4) / (2 + 4))


def test_bootstrap_does_not_depend_on_row_order():
    """m4: the CI must be the same whatever order the rows come in."""
    rng = np.random.default_rng(1)
    rows = []
    for i in range(20):
        b = 3 + rng.random()
        rows += [_row("baseline", f"s{i:02d}", 0, b), _row("oracle", f"s{i:02d}", 0, 1.0),
                 _row("mrac", f"s{i:02d}", 0, 1 + 2 * rng.random())]
    a = gap_closed(rows, "mrac")
    b = gap_closed(rows[::-1], "mrac")
    assert np.allclose(a, b)


def test_lowest_speed_flags_missing_and_errored_runs():
    """m3 / M4: a missing or errored speed must not be skipped silently."""
    ok = lambda r: r["recovered"]
    full = [_row("b", f"v{v}", v, 1.0, rec=v >= 4) for v in range(0, 9, 2)]
    assert lowest_from_which_all(full, "V", ok) == (4, True)
    gap = [r for r in full if r["V"] != 6]
    assert lowest_from_which_all(gap, "V", ok, grid=range(0, 9, 2)) == (8, False)
    err = [dict(r, error="boom") if r["V"] == 6 else r for r in full]
    assert lowest_from_which_all(err, "V", ok) == (8, False)


def test_by_scenario_is_sorted():
    rows = [_row("a", "s2", 0, 1.0), _row("a", "s1", 0, 1.0)]
    assert list(by_scenario(rows)) == ["s1", "s2"]
