"""Checks for the MRAC-INDI estimator. Run: pytest -q"""
import numpy as np

from src.scenarios import Scenario, Fault
from src.controllers import run_scenario


def test_mrac_keeps_healthy_rotors_healthy():
    m = run_scenario("mrac", Scenario("t", None))
    assert m["healthy_dev"] <= 0.15


def test_mrac_finds_the_damaged_rotor():
    """Abrupt 50 % loss. The estimate oscillates at ~13 Hz between diagonal partners (see
    results/stage2_compare.png), so judge its AVERAGE over the last 0.5 s of the window."""
    for rotor in range(4):
        r = run_scenario("mrac", Scenario("t", Fault(rotor, 0.5, 6.0)), keep_trace=True)
        tr = r["trace"]
        tail = (tr["t"] >= 8.0) & (tr["t"] < 8.5)
        est = tr["eta_hat"][tail].mean(axis=0)
        assert int(np.argmin(est)) == rotor, (rotor, est)
        assert abs(est[rotor] - 0.5) < 0.1, (rotor, est)


def test_mrac_reduces_the_upset():
    s = Scenario("t", Fault(0, 0.65, 6.0))
    assert run_scenario("mrac", s)["upset_deg"] < 0.85 * run_scenario("baseline", s)["upset_deg"]
