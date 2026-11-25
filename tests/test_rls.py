"""Checks for the RLS-INDI estimator. Run: pytest -q"""
import numpy as np

from src.scenarios import Scenario, Fault
from src.controllers import run_scenario


def test_rls_keeps_healthy_rotors_healthy():
    """No fault, no gust: the estimate stays within 0.15 of 1 (tuning rule (c))."""
    m = run_scenario("rls", Scenario("t", None))
    assert m["healthy_dev"] <= 0.15


def test_rls_finds_the_damaged_rotor():
    """Abrupt 50 % loss: at the end of the window the damaged rotor has the lowest estimate,
    within 0.2 of the truth. (Healthy diagonal partners can be off by ~0.1-0.2: the X layout
    only separates them through the noisy yaw and thrust rows; see src/rls.py.)"""
    for rotor in range(4):
        r = run_scenario("rls", Scenario("t", Fault(rotor, 0.5, 6.0)), keep_trace=True)
        est = r["trace"]["eta_hat"][-1]
        assert int(np.argmin(est)) == rotor, (rotor, est)
        assert abs(est[rotor] - 0.5) < 0.2, (rotor, est)


def test_rls_reduces_the_upset():
    s = Scenario("t", Fault(0, 0.65, 6.0))
    assert run_scenario("rls", s)["upset_deg"] < 0.7 * run_scenario("baseline", s)["upset_deg"]
