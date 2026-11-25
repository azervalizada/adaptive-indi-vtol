"""Checks for L1-INDI and for the delay / loss-of-control machinery. Run: pytest -q"""
import numpy as np

from src.params import QuadParams
from src.quadrotor import Quadrotor
from src.sim import simulate
from src.scenarios import Scenario, Fault, fault_metrics
from src.controllers import run_scenario


def test_l1_does_not_disturb_the_healthy_vehicle():
    b = run_scenario("baseline", Scenario("t", None))
    m = run_scenario("l1", Scenario("t", None))
    assert m["att_rms_deg"] <= 1.05 * b["att_rms_deg"]


def test_l1_reduces_the_upset():
    s = Scenario("t", Fault(0, 0.65, 6.0))
    assert run_scenario("l1", s)["upset_deg"] < 0.6 * run_scenario("baseline", s)["upset_deg"]


class _StepController:
    """Commands +1.5 N on rotor 1 from t = 1.0 s; used to measure the transport delay."""
    def __init__(self):
        self.last = {"q_des": np.array([1.0, 0, 0, 0]), "w_cmd": np.zeros(3)}
        self.k = 0

    def update(self, meas, ref):
        self.k += 1
        u = np.full(4, 2.4525)
        if self.k * 0.002 > 1.0 + 1e-9:
            u[0] = 4.0
        return u


def _applied_at(delay_s):
    """Run simulate() with the step controller and return the exact time [s] at which the
    rotor-1 step reached the motor. The motor is first order, T(t) = T0 + dT (1 - exp(-(t - ta)/tau)),
    so one logged sample after the rise gives ta to well below 1 ms despite the 2 ms log."""
    p = QuadParams()
    plant = Quadrotor(p, gyro_noise_std=0.0, acc_noise_std=0.0)
    r = simulate(plant, _StepController(), t_end=1.1, delay_s=delay_s)
    T0, dT = 2.4525, 4.0 - 2.4525
    i = np.argmax(r["T_motor"][:, 0] > T0 + 1e-6)
    frac = (r["T_motor"][i, 0] - T0) / dT
    return r["t"][i] + p.tau_motor * np.log(1.0 - frac)


def test_delay_is_exactly_what_was_asked():
    """In simulate(), the step must reach the motor exactly delay_s later than without delay.
    Guards the off-by-one found on 27 Oct (a buffer of n+1 entries added 1 ms)."""
    base = _applied_at(0.0)
    for delay in (0.004, 0.005, 0.010, 0.020):
        extra = _applied_at(delay) - base
        assert abs(extra - delay) < 2e-4, (delay, extra)
    a = simulate(Quadrotor(QuadParams()), _StepController(), t_end=1.1)
    b = simulate(Quadrotor(QuadParams()), _StepController(), t_end=1.1, delay_s=0.0)
    np.testing.assert_array_equal(a["T_motor"], b["T_motor"])
