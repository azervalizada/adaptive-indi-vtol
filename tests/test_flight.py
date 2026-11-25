"""Closed-loop checks: short simulated flights (~1 s of CPU each). Run: pytest -q

These guard the results the paper will rely on. If a code change breaks one of them,
something physical changed, not just a number.
"""
import numpy as np

from src.params import QuadParams, INDIGains
from src.quadrotor import Quadrotor
from src.indi import INDIController
from src.oracle import OracleINDI
from src.sim import simulate
from src.scenarios import Scenario, Fault
from src.controllers import CONTROLLERS, run_scenario

DT = 0.002


def test_every_controller_flies_without_fault():
    """No fault: every registered controller holds 2 m and stays close to the baseline."""
    s = Scenario("t", None)
    base = run_scenario("baseline", s)
    for name in CONTROLLERS:
        m = run_scenario(name, s)
        assert not m["crashed"], name
        assert m["alt_dev_m"] < 0.01, name
        assert m["att_rms_deg"] < 1.5 * base["att_rms_deg"] + 0.05, name   # adaptation must not hurt


def test_runs_are_reproducible():
    s = Scenario("t", Fault(1, 0.6, 5.5, 1.0), seed=7, mass_scale=1.03, J_scale=(0.95, 1.05, 1.0))
    a, b = run_scenario("baseline", s), run_scenario("baseline", s)
    assert a == b


def test_oracle_equals_baseline_when_healthy():
    """With eta = 1 the oracle's effectiveness matrix IS the nominal one."""
    s = Scenario("t", None)
    assert run_scenario("baseline", s) == {**run_scenario("oracle", s), "controller": "baseline"}


def test_oracle_beats_baseline_after_abrupt_fault():
    for rotor in range(4):
        s = Scenario("t", Fault(rotor, 0.5, 6.0))
        b, o = run_scenario("baseline", s), run_scenario("oracle", s)
        assert o["upset_deg"] < 0.7 * b["upset_deg"], rotor


def test_stage1_altitude_sag_matches_formula():
    """Stage 1 result: rpm-thrust INDI sags by (mg/4)(1/eta - 1) / (m kp_z). Within 5 %."""
    p, k, eta = QuadParams(), INDIGains(), 0.6
    plant = Quadrotor(p)
    ctrl = INDIController(QuadParams(), k, DT, thrust_feedback="model")
    r = simulate(plant, ctrl, t_end=8.95, fault_time=6.0, fault_eta=[eta, 1, 1, 1])
    m = (r["t"] >= 8.7)
    sag = np.mean(r["pos"][m, 2] - r["ref"][m, 2])
    expected = (p.mass * p.g / 4) * (1 / eta - 1) / (p.mass * k.kp_pos[2])
    assert abs(sag - expected) / expected < 0.05


def test_accel_thrust_removes_the_sag():
    """Stage 1b result: with accelerometer thrust feedback the sag is below 1 cm."""
    m = run_scenario("baseline", Scenario("t", Fault(0, 0.5, 6.0)))
    assert m["alt_dev_m"] < 0.05 and not m["crashed"]


def test_infeasible_fault_crashes():
    """Below eta ~0.41 rotor 1 cannot carry mg/4: hover is physically impossible.
    If this ever stops crashing, the physics changed."""
    plant = Quadrotor(QuadParams())
    ctrl = OracleINDI(QuadParams(), INDIGains(), DT, plant)
    r = simulate(plant, ctrl, fault_time=6.0, fault_eta=[0.3, 1, 1, 1])
    assert r["diverged_at"] is not None
