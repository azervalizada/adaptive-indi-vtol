"""Checks for WLS allocation and the quadplane INDI (Stage 3). Run: pytest -q"""
import numpy as np
import pytest
from scipy.optimize import lsq_linear

from src.wls import wls_alloc
from src.qp_params import QPParams
from src.quadplane import Quadplane
from src.qp_indi import QPINDIController, QPOracleINDI, QPGains
from src.qp_sim import simulate_qp
from src.scenarios import Fault


def test_wls_matches_a_reference_solver():
    """Same optimum as SciPy's bounded least squares, including cases with limits active."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        G = rng.normal(size=(4, 6))
        v = rng.normal(size=4) * rng.uniform(0.1, 5)
        lo, hi = -rng.uniform(0, 1, 6), rng.uniform(0, 1, 6)
        pref, Wv, Wu = rng.normal(size=6) * 0.1, np.array([1, 1, .3, .3]), rng.uniform(.1, 1, 6)
        du = wls_alloc(G, v, lo, hi, pref, Wv, Wu, 100.0)
        A = np.vstack([10.0 * Wv[:, None] * G, np.diag(Wu)])
        b = np.concatenate([10.0 * Wv * v, Wu * pref])
        ref = lsq_linear(A, b, bounds=(lo, hi), method="bvls", tol=1e-12).x
        f = lambda x: np.sum((A @ x - b) ** 2)
        assert f(du) <= f(ref) * (1 + 1e-9) + 1e-12
        assert np.all(du >= lo - 1e-12) and np.all(du <= hi + 1e-12)


def test_wls_uses_surfaces_only_when_they_are_effective():
    """Hover: surface columns are zero, so the surfaces stay put. At speed they share the load."""
    c = QPINDIController(QPParams(), QPGains(), 0.002)
    v = np.array([5.0, 0.0, 0.0, 0.0])                       # roll acceleration demand
    lo, hi = c.u_min - 12.0 * np.r_[np.ones(4), 0, 0], c.u_max - 12.0 * np.r_[np.ones(4), 0, 0]
    for Va, uses_surface in ((0.0, False), (12.0, True)):
        du = wls_alloc(c.effectiveness(Va), v, lo, hi, np.zeros(6), c.k.W_v, c.W_u, c.k.gamma)
        assert (abs(du[4]) > 1e-3) == uses_surface
        np.testing.assert_allclose(c.effectiveness(Va) @ du, v, atol=1e-2)


def _run(V, fault=None, oracle=False, t_end=7.0):
    pl = Quadplane(QPParams())
    c = QPOracleINDI(QPParams(), QPGains(), 0.002, pl) if oracle else QPINDIController(QPParams(), QPGains(), 0.002)
    return simulate_qp(pl, c, V, t_end=t_end, fault=fault)


@pytest.mark.parametrize("V", [0.0, 10.0])
def test_steady_flight_is_held(V):
    r = _run(V, t_end=4.0)
    assert r["diverged_at"] is None
    assert r["att_err_deg"][r["t"] > 1].max() < 0.2
    assert np.abs(r["pos"][:, 2] + 20).max() < 0.01
    assert np.abs(r["pos"][:, 0:2] - r["ref"][:, 0:2]).max() < 0.05


def test_fault_at_speed_is_absorbed_and_surfaces_help():
    """Rotor 1 at 60 % in 8 m/s flight: recovered, and the surfaces carry part of the moment."""
    r = _run(8.0, Fault(0, 0.6, 5.0))
    w = r["t"] >= 5.0
    assert r["att_err_deg"][w].max() < 1.5
    assert r["att_err_deg"][-1] < 0.2
    assert abs(np.degrees(r["x_act"][-1, 5])) > 1.0          # aileron deflected after the fault


def test_complete_rotor_loss_is_not_recovered_in_hover_but_recovered_at_8_mps():
    """Judged with the Stage 3 metric itself (qp_metrics). In hover the front rotor's loss leaves the
    two diagonal rotors at full thrust (= weight): roll/pitch and altitude are held but yaw is given
    up, the heading spins away -> not recovered. (Before the post-stall moment fix of 28 Oct, review
    M1, this case was 'lost': the unbounded C_m_alpha*alpha at large angles threw the aircraft down.)
    At 8 m/s the surfaces take over and the aircraft fully recovers."""
    from src.qp_controllers import run_qp
    from src.qp_scenarios import QPScenario
    h = run_qp("baseline", QPScenario("t", 0.0, Fault(0, 0.0, 5.0)))
    assert not h["recovered"] and h["heading_final_deg"] > 5.0 and h["tilt_peak_deg"] > 10.0
    m = run_qp("baseline", QPScenario("t", 8.0, Fault(0, 0.0, 5.0)))
    assert m["recovered"] and m["tilt_peak_deg"] < 5.0
