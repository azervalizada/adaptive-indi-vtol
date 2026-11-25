"""Stage 4 (transition) scenarios and metrics (added 30 Oct 2026; the independent review noted that
tr_scenarios, run_tr and the outrun logic had no test). No simulation here: synthetic traces."""
import numpy as np
import pytest

from src.tr_scenarios import (transition_reference, TRScenario, tr_grid, tr_nominal, tr_metrics,
                              T_START, V_CRUISE, ACCELS)


@pytest.mark.parametrize("accel", ACCELS)
def test_reference_is_continuous_and_consistent(accel):
    ref = transition_reference(accel)
    dt = 1e-3
    t = np.arange(0.0, T_START + V_CRUISE / accel + 3.0, dt)
    P = np.array([ref(x)["pos"] for x in t])
    V = np.array([ref(x)["vel"] for x in t])
    A = np.array([ref(x)["acc"] for x in t])
    assert np.max(np.abs(np.diff(P, axis=0))) < V_CRUISE * dt * 1.01      # no position jump
    assert np.max(np.abs(np.diff(V[:, 0]))) <= accel * dt * 1.01           # no velocity jump
    # velocity is the derivative of position, acceleration of velocity (away from the two corners)
    assert np.allclose(np.gradient(P[:, 0], dt)[5:-5], V[5:-5, 0], atol=accel * dt * 2)
    corners = (np.abs(t - T_START) < 0.01) | (np.abs(t - T_START - V_CRUISE / accel) < 0.01)
    assert np.allclose(np.gradient(V[:, 0], dt)[~corners][5:-5], A[~corners][5:-5, 0], atol=1e-6)
    assert V[0, 0] == 0.0 and abs(V[-1, 0] - V_CRUISE) < 1e-12
    assert np.all(P[:, 2] == P[0, 2]) and np.all(P[:, 1] == 0.0)            # altitude and track held


def test_fault_time_and_window():
    sc = TRScenario("t", 2.0, 4.0, 1, 0.0)
    assert sc.fault.t0 == pytest.approx(T_START + 4.0 / 2.0)
    assert sc.fault.rotor == 1 and sc.fault.eta == 0.0
    # the window covers the rest of the acceleration + 2 s, and at least 4 s after the fault
    assert sc.t_end >= T_START + V_CRUISE / 2.0 + 2.0
    late = TRScenario("t", 1.0, 12.0, 0, 0.0)
    assert late.t_end >= late.fault.t0 + 4.0
    assert all(s.fault is None for s in tr_nominal())


def test_grid_is_complete_and_unique():
    g = tr_grid()
    assert len(g) == 84 and len({s.label() for s in g}) == 84


def _trace(sc, final_speed, tilt_end=0.2, tilt_max=5.0):
    t = np.arange(0.0, sc.t_end, 0.002)
    n = len(t)
    tilt = np.where(t < sc.t0, 0.01, tilt_end)
    tilt[np.argmin(np.abs(t - sc.t0 - 0.5))] = tilt_max
    pos = np.zeros((n, 3)); pos[:, 2] = -20.0
    return {"t": t, "tilt_err_deg": tilt, "heading_err_deg": np.full(n, 1.0), "pos": pos, "ref": pos.copy(),
            "u_cmd": np.tile([10.0, 10, 10, 10, 5, 0, 0], (n, 1)), "T_max": 24.5, "delta_max": 0.436,
            "diverged_at": None, "Va": np.full(n, final_speed),
            "eta_hat": np.ones((n, 4)), "eta": np.ones((n, 4))}


def test_metrics_completed_needs_recovery_and_cruise_speed():
    sc = TRScenario("t", 3.0, 0.0, 0, 0.3)            # fault at T_START: healthy window must not be empty
    m = tr_metrics(_trace(sc, V_CRUISE - 0.2), sc)
    assert m["recovered"] and m["completed"] and m["final_speed_err"] == pytest.approx(0.2)
    assert m["healthy_dev"] is not None
    m = tr_metrics(_trace(sc, V_CRUISE - 1.5), sc)       # recovered but too slow
    assert m["recovered"] and not m["completed"]
    m = tr_metrics(_trace(sc, V_CRUISE, tilt_end=2.0), sc)   # at cruise, but tilt not recovered
    assert not m["recovered"] and not m["completed"]


def test_metrics_lost_run():
    sc = TRScenario("t", 2.0, 6.0, 1, 0.0)
    m = tr_metrics(_trace(sc, V_CRUISE, tilt_max=120.0), sc)
    assert m["lost"] and not m["completed"] and m["final_speed_err"] is None
