"""Physics checks for the quadplane plant (Stage 3). Run: pytest -q"""
import numpy as np
import pytest

from src.qp_params import QPParams
from src.quadplane import Quadplane


@pytest.fixture
def qp():
    return Quadplane(QPParams())


@pytest.mark.parametrize("Va", [0.0, 5.0, 10.0, 15.0])
def test_trim_is_an_equilibrium(qp, Va):
    """Level flight: no linear acceleration, no pitch acceleration. Roll/yaw may be slightly off
    because of the pusher torque (left to the controller by design)."""
    x, u = qp.trim(Va)
    d = qp.derivatives(x, u)
    np.testing.assert_allclose(d[3:6], 0.0, atol=1e-9)
    assert abs(d[11]) < 1e-9
    assert np.all(np.abs(d[10:13]) < 0.3)
    np.testing.assert_allclose(d[13:20], 0.0, atol=1e-9)


def test_wing_takes_over_the_weight_with_speed(qp):
    """Rotor thrust needed in level flight falls as the wing produces lift."""
    T = [qp.trim(Va)[1][0] for Va in (0.0, 5.0, 10.0, 15.0)]
    assert np.isclose(T[0], 5.0 * 9.81 / 4)
    assert all(a > b for a, b in zip(T, T[1:]))


def test_no_aerodynamics_without_airspeed(qp):
    F, M, Va, *_ = qp.aero(np.zeros(3), np.array([0.3, -0.2, 0.1]), 0.4, -0.4)
    assert Va == 0.0 and not F.any() and not M.any()


def test_surface_authority_grows_with_dynamic_pressure(qp):
    p = qp.p
    a5, a10 = p.surface_moment_per_rad(5.0), p.surface_moment_per_rad(10.0)
    np.testing.assert_allclose(a10, 4.0 * a5)
    # the linearised authority matches the full aerodynamic model for a small deflection
    v = np.array([10.0, 0.0, 0.0])
    _, M0, *_ = qp.aero(v, np.zeros(3), 0.0, 0.0)
    _, Ma, *_ = qp.aero(v, np.zeros(3), 0.01, 0.0)
    np.testing.assert_allclose((Ma - M0) / 0.01, a10[:, 0], rtol=1e-6)


def test_aerodynamic_signs(qp):
    v = np.array([10.0, 0.0, 0.0])
    _, M0, *_ = qp.aero(v, np.zeros(3), 0.0, 0.0)
    _, Ma, *_ = qp.aero(v, np.zeros(3), 0.1, 0.0)
    _, Me, *_ = qp.aero(v, np.zeros(3), 0.0, 0.1)
    assert Ma[0] > M0[0]            # positive aileron -> positive roll moment (right wing down)
    assert Me[1] < M0[1]            # positive elevator (trailing edge down) -> nose down
    F1, M1, *_ = qp.aero(np.array([10.0, 0.0, 10.0 * np.tan(0.05)]), np.zeros(3), 0.0, 0.0)
    F0, _, *_ = qp.aero(v, np.zeros(3), 0.0, 0.0)
    assert M1[1] < M0[1]            # more angle of attack -> nose-down moment (static stability)
    assert F1[2] < F0[2]            # more angle of attack -> more lift (negative body z)
    _, Mp, *_ = qp.aero(v, np.array([1.0, 0.0, 0.0]), 0.0, 0.0)
    assert Mp[0] < M0[0]            # roll damping opposes roll rate


def test_accelerometer_sees_wing_lift_and_faults_rpm_does_not(qp):
    qp.acc_noise_std = 0.0
    x, u = qp.trim(10.0)
    a_trim = qp.sensors(x)["acc"][2]
    np.testing.assert_allclose(a_trim, -9.81, atol=1e-9)    # rotors + wing together carry the weight
    qp.set_fault([0.5, 1, 1, 1])
    s = qp.sensors(x)
    assert s["acc"][2] > a_trim + 0.5                        # less real thrust is measured ...
    np.testing.assert_allclose(s["T_motor"], x[13:17])       # ... but rpm says nothing changed


def test_level_flight_is_held_open_loop_for_a_moment(qp):
    """Trim inputs held for 0.2 s: the aircraft stays close to level flight (sanity check of
    the integration; it is not expected to stay trimmed for long without control)."""
    x, u = qp.trim(10.0)
    for _ in range(200):
        x = qp.step(x, u, 0.001)
    assert np.linalg.norm(x[3:6] - [10.0, 0, 0]) < 0.05
    assert np.degrees(np.linalg.norm(x[7:10]) * 2) < 1.0


def test_static_moments_are_bounded_and_continuous_at_all_angles(qp):
    """Review finding M1 (28 Oct): the [BM] linear C_m_alpha*alpha jumped at alpha = +-180 deg and
    the sideslip terms grew without bound. Sweep the flow direction all around the aircraft."""
    Va = 8.0
    prev = None
    worst_jump = 0.0
    peak = 0.0
    for a in np.linspace(-np.pi, np.pi, 3601):
        v = Va * np.array([np.cos(a), 0.0, np.sin(a)])
        _, M, *_ = qp.aero(v, np.zeros(3), 0.0, 0.0)
        if prev is not None:
            worst_jump = max(worst_jump, abs(M[1] - prev))
        prev = M[1]
        peak = max(peak, abs(M[1]))
    qbar_Sc = 0.5 * qp.p.rho * Va ** 2 * qp.p.S * qp.p.c
    assert peak < 1.5 * qbar_Sc                     # |C_m| stays O(1)
    # end points (-180 and +180 deg) are the same flow: no jump across the wrap
    _, Ma, *_ = qp.aero(Va * np.array([-1.0, 0.0, 1e-9]), np.zeros(3), 0.0, 0.0)
    _, Mb, *_ = qp.aero(Va * np.array([-1.0, 0.0, -1e-9]), np.zeros(3), 0.0, 0.0)
    assert abs(Ma[1] - Mb[1]) < 1e-6
    # sideways flow: sideslip moments saturate (beta = 90 deg gives the same as ~30 deg)
    _, M90, *_ = qp.aero(np.array([0.0, Va, 0.0]), np.zeros(3), 0.0, 0.0)
    b = np.radians(30.0)
    _, M30, *_ = qp.aero(Va * np.array([np.cos(b), np.sin(b), 0.0]), np.zeros(3), 0.0, 0.0)
    assert abs(M90[2]) < 1.4 * abs(M30[2])          # tanh(1) = 0.76 of saturation at 30 deg


def test_sideways_flow_is_continuous(qp):
    """Second review M1 (4 Nov): near beta = 90 deg a full-size lift pointed by alpha = atan2(w, u) flipped
    sign with the tiny u, w (F_z -61 N <-> +61 N at 10 m/s). Longitudinal terms now use 0.5 rho (u^2 + w^2)."""
    Va = 10.0
    F = [qp.aero(np.array([eu, Va, ew]), np.zeros(3), 0.0, 0.0)[0] for eu, ew in
         [(0.01, 0.01), (0.01, -0.01), (-0.01, 0.01), (-0.01, -0.01), (0.0, 0.0)]]
    for f in F:
        assert np.allclose(f, F[-1], atol=0.05)
    # and nothing changes in straight flight (beta = 0): trim and G(V) rely on it
    f0, m0, *_ = qp.aero(np.array([Va, 0.0, 0.0]), np.zeros(3), 0.0, 0.0)
    qS = 0.5 * qp.p.rho * Va ** 2 * qp.p.S
    assert np.isclose(-f0[2], qS * qp.p.C_L_0, rtol=1e-3)


def test_post_stall_moment_matches_post_stall_force(qp):
    """Second review M2 (4 Nov): beyond stall the pitch moment is the force model's own normal force
    C_N = CL cos(a) + CD sin(a) = -F_z / (qbar S) acting at the linear static margin, so the centre of
    pressure stays at that margin (before: up to 2.8 c aft at 170 deg). Parasitic drag (kept beyond
    stall) adds C_D_p sin(a) to the normal force but not to the moment, hence the tolerance."""
    Va = 10.0
    qS = 0.5 * qp.p.rho * Va ** 2 * qp.p.S
    margin = -qp.p.C_m_alpha / qp.p.C_L_alpha
    for a in np.radians([60.0, 90.0, 120.0, 150.0, 170.0, -90.0]):
        F, M, *_ = qp.aero(Va * np.array([np.cos(a), 0.0, np.sin(a)]), np.zeros(3), 0.0, 0.0)
        CN = -F[2] / qS
        Cm = M[1] / (qS * qp.p.c)
        assert abs(Cm + margin * CN) <= margin * qp.p.C_D_p + 1e-6, (np.degrees(a), Cm, CN)
