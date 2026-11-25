"""Fast checks of the building blocks (no full flight). Run: pytest -q"""
import numpy as np

from src.params import QuadParams
from src.rotations import cross3, norm3, quat_to_rot, rot_to_quat, quat_mult, quat_conj
from src.quadrotor import Quadrotor
from src.scenarios import Fault


def test_rotor_moment_signs():
    """More thrust on rotor 1 (front-right) must roll the vehicle left and pitch the nose up;
    rotors 1 and 3 must yaw one way, 2 and 4 the other."""
    B = QuadParams().B_moments()
    assert B[0, 0] < 0 and B[1, 0] > 0           # roll left, nose up
    assert B[0, 2] > 0 and B[1, 2] < 0           # rotor 3 (rear-left): the opposite
    assert np.sign(B[2, 0]) == np.sign(B[2, 2]) != np.sign(B[2, 1]) == np.sign(B[2, 3])


def test_equal_thrust_gives_no_moment():
    """Four equal thrusts: no rotation at all (symmetric layout, yaw torques cancel)."""
    B = QuadParams().B_moments()
    np.testing.assert_allclose(B @ np.full(4, 2.5), 0.0, atol=1e-12)


def test_fast_helpers_match_numpy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(cross3(a, b), np.cross(a, b), atol=1e-12)
        assert abs(norm3(a) - np.linalg.norm(a)) < 1e-12


def test_quaternion_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        np.testing.assert_allclose(rot_to_quat(quat_to_rot(q)), q, atol=1e-9)
        np.testing.assert_allclose(quat_mult(q, quat_conj(q)), [1, 0, 0, 0], atol=1e-12)


def test_hover_is_an_equilibrium():
    """Healthy vehicle, level, motors at mg/4: nothing should change."""
    p = QuadParams()
    plant = Quadrotor(p)
    x = plant.initial_state(pos=(0.0, 0.0, -2.0))
    np.testing.assert_allclose(plant.derivatives(x, x[13:17]), 0.0, atol=1e-12)


def test_accelerometer_sees_the_fault_rpm_does_not():
    p = QuadParams()
    plant = Quadrotor(p, acc_noise_std=0.0)
    x = plant.initial_state()
    plant.set_fault([0.5, 1, 1, 1])
    s = plant.sensors(x)
    hover = p.mass * p.g / 4
    np.testing.assert_allclose(s["T_motor"], hover)                     # RPM: looks healthy
    assert abs(-p.mass * s["acc"][2] - 3.5 * hover) < 1e-12            # accel: 3.5 of 4 rotors


def test_fault_profiles():
    abrupt = Fault(rotor=2, eta=0.5, t0=6.0)
    np.testing.assert_allclose(abrupt.eta_at(5.999), [1, 1, 1, 1])
    np.testing.assert_allclose(abrupt.eta_at(6.0), [1, 1, 0.5, 1])
    ramp = Fault(rotor=0, eta=0.6, t0=6.0, ramp=1.0)
    np.testing.assert_allclose(ramp.eta_at(6.5), [0.8, 1, 1, 1])
    np.testing.assert_allclose(ramp.eta_at(7.5), [0.6, 1, 1, 1])
