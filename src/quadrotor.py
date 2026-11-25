"""'True' quadrotor plant. Faults are injected here; the controller never sees them directly.

State x (17): p_NED(3), v_NED(3), q(4, scalar first, body->NED), omega_body(3), T_motor(4)
T_motor is the thrust a healthy rotor would produce at its current speed. With RPM
feedback this is what the flight computer measures. A damaged propeller (fault) turns
it into eta_i * T_motor_i of real thrust, which the RPM sensor cannot see.
"""
import numpy as np
from .rotations import quat_to_rot  # noqa: F401  (kept for readers: the math in derivatives() expands it)


class Quadrotor:
    def __init__(self, params, rng=None, gyro_noise_std=0.002, acc_noise_std=0.05, seed=None):
        """seed=None keeps the Stage 1 noise streams (seeds 0 and 1); an integer gives a new,
        reproducible pair of streams (used by the Monte Carlo scenarios)."""
        self.p = params
        self.B = params.B_moments()
        self.eta = np.ones(4)                 # rotor effectiveness, 1 = healthy
        if seed is not None:
            ss_gyro, ss_acc = np.random.SeedSequence(seed).spawn(2)
            rng, rng_acc = np.random.default_rng(ss_gyro), np.random.default_rng(ss_acc)
        else:
            rng_acc = np.random.default_rng(1)   # separate stream: gyro noise stays as in Stage 1
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.gyro_noise_std = gyro_noise_std  # rad/s
        self.acc_noise_std = acc_noise_std    # m/s^2
        self.rng_acc = rng_acc
        self.disturbance_moment = np.zeros(3)
        # constants precomputed once (used thousands of times per simulated second)
        self._J = params.J.tolist()
        self._J_inv = np.linalg.inv(params.J).tolist()
        self._inv_tau = 1.0 / params.tau_motor

    def set_fault(self, eta):
        self.eta = np.asarray(eta, dtype=float).copy()

    def initial_state(self, pos=(0.0, 0.0, 0.0)):
        x = np.zeros(17)
        x[0:3] = pos
        x[6] = 1.0
        x[13:17] = self.p.mass * self.p.g / 4.0
        return x

    def derivatives(self, x, u_cmd):
        """Time derivative of the state. u_cmd must already be clipped to [0, T_max].

        Written with plain Python floats instead of small NumPy arrays: for 3- and 4-element
        vectors the NumPy call overhead dominates, and this function runs 4 times per 1 ms step.
        The equations are the same as:
            a     = R(q) @ (0, 0, -sum(T_real)) / m + (0, 0, g)
            w_dot = J^-1 (B @ T_real + d - w x (J w))
            q_dot = 0.5 * q (x) (0, w)
            T_dot = (u_cmd - T_motor) / tau_motor
        """
        p = self.p
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        wx, wy, wz = x[10], x[11], x[12]
        T_motor = x[13:17]
        T_real = self.eta * T_motor

        # thrust acts along -z_body; -z_body in NED is minus the 3rd column of R(q)
        f = float(T_real.sum()) / p.mass
        ax = -f * 2.0 * (qx * qz + qw * qy)
        ay = -f * 2.0 * (qy * qz - qw * qx)
        az = -f * (1.0 - 2.0 * (qx * qx + qy * qy)) + p.g

        # rotational dynamics: J w_dot = tau - w x (J w)
        J = self._J
        Jw0 = J[0][0] * wx + J[0][1] * wy + J[0][2] * wz
        Jw1 = J[1][0] * wx + J[1][1] * wy + J[1][2] * wz
        Jw2 = J[2][0] * wx + J[2][1] * wy + J[2][2] * wz
        tau = self.B @ T_real + self.disturbance_moment
        r0 = tau[0] - (wy * Jw2 - wz * Jw1)
        r1 = tau[1] - (wz * Jw0 - wx * Jw2)
        r2 = tau[2] - (wx * Jw1 - wy * Jw0)
        Ji = self._J_inv

        dx = np.empty(17)
        dx[0:3] = x[3:6]
        dx[3] = ax
        dx[4] = ay
        dx[5] = az
        # q_dot = 0.5 * q (x) (0, w)
        dx[6] = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dx[7] = 0.5 * (qw * wx + qy * wz - qz * wy)
        dx[8] = 0.5 * (qw * wy - qx * wz + qz * wx)
        dx[9] = 0.5 * (qw * wz + qx * wy - qy * wx)
        dx[10] = Ji[0][0] * r0 + Ji[0][1] * r1 + Ji[0][2] * r2
        dx[11] = Ji[1][0] * r0 + Ji[1][1] * r1 + Ji[1][2] * r2
        dx[12] = Ji[2][0] * r0 + Ji[2][1] * r1 + Ji[2][2] * r2
        dx[13:17] = (u_cmd - T_motor) * self._inv_tau
        return dx

    def step(self, x, u_cmd, dt):
        """RK4 with zero-order hold on u_cmd."""
        u_cmd = np.clip(u_cmd, 0.0, self.p.T_max)
        k1 = self.derivatives(x, u_cmd)
        k2 = self.derivatives(x + 0.5 * dt * k1, u_cmd)
        k3 = self.derivatives(x + 0.5 * dt * k2, u_cmd)
        k4 = self.derivatives(x + dt * k3, u_cmd)
        xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] /= np.linalg.norm(xn[6:10])
        return xn

    def sensors(self, x):
        """What the flight computer measures: pose, velocity, noisy gyro, noisy accelerometer,
        rotor thrust from RPM.

        The accelerometer measures specific force in the body frame (everything except gravity).
        Thrust is the only non-gravity force in this model, so it reads (0, 0, -T_real/m):
        it DOES see a damaged propeller, unlike the RPM-based thrust.
        """
        gyro = x[10:13] + self.gyro_noise_std * self.rng.standard_normal(3)
        T_real = self.eta * x[13:17]
        acc = np.array([0.0, 0.0, -np.sum(T_real) / self.p.mass]) + self.acc_noise_std * self.rng_acc.standard_normal(3)
        return {
            "pos": x[0:3].copy(),
            "vel": x[3:6].copy(),
            "q": x[6:10].copy(),
            "gyro": gyro,
            "acc": acc,
            "T_motor": x[13:17].copy(),
        }
