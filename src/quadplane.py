"""'True' quadplane plant (Stage 3). Same role as quadrotor.py: faults live only here.

State x (20):
    p_NED(3) v_NED(3) q(4, body->NED) omega_body(3) T_lift(4) T_push(1) delta_a delta_e
Input u (7):
    T_lift commands(4) [N], T_push command [N], delta_a, delta_e commands [rad]

Forces and moments in the body frame (FRD):
    lift rotors   F = (0, 0, -sum(eta_i T_i)),        M = B_rotor @ (eta * T)
    pusher        F = (T_push, 0, 0),                 M = (-k_m_push T_push, 0, 0)
    wing          lift, drag, side force and moments from the [BM] coefficient model (aero())
    gravity       R^T (0, 0, m g)
    gust          optional extra body moment (as in Stage 1-2) and a wind vector (NED)

Aerodynamic model (Beard & McLain, ch. 4), with alpha = atan2(w_r, u_r), beta = asin(v_r / Va)
of the air-relative body velocity (u_r, v_r, w_r):
    C_L(alpha) = (1 - s) (C_L_0 + C_L_alpha alpha) + s * 2 sign(alpha) sin^2(alpha) cos(alpha)
    s(alpha)   = blending function that switches to a flat plate beyond alpha0 (stall)
    C_D(alpha) = (1 - s) (C_D_p + (C_L_0 + C_L_alpha alpha)^2 / (pi e AR)) + s * 2 |sin^3(alpha)|
                 (the flat-plate part replaces [BM]'s polar beyond stall, see qp_params.py)
    lift and drag act in the stability frame and are rotated by alpha into the body frame.
    C_m,static = (1 - s)(C_m_0 + C_m_alpha alpha) + s * 2 sin(alpha)|sin(alpha)| C_m_alpha / C_L_alpha
                 (the force model's flat-plate normal force at the linear static margin; continuous)
    lift, drag and pitch moment use the in-plane dynamic pressure 0.5 rho (u^2 + w^2) = qbar cos^2(beta)
    (4 Nov 2026, second review M1/M2); side force, roll and yaw use qbar
    sideslip enters C_Y, C_l, C_n as beta_s tanh(beta / beta_s), beta_s = 30 deg
    (both added 28 Oct 2026 after the independent review, finding M1)
Every aerodynamic force and moment carries the factor 0.5 rho Va^2: no airspeed, no aerodynamics.
"""
import math
import numpy as np

from .rotations import cross3


class Quadplane:
    def __init__(self, params, gyro_noise_std=0.002, acc_noise_std=0.05, airspeed_noise_std=0.2,
                 seed=None):
        self.p = params
        self.B = params.B_rotor_moments()
        self.eta = np.ones(4)
        ss = np.random.SeedSequence(0 if seed is None else seed).spawn(3)
        self.rng_gyro, self.rng_acc, self.rng_air = (np.random.default_rng(s) for s in ss)
        self.gyro_noise_std = gyro_noise_std
        self.acc_noise_std = acc_noise_std
        self.airspeed_noise_std = airspeed_noise_std
        self.disturbance_moment = np.zeros(3)
        self.wind_ned = np.zeros(3)
        self._J = params.J
        self._J_inv = np.linalg.inv(params.J)
        self._umax = np.array([params.T_max] * 4 + [params.T_push_max, params.delta_max, params.delta_max])
        self._umin = np.array([0.0] * 5 + [-params.delta_max, -params.delta_max])

    def set_fault(self, eta):
        self.eta = np.asarray(eta, dtype=float).copy()

    # ------------------------------------------------------------------ aerodynamics
    def aero(self, v_rel_body, w, delta_a, delta_e):
        """Aerodynamic force and moment in the body frame. Returns (F, M, Va, alpha, beta)."""
        p = self.p
        u, v, wz = v_rel_body
        Va = math.sqrt(u * u + v * v + wz * wz)
        if Va < 1e-3:
            return np.zeros(3), np.zeros(3), Va, 0.0, 0.0
        alpha = math.atan2(wz, u)
        beta = math.asin(max(-1.0, min(1.0, v / Va)))
        qbar = 0.5 * p.rho * Va * Va
        # longitudinal terms (lift, drag, pitch moment) see only the flow in the body x-z plane:
        # 0.5 rho (u^2 + w^2) = qbar cos^2(beta). Added 4 Nov 2026 (second review, M1): with the full qbar,
        # in sideways flow (|beta| -> 90 deg) a full-size lift/drag pointed in a direction set by the tiny
        # u, w (alpha = atan2(w, u)) and flipped sign with them (F_z -61 N <-> +61 N at 10 m/s).
        qlon = 0.5 * p.rho * (u * u + wz * wz)
        c, b = p.c, p.b
        P, Q, R = w
        # stall blending
        e1 = math.exp(min(700.0, -p.M_blend * (alpha - p.alpha0)))
        e2 = math.exp(min(700.0, p.M_blend * (alpha + p.alpha0)))
        s = (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))
        sa, ca = math.sin(alpha), math.cos(alpha)
        CL_lin = p.C_L_0 + p.C_L_alpha * alpha
        CL = (1.0 - s) * CL_lin + s * 2.0 * math.copysign(1.0, alpha) * sa * sa * ca
        # parasitic drag C_D_p kept beyond stall (second review m3: it vanished there)
        CD = (1.0 - s) * (p.C_D_p + CL_lin ** 2 / (math.pi * p.e_oswald * p.AR)) + s * (p.C_D_p + 2.0 * abs(sa) ** 3)
        # rate and control terms (rate terms use c/(2Va), b/(2Va))
        CL += p.C_L_q * c / (2 * Va) * Q + p.C_L_delta_e * delta_e
        CD += p.C_D_q * c / (2 * Va) * Q + p.C_D_delta_e * abs(delta_e)
        # post-stall static moments (28 Oct 2026, independent review M1): the [BM] linear terms
        # C_m_alpha*alpha and C_*_beta*beta grow without bound and C_m jumps at alpha = +-180 deg.
        #  * pitch: same stall blend as lift; beyond stall the moment is the flat-plate normal force of
        #    the force model itself, C_N = CL cos(a) + CD sin(a) = 2 sin(a)|sin(a)|, acting at the linear
        #    static margin -C_m_alpha/C_L_alpha (continuous and periodic in alpha, zero at 180 deg).
        #    Corrected 4 Nov 2026 (second review M2: it used 2 sin(a), twice the force model's value at 30 deg)
        #  * sideslip: beta_s tanh(beta/beta_s), slope 1 at 0, saturates at beta_s (vertical-tail stall)
        Cm_static = (1.0 - s) * (p.C_m_0 + p.C_m_alpha * alpha) + s * 2.0 * sa * abs(sa) * (p.C_m_alpha / p.C_L_alpha)
        bs = p.beta_sat * math.tanh(beta / p.beta_sat)
        Fx = qlon * p.S * (-CD * ca + CL * sa)
        Fz = qlon * p.S * (-CD * sa - CL * ca)
        Fy = qbar * p.S * (p.C_Y_0 + p.C_Y_beta * bs + p.C_Y_p * b / (2 * Va) * P
                           + p.C_Y_r * b / (2 * Va) * R + p.C_Y_delta_a * delta_a)
        L = qbar * p.S * b * (p.C_ell_0 + p.C_ell_beta * bs + p.C_ell_p * b / (2 * Va) * P
                              + p.C_ell_r * b / (2 * Va) * R + p.C_ell_delta_a * delta_a)
        M = qlon * p.S * c * (Cm_static + p.C_m_q * c / (2 * Va) * Q + p.C_m_delta_e * delta_e)
        N = qbar * p.S * b * (p.C_n_0 + p.C_n_beta * bs + p.C_n_p * b / (2 * Va) * P
                              + p.C_n_r * b / (2 * Va) * R + p.C_n_delta_a * delta_a)
        return np.array([Fx, Fy, Fz]), np.array([L, M, N]), Va, alpha, beta

    @staticmethod
    def rot(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    def forces_moments(self, x):
        """Non-gravity body force and total body moment at state x."""
        p = self.p
        q, w = x[6:10], x[10:13]
        T_real = self.eta * x[13:17]
        Tp, da, de = x[17], x[18], x[19]
        R = self.rot(q)
        v_rel = R.T @ (x[3:6] - self.wind_ned)
        Fa, Ma, Va, alpha, beta = self.aero(v_rel, w, da, de)
        F = Fa + np.array([Tp, 0.0, -T_real.sum()])
        M = Ma + self.B @ T_real + np.array([-p.k_m_push * Tp, 0.0, 0.0]) + self.disturbance_moment
        return F, M, R, Va

    # ------------------------------------------------------------------ dynamics
    def derivatives(self, x, u):
        p = self.p
        F, M, R, _ = self.forces_moments(x)
        w = x[10:13]
        qw, qx, qy, qz = x[6:10]
        wx, wy, wz = w
        dx = np.empty(20)
        dx[0:3] = x[3:6]
        dx[3:6] = R @ F / p.mass + np.array([0.0, 0.0, p.g])
        dx[6] = 0.5 * (-qx * wx - qy * wy - qz * wz)
        dx[7] = 0.5 * (qw * wx + qy * wz - qz * wy)
        dx[8] = 0.5 * (qw * wy - qx * wz + qz * wx)
        dx[9] = 0.5 * (qw * wz + qx * wy - qy * wx)
        dx[10:13] = self._J_inv @ (M - cross3(w, self._J @ w))
        dx[13:18] = (u[0:5] - x[13:18]) / p.tau_motor
        dx[18:20] = (u[5:7] - x[18:20]) / p.tau_servo
        return dx

    def step(self, x, u, dt):
        u = np.clip(u, self._umin, self._umax)
        k1 = self.derivatives(x, u)
        k2 = self.derivatives(x + 0.5 * dt * k1, u)
        k3 = self.derivatives(x + 0.5 * dt * k2, u)
        k4 = self.derivatives(x + dt * k3, u)
        xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        xn[6:10] /= np.linalg.norm(xn[6:10])
        return xn

    def sensors(self, x):
        """Gyro and accelerometer (specific force, now including aerodynamics), rpm-based thrust of
        lift rotors and pusher, pitot airspeed. Servo positions are NOT measured (the controller
        models them), as on most small aircraft."""
        F, _, R, Va = self.forces_moments(x)
        return {
            "pos": x[0:3].copy(), "vel": x[3:6].copy(), "q": x[6:10].copy(),
            "gyro": x[10:13] + self.gyro_noise_std * self.rng_gyro.standard_normal(3),
            "acc": F / self.p.mass + self.acc_noise_std * self.rng_acc.standard_normal(3),
            "T_motor": x[13:17].copy(), "T_push": float(x[17]),
            "airspeed": max(0.0, Va + self.airspeed_noise_std * self.rng_air.standard_normal()),
        }

    # ------------------------------------------------------------------ trim
    def trim(self, Va, theta=0.0, z=-20.0):
        """Steady level flight north at airspeed Va (no wind) with pitch theta.

        With equal lift-rotor thrusts, the elevator alone balances the pitching moment:
            C_m_0 + C_m_alpha alpha + C_m_delta_e delta_e = 0,  alpha = theta
        then the pusher balances x and the rotors balance z:
            T_push = m g sin(theta) - F_aero,x,   4 T = F_aero,z + m g cos(theta)
        Roll/yaw torques of the pusher (<= 0.2 Nm) are left to the controller.
        Returns (state, input)."""
        p = self.p
        de = 0.0 if Va <= 0 else -(p.C_m_0 + p.C_m_alpha * theta) / p.C_m_delta_e
        de = float(np.clip(de, -p.delta_max, p.delta_max))
        q = np.array([np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0])
        v_body = np.array([Va * np.cos(theta), 0.0, Va * np.sin(theta)])
        Fa, _, _, _, _ = self.aero(v_body, np.zeros(3), 0.0, de)
        Tp = p.mass * p.g * np.sin(theta) - Fa[0]
        T = (Fa[2] + p.mass * p.g * np.cos(theta)) / 4.0
        x = np.zeros(20)
        x[0:3] = (0.0, 0.0, z)
        x[3:6] = (Va, 0.0, 0.0)
        x[6:10] = q
        x[13:17] = T
        x[17] = Tp
        x[19] = de
        u = np.array([T, T, T, T, Tp, 0.0, de])
        return x, u
