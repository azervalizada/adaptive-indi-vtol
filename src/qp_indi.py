r"""Baseline INDI for the quadplane in hover and slow forward flight (Stage 3).

Same structure as indi.py (position -> attitude -> rate -> INDI), with three changes:

1. Over-actuation. Moments and the body-z force come from 6 actuators:
       u = [T_1..T_4 (lift rotors, N), delta_a, delta_e (rad)]
   G(Va) is 4 x 6 and depends on airspeed: the surface columns grow with 0.5 rho Va^2.
       rows 1-3: angular acceleration  J^-1 [B_rotor | B_surface(Va)]
       row  4  : body -z force [N]     [1 1 1 1 | 0, qbar S C_L_delta_e]   (elevator changes lift)
   G du = v has infinitely many solutions, so the INDI step is a weighted least-squares
   allocation with actuator limits (wls.py) instead of a matrix inverse.
2. Forward thrust from the pusher, incrementally like the z channel:
       T_push = T_push_f + m (f_x,des - a_x,meas)       (a_x,meas: filtered accelerometer, body x)
3. Attitude reference for a quadplane: pitch is HELD at theta_ref (the pusher, not the tilt,
   drives the aircraft forward, as on real quadplanes); roll comes from the lateral force demand;
   yaw follows the reference heading.

What the controller knows: nominal parameters, measured airspeed, gyro, accelerometer, rotor
thrust from rpm, pusher thrust. Servo positions are not measured: a first-order servo model in the
controller provides them (same synchronisation filter as everything else).

Allocation preferences (secondary objective, design choice of 27 Oct): actuators are normalised
by their range and weighted equally; preferred state = equal lift rotors, surfaces at zero, pulled
towards with a time constant of 0.2 s. With these weights the allocation itself decides how load
moves from rotors to surfaces as airspeed grows (cheapest normalised actuator first).
"""
from dataclasses import dataclass, field
import numpy as np

from .indi import SecondOrderFilter
from .rotations import quat_mult, quat_conj
from .wls import wls_alloc


@dataclass
class QPGains:
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 2.5]))
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 3.0]))
    a_max_xy: float = 3.0
    roll_max: float = np.radians(30.0)
    theta_ref: float = 0.0
    k_att: np.ndarray = field(default_factory=lambda: np.array([6.0, 6.0, 2.0]))
    k_rate: np.ndarray = field(default_factory=lambda: np.array([20.0, 20.0, 8.0]))
    filt_wn: float = 50.0
    filt_zeta: float = 0.55
    # allocation
    W_v: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.3, 0.3]))
    gamma: float = 1000.0
    tau_pref: float = 0.2


def euler_to_quat(phi, theta, psi):
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class QPINDIController:
    def __init__(self, params_nominal, gains, dt):
        p = self.p = params_nominal
        self.k = gains
        self.dt = dt
        self.J = p.J
        self._J_inv = np.linalg.inv(p.J)
        self._Ga_rot = self._J_inv @ p.B_rotor_moments()        # 3 x 4
        self.u_min = np.array([0.0] * 4 + [-p.delta_max] * 2)
        self.u_max = np.array([p.T_max] * 4 + [p.delta_max] * 2)
        self.W_u = 1.0 / (self.u_max - self.u_min)               # normalise by range
        self._servo = 1.0 - np.exp(-dt / p.tau_servo)
        self._pref = dt / gains.tau_pref
        self.delta_hat = None                                    # servo model state [da, de]
        self.f_gyro = SecondOrderFilter(3, gains.filt_wn, gains.filt_zeta)
        self.f_act = None                                        # 6 actuators, set on first call
        self.f_acc = None
        self.f_push = None

    # ---- model-dependent part (same hooks as indi.py) ----
    def observe(self, u_f, wdot_f, acc_f, meas):
        pass

    def effectiveness(self, Va):
        """4 x 6 control effectiveness at airspeed Va (nominal model)."""
        p = self.p
        G = np.zeros((4, 6))
        G[0:3, 0:4] = self._Ga_rot
        G[0:3, 4:6] = self._J_inv @ p.surface_moment_per_rad(Va)
        G[3, 0:4] = 1.0
        G[3, 5] = 0.5 * p.rho * Va * Va * p.S * p.C_L_delta_e
        return G

    def augment(self, nu, meas):
        return nu

    # ---- outer loop ----
    def attitude_reference(self, meas, ref):
        k, p = self.k, self.p
        a_des = k.kp_pos * (ref["pos"] - meas["pos"]) + k.kd_pos * (ref["vel"] - meas["vel"])
        if "acc" in ref:                               # feedforward of the reference acceleration (Stage 4
            a_des = a_des + ref["acc"]                 # transition); Stage 3 references have none
        n_xy = float(np.hypot(a_des[0], a_des[1]))
        if n_xy > k.a_max_xy:
            a_des[0:2] *= k.a_max_xy / n_xy
        f = a_des - np.array([0.0, 0.0, p.g])          # desired specific force (NED), per kg
        psi = ref.get("yaw", 0.0)
        c, s = np.cos(psi), np.sin(psi)
        f_h = np.array([c * f[0] + s * f[1], -s * f[0] + c * f[1], f[2]])   # heading frame
        phi = float(np.clip(np.arctan2(f_h[1], -f_h[2]), -k.roll_max, k.roll_max))
        q_des = euler_to_quat(phi, k.theta_ref, psi)
        R = self._rot(q_des)
        f_b = R.T @ f                                  # desired specific force in the desired body frame
        return q_des, f_b

    @staticmethod
    def _rot(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

    # ---- full update ----
    def update(self, meas, ref):
        p, k = self.p, self.k
        if self.f_act is None:                         # start the filters on the first measurement
            self.delta_hat = np.zeros(2) if "delta0" not in ref else np.array(ref["delta0"], float)
            u0 = np.concatenate([meas["T_motor"], self.delta_hat])
            self.f_act = SecondOrderFilter(6, k.filt_wn, k.filt_zeta, init=u0)
            self.f_acc = SecondOrderFilter(3, k.filt_wn, k.filt_zeta, init=meas["acc"])
            self.f_push = SecondOrderFilter(1, k.filt_wn, k.filt_zeta, init=[meas["T_push"]])
            self.f_gyro.y = meas["gyro"].copy()

        q_des, f_b = self.attitude_reference(meas, ref)
        q_e = quat_mult(quat_conj(q_des), meas["q"])
        if q_e[0] < 0:
            q_e = -q_e
        w_cmd = -2.0 * k.k_att * q_e[1:4]

        _, wdot_f = self.f_gyro.update(meas["gyro"], self.dt)
        u_f, _ = self.f_act.update(np.concatenate([meas["T_motor"], self.delta_hat]), self.dt)
        acc_f, _ = self.f_acc.update(meas["acc"], self.dt)
        Tp_f, _ = self.f_push.update(np.array([meas["T_push"]]), self.dt)

        self.observe(u_f, wdot_f, acc_f, meas)
        nu = k.k_rate * (w_cmd - meas["gyro"])
        nu = self.augment(nu, meas)

        Va = meas["airspeed"]
        G = self.effectiveness(Va)
        v = np.concatenate([nu - wdot_f, [p.mass * (-f_b[2]) - (-p.mass * acc_f[2])]])
        u_pref = np.concatenate([np.full(4, u_f[0:4].mean()), np.zeros(2)])
        du = wls_alloc(G, v, self.u_min - u_f, self.u_max - u_f, self._pref * (u_pref - u_f),
                       k.W_v, self.W_u, k.gamma)
        u = np.clip(u_f + du, self.u_min, self.u_max)

        T_push = float(np.clip(Tp_f[0] + p.mass * (f_b[0] - acc_f[0]), 0.0, p.T_push_max))
        self.delta_hat = self.delta_hat + self._servo * (u[4:6] - self.delta_hat)

        self.last = {"q_des": q_des, "w_cmd": w_cmd, "nu": nu, "wdot_f": wdot_f, "Va": Va}
        return np.concatenate([u[0:4], [T_push], u[4:6]])


class QPOracleINDI(QPINDIController):
    """Knows the true rotor effectiveness (reference, as in Stage 1-2)."""
    def __init__(self, params_nominal, gains, dt, plant):
        super().__init__(params_nominal, gains, dt)
        self.plant = plant

    def effectiveness(self, Va):
        G = super().effectiveness(Va)
        G[:, 0:4] *= self.plant.eta[None, :]
        return G
