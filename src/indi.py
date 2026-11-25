"""Baseline INDI flight controller (position -> attitude -> INDI angular-acceleration loop).

The controller holds its OWN copy of the vehicle model (nominal parameters). It never
reads the plant's fault state. The only model-dependent part of the inner loop is the
control-effectiveness matrix G = [J^-1 B_moments; 1 1 1 1]; adaptive variants (Stage 2)
will override `effectiveness()` to update it online.
"""
import numpy as np
from .rotations import quat_mult, quat_conj, rot_to_quat, cross3, norm3


class SecondOrderFilter:
    """y'' = wn^2 (u - y) - 2 zeta wn y'. Returns filtered value and its derivative."""

    def __init__(self, n, wn, zeta, init=None):
        self.wn, self.zeta = wn, zeta
        self.y = np.zeros(n) if init is None else np.array(init, dtype=float)
        self.yd = np.zeros(n)

    def update(self, u, dt):
        ydd = self.wn ** 2 * (u - self.y) - 2.0 * self.zeta * self.wn * self.yd
        self.yd = self.yd + dt * ydd          # semi-implicit Euler
        self.y = self.y + dt * self.yd
        return self.y, self.yd


class INDIController:
    """thrust_feedback:
      "accel" (default) - total thrust is incremental too: the current thrust is MEASURED
                          from the accelerometer (T_now = -m * a_z,body), as in practical INDI.
      "model"           - the current thrust is ESTIMATED from the rotor speeds (Stage 1 version);
                          a damaged propeller makes this estimate too high.
    """

    def __init__(self, params_nominal, gains, dt, thrust_feedback="accel"):
        if thrust_feedback not in ("accel", "model"):
            raise ValueError("thrust_feedback must be 'accel' or 'model'")
        self.p = params_nominal
        self.k = gains
        self.dt = dt
        self.thrust_feedback = thrust_feedback
        self.J = params_nominal.J
        self.B_full = params_nominal.B_full()
        self._G_nominal = self.B_full.copy()
        self._G_nominal[0:3, :] = np.linalg.solve(self.J, self._G_nominal[0:3, :])
        hover = params_nominal.mass * params_nominal.g / 4.0
        self.f_gyro = SecondOrderFilter(3, gains.filt_wn, gains.filt_zeta)
        self.f_act = SecondOrderFilter(4, gains.filt_wn, gains.filt_zeta, init=np.full(4, hover))
        self.f_acc = SecondOrderFilter(1, gains.filt_wn, gains.filt_zeta, init=[-params_nominal.g])

    # ---- model-dependent part (overridden by adaptive variants) ----
    def observe(self, u_f, wdot_f, acc_z_f, meas=None):
        """Called every control step with the synchronised filtered signals:
        u_f (rotor thrusts from rpm, N), wdot_f (angular acceleration, rad/s^2),
        acc_z_f (body-z specific force, m/s^2), and the raw sensor dict `meas`.
        Baseline INDI does not learn: no-op."""
        pass

    def effectiveness(self):
        """Maps rotor-thrust increments to [angular-acceleration increments; total-thrust increment]."""
        return self._G_nominal.copy()

    def augment(self, nu, meas):
        """Called every control step with the desired angular acceleration nu (rad/s^2) before
        it goes to the INDI allocation; returns the nu actually used. Lets an adaptive element
        add a compensation term (L1). Baseline INDI: unchanged."""
        return nu

    # ---- outer loops ----
    def attitude_reference(self, meas, ref):
        k, p = self.k, self.p
        a_des = (k.kp_pos * (ref["pos"] - meas["pos"])
                 + k.kd_pos * (ref.get("vel", np.zeros(3)) - meas["vel"]))
        n_xy = float(np.hypot(a_des[0], a_des[1]))
        if n_xy > k.a_max_xy:
            a_des[0:2] *= k.a_max_xy / n_xy
        f = p.mass * (a_des - np.array([0.0, 0.0, p.g]))   # required force (NED); thrust acts along -z_b
        T_des = norm3(f)
        z_b = -f / T_des
        yaw = ref.get("yaw", 0.0)
        x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_b = cross3(z_b, x_c)
        y_b /= norm3(y_b)
        x_b = cross3(y_b, z_b)
        q_des = rot_to_quat(np.column_stack([x_b, y_b, z_b]))
        return q_des, T_des

    # ---- full update ----
    def update(self, meas, ref):
        q_des, T_des = self.attitude_reference(meas, ref)
        q_e = quat_mult(quat_conj(q_des), meas["q"])
        if q_e[0] < 0:
            q_e = -q_e
        w_cmd = -2.0 * self.k.k_att * q_e[1:4]

        w_f, wdot_f = self.f_gyro.update(meas["gyro"], self.dt)
        u_f, _ = self.f_act.update(meas["T_motor"], self.dt)
        acc_z_f, _ = self.f_acc.update(np.array([meas["acc"][2]]), self.dt)

        self.observe(u_f, wdot_f, acc_z_f[0], meas)           # adaptive variants learn here

        nu = self.k.k_rate * (w_cmd - meas["gyro"])          # desired angular acceleration
        nu = self.augment(nu, meas)                           # adaptive compensation (L1)
        G = self.effectiveness()
        if self.thrust_feedback == "accel":
            T_now = -self.p.mass * acc_z_f[0]                 # thrust actually produced (measured)
        else:
            T_now = G[3, :] @ u_f                             # thrust the model believes is produced
        rhs = np.concatenate([nu - wdot_f, [T_des - T_now]])
        du = np.linalg.solve(G, rhs)
        u_cmd = np.clip(u_f + du, 0.0, self.p.T_max)

        self.last = {"q_des": q_des, "T_des": T_des, "w_cmd": w_cmd, "nu": nu, "wdot_f": wdot_f}
        return u_cmd
