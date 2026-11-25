r"""RLS, MRAC and L1 on the quadplane INDI (Stage 3). The learning rules are those of Stage 2
(rls.py, mrac.py, l1.py); only what changes on a quadplane is new here.

What changes
------------
* On a quadplane the angular acceleration and the z-force also contain the wing (damping, static
  stability, surfaces, lift), the pusher's torque and the gyroscopic term. RLS and MRAC get the SAME
  knowledge of this non-rotor part (class KnownPart below, 28 Oct 2026 after the independent
  review, finding C1: before, RLS removed only the surface increment and at speed blamed the
  wing's damping on the rotors; noise-free batch LS at 8 m/s gave eta = [-0.13, -3.57, 0.46, 0.54]
  for a true [0.6, 1, 1, 1]):
      o = [ J^-1 (M_aero,nom + M_push - w x J w) ;  -F_aero,z,nom ]   (from measured v, q, w, T_push
                                                                       and the servo model, no wind)
  passed through the same synchronisation filter as the gyro / actuators / accelerometer, so
      RLS:   y = D([wdot_f ; -m a_z,f] - o_f)  =  G_rot diag(D u_rot,f) eta
      MRAC:  predictor  w_hat_dot = J^-1 (B eta T) + o_ang + ...,  thrust channel -m a_z,f + F_z,f
  (D = change over L control steps, as in rls.py).
  aero_model = False (sensitivity variant, review M3): o keeps only what the baseline INDI already
  uses, the surface columns of G(Va), plus the pusher torque and the gyroscopic term.
* L1 is unchanged: it never needed a model of the vehicle (everything INDI misses is sigma).

The plug-in points are the same hooks as in Stage 2: observe(), effectiveness(), augment().
"""
from dataclasses import dataclass
import numpy as np

from .qp_indi import QPINDIController
from .quadplane import Quadplane
from .rotations import cross3
from .indi import SecondOrderFilter

# Estimators start learning only after the synchronisation filters have settled from their start
# values (27 Oct: without this, RLS learned from the start-up transient of the accelerometer filter
# in the first 0.05 s and froze at its limits in hover). Faults start at t >= 4 s, so it costs nothing.
WARMUP_S = 0.5


class KnownPart:
    """Nominal non-rotor part o (4,) of [angular acceleration; rotor thrust], filtered with the
    controller's synchronisation filter (same wn, zeta). Shared by RLS and MRAC."""
    def __init__(self, ctrl, aero_model=True):
        self.c = ctrl
        self.aero_model = aero_model
        self._model = Quadplane(ctrl.p)                          # nominal aerodynamics only
        self.f = None
        self.o = np.zeros(4)
        self.o_f = np.zeros(4)

    def update(self, meas):
        c, p = self.c, self.c.p
        w = meas["gyro"]
        if self.aero_model:
            R = c._rot(meas["q"])
            v_b = R.T @ meas["vel"]                              # no wind knowledge
            Fa, Ma, *_ = self._model.aero(v_b, w, c.delta_hat[0], c.delta_hat[1])
            Fz = Fa[2]
        else:
            G = QPINDIController.effectiveness(c, meas["airspeed"])
            Ma = c.J @ (G[0:3, 4:6] @ c.delta_hat)
            Fz = -G[3, 5] * c.delta_hat[1]
        M = Ma + np.array([-p.k_m_push * meas["T_push"], 0.0, 0.0]) - cross3(w, c.J @ w)
        self.o = np.concatenate([c._J_inv @ M, [-Fz]])
        if self.f is None:
            self.f = SecondOrderFilter(4, c.k.filt_wn, c.k.filt_zeta, init=self.o)
        self.o_f, _ = self.f.update(self.o, c.dt)
        return self.o, self.o_f


# ------------------------------------------------------------------------------------- RLS
@dataclass
class QPRLSSettings:
    # lag, threshold, forget, p0: chosen by `python tune_qp.py rls` (re-tuned 28 Oct 2026 after the
    # review fixes, equal-budget procedure, with eta_min = 0.1: 72 candidates, 36 pass). threshold and
    # p0 interior (all neighbours evaluated); lag = 1 and forget = 1 (no forgetting) are bounds.
    # Do not hand-edit; re-run the tuning (tests/test_qp_adaptive.py checks these against
    # results/tuning_qp_rls.csv).
    lag: int = 1
    threshold: float = 0.05      # dead-zone on |D u_rot,f| [N] per sqrt(step)
    forget: float = 1.0
    p0: float = 0.01
    p_max: float = 1.0
    # 0.1 as in Stage 2 (decision of 28 Oct 2026, Azer, after the review reruns): with 0, an abrupt
    # fault drove all four estimates to 0 within 20-50 ms; the allocation then stopped moving the
    # rotors, no new information arrived and the estimates stayed locked at 0 (Monte Carlo gap
    # closed -42 % with 0, 111 % with 0.1). eta_min = 0 is reported as the sensitivity variant rls_eta0.
    eta_min: float = 0.1
    eta_max: float = 1.5
    # per-row noise std of the one-step increments, measured in fault-free flight (27 Oct, baseline
    # INDI, t = 2-5 s): hover 0.0110 / 0.0108 / 0.0110 / 0.0068, 8 m/s 0.0110 / 0.0105 / 0.0110 / 0.0066
    # roll, pitch, yaw angular acceleration [rad/s^2], thrust [N]
    row_noise: tuple = (0.011, 0.0107, 0.011, 0.0067)
    aero_model: bool = True       # subtract the nominal non-rotor part (KnownPart); False: review M3 variant


class QPRLSINDI(QPINDIController):
    def __init__(self, params_nominal, gains, dt, settings=None):
        super().__init__(params_nominal, gains, dt)
        self.s = settings or QPRLSSettings()
        self.eta_hat = np.ones(4)
        self.P = self.s.p0 * np.eye(4)
        self._w = 1.0 / (np.array(self.s.row_noise) * np.sqrt(self.s.lag))
        self._hist = []
        self._G_rot = super().effectiveness(0.0)[:, 0:4]       # 4 x 4, airspeed-independent
        self._known = KnownPart(self, self.s.aero_model)
        self._k = 0

    def regression(self, u_f, wdot_f, acc_f, meas):
        """One weighted regression row (y, Phi) with y = Phi eta (+ noise), or None (warm-up,
        history filling, dead-zone). Separate from observe() so tests can check y = Phi eta."""
        self._k += 1
        _, o_f = self._known.update(meas)              # runs from the start so its filter is settled
        if self._k * self.dt < WARMUP_S:              # filters still settling from their start values
            return None
        y_raw = np.array([wdot_f[0], wdot_f[1], wdot_f[2], -self.p.mass * acc_f[2]]) - o_f
        self._hist.append((u_f.copy(), y_raw))
        if len(self._hist) <= self.s.lag:
            return None
        u_old, y_old = self._hist.pop(0)
        du = u_f - u_old
        if np.linalg.norm(du[0:4]) < self.s.threshold * np.sqrt(self.s.lag):
            return None
        y = (y_raw - y_old) * self._w                 # known non-rotor part already removed (o_f)
        Phi = (self._G_rot * du[None, 0:4]) * self._w[:, None]
        return y, Phi

    def observe(self, u_f, wdot_f, acc_f, meas):
        row = self.regression(u_f, wdot_f, acc_f, meas)
        if row is None:
            return
        y, Phi = row
        P = self.P
        K = P @ Phi.T @ np.linalg.inv(Phi @ P @ Phi.T + np.eye(4))
        self.eta_hat = np.clip(self.eta_hat + K @ (y - Phi @ self.eta_hat), self.s.eta_min, self.s.eta_max)
        P = (P - K @ Phi @ P) / self.s.forget
        np.fill_diagonal(P, np.minimum(np.diag(P), self.s.p_max))
        self.P = 0.5 * (P + P.T)

    def effectiveness(self, Va):
        G = super().effectiveness(Va)
        G[:, 0:4] *= self.eta_hat[None, :]
        return G


# ------------------------------------------------------------------------------------ MRAC
@dataclass
class QPMRACSettings:
    # chosen by `python tune_qp.py mrac` (re-tuned 28 Oct 2026 after the review fixes: 62 candidates,
    # 45 pass). a, gamma_eta, gamma_thrust interior; gamma_d = 0 is the lower bound. Do not hand-edit.
    a: float = 30.0
    gamma_eta: float = 16.0
    gamma_d: float = 0.0
    gamma_thrust: float = 1.0
    eta_min: float = 0.0
    eta_max: float = 1.5
    d_max: float = 50.0
    aero_model: bool = True       # nominal non-rotor part (KnownPart); False: review M3 variant


class QPMRACINDI(QPINDIController):
    def __init__(self, params_nominal, gains, dt, settings=None):
        super().__init__(params_nominal, gains, dt)
        self.s = settings or QPMRACSettings()
        self.eta_hat = np.ones(4)
        self.d_hat = np.zeros(3)
        self.w_hat = None
        self._known = KnownPart(self, self.s.aero_model)
        self._B_rot = params_nominal.B_rotor_moments()
        self._k = 0

    def observe(self, u_f, wdot_f, acc_f, meas):
        s, dt, p = self.s, self.dt, self.p
        w, T = meas["gyro"], meas["T_motor"]
        self._k += 1
        o, o_f = self._known.update(meas)
        if self.w_hat is None or self._k * dt < WARMUP_S:
            self.w_hat = w.copy()
            return
        e = self.w_hat - w
        w_hat_dot = self._J_inv @ (self._B_rot @ (self.eta_hat * T)) + o[0:3] + self.d_hat - s.a * e
        self.w_hat = self.w_hat + dt * w_hat_dot
        eta_dot = -s.gamma_eta * T * ((self._J_inv @ self._B_rot).T @ e)
        if s.gamma_thrust > 0.0:
            # accelerometer minus nominal lift, both filtered (review m12: was unfiltered lift)
            T_rot_meas = -p.mass * acc_f[2] - o_f[3]
            e_f = float(self.eta_hat @ u_f[0:4]) - T_rot_meas
            eta_dot = eta_dot - s.gamma_thrust * u_f[0:4] * e_f
        self.eta_hat = np.clip(self.eta_hat + dt * eta_dot, s.eta_min, s.eta_max)
        self.d_hat = np.clip(self.d_hat - dt * s.gamma_d * e, -s.d_max, s.d_max)

    def effectiveness(self, Va):
        G = super().effectiveness(Va)
        G[:, 0:4] *= self.eta_hat[None, :]
        return G


# -------------------------------------------------------------------------------------- L1
@dataclass
class QPL1Settings:
    # chosen by `python tune_qp.py l1` (re-tuned 28 Oct 2026: 51 candidates, all pass). a interior;
    # wc = 1280 rad/s is the top of the ladder (Nyquist of the 500 Hz loop is 1571 rad/s): the filter
    # is then nearly transparent. Its delay margin is ~20 ms at wc = 320, 640 and 1280 alike (set by
    # the predictor pole a, checked 28 Oct). Do not hand-edit.
    a: float = 30.0
    wc: float = 1280.0
    sigma_max: float = 200.0


class QPL1INDI(QPINDIController):
    """Same three blocks as l1.py: predictor with the nominal motor lag, piecewise-constant
    adaptation, low-pass filtered compensation subtracted from nu."""
    def __init__(self, params_nominal, gains, dt, settings=None):
        super().__init__(params_nominal, gains, dt)
        self.s = settings or QPL1Settings()
        a = self.s.a
        self._ead = np.exp(-a * dt)
        self._phi = (1.0 - self._ead) / a
        self._lp = 1.0 - np.exp(-self.s.wc * dt)
        self._act = 1.0 - np.exp(-dt / params_nominal.tau_motor)
        self.w_hat = None
        self.sigma_hat = np.zeros(3)
        self.u_ad = np.zeros(3)
        self.nu_pred = np.zeros(3)

    def augment(self, nu, meas):
        s = self.s
        w = meas["gyro"]
        if self.w_hat is None:
            self.w_hat = w.copy()
        e = self.w_hat - w
        self.sigma_hat = np.clip(-self._ead / self._phi * e, -s.sigma_max, s.sigma_max)
        self.u_ad = self.u_ad + self._lp * (self.sigma_hat - self.u_ad)
        nu_a = nu - self.u_ad
        self.nu_pred = self.nu_pred + self._act * (nu_a - self.nu_pred)
        self.w_hat = w + self._ead * e + self._phi * (self.nu_pred + self.sigma_hat)
        return nu_a
