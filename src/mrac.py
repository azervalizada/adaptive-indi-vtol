r"""INDI with predictor-based (indirect) MRAC estimation of the rotor effectiveness.

Same unknowns as RLS (eta_hat, one effectiveness per rotor), same plug-in point
(G = G_nominal * diag(eta_hat)); only the LEARNING RULE differs. Close to the PMRAC of
Franceschini & Invernizzi (2026) and the predictor-based adaptive allocation of Falconi (TUM, 2021).

Idea
----
Run a small model of the rotational dynamics next to the real vehicle (the "predictor") and
feed it the same rotor thrusts. If the predicted rotation rate drifts away from the measured
one, the model's guess of eta (and of an unknown external moment d) is wrong, and the
adaptation law nudges them in the direction that shrinks the difference.

Equations (body rates w, rotor thrusts T from rpm, G_a = J^-1 B = first 3 rows of G_nominal)
-------------------------------------------------------------------------------------------
    true vehicle:  w_dot     = G_a diag(eta) T     + d     - J^-1 (w x J w)
    predictor:     w_hat_dot = G_a diag(eta_hat) T + d_hat - J^-1 (w x J w) - a (w_hat - w)
    error:         e = w_hat - w   ->   e_dot = -a e + G_a diag(T) (eta_hat - eta) + (d_hat - d)

    Lyapunov function V = 1/2 e'e + 1/2 eta~' Gamma_eta^-1 eta~ + 1/2 d~' Gamma_d^-1 d~ gives
    V_dot = -a e'e (never positive) with the adaptation laws

        eta_hat_dot = -Gamma_eta * diag(T) G_a' e
        d_hat_dot   = -Gamma_d * e

    So e -> 0 and the estimates stay bounded. (They converge to the truth only if the thrusts
    vary enough: "persistent excitation". In steady hover eta and d can trade off against each
    other, which is exactly what the gust check probes.)

Thrust channel (algebraic, like the RLS thrust row): measured thrust f = -m * acc_z_f vs
predicted sum(eta_hat * u_f); gradient step  eta_hat_dot += -gamma_thrust * u_f * (f_pred - f).

d_hat is the textbook "matched uncertainty" term of MRAC (Lavretsky & Wise). It is internal to
the estimator: INDI already rejects external moments through the measured angular acceleration.

Safeguard: projection = eta_hat kept in [eta_min, eta_max], d_hat in [-d_max, d_max].

Note: the thrust channel uses the controller's NOMINAL mass. When the true mass differs
(Monte Carlo: +-5 %), that mismatch shows up as all four eta_hat shifted by about the mass
ratio. This is model mismatch acting on the estimator, which is what the Monte Carlo set probes.
"""
from dataclasses import dataclass
import numpy as np

from .indi import INDIController
from .rotations import cross3


@dataclass
class MRACSettings:
    # a, gamma_eta, gamma_d, gamma_thrust: chosen by `python tune.py mrac` on the tuning set
    # (re-tuned 30 Oct 2026, equal-budget procedure: 70 candidates; a, gamma_eta, gamma_thrust
    # interior, gamma_d = 0 is the lower bound). Do not hand-edit; re-run the tuning.
    a: float = 10.0              # predictor pole [1/s]: how hard the predictor is pulled to the gyro
    gamma_eta: float = 1.0       # adaptation gain for eta
    gamma_d: float = 0.0         # adaptation gain for the external-moment estimate (0 = off)
    gamma_thrust: float = 20.0   # gradient gain of the thrust channel (0 = off)
    eta_min: float = 0.1
    eta_max: float = 1.5
    d_max: float = 50.0          # rad/s^2


class MRACINDI(INDIController):
    def __init__(self, params_nominal, gains, dt, settings=None, thrust_feedback="accel"):
        super().__init__(params_nominal, gains, dt, thrust_feedback)
        self.s = settings or MRACSettings()
        self.eta_hat = np.ones(4)
        self.d_hat = np.zeros(3)
        self.w_hat = None
        self._Ga = self._G_nominal[0:3, :]
        self._J_inv = np.linalg.inv(self.J)

    def observe(self, u_f, wdot_f, acc_z_f, meas=None):
        s, dt = self.s, self.dt
        w = meas["gyro"]
        T = meas["T_motor"]
        if self.w_hat is None:
            self.w_hat = w.copy()
            return
        e = self.w_hat - w

        # predictor (explicit Euler at the control rate)
        gyro_term = self._J_inv @ cross3(w, self.J @ w)
        w_hat_dot = self._Ga @ (self.eta_hat * T) + self.d_hat - gyro_term - s.a * e
        self.w_hat = self.w_hat + dt * w_hat_dot

        # adaptation laws (Lyapunov-based), then projection onto the allowed box
        eta_dot = -s.gamma_eta * T * (self._Ga.T @ e)
        if s.gamma_thrust > 0.0:
            e_f = float(self.eta_hat @ u_f) - (-self.p.mass * acc_z_f)
            eta_dot = eta_dot - s.gamma_thrust * u_f * e_f
        self.eta_hat = np.clip(self.eta_hat + dt * eta_dot, s.eta_min, s.eta_max)
        self.d_hat = np.clip(self.d_hat - dt * s.gamma_d * e, -s.d_max, s.d_max)

    def effectiveness(self):
        return self._G_nominal * self.eta_hat[None, :]
