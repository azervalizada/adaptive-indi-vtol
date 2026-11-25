r"""INDI with a recursive-least-squares (RLS) estimate of the rotor effectiveness.

Idea
----
INDI already measures how the vehicle responds to each command increment. If rotor i is
damaged, the response to a change in its thrust is eta_i times smaller than the nominal model
says. RLS keeps a running estimate eta_hat of the four eta_i and the controller uses
G = G_nominal * diag(eta_hat) instead of G_nominal.

Measurement model (same synchronised filters as INDI, so the signals line up in time)
--------------------------------------------------------------------------------------
Over L control steps, with Delta = value now minus value L steps ago:

    [ Delta wdot_f      ]                                  4 outputs: roll, pitch, yaw
    [ -m * Delta acc_z_f ]  =  G_nominal * diag(Delta u_f) * eta     angular acceleration
                               \_______  Phi (4 x 4) ____/            and total thrust

This is linear in eta, so the standard RLS update applies:

    e   = y - Phi eta_hat                    prediction error
    K   = P Phi^T (Phi P Phi^T + I)^-1       gain: how much to trust the new data
    eta_hat <- eta_hat + K e
    P   <- (P - K Phi P) / lambda            lambda < 1 slowly forgets old data,
                                             so a NEW fault can be learned

Practical safeguards
--------------------
* Rows are divided by their measured noise level, so a noisy channel (yaw) counts less.
* Dead-zone: no update when the command barely changed (|Delta u_f| < threshold). In steady
  hover the increments are mostly sensor noise that the controller itself reacts to, which
  biases closed-loop estimates (seen in the offline check of 26 Oct: eta_hat ~1.5 on a healthy
  vehicle without the dead-zone).
* eta_hat is kept in [eta_min, eta_max] and P is capped, so the estimate cannot run away.

Known limitation (physics of the X layout): diagonal rotors (1-3, 2-4) have opposite roll and
pitch effects, so only the noisy yaw and thrust rows tell them apart.
"""
from dataclasses import dataclass
import numpy as np

from .indi import INDIController


@dataclass
class RLSSettings:
    # lag, threshold, forget, p0: chosen by `python tune.py rls` on the tuning set (re-tuned 30 Oct 2026
    # with the equal-budget procedure: 68 candidates, all four interior). Do not hand-edit; re-run the
    # tuning (tests/test_controllers_tuned.py checks these against results/tuning_rls.csv).
    lag: int = 10                # L: increments over L control steps (10 x 2 ms = 20 ms)
    threshold: float = 0.005     # dead-zone on |Delta u_f| [N] per sqrt(step)
    forget: float = 0.97         # lambda per update
    p0: float = 0.3              # initial covariance (how unsure we start)
    p_max: float = 1.0           # cap on each diagonal entry of P
    eta_min: float = 0.1
    eta_max: float = 1.5
    # per-row noise std of the increments over ONE step, measured in hover (26 Oct):
    # roll, pitch, yaw angular acceleration [rad/s^2], thrust [N]
    row_noise: tuple = (0.011, 0.010, 0.011, 0.0013)


class RLSINDI(INDIController):
    def __init__(self, params_nominal, gains, dt, settings=None, thrust_feedback="accel"):
        super().__init__(params_nominal, gains, dt, thrust_feedback)
        self.s = settings or RLSSettings()
        self.eta_hat = np.ones(4)
        self.P = self.s.p0 * np.eye(4)
        # row weights for a difference over L steps: noise of a sum of L increments grows ~ sqrt(L)
        self._w = 1.0 / (np.array(self.s.row_noise) * np.sqrt(self.s.lag))
        self._hist = []          # last L+1 samples of (u_f, y_raw)
        self.n_updates = 0

    def observe(self, u_f, wdot_f, acc_z_f, meas=None):
        y_raw = np.array([wdot_f[0], wdot_f[1], wdot_f[2], -self.p.mass * acc_z_f])
        self._hist.append((u_f.copy(), y_raw))
        if len(self._hist) <= self.s.lag:
            return
        u_old, y_old = self._hist.pop(0)
        du = u_f - u_old
        if np.linalg.norm(du) < self.s.threshold * np.sqrt(self.s.lag):
            return                                   # dead-zone: not enough excitation
        Phi = (self._G_nominal * du[None, :]) * self._w[:, None]
        y = (y_raw - y_old) * self._w
        P = self.P
        S = Phi @ P @ Phi.T + np.eye(4)
        K = P @ Phi.T @ np.linalg.inv(S)
        self.eta_hat = np.clip(self.eta_hat + K @ (y - Phi @ self.eta_hat), self.s.eta_min, self.s.eta_max)
        P = (P - K @ Phi @ P) / self.s.forget
        np.fill_diagonal(P, np.minimum(np.diag(P), self.s.p_max))
        self.P = 0.5 * (P + P.T)                     # keep it symmetric against round-off
        self.n_updates += 1

    def effectiveness(self):
        return self._G_nominal * self.eta_hat[None, :]
