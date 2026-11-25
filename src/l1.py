r"""INDI with L1 adaptive augmentation (piecewise-constant adaptation law).

Different in kind from RLS and MRAC: L1 does NOT estimate the rotor effectiveness. It
estimates the lumped uncertainty sigma ("how differently does the vehicle rotate than INDI
expects?") and subtracts a LOW-PASS FILTERED copy of it from INDI's command. So it compensates
a fault without diagnosing it. Close to Raos, Gozzini & Invernizzi (EuroGNC 2026), "INDI flight
control laws with L1 adaptive augmentation", and to Hovakimyan & Cao (2010), ch. 2-3.

Reference model
---------------
With synchronised filters, INDI turns the angular-acceleration command nu into the achieved
angular acceleration through (approximately) the actuator dynamics A(s) (Smeur et al., 2016):

    w_dot  ~=  A(s) [nu_a]  +  sigma          nu_a = nu - u_ad  (what INDI actually receives)

sigma collects everything INDI's nominal model misses: a damaged rotor (wrong G), gusts, etc.

The three L1 blocks (all per axis, at the control rate Ts)
----------------------------------------------------------
1. State predictor:   w_hat_dot = -a (w_hat - w) + A(s)[nu_a] + sigma_hat
2. Adaptation (piecewise constant, Hovakimyan & Cao 2010, sec. 3.3):
       e = w_hat - w,   Phi = (1 - exp(-a Ts)) / a,
       sigma_hat = -exp(-a Ts) / Phi * e
   i.e. sigma_hat is recomputed from scratch every sample so that the predictor would hit the
   measurement one step later. Adaptation is "as fast as the sampling".
3. Low-pass filter C(s) = wc / (s + wc) on the compensation:
       u_ad = C(s)[sigma_hat],    nu_a = nu - u_ad
   C(s) decides how much of the fast estimate reaches the motors: this is what keeps fast
   adaptation from turning into oscillation (the MRAC trade-off).

A(s) in the predictor: "motor" = first-order lag with the nominal motor time constant;
"none" = nu_a acts instantly. Chosen on FAULT-FREE flight only (see src/README notes /
decision log): the model that keeps sigma_hat smallest on a healthy vehicle.

Thrust is left to INDI's accelerometer channel, as in the baseline.
"""
from dataclasses import dataclass
import numpy as np

from .indi import INDIController


@dataclass
class L1Settings:
    # a, wc: chosen by `python tune.py l1` on the tuning set (re-tuned 30 Oct 2026, equal-budget
    # procedure: 56 candidates, rules (a)(b); (c) does not apply). Both at a ladder end: wc = 1280
    # rad/s near the Nyquist rate of the 500 Hz loop, a = 0.1 the a -> 0 limit (the score has
    # converged: 0.4918 / 0.4912 / 0.4910 deg at a = 1 / 0.3 / 0.1; as a -> 0 the law tends to
    # sigma_hat = -e / Ts). In this delay-free simulation a faster L1 is always better; see the
    # delay robustness check. Do not hand-edit; re-run the tuning.
    a: float = 0.1               # predictor pole [1/s]
    wc: float = 1280.0           # low-pass cut-off of C(s) [rad/s]
    pred_model: str = "motor"    # A(s) in the predictor: "motor" or "none"
    sigma_max: float = 200.0     # rad/s^2, safety clip on the compensation


class L1INDI(INDIController):
    def __init__(self, params_nominal, gains, dt, settings=None, thrust_feedback="accel"):
        super().__init__(params_nominal, gains, dt, thrust_feedback)
        self.s = settings or L1Settings()
        if self.s.pred_model not in ("motor", "none"):
            raise ValueError("pred_model must be 'motor' or 'none'")
        a, Ts = self.s.a, dt
        self._ead = np.exp(-a * Ts)                     # predictor error decay over one step
        self._phi = (1.0 - self._ead) / a               # integral of exp(-a t) over one step
        self._lp = 1.0 - np.exp(-self.s.wc * Ts)        # exact discretisation of C(s)
        self._act = 1.0 - np.exp(-Ts / params_nominal.tau_motor)
        self.w_hat = None
        self.sigma_hat = np.zeros(3)
        self.u_ad = np.zeros(3)
        self.nu_pred = np.zeros(3)                      # A(s)[nu_a]

    def augment(self, nu, meas):
        s = self.s
        w = meas["gyro"]
        if self.w_hat is None:                          # first call: start the predictor on the gyro
            self.w_hat = w.copy()
        e = self.w_hat - w

        # 2. piecewise-constant adaptation
        self.sigma_hat = np.clip(-self._ead / self._phi * e, -s.sigma_max, s.sigma_max)
        # 3. low-pass filtered compensation
        self.u_ad = self.u_ad + self._lp * (self.sigma_hat - self.u_ad)
        nu_a = nu - self.u_ad

        # 1. predictor, propagated exactly over one step (w held constant over the step)
        if s.pred_model == "motor":
            self.nu_pred = self.nu_pred + self._act * (nu_a - self.nu_pred)
        else:
            self.nu_pred = nu_a
        self.w_hat = w + self._ead * e + self._phi * (self.nu_pred + self.sigma_hat)
        return nu_a
