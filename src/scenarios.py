"""Fixed test scenarios and metrics.

The scenarios and metrics were fixed BEFORE any adaptive controller was run (26 Oct 2026), so
the comparison cannot be tuned to favour one method. Later additions, each recorded in the
plan's decision log and made before the methods that follow them existed:
  * upset window extended by the ramp time for ramp faults (before RLS)
  * tuning rule (c) and the gust check (after RLS, before MRAC/L1)
  * eta_err_end averaged over the last 0.5 s (after MRAC; reporting only, not used in tuning)
Change anything here only with a note in the decision log.

Timeline of every scenario (default_reference in sim.py):
    0-3 s hover at 2 m | 3 s step to (2, 1) m | quiet from ~5 s | 9 s step back home
The fault starts at t0 in [5.0, 6.5] s, where the fault-free attitude error is <= 0.22 deg,
and every metric is measured in the window [t0, t0 + 2.5 s], which ends before the 9 s step.
So the metrics see the fault, not the manoeuvre.
"""
from dataclasses import dataclass
import numpy as np

WINDOW_S = 2.5          # metric window after the fault
UPSET_S = 0.5           # primary metric looks at the first half second
RECOVERED_DEG = 1.0     # "recovered" = attitude error stays below this (noise floor is ~0.2 deg)


@dataclass(frozen=True)
class Fault:
    """Loss of effectiveness of one rotor.

    rotor: 0..3 (rotor 1..4 in the docs)
    eta:   remaining effectiveness after the fault, e.g. 0.5 = half thrust
    t0:    fault start [s]
    ramp:  0 = abrupt (broken propeller); > 0 = linear decay over `ramp` seconds (e.g. overheating motor)
    """
    rotor: int
    eta: float
    t0: float
    ramp: float = 0.0

    def eta_at(self, t):
        out = np.ones(4)
        if t < self.t0:
            return out
        if self.ramp > 0.0 and t < self.t0 + self.ramp:
            frac = (t - self.t0) / self.ramp
            out[self.rotor] = 1.0 + frac * (self.eta - 1.0)
        else:
            out[self.rotor] = self.eta
        return out


@dataclass(frozen=True)
class Gust:
    """External moment on a HEALTHY vehicle (wind gust, payload shift) for `duration` seconds.

    An adaptive scheme should reject it like any INDI does and must NOT read it as a rotor fault.
    Added 26 Oct 2026 after the first RLS benchmark (see decision log), before MRAC/L1 exist.
    """
    t0: float
    moment: tuple                 # body-frame moment [Nm]: roll, pitch, yaw
    duration: float = 0.5


@dataclass(frozen=True)
class Scenario:
    name: str
    fault: Fault = None
    gust: Gust = None
    seed: int = 0                 # sensor-noise seed
    mass_scale: float = 1.0       # true vehicle vs. the controller's nominal model
    J_scale: tuple = (1.0, 1.0, 1.0)
    t_end: float = 9.0            # the metric window always ends by 9 s
    delay_s: float = 0.0          # extra, unmodelled transport delay on the motor commands [s]

    def label(self):
        base = self._label()
        return base if self.delay_s == 0 else f"{base}, +{self.delay_s * 1000:g} ms delay"

    def _label(self):
        if self.gust is not None:
            g = self.gust
            return f"{self.name}: gust {tuple(round(m, 3) for m in g.moment)} Nm for {g.duration:g}s, t0 {g.t0:.2f}s"
        if self.fault is None:
            return f"{self.name}: no fault"
        f = self.fault
        kind = "abrupt" if f.ramp == 0 else f"ramp {f.ramp:g}s"
        return f"{self.name}: rotor {f.rotor + 1}, eta {f.eta:.2f}, {kind}, t0 {f.t0:.2f}s"


# ---------------------------------------------------------------------------------------
# Scenario sets
# ---------------------------------------------------------------------------------------
def benchmark_grid():
    """Deterministic grid: 4 rotors x 3 severities x 2 fault types = 24 scenarios.

    Perfect nominal model, fixed noise seed, fault at 6.0 s. This is the main comparison table.
    """
    out = []
    for rotor in range(4):
        for eta in (0.8, 0.65, 0.5):
            for ramp in (0.0, 1.0):
                out.append(Scenario("grid", Fault(rotor, eta, 6.0, ramp)))
    return out


def monte_carlo(n, seed=2026):
    """Random scenarios for the robustness study (Stage 5 on the quadplane, tried here first).

    Fault: random rotor, eta ~ U[0.5, 0.9], abrupt or 1 s ramp (50/50), t0 ~ U[5.0, 6.5].
    Model mismatch: true mass x U[0.95, 1.05], true inertia x U[0.9, 1.1] per axis.
    Noise: a different seed per run.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fault = Fault(rotor=int(rng.integers(4)),
                      eta=float(rng.uniform(0.5, 0.9)),
                      t0=float(rng.uniform(5.0, 6.5)),
                      ramp=float(rng.choice([0.0, 1.0])))
        out.append(Scenario(f"mc{i:03d}", fault, seed=1000 + i,
                            mass_scale=float(rng.uniform(0.95, 1.05)),
                            J_scale=tuple(float(s) for s in rng.uniform(0.9, 1.1, 3))))
    return out


def tuning_set():
    """Scenarios used ONLY to tune adaptive controllers; never reported as results.

    Kept apart from the benchmark grid (different severities, fault times and noise seeds), so
    a method cannot be tuned on the exam questions. Tuning rule, the same for every method:
      choose the settings with the lowest mean upset_deg over the faulty scenarios here,
      subject to (a) no crash, (b) fault-free att_rms_deg <= 1.05 x baseline and
      (c) [added 26 Oct, see decision log] healthy_dev <= 0.15 in every tuning scenario:
          an estimator must recognise healthy rotors as healthy. Without (c) the RLS search
          picked a setting whose estimates sat at the clip limit (~1.45 on healthy rotors) and
          still scored well on upset_deg by acting as an erratic high-gain loop.
    """
    faults = [Fault(1, 0.55, 5.4), Fault(3, 0.75, 5.8), Fault(0, 0.6, 5.6, 1.0), Fault(2, 0.7, 5.2, 1.0)]
    out = [Scenario(f"tune{i}", f, seed=90 + i) for i, f in enumerate(faults)]
    out.append(Scenario("tune-nominal", None, seed=95))
    return out


def gust_check():
    """Healthy vehicle, step moment for 0.5 s at 6.0 s. Max rotor moment authority is ~0.8 Nm
    in roll/pitch and ~0.06 Nm in yaw, so these are moderate disturbances."""
    moments = [(0.05, 0.0, 0.0), (0.0, 0.1, 0.0), (0.07, 0.07, 0.0), (0.0, 0.0, 0.02)]
    return [Scenario(f"gust{i}", gust=Gust(6.0, m)) for i, m in enumerate(moments)]


def nominal_check():
    """No fault: an adaptive scheme must not make the healthy vehicle worse."""
    return [Scenario("nominal", None)]


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def fault_metrics(res, scenario):
    """Metrics in the window after the fault or gust (or 5.0-7.5 s for an undisturbed run).

    upset_deg     PRIMARY. Peak attitude error from the fault start until 0.5 s after the fault
                  is fully developed, i.e. [t0, t0 + ramp + 0.5 s]. (For a ramp the damage is
                  still growing during the ramp; looking only at [t0, t0 + 0.5 s] would hide it.)
    att_rms_deg   RMS attitude error over the 2.5 s window.
    recovery_s    Time until the attitude error stays below 1 deg for the rest of the window
                  (2.5 = never recovered inside the window, 0 = never left).
    alt_dev_m     Peak altitude error in the window.
    sat_frac      Fraction of the window with at least one motor at 0 or T_max.
    crashed       Loss of control: simulation diverged OR attitude error > 90 deg after t0.
    For controllers that estimate eta (log key "eta_hat"), also:
    healthy_dev   Max |eta_hat - 1| while every rotor is healthy (4.5 s until the fault, or
                  4.5-9 s without a fault, gusts included). A sound estimator keeps healthy
                  rotors near 1; a large value under a gust is a FALSE fault detection.
    eta_err_end   Mean |eta_hat - eta_true| over the four rotors AND over the last 0.5 s of the
                  window. (Until 26 Oct it was read at the single last sample, which hides an
                  oscillating estimate that happens to cross the truth; found on MRAC.)
    """
    if scenario.fault is not None:
        t0, ramp = scenario.fault.t0, scenario.fault.ramp
    elif scenario.gust is not None:
        t0, ramp = scenario.gust.t0, 0.0
    else:
        t0, ramp = 5.0, 0.0
    t = res["t"]
    # loss of control: the simulation diverged, or the attitude error exceeded 90 deg at any time
    # after the fault/gust (vehicle on its side or inverted). The second test was added on 27 Oct:
    # a vehicle that flips but stays within 100 m was not flagged by the divergence test alone.
    lost = res["diverged_at"] is not None or bool(np.any(res["att_err_deg"][t >= t0] > 90.0))
    if lost:
        return {"upset_deg": np.nan, "att_rms_deg": np.nan, "recovery_s": np.nan,
                "alt_dev_m": np.nan, "sat_frac": np.nan, "crashed": True}
    w = (t >= t0) & (t < t0 + WINDOW_S)
    up = (t >= t0) & (t < t0 + ramp + UPSET_S)
    err = res["att_err_deg"]
    tw, ew = t[w], err[w]
    above = np.nonzero(ew >= RECOVERED_DEG)[0]
    recovery = 0.0 if len(above) == 0 else min(WINDOW_S, tw[above[-1]] - t0 + (tw[1] - tw[0]))
    u = res["u_cmd"][w]
    extra = {}
    if "eta_hat" in res:
        eh = res["eta_hat"]
        t_end_h = t0 if scenario.fault is not None else 9.0
        h = (t >= 4.5) & (t < t_end_h)
        tail = w & (t >= t0 + WINDOW_S - 0.5)
        extra = {"healthy_dev": float(np.max(np.abs(eh[h] - 1.0))),
                 "eta_err_end": float(np.mean(np.abs(eh[tail] - res["eta"][tail])))}
    return {
        **extra,
        "upset_deg": float(np.max(err[up])),
        "att_rms_deg": float(np.sqrt(np.mean(ew ** 2))),
        "recovery_s": float(recovery),
        "alt_dev_m": float(np.max(np.abs(res["pos"][w, 2] - res["ref"][w, 2]))),
        "sat_frac": float(np.mean(np.any((u >= res["T_max"] - 1e-3) | (u <= 1e-6), axis=1))),
        "crashed": False,
    }
