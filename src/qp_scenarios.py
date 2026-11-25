"""Stage 3 (quadplane) scenarios and metrics, fixed on 27 Oct 2026 BEFORE any adaptive controller
was run on the quadplane (decision log: "Aşama 3 metrikleri").

Why new metrics (vs scenarios.py): the quadplane is large and slow (inertia ~50x the quadrotor's).
After a severe fault its attitude can degrade over SECONDS, not in the first 0.5 s, and without a
rudder the heading may drift while the aircraft stays upright. So:
  * window of 4 s after the fault is fully developed (t0 + ramp + 4 s)
  * tilt (direction of the body z axis = thrust/lift direction) is the primary quantity; heading is
    reported separately, it is not a safety problem by itself on an aircraft without a rudder.

Flight: steady level flight north at airspeed V (V = 0: hover), see qp_sim.py.
"""
from dataclasses import dataclass, replace
import numpy as np

from .scenarios import Fault, Gust

WINDOW_S = 4.0
TAIL_S = 0.5            # "final" values = mean over the last 0.5 s of the window
LOST_TILT_DEG = 90.0
LOST_ALT_M = 5.0        # added 27 Oct after the first baseline/oracle run, see qp_metrics()
RECOVERED_TILT_DEG = 1.0
RECOVERED_HEADING_DEG = 5.0


@dataclass(frozen=True)
class QPScenario:
    name: str
    V: float = 0.0
    fault: Fault = None
    gust: Gust = None
    seed: int = 0
    mass_scale: float = 1.0
    J_scale: tuple = (1.0, 1.0, 1.0)
    surf_scale: tuple = (1.0, 1.0)     # true aileron, elevator effectiveness vs the nominal model
    delay_s: float = 0.0
    noise_scale: float = 1.0           # model-sensitivity analysis (28 Oct): sensor noise x this
    tau_scale: float = 1.0             # model-sensitivity analysis: true motor time constant x this
    # aerodynamic mismatch and wind (28 Oct 2026, review M3): true aero coefficients x these factors,
    # keys from AERO_KEYS; constant wind (NED, m/s) that no controller knows
    aero_scale: tuple = ()             # ((name, factor), ...)
    wind: tuple = (0.0, 0.0, 0.0)

    @property
    def t0(self):
        if self.fault is not None:
            return self.fault.t0
        if self.gust is not None:
            return self.gust.t0
        return 5.0

    @property
    def t_end(self):
        ramp = self.fault.ramp if self.fault is not None else 0.0
        return self.t0 + ramp + WINDOW_S + 0.01

    def label(self):
        s = f"{self.name}: V {self.V:g} m/s"
        if self.fault is not None:
            f = self.fault
            kind = "abrupt" if f.ramp == 0 else f"ramp {f.ramp:g}s"
            s += f", rotor {f.rotor + 1}, eta {f.eta:.2f}, {kind}, t0 {f.t0:.2f}s"
        elif self.gust is not None:
            s += f", gust {tuple(round(m, 3) for m in self.gust.moment)} Nm"
        else:
            s += ", no fault"
        return s + ("" if self.delay_s == 0 else f", +{self.delay_s * 1000:g} ms delay")


SPEEDS = (0.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def qp_grid():
    """4 rotors x eta {0.6, 0.3, 0.0} x V {0, 4, 6, 8, 10, 12}, abrupt at 5.0 s: 72 scenarios.
    eta 0.3 and 0.0 are physically infeasible in hover (hover needs eta >= 0.5): those runs show
    the boundary and are excluded from 'gap closed' when lost."""
    return [QPScenario("grid", V, Fault(r, e, 5.0)) for e in (0.6, 0.3, 0.0) for V in SPEEDS for r in range(4)]


def qp_speed_sweep(etas=(0.3, 0.15, 0.0), rotors=(0, 1, 2, 3)):
    """Minimum safe airspeed: V = 0, 1, ..., 12 m/s for each (rotor, eta), abrupt at 5.0 s."""
    return [QPScenario("sweep", float(V), Fault(r, e, 5.0)) for e in etas for r in rotors for V in range(13)]


def qp_monte_carlo(n, seed=2027):
    """Random fault, speed and model mismatch, including the aerodynamic control derivatives
    (the surfaces' real effectiveness is uncertain on a real aircraft: +-20 %)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        f = Fault(int(rng.integers(4)), float(rng.uniform(0.0, 0.9)), float(rng.uniform(4.0, 5.0)),
                  float(rng.choice([0.0, 1.0])))
        out.append(QPScenario(f"mc{i:03d}", float(rng.uniform(0.0, 12.0)), f, seed=2000 + i,
                              mass_scale=float(rng.uniform(0.95, 1.05)),
                              J_scale=tuple(float(s) for s in rng.uniform(0.9, 1.1, 3)),
                              surf_scale=tuple(float(s) for s in rng.uniform(0.8, 1.2, 2))))
    return out


AERO_KEYS = ("C_L_alpha", "C_m_alpha", "C_m_q", "C_ell_p", "C_ell_beta", "C_n_beta", "C_n_r", "C_D_p",
             "C_Y_beta")


def qp_monte_carlo_aero(n, seed=2028, scale=0.3, wind_max=3.0):
    """Monte Carlo of qp_monte_carlo PLUS (review M3, 28 Oct 2026) the aerodynamic model the
    controllers know is wrong: every coefficient in AERO_KEYS of the true aircraft is x U(1-scale,
    1+scale), and a constant horizontal wind of U(0, wind_max) m/s from a random direction blows
    (no controller knows it; MRAC/RLS use ground velocity as air velocity)."""
    rng = np.random.default_rng(seed)
    out = []
    for i, sc in enumerate(qp_monte_carlo(n)):
        a = tuple((k, float(rng.uniform(1 - scale, 1 + scale))) for k in AERO_KEYS)
        w, d = float(rng.uniform(0, wind_max)), float(rng.uniform(0, 2 * np.pi))
        out.append(replace(sc, name=f"mca{i:03d}", aero_scale=a, wind=(w * np.cos(d), w * np.sin(d), 0.0)))
    return out


def qp_tuning_set():
    """ONLY for tuning; never reported. Different rotors, severities, speeds and seeds."""
    s = [QPScenario("tune0", 5.0, Fault(1, 0.45, 4.6), seed=90),
         QPScenario("tune1", 9.0, Fault(3, 0.2, 4.8), seed=91),
         QPScenario("tune2", 2.0, Fault(0, 0.7, 4.4, 1.0), seed=92),
         QPScenario("tune3", 11.0, Fault(2, 0.1, 4.2, 1.0), seed=93)]
    s.append(QPScenario("tune-nominal", 7.0, seed=95))
    return s


def qp_gust_check():
    """Healthy aircraft, 0.5 s step moment at 5.0 s, in hover and at 8 m/s. Rotor authority is
    ~8.6 Nm in roll/pitch and ~0.6 Nm in yaw, so these are moderate (~10 %)."""
    moments = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.7, 0.7, 0.0), (0.0, 0.0, 0.1)]
    return [QPScenario(f"gust{i}", V, gust=Gust(5.0, m)) for V in (0.0, 8.0) for i, m in enumerate(moments)]


def qp_nominal_check():
    return [QPScenario("nominal", V) for V in (0.0, 8.0)]


def qp_metrics(res, sc):
    """Metrics in the window [t0, t0 + ramp + 4 s].

    tilt_peak_deg      PRIMARY. Peak tilt error (angle between desired and actual body z axis).
    tilt_final_deg     Mean tilt error over the last 0.5 s (steady-state).
    tilt_rms_deg       RMS tilt error over the window (used only by tuning rule (b), added 27 Oct).
    heading_peak_deg   Peak heading error; heading_final_deg its last-0.5-s mean.
    alt_dev_m          Peak altitude error.   lat_dev_m: peak sideways drift.
    rotor_sat_frac     Fraction of the window with a lift rotor at 0 or T_max.
    surf_sat_frac      Fraction of the window with a surface at its limit.
    lost               Diverged, or tilt error > 90 deg at any time after t0, or altitude error
                       > 5 m in the window. The altitude test was added on 27 Oct 2026 after the first
                       baseline/oracle run, BEFORE any adaptive method ran on the quadplane: complete
                       loss of a front rotor in hover gave a slow, uncontrolled descent (10 m in 4 s,
                       tilt 44 deg) that never crossed 90 deg and was not flagged.
    recovered          Not lost, tilt_final < 1 deg and heading_final < 5 deg.
    For estimators (log key eta_hat): healthy_dev (max |eta_hat - 1| from 3.5 s until the fault,
    or until the end without a fault) and eta_err_end (mean |eta_hat - eta| over the last 0.5 s).
    """
    t = res["t"]
    t0 = sc.t0
    t1 = sc.t_end - 0.01
    lost = res["diverged_at"] is not None or bool(np.any(res["tilt_err_deg"][t >= t0] > LOST_TILT_DEG))
    w = (t >= t0) & (t < t1)
    if not lost and np.max(np.abs(res["pos"][w, 2] - res["ref"][w, 2])) > LOST_ALT_M:
        lost = True
    if lost:
        return {"lost": True, "recovered": False}
    tail = w & (t >= t1 - TAIL_S)
    tilt, head = res["tilt_err_deg"], res["heading_err_deg"]
    u = res["u_cmd"][w]
    out = {
        "lost": False,
        "tilt_peak_deg": float(tilt[w].max()),
        "tilt_final_deg": float(tilt[tail].mean()),
        "tilt_rms_deg": float(np.sqrt(np.mean(tilt[w] ** 2))),   # for tuning rule (b); added 27 Oct
        "heading_peak_deg": float(head[w].max()),
        "heading_final_deg": float(head[tail].mean()),
        "alt_dev_m": float(np.max(np.abs(res["pos"][w, 2] - res["ref"][w, 2]))),
        "lat_dev_m": float(np.max(np.abs(res["pos"][w, 1] - res["ref"][w, 1]))),
        "rotor_sat_frac": float(np.mean(np.any((u[:, 0:4] >= res["T_max"] - 1e-3) | (u[:, 0:4] <= 1e-6), axis=1))),
        "surf_sat_frac": float(np.mean(np.any(np.abs(u[:, 5:7]) >= res["delta_max"] - 1e-6, axis=1))),
    }
    out["recovered"] = out["tilt_final_deg"] < RECOVERED_TILT_DEG and out["heading_final_deg"] < RECOVERED_HEADING_DEG
    if "eta_hat" in res:
        eh = res["eta_hat"]
        # healthy window: from 3.5 s (or 1 s before an earlier fault, Stage 4: fault at 3.0 s; fixed
        # 28 Oct after a crash on an empty window) until the fault; learning starts at 0.5 s (warm-up)
        h = (t >= min(3.5, t0 - 1.0)) & (t < (t0 if sc.fault is not None else t1))
        out["healthy_dev"] = float(np.max(np.abs(eh[h] - 1.0))) if h.any() else None
        out["eta_err_end"] = float(np.mean(np.abs(eh[tail] - res["eta"][tail])))
    return out
