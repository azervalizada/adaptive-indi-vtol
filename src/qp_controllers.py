"""Registry of quadplane controllers and the function that runs one Stage 3 scenario.
Same idea as controllers.py: one line per controller, every script runs them identically."""
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from .qp_params import QPParams
from .quadplane import Quadplane
from .qp_indi import QPINDIController, QPOracleINDI, QPGains
from .qp_sim import simulate_qp
from .qp_scenarios import qp_metrics

DT_CTRL = 0.002

QP_CONTROLLERS = {
    "baseline": lambda pn, g, dt, plant: QPINDIController(pn, g, dt),
    "oracle":   lambda pn, g, dt, plant: QPOracleINDI(pn, g, dt, plant),
}


def make_qp_plant(sc):
    """True aircraft = nominal parameters x the scenario's model mismatch."""
    n = QPParams()
    true = replace(n, mass=n.mass * sc.mass_scale, J=n.J @ np.diag(sc.J_scale),
                   C_ell_delta_a=n.C_ell_delta_a * sc.surf_scale[0],
                   C_n_delta_a=n.C_n_delta_a * sc.surf_scale[0],
                   C_m_delta_e=n.C_m_delta_e * sc.surf_scale[1],
                   tau_motor=n.tau_motor * getattr(sc, "tau_scale", 1.0))
    if getattr(sc, "aero_scale", ()):                       # review M3: aero model mismatch
        true = replace(true, **{k: getattr(n, k) * f for k, f in sc.aero_scale})
    k = getattr(sc, "noise_scale", 1.0)
    plant = Quadplane(true, seed=sc.seed, gyro_noise_std=0.002 * k, acc_noise_std=0.05 * k,
                      airspeed_noise_std=0.2 * k)
    plant.wind_ned = np.array(getattr(sc, "wind", (0.0, 0.0, 0.0)), dtype=float)
    return plant


def controller_factory(name):
    """Main controllers (QP_CONTROLLERS) and the sensitivity variants (src/qp_variants.py)."""
    if name in QP_CONTROLLERS:
        return QP_CONTROLLERS[name]
    from . import qp_variants
    return qp_variants.make(name)


def run_qp(name, sc, keep_trace=False):
    plant = make_qp_plant(sc)
    ctrl = controller_factory(name)(QPParams(), QPGains(), DT_CTRL, plant)
    res = simulate_qp(plant, ctrl, sc.V, t_end=sc.t_end, fault=sc.fault, gust=sc.gust, delay_s=sc.delay_s)
    out = {"controller": name, "scenario": sc.label(), "V": sc.V,
           "rotor": None if sc.fault is None else sc.fault.rotor + 1,
           "eta": None if sc.fault is None else sc.fault.eta, **qp_metrics(res, sc)}
    if keep_trace:
        out["trace"] = res
    return out


def error_row(name, sc, e):
    """A run that raised: counted as LOST (conservative) and flagged (review M4, 28 Oct 2026: errored
    transitions used to be recorded as survived and the error text was dropped from the CSV)."""
    return {"controller": name, "scenario": sc.label(), "lost": True, "recovered": False,
            "completed": False, "error": repr(e)}


def _job(job):
    try:
        return run_qp(*job)
    except Exception as e:                 # one failing run must not lose all others
        out = error_row(*job, e)
        out.update(V=job[1].V, rotor=None if job[1].fault is None else job[1].fault.rotor + 1,
                   eta=None if job[1].fault is None else job[1].fault.eta)
        return out


def run_qp_many(names, scenarios, processes=None):
    jobs = [(c, s) for s in scenarios for c in names]
    with Pool(processes) as pool:
        return pool.map(_job, jobs)


# adaptive controllers (Stage 3d); settings come from tune_qp.py on the tuning set
from .qp_adaptive import QPRLSINDI, QPMRACINDI, QPL1INDI   # noqa: E402

QP_CONTROLLERS["rls"] = lambda pn, g, dt, plant: QPRLSINDI(pn, g, dt)
QP_CONTROLLERS["mrac"] = lambda pn, g, dt, plant: QPMRACINDI(pn, g, dt)
QP_CONTROLLERS["l1"] = lambda pn, g, dt, plant: QPL1INDI(pn, g, dt)


def run_tr(name, sc, keep_trace=False):
    """Stage 4: fly one transition scenario (tr_scenarios.TRScenario) with one controller."""
    from .tr_scenarios import transition_reference, tr_metrics
    plant = make_qp_plant(sc)
    ctrl = controller_factory(name)(QPParams(), QPGains(), DT_CTRL, plant)
    res = simulate_qp(plant, ctrl, 0.0, t_end=sc.t_end, fault=sc.fault, delay_s=sc.delay_s,
                      reference=transition_reference(sc.accel))
    out = {"controller": name, "scenario": sc.label(), "accel": sc.accel, "V_f": sc.V_f,
           "rotor": None if sc.fault is None else sc.rotor + 1,      # review m17: was 1 without a fault
           "eta": None if sc.fault is None else sc.eta,
           **tr_metrics(res, sc)}
    if keep_trace:
        out["trace"] = res
    return out
