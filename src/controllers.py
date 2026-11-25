"""Registry of controllers and the function that runs one scenario.

To add a controller (e.g. RLS in Stage 2): write its class, then add ONE line to CONTROLLERS.
Every benchmark script picks it up from here, so all controllers are run in exactly the same way.
"""
import numpy as np
from dataclasses import replace

from .params import QuadParams, INDIGains
from .quadrotor import Quadrotor
from .indi import INDIController
from .oracle import OracleINDI
from .rls import RLSINDI
from .mrac import MRACINDI
from .l1 import L1INDI
from .sim import simulate
from .scenarios import fault_metrics

DT_CTRL = 0.002

# name -> function(params_nominal, gains, dt, plant) -> controller
CONTROLLERS = {
    "baseline": lambda pn, g, dt, plant: INDIController(pn, g, dt, thrust_feedback="accel"),
    "rls":      lambda pn, g, dt, plant: RLSINDI(pn, g, dt),
    "mrac":     lambda pn, g, dt, plant: MRACINDI(pn, g, dt),
    "l1":       lambda pn, g, dt, plant: L1INDI(pn, g, dt),
    "oracle":   lambda pn, g, dt, plant: OracleINDI(pn, g, dt, plant, thrust_feedback="accel"),
}


def make_plant(scenario):
    """True vehicle: the nominal parameters scaled by the scenario's model mismatch."""
    nominal = QuadParams()
    true = replace(nominal,
                   mass=nominal.mass * scenario.mass_scale,
                   J=nominal.J @ np.diag(scenario.J_scale))
    return Quadrotor(true, seed=scenario.seed)


def run_scenario(controller_name, scenario, keep_trace=False):
    """Fly one scenario with one controller. Returns the metrics (and the full log if asked)."""
    plant = make_plant(scenario)
    ctrl = CONTROLLERS[controller_name](QuadParams(), INDIGains(), DT_CTRL, plant)
    res = simulate(plant, ctrl, t_end=scenario.t_end, fault=scenario.fault, gust=scenario.gust,
                   delay_s=scenario.delay_s)
    out = {"controller": controller_name, "scenario": scenario.label(), **fault_metrics(res, scenario)}
    if keep_trace:
        out["trace"] = res
    return out


def _run_job(job):
    return run_scenario(*job)


def run_many(controller_names, scenarios, processes=None):
    """All controllers x all scenarios, one simulation per CPU core at a time."""
    from multiprocessing import Pool
    jobs = [(c, s) for s in scenarios for c in controller_names]
    with Pool(processes) as pool:
        return pool.map(_run_job, jobs)
