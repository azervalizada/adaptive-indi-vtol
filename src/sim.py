"""Simulation loop, reference trajectory and simple metrics.

The fixed test scenarios and the metrics used for comparisons live in scenarios.py.
"""
import numpy as np
from .rotations import quat_to_euler, quat_mult, quat_conj


def default_reference(t):
    """Hover at 2 m, step to (2, 1) m at t=3 s, back to origin at t=9 s. NED, so z = -2."""
    if t < 3.0:
        pos = np.array([0.0, 0.0, -2.0])
    elif t < 9.0:
        pos = np.array([2.0, 1.0, -2.0])
    else:
        pos = np.array([0.0, 0.0, -2.0])
    return {"pos": pos, "yaw": 0.0}


def simulate(plant, controller, t_end=14.0, dt_plant=0.001, ctrl_every=2,
             fault_time=None, fault_eta=None, fault=None, gust=None, delay_s=0.0,
             reference=default_reference):
    """Fly the plant with the controller from hover at 2 m.

    Faults, either:
      fault_time + fault_eta   abrupt change of all four effectiveness values (Stage 1 style)
      fault                    a scenarios.Fault (abrupt or ramp on one rotor)
    External disturbance: gust = a scenarios.Gust (step moment on the body)
    delay_s: extra transport delay between the controller's command and the motors, NOT known
             to the controller (robustness check; real systems have a few ms from bus, ESC, ...)
    """
    n = int(round(t_end / dt_plant))
    fault_end = None if fault is None else fault.t0 + fault.ramp + dt_plant
    x = plant.initial_state(pos=(0.0, 0.0, -2.0))
    u = x[13:17].copy()
    n_delay = int(round(delay_s / dt_plant))
    u_hist = [u.copy()] * n_delay                     # commands of the last n_delay plant steps
    log = {k: [] for k in ["t", "pos", "ref", "euler", "w", "u_cmd", "T_motor", "eta", "att_err_deg", "rate_err"]}
    for i in range(n):
        t = i * dt_plant
        if fault_time is not None and abs(t - fault_time) < 0.5 * dt_plant:
            plant.set_fault(fault_eta)
        if fault is not None and fault.t0 - 0.5 * dt_plant <= t <= fault_end:
            plant.set_fault(fault.eta_at(max(t, fault.t0)))
        if gust is not None:
            on = gust.t0 - 0.5 * dt_plant <= t < gust.t0 + gust.duration - 0.5 * dt_plant
            plant.disturbance_moment = np.array(gust.moment) if on else np.zeros(3)
        ref = reference(t)
        if i % ctrl_every == 0:
            u = controller.update(plant.sensors(x), ref)
            log["t"].append(t)
            log["pos"].append(x[0:3].copy())
            log["ref"].append(ref["pos"].copy())
            log["euler"].append(quat_to_euler(x[6:10]))
            log["w"].append(x[10:13].copy())
            log["u_cmd"].append(u.copy())
            log["T_motor"].append(x[13:17].copy())
            log["eta"].append(plant.eta.copy())
            # tracking quality of the inner loops: how far the vehicle is from what the
            # outer loop asked for (attitude), and from what the attitude loop asked for (rate)
            q_e = quat_mult(quat_conj(controller.last["q_des"]), x[6:10])
            log["att_err_deg"].append(np.degrees(2.0 * np.arccos(min(1.0, abs(q_e[0])))))
            log["rate_err"].append(np.linalg.norm(controller.last["w_cmd"] - x[10:13]))
            if hasattr(controller, "eta_hat"):            # adaptive controllers: their estimate
                log.setdefault("eta_hat", []).append(controller.eta_hat.copy())
        u_hist.append(u)
        u_applied = u_hist.pop(0) if n_delay > 0 else u
        x = plant.step(x, u_applied, dt_plant)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[0:3]) > 100:
            log["diverged_at"] = t
            break
    out = {k: np.array(v) for k, v in log.items() if k != "diverged_at"}
    out["diverged_at"] = log.get("diverged_at")
    out["T_max"] = plant.p.T_max
    return out


def metrics(res, t_from):
    m = res["t"] >= t_from
    e = res["pos"][m] - res["ref"][m]
    return {
        "pos_rms_m": float(np.sqrt(np.mean(np.sum(e ** 2, axis=1)))),
        "z_max_dev_m": float(np.max(np.abs(e[:, 2]))),
        "yaw_max_deg": float(np.degrees(np.max(np.abs(res["euler"][m, 2])))),
        "tilt_max_deg": float(np.degrees(np.max(np.abs(res["euler"][m, 0:2])))),
        "att_rms_deg": float(np.sqrt(np.mean(res["att_err_deg"][m] ** 2))),
        "att_max_deg": float(np.max(res["att_err_deg"][m])),
        "rate_rms": float(np.sqrt(np.mean(res["rate_err"][m] ** 2))),
        "sat_frac": float(np.mean(np.any((res["u_cmd"][m] >= res.get("T_max", 6.0) - 1e-3) | (res["u_cmd"][m] <= 1e-6), axis=1))),
        "diverged_at": res["diverged_at"],
    }
