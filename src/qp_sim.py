"""Simulation loop for the quadplane (Stage 3). Same clocks and delay buffer as sim.py.

Flight: starts trimmed in level flight north at airspeed V (V = 0: hover) at 20 m altitude and
keeps flying straight and level at V. The reference moves with the aircraft:
    pos_ref(t) = (V t, 0, -20),  vel_ref = (V, 0, 0),  yaw 0
so every scenario is "steady flight at speed V", and a fault or gust is the only event.
"""
import numpy as np
from .rotations import quat_mult, quat_conj, quat_to_euler


def _zaxis(q):
    """Body z axis in NED = third column of R(q)."""
    w, x, y, z = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def steady_reference(V, z=-20.0):
    def ref(t):
        return {"pos": np.array([V * t, 0.0, z]), "vel": np.array([V, 0.0, 0.0]), "yaw": 0.0}
    return ref


def simulate_qp(plant, controller, V, t_end=9.0, dt_plant=0.001, ctrl_every=2, fault=None,
                gust=None, delay_s=0.0, reference=None):
    """reference=None: steady flight at V (Stage 3). Otherwise a function t -> {pos, vel, yaw[, acc]};
    the flight then starts trimmed at V (Stage 4: V = 0, hover) and follows it."""
    n = int(round(t_end / dt_plant))
    x, u = plant.trim(V)
    if reference is None:
        reference = steady_reference(V)
    fault_end = None if fault is None else fault.t0 + fault.ramp + dt_plant
    n_delay = int(round(delay_s / dt_plant))
    u_hist = [u.copy()] * n_delay
    keys = ["t", "pos", "ref", "euler", "w", "u_cmd", "x_act", "eta", "att_err_deg", "Va"]
    log = {k: [] for k in keys}
    first = True
    for i in range(n):
        t = i * dt_plant
        if fault is not None and fault.t0 - 0.5 * dt_plant <= t <= fault_end:
            plant.set_fault(fault.eta_at(max(t, fault.t0)))
        if gust is not None:
            on = gust.t0 - 0.5 * dt_plant <= t < gust.t0 + gust.duration - 0.5 * dt_plant
            plant.disturbance_moment = np.array(gust.moment) if on else np.zeros(3)
        if i % ctrl_every == 0:
            ref = reference(t)
            if first:                                  # tell the servo model where the trim started
                ref["delta0"] = x[18:20].copy()
                first = False
            meas = plant.sensors(x)
            u = controller.update(meas, ref)
            log["t"].append(t)
            log["pos"].append(x[0:3].copy())
            log["ref"].append(ref["pos"].copy())
            log["euler"].append(quat_to_euler(x[6:10]))
            log["w"].append(x[10:13].copy())
            log["u_cmd"].append(u.copy())
            log["x_act"].append(x[13:20].copy())
            log["eta"].append(plant.eta.copy())
            log["Va"].append(meas["airspeed"])
            q_des = controller.last["q_des"]
            q_e = quat_mult(quat_conj(q_des), x[6:10])
            log["att_err_deg"].append(np.degrees(2.0 * np.arccos(min(1.0, abs(q_e[0])))))
            # tilt error: angle between the desired and the actual body z axis (thrust direction);
            # heading error: yaw difference. Their split is the Stage 3 metric (decision 27 Oct).
            zb, zd = _zaxis(x[6:10]), _zaxis(q_des)
            log.setdefault("tilt_err_deg", []).append(np.degrees(np.arccos(np.clip(zb @ zd, -1.0, 1.0))))
            dyaw = quat_to_euler(x[6:10])[2] - quat_to_euler(q_des)[2]
            log.setdefault("heading_err_deg", []).append(abs(np.degrees((dyaw + np.pi) % (2 * np.pi) - np.pi)))
            if hasattr(controller, "eta_hat"):
                log.setdefault("eta_hat", []).append(controller.eta_hat.copy())
        if n_delay > 0:                            # review m15: without delay the history grew forever
            u_hist.append(u)
            u_applied = u_hist.pop(0)
        else:
            u_applied = u
        x = plant.step(x, u_applied, dt_plant)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[0:3] - reference(t)["pos"]) > 100:
            log["diverged_at"] = t
            break
    out = {k: np.array(v) for k, v in log.items() if k != "diverged_at"}
    out["diverged_at"] = log.get("diverged_at")
    out["T_max"] = plant.p.T_max
    out["delta_max"] = plant.p.delta_max
    return out
