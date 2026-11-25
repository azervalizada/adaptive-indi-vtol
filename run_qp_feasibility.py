"""Static feasibility (Stage 3, added 27 Oct 2026): can the aircraft with a damaged rotor hold steady,
straight, level flight at airspeed V at all, with actuators within their limits? Independent of any
controller: it asks whether an equilibrium EXISTS, not whether a controller reaches it.
Compared with the simulated minimum safe airspeed (run_qp_benchmark.py --sweep) it shows whether the
limit comes from the physics or from the controller.

    python run_qp_feasibility.py     ->  results/qp_feasibility.txt and .csv (~25 min on 2 cores)

Unknowns: rotor thrusts, delta_a, delta_e, T_push, and the flight condition: sideslip beta and
bank phi (pitch theta fixed at 0 as in the controller, or free). Equations: 3 forces, 3 moments = 0,
and the flight path is level: vertical velocity (NED) = 0 (added 28 Oct 2026, review m6: with both
sideslip and bank non-zero the equilibria found before descended at 0.03-0.2 m/s)."""
import sys
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from src.qp_params import QPParams
from src.quadplane import Quadplane

p = QPParams()
qp = Quadplane(p)


def residual(z, V, dead, eta, theta_free):
    T = z[0:4].copy()
    T[dead] *= eta
    da, de, Tp, beta, phi, alpha = z[4:10]
    theta = z[10] if theta_free else 0.0
    # angle of attack is its own unknown (fixed 28 Oct 2026: it was tied to the pitch, alpha = theta,
    # which together with the level-flight condition over-constrained the problem: at pitch 0 level
    # flight then needed bank or sideslip exactly 0; a simulated equilibrium at 6 m/s was 'infeasible')
    vb = V * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
    Fa, Ma, *_ = qp.aero(vb, np.zeros(3), da, de)
    g_b = p.mass * p.g * np.array([-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)])
    F = Fa + np.array([Tp, 0, -T.sum()]) + g_b
    M = Ma + qp.B @ T + np.array([-p.k_m_push * Tp, 0, 0])
    # vertical (down) velocity of the body-frame velocity vb at roll phi, pitch theta (yaw drops out)
    v_down = (-np.sin(theta) * vb[0] + np.sin(phi) * np.cos(theta) * vb[1]
              + np.cos(phi) * np.cos(theta) * vb[2])
    return np.concatenate([F / p.mass, M / np.diag(p.J), [v_down / max(V, 1.0)]])


def feasible(args):
    V, dead, eta, theta_rng, beta_max = args
    theta_free = theta_rng is not None          # (low, high) pitch bounds, or None for pitch = 0
    lo = np.r_[np.zeros(4), -p.delta_max, -p.delta_max, 0.0, -beta_max, -0.5, -0.4]
    hi = np.r_[np.full(4, p.T_max), p.delta_max, p.delta_max, p.T_push_max, beta_max, 0.5, 0.4]
    if theta_free:
        lo, hi = np.r_[lo, theta_rng[0]], np.r_[hi, theta_rng[1]]
    for s in range(8):
        z0 = np.random.default_rng(s).uniform(lo, hi)
        if theta_free and s % 2 == 0:
            z0[10] = 0.0            # half the starts at pitch 0: a range that contains 0 must be at least as feasible
        sol = least_squares(residual, z0, bounds=(lo, hi), args=(V, dead, eta, theta_free),
                            xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=3000)
        if np.max(np.abs(sol.fun)) < 1e-6:
            return True
    return False


# case 3 added 28 Oct 2026: after the level-flight constraint (review m6), pitch = 0 EXACTLY couples bank
# and sideslip (level flight then needs phi = 0 or beta = 0), which is stricter than the simulation's
# 'recovered' (attitude within 1 deg). |pitch| <= 1 deg matches that tolerance.
CASES = [("pitch 0, |sideslip| <= 5 deg  (controller's pitch reference + 'recovered' heading criterion)", None, np.radians(5.0)),
         ("pitch 0, |sideslip| <= 20 deg (heading allowed to drift)", None, np.radians(20.0)),
         ("pitch free (-11..17 deg), |sideslip| <= 20 deg", (-0.2, 0.3), np.radians(20.0)),
         ("|pitch| <= 1 deg, |sideslip| <= 5 deg (the 1 deg attitude tolerance of 'recovered')",
          (-np.radians(1.0), np.radians(1.0)), np.radians(5.0))]

if __name__ == "__main__":
    import os
    Vs = np.arange(0, 12.01, 0.5)
    lines = ["Static minimum airspeed [m/s] for steady level flight after a rotor fault (0.5 m/s grid);",
             "bank |phi| <= 29 deg, actuators within limits, level flight path. Rotor 1, 4 front; 2, 3 rear.", ""]
    table = []
    for ci, (title, theta_free, bmax) in enumerate(CASES):
        jobs = [(V, d, e, theta_free, bmax) for e in (0.3, 0.15, 0.0) for d in range(4) for V in Vs]
        with Pool() as pool:
            ok = dict(zip(jobs, pool.map(feasible, jobs)))
        lines.append(title)
        for e in (0.3, 0.15, 0.0):
            cells = []
            for d in range(4):
                vmin = None
                for V in Vs[::-1]:
                    if not ok[(V, d, e, theta_free, bmax)]:
                        break
                    vmin = V
                cells.append(f"rotor {d + 1}: {vmin}")
                table.append({"case": ci, "case_title": title, "eta": e, "rotor": d + 1,
                              "v_min": "" if vmin is None else float(vmin)})
            lines.append(f"  eta {e:4.2f}:  " + ",  ".join(cells))
        lines.append("")
        print("\n".join(lines[-5:]), flush=True)
    os.makedirs("results", exist_ok=True)
    open("results/qp_feasibility.txt", "w").write("\n".join(lines) + "\n")
    import csv
    with open("results/qp_feasibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
