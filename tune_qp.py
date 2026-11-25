"""Tune an adaptive controller for the QUADPLANE on the Stage 3 tuning set (never on the benchmark).

    python tune_qp.py rls | mrac | l1        main results           (~20-35 min each on 2 cores)
    python tune_qp.py rls --eta-min 0        sensitivity: lower bound of eta_hat 0 (main RLS: 0.1 since
                                             28 Oct 2026); python tune_qp.py mrac --eta-min 0.1 likewise
    python tune_qp.py rls --no-aero          sensitivity (review M3): estimator without the nominal
                                             aero model (aero_model=False), also for mrac

Rule (the Stage 2 rule, same for every method):
  lowest mean tilt_peak_deg over the faulty tuning scenarios, subject to
  (a) no loss of control in any tuning scenario,
  (b) fault-free tilt_rms_deg <= 1.05 x baseline + 0.01 deg,
  (c) healthy_dev <= 0.15 in every tuning scenario (only for methods that estimate eta).
  (b) has an absolute 0.01 deg tolerance on the quadplane (added 27 Oct): its fault-free noise floor is
  ~0.002 deg (inertia ~50x the quadrotor's), where 5 % is below the run-to-run noise.

Search (changed 28 Oct 2026 after the independent review, finding M5: before, the grids had 72, 72
and 28 candidates and the RLS winner sat on the edge of three of its four axes). Now the SAME
procedure and budget for every method:
  1. coarse grid of exactly 48 candidates (COARSE),
  2. local pattern search from the best admissible candidate on each parameter's ladder (LADDER):
     evaluate the +-1 neighbours on every axis, move to the best admissible one if it is better,
     repeat; stop when no neighbour improves (a local optimum with every neighbour checked) or after
     48 more candidates. A winner at the END of a ladder is a physical bound (lag = 1, gamma_d = 0,
     ...) and is reported as such.
Writes results/tuning_qp_<method><suffix>.csv (column 'stage': coarse / refine), prints the winner.
"""
import csv
import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from src.qp_params import QPParams
from src.qp_indi import QPGains
from src.qp_sim import simulate_qp
from src.qp_scenarios import qp_tuning_set, qp_metrics
from src.qp_controllers import make_qp_plant, run_qp, DT_CTRL
from src.qp_adaptive import (QPRLSINDI, QPRLSSettings, QPMRACINDI, QPMRACSettings,
                             QPL1INDI, QPL1Settings)

# ordered ladders of values; the pattern search moves one step along them. Ends = physical/practical
# bounds (lag >= 1 step, gamma_d >= 0, L1 filter below the 250 Hz Nyquist rate of the 500 Hz loop ...)
LADDER = {
    "rls": {"lag": [1, 2, 3, 5, 7, 10, 15, 20],
            "threshold": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
            "forget": [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 1.0],
            "p0": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "mrac": {"a": [3.0, 10.0, 30.0, 100.0, 300.0],
             "gamma_eta": [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0],
             "gamma_d": [0.0, 3.0, 10.0, 30.0, 100.0, 300.0],
             "gamma_thrust": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    "l1": {"a": [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
           "wc": [5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 57.0, 80.0, 113.0, 160.0, 226.0, 320.0, 453.0,
                  640.0, 905.0, 1280.0]},
}
# 48 coarse candidates per method
COARSE = {
    "rls": {"lag": [1, 5, 10], "threshold": [0.01, 0.05, 0.2, 0.5], "forget": [0.98, 0.995], "p0": [0.01, 0.1]},
    "mrac": {"a": [10.0, 30.0, 100.0], "gamma_eta": [1.0, 4.0, 16.0, 64.0], "gamma_d": [0.0, 30.0],
             "gamma_thrust": [0.03, 0.1]},
    "l1": {"a": [10.0, 30.0, 100.0, 300.0], "wc": [10.0, 20.0, 40.0, 57.0, 80.0, 113.0, 160.0, 226.0, 320.0,
                                                   453.0, 640.0, 905.0]},
}
CLASSES = {"rls": (QPRLSSettings, QPRLSINDI), "mrac": (QPMRACSettings, QPMRACINDI), "l1": (QPL1Settings, QPL1INDI)}
REFINE_BUDGET = 48
# kept for older scripts (run_qp_sensitivity.winner): (settings class, controller class, ladders)
SEARCH = {m: (*CLASSES[m], LADDER[m]) for m in LADDER}


ETA_MIN = None      # set by --eta-min: sensitivity variants only


def run_candidate(job):
    method, values, extra = job
    S, C = CLASSES[method]
    settings = replace(S(), **dict(zip(LADDER[method].keys(), values)), **extra)
    out = []
    for sc in qp_tuning_set():
        plant = make_qp_plant(sc)
        res = simulate_qp(plant, C(QPParams(), QPGains(), DT_CTRL, settings), sc.V, t_end=sc.t_end,
                          fault=sc.fault, gust=sc.gust)
        out.append((sc, qp_metrics(res, sc)))
    return values, out


def score_row(method, keys, values, out, limit_b, stage):
    faulty = [m for sc, m in out if sc.fault is not None]
    nominal = [m for sc, m in out if sc.fault is None][0]
    lost = any(m["lost"] for _, m in out)
    score = np.nan if lost else float(np.mean([m["tilt_peak_deg"] for m in faulty]))
    hd = max((m.get("healthy_dev") or 0.0 for _, m in out if not m["lost"]), default=np.nan)
    ok_a = not lost
    ok_b = ok_a and nominal["tilt_rms_deg"] <= limit_b
    ok_c = ok_a and (method == "l1" or hd <= 0.15)
    return {**dict(zip(keys, values)), "stage": stage, "score": score,
            "nominal_rms": nominal.get("tilt_rms_deg", np.nan), "healthy_dev": hd,
            "ok_a": ok_a, "ok_b": ok_b, "ok_c": ok_c, "ok": bool(ok_a and ok_b and ok_c)}


def main():
    method = sys.argv[1]
    extra, suffix = {}, ""
    if "--eta-min" in sys.argv:
        extra["eta_min"] = float(sys.argv[sys.argv.index("--eta-min") + 1])
        suffix += f"_etamin{extra['eta_min']:g}"
    if "--no-aero" in sys.argv:
        extra["aero_model"] = False
        suffix += "_noaero"
    ladder = LADDER[method]
    keys = list(ladder)
    base_nom = [run_qp("baseline", sc) for sc in qp_tuning_set() if sc.fault is None][0]
    limit_b = 1.05 * base_nom["tilt_rms_deg"] + 0.01
    rows, seen = [], {}

    def evaluate(cands, stage):
        cands = [c for c in dict.fromkeys(cands) if c not in seen]
        with Pool() as pool:
            for values, out in pool.map(run_candidate, [(method, c, extra) for c in cands]):
                r = score_row(method, keys, values, out, limit_b, stage)
                rows.append(r)
                seen[values] = r
        return cands

    evaluate(list(itertools.product(*COARSE[method].values())), "coarse")
    print(f"{method}{suffix}: coarse grid {len(rows)} candidates, {sum(r['ok'] for r in rows)} admissible", flush=True)

    def best():
        ok = [r for r in rows if r["ok"]]
        return min(ok, key=lambda r: r["score"]) if ok else None

    cur, used = best(), 0
    while cur is not None and used < REFINE_BUDGET:
        idx = [ladder[k].index(cur[k]) for k in keys]
        nb = []
        for j, k in enumerate(keys):
            for step in (-1, 1):
                i = idx[j] + step
                if 0 <= i < len(ladder[k]):
                    v = [cur[kk] for kk in keys]
                    v[j] = ladder[k][i]
                    nb.append(tuple(v))
        new = [c for c in dict.fromkeys(nb) if c not in seen][:REFINE_BUDGET - used]
        if not new:
            break
        used += len(evaluate(new, "refine"))
        nxt = best()
        print(f"   refine: +{len(new)} (total refine {used}), best score {nxt['score']:.4f}", flush=True)
        if nxt is cur:
            break
        cur = nxt

    os.makedirs("results", exist_ok=True)
    with open(f"results/tuning_qp_{method}{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sorted([r for r in rows if r["ok"]], key=lambda r: r["score"])
    print(f"{method}{suffix}: {len(rows)} candidates ({used} refine), {len(ok)} pass the rule "
          f"(limit for (b): {limit_b:.4f} deg)")
    for r in ok[:10]:
        print("  ", {k: r[k] for k in keys}, f"score {r['score']:.4f}  healthy_dev {r['healthy_dev']:.3f}  rms {r['nominal_rms']:.4f}")
    if ok:
        win = ok[0]
        print("WINNER:", {k: win[k] for k in keys})
        for k in keys:
            i = ladder[k].index(win[k])
            nbrs = [ladder[k][j] for j in (i - 1, i + 1) if 0 <= j < len(ladder[k])]
            checked = all(tuple(win[kk] if kk != k else v for kk in keys) in seen for v in nbrs)
            where = "ladder end (bound)" if i in (0, len(ladder[k]) - 1) else "interior"
            print(f"   {k} = {win[k]}: {where}, neighbours {'all evaluated' if checked else 'NOT all evaluated (budget)'}")
    free = sorted([r for r in rows if r["ok_a"]], key=lambda r: r["score"])
    if free:
        print("unconstrained best (rule (a) only, reported, not used):", {k: free[0][k] for k in keys},
              f"score {free[0]['score']:.4f} healthy_dev {free[0]['healthy_dev']:.3f}")


if __name__ == "__main__":
    main()
