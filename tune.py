"""Tune an adaptive controller for the QUADROTOR on the TUNING set (never on the benchmark scenarios).

    python tune.py rls | mrac | l1        (~10 min each on 2 cores)

Rule (same for every method, see scenarios.tuning_set):
  lowest mean upset_deg over the faulty tuning scenarios, subject to
  (a) no crash, (b) fault-free att_rms_deg <= 1.05 x baseline, (c) healthy_dev <= 0.15 everywhere
  (healthy_dev only exists for methods that estimate eta; L1 does not, so (c) cannot apply).

Search (changed 30 Oct 2026 to the procedure of tune_qp.py, after the independent review found
unequal budgets and winners on grid edges; before: fixed grids of 72 / 72 / 28 candidates, L1's
winner on the edge of both axes). Same procedure and budget for every method:
  1. coarse grid of exactly 48 candidates (COARSE),
  2. local pattern search from the best admissible candidate along each parameter's ordered ladder
     (LADDER): +-1 step on every axis, move if better, stop when no neighbour improves or after 48
     more candidates. A winner at the end of a ladder is a physical/practical bound. (30 Oct: the
     first run ended at the ladder end for MRAC gamma_thrust = 10 and L1 a = 1, which are not
     physical bounds; those two ladders were extended and the two methods re-tuned.)
Writes results/tuning_<method>.csv (column 'stage': coarse / refine), prints the winner.
"""
import csv
import itertools
import os
import sys
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from src.params import QuadParams, INDIGains
from src.sim import simulate
from src.scenarios import tuning_set, fault_metrics
from src.controllers import run_scenario, make_plant, DT_CTRL
from src.rls import RLSINDI, RLSSettings
from src.mrac import MRACINDI, MRACSettings
from src.l1 import L1INDI, L1Settings

LADDER = {
    "rls": {"lag": [1, 2, 3, 5, 7, 10, 15, 20, 30],
            "threshold": [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1],
            "forget": [0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 1.0],
            "p0": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
    "mrac": {"a": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
             "gamma_eta": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
             "gamma_d": [0.0, 3.0, 10.0, 30.0, 100.0, 300.0],
             "gamma_thrust": [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]},
    # pred_model is fixed to "motor" (chosen on fault-free flight, not tuned); wc up to the Nyquist side
    # of the 500 Hz loop (1571 rad/s)
    "l1": {"a": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
           "wc": [5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 57.0, 80.0, 113.0, 160.0, 226.0, 320.0, 453.0,
                  640.0, 905.0, 1280.0]},
}
COARSE = {
    "rls": {"lag": [1, 5, 10], "threshold": [0.0, 0.005, 0.01, 0.02], "forget": [0.98, 0.995], "p0": [0.01, 0.1]},
    "mrac": {"a": [10.0, 30.0, 100.0], "gamma_eta": [0.3, 1.0, 3.0, 10.0], "gamma_d": [0.0, 30.0],
             "gamma_thrust": [0.5, 2.0]},
    "l1": {"a": [3.0, 10.0, 30.0, 100.0], "wc": [10.0, 20.0, 40.0, 57.0, 80.0, 113.0, 160.0, 226.0, 320.0,
                                                 453.0, 640.0, 905.0]},
}
CLASSES = {"rls": (RLSSettings, RLSINDI), "mrac": (MRACSettings, MRACINDI), "l1": (L1Settings, L1INDI)}
REFINE_BUDGET = 48
SEARCH = {m: (*CLASSES[m], LADDER[m]) for m in LADDER}     # kept for older scripts


def run_candidate(job):
    method, values = job
    S, C = CLASSES[method]
    settings = replace(S(), **dict(zip(LADDER[method].keys(), values)))
    out = []
    for sc in tuning_set():
        plant = make_plant(sc)
        res = simulate(plant, C(QuadParams(), INDIGains(), DT_CTRL, settings), t_end=sc.t_end,
                       fault=sc.fault, gust=sc.gust)
        out.append((sc, fault_metrics(res, sc)))
    return values, out


def score_row(method, keys, values, out, limit_b, stage):
    faulty = [m for sc, m in out if sc.fault is not None]
    nom = [m for sc, m in out if sc.fault is None][0]
    crashed = any(m["crashed"] for _, m in out)
    hd = max(m.get("healthy_dev", 0.0) for _, m in out) if not crashed else np.nan
    ok_a = not crashed
    ok_b = ok_a and nom["att_rms_deg"] <= limit_b
    ok_c = ok_a and (method == "l1" or hd <= 0.15)
    return {**dict(zip(keys, values)), "stage": stage,
            "score": np.nan if crashed else float(np.mean([m["upset_deg"] for m in faulty])),
            "nominal_rms": nom["att_rms_deg"], "healthy_dev": hd,
            "eta_err_end": float(np.mean([m.get("eta_err_end", np.nan) for m in faulty])),
            "ok_a": ok_a, "ok_b": ok_b, "ok_c": ok_c, "ok": bool(ok_a and ok_b and ok_c)}


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "rls"
    ladder = LADDER[method]
    keys = list(ladder)
    nominal = [sc for sc in tuning_set() if sc.fault is None][0]
    limit_b = 1.05 * run_scenario("baseline", nominal)["att_rms_deg"]
    base_faulty = np.mean([run_scenario("baseline", sc)["upset_deg"] for sc in tuning_set() if sc.fault])
    rows, seen = [], {}

    def evaluate(cands, stage):
        cands = [c for c in dict.fromkeys(cands) if c not in seen]
        with Pool() as pool:
            for values, out in pool.map(run_candidate, [(method, c) for c in cands]):
                r = score_row(method, keys, values, out, limit_b, stage)
                rows.append(r)
                seen[values] = r
        return cands

    def best():
        ok = [r for r in rows if r["ok"]]
        return min(ok, key=lambda r: r["score"]) if ok else None

    evaluate(list(itertools.product(*COARSE[method].values())), "coarse")
    print(f"{method}: coarse grid {len(rows)} candidates, {sum(r['ok'] for r in rows)} admissible", flush=True)
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
    with open(f"results/tuning_{method}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    ok = sorted([r for r in rows if r["ok"]], key=lambda r: r["score"])
    print(f"baseline on tuning set: mean upset {base_faulty:.3f} deg; limit for (b): {limit_b:.4f} deg")
    print(f"{method}: {len(rows)} candidates ({used} refine), {len(ok)} pass the rule")
    for r in ok[:10]:
        print("  ", {k: r[k] for k in keys}, f"upset {r['score']:.4f}  healthy_dev {r['healthy_dev']:.3f}  rms {r['nominal_rms']:.4f}")
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
