"""Quadplane: robustness to an UNMODELLED transport delay on all actuator commands (Stage 3).
Same check as run_robustness.py (Stage 2), on the quadplane. None of the controllers knows the delay.

Two sets, each with its own margin (28 Oct 2026, review M2: the 8 m/s set alone gave the baseline
">= 60 ms", but in hover its margin is ~44 ms):
  8 m/s:  fault-free, 1 Nm pitch gust, rotor 1 at 30 %, and rotor 3 at 60 % at 4 m/s
  hover:  fault-free, 1 Nm pitch gust, rotor 1 at 60 %
Delays in 5 ms steps. The margin is an OSCILLATION-THRESHOLD margin (largest delay before the healthy
aircraft's tilt RMS exceeds the threshold, or a loss), not a linear stability margin (review m7). Cell = mean tilt_peak_deg; '~' = fault-free tilt RMS > 0.1 deg (oscillating; the
fault-free noise floor on the quadplane is ~0.003 deg, so 0.1 deg is ~30x it; Stage 2 used 0.5 deg
with a floor of 0.16 deg); LOST = loss of control in any of the four.

    python run_qp_robustness.py     ->  results/qp_robustness_delay.csv / .txt   (~5 min on 2 cores)
"""
import csv
import os
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from src.qp_controllers import QP_CONTROLLERS, run_qp, make_qp_plant, DT_CTRL
from src.qp_params import QPParams
from src.qp_indi import QPGains
from src.qp_adaptive import QPL1INDI, QPL1Settings
from src.qp_sim import simulate_qp
from src.qp_scenarios import qp_metrics
from src.qp_scenarios import QPScenario
from src.scenarios import Fault, Gust

DELAYS_MS = list(range(0, 65, 5))
OSC_DEG = 0.1
SETS = {"8 m/s": [QPScenario("nominal", 8.0),
                  QPScenario("gust", 8.0, gust=Gust(5.0, (0.0, 1.0, 0.0))),
                  QPScenario("fault", 8.0, Fault(0, 0.3, 5.0)),
                  QPScenario("fault", 4.0, Fault(2, 0.6, 5.0))],
        "hover": [QPScenario("nominal", 0.0),
                  QPScenario("gust", 0.0, gust=Gust(5.0, (0.0, 1.0, 0.0))),
                  QPScenario("fault", 0.0, Fault(0, 0.6, 5.0))]}


# other L1 settings, as in Stage 2 (observation after the delay check, not tuning results)
L1_VARIANTS = {"l1_a100": (100.0, 1280.0), "l1_a300": (300.0, 1280.0), "l1_wc320": (30.0, 320.0)}


def run(job):
    name, d, sc, setname = job
    sc = replace(sc, delay_s=d / 1000.0)
    if name in L1_VARIANTS:
        a, wc = L1_VARIANTS[name]
        plant = make_qp_plant(sc)
        ctrl = QPL1INDI(QPParams(), QPGains(), DT_CTRL, replace(QPL1Settings(), a=a, wc=wc))
        res = simulate_qp(plant, ctrl, sc.V, t_end=sc.t_end, fault=sc.fault, gust=sc.gust, delay_s=sc.delay_s)
        r = {"scenario": sc.label(), **qp_metrics(res, sc)}
    else:
        r = run_qp(name, sc)
    return {"set": setname, "controller": name, "delay_ms": d, **r}


def main():
    names = list(QP_CONTROLLERS) + list(L1_VARIANTS)
    jobs = [(n, d, s, k) for k, scs in SETS.items() for n in names for d in DELAYS_MS for s in scs]
    with Pool() as pool:
        rows = pool.map(run, jobs)
    os.makedirs("results", exist_ok=True)
    keys = ["set", "controller", "delay_ms", "scenario", "lost", "tilt_peak_deg", "tilt_rms_deg", "tilt_final_deg"]
    with open("results/qp_robustness_delay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["Quadplane, unmodelled actuator delay. Cell = mean tilt_peak_deg over the set;",
             f"'~' = fault-free tilt RMS > {OSC_DEG} deg (oscillating), LOST = loss of control.",
             "Margin = largest delay before the first ~ or LOST (oscillation-threshold margin, 5 ms grid)."]
    for k in SETS:
        lines += ["", f"Set: {k}", f"{'controller':<14}" + "".join(f"{d:>6}" for d in DELAYS_MS) + "   margin"]
        for n in names:
            cells, margin, broken = [], None, False
            for d in DELAYS_MS:
                rr = [r for r in rows if r["set"] == k and r["controller"] == n and r["delay_ms"] == d]
                if any(r["lost"] for r in rr):
                    cells.append(f"{'LOST':>6}")
                    broken = True
                    continue
                nom = [r for r in rr if "no fault" in r["scenario"]][0]
                osc = nom["tilt_rms_deg"] > OSC_DEG
                cells.append(f"{np.mean([r['tilt_peak_deg'] for r in rr]):5.2f}{'~' if osc else ' '}")
                if osc:
                    broken = True
                elif not broken:
                    margin = d
            lines.append(f"{n:<14}" + "".join(cells) + f"   {'>=' if not broken else ''}{margin} ms")
    text = "\n".join(lines)
    print(text)
    open("results/qp_robustness_delay.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
