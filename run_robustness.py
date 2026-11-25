"""Robustness to an UNMODELLED transport delay on the motor commands (added 27 Oct 2026).

None of the controllers knows about the delay. Real systems have a few ms between the flight
computer and the motors (bus, ESC, computation). Adaptive schemes that act aggressively can look
excellent in a delay-free simulation and lose their stability margin; L1 theory states this
trade-off explicitly for the filter bandwidth wc.

For every controller and delay: 4 scenarios (fault-free, 0.1 Nm pitch gust, rotor 1 at 65 %
abrupt, rotor 3 at 50 % abrupt). Reported: crashes and mean upset_deg, and the fault-free
attitude RMS (oscillation shows up there first).

    python run_robustness.py     ->  results/robustness_delay.csv, results/robustness_delay.txt
"""
import csv
import os
from dataclasses import replace
from multiprocessing import Pool
import numpy as np

from src.params import QuadParams, INDIGains
from src.sim import simulate
from src.scenarios import Scenario, Fault, Gust, fault_metrics
from src.controllers import CONTROLLERS, make_plant, DT_CTRL
from src.l1 import L1INDI, L1Settings

DELAYS_MS = [0, 5, 10, 15, 18, 20, 22, 25, 30]
OSC_DEG = 0.5      # fault-free attitude RMS above this = oscillating (healthy noise floor is 0.164)
BASE_SCENARIOS = [Scenario("nominal", None),
                  Scenario("gust", gust=Gust(6.0, (0.0, 0.1, 0.0))),
                  Scenario("fault", Fault(0, 0.65, 6.0)),
                  Scenario("fault", Fault(2, 0.5, 6.0))]
# the tuned L1 (a = 10, wc = 320) plus other predictor poles a: a, not wc, sets the delay margin
# (exploratory check of 27 Oct); one narrow-filter case shows that a narrow C(s) does not help
L1_VARIANTS = {"l1_a10": (10.0, 1280.0), "l1_a100": (100.0, 1280.0), "l1_a300": (300.0, 1280.0),
               "l1_wc320": (10.0, 320.0)}   # 30 Oct: re-centred on the re-tuned L1 (a 0.1, wc 1280); l1_wc320 = old setting
NAMES = list(CONTROLLERS) + list(L1_VARIANTS)


def make_controller(name, plant):
    if name in L1_VARIANTS:
        a, wc = L1_VARIANTS[name]
        return L1INDI(QuadParams(), INDIGains(), DT_CTRL, replace(L1Settings(), a=a, wc=wc))
    return CONTROLLERS[name](QuadParams(), INDIGains(), DT_CTRL, plant)


def run(job):
    name, delay_ms, sc = job
    sc = replace(sc, delay_s=delay_ms / 1000.0)
    plant = make_plant(sc)
    res = simulate(plant, make_controller(name, plant), t_end=sc.t_end, fault=sc.fault,
                   gust=sc.gust, delay_s=sc.delay_s)
    m = fault_metrics(res, sc)
    return {"controller": name, "delay_ms": delay_ms, "scenario": sc.label(), **m}


def main():
    jobs = [(n, d, s) for n in NAMES for d in DELAYS_MS for s in BASE_SCENARIOS]
    with Pool() as pool:
        rows = pool.map(run, jobs)
    os.makedirs("results", exist_ok=True)
    keys = ["controller", "delay_ms", "scenario", "upset_deg", "att_rms_deg", "sat_frac", "crashed"]
    with open("results/robustness_delay.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def cell_state(n, d):
        rr = [r for r in rows if r["controller"] == n and r["delay_ms"] == d]
        nom = [r for r in rr if "no fault" in r["scenario"]][0]
        if any(r["crashed"] for r in rr):
            return "LOST", None
        return ("osc" if nom["att_rms_deg"] > OSC_DEG else "ok"), float(np.mean([r["upset_deg"] for r in rr]))

    lines = ["Unmodelled motor-command delay. Cell = mean upset_deg over 4 scenarios;",
             f"'~' = fault-free attitude RMS > {OSC_DEG} deg (oscillating), LOST = loss of control",
             "(diverged or attitude error > 90 deg). Margin = largest delay before the first ~ or LOST.", ""]
    lines.append(f"{'controller':<14}" + "".join(f"{d:>6} ms" for d in DELAYS_MS) + "   margin")
    for n in NAMES:
        cells, margin, broken = [], None, False
        for d in DELAYS_MS:
            state, v = cell_state(n, d)
            cells.append(f"{'LOST':>9}" if state == "LOST" else f"{v:8.2f}{'~' if state == 'osc' else ' '}")
            if state != "ok":
                broken = True
            elif not broken:
                margin = d
        lines.append(f"{n:<14}" + "".join(cells) + f"   {'>=' if not broken else ''}{margin} ms")
    text = "\n".join(lines)
    print(text)
    open("results/robustness_delay.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
