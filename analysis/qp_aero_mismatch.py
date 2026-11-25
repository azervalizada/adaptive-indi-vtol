"""Aerodynamic-model and air-data mismatch (review M3, 28 Oct 2026). MRAC and RLS use the nominal aero
model with ground velocity as air velocity; here the true aircraft's aero coefficients are off by
up to +-30 % and an unknown constant wind of up to 3 m/s blows from a random direction
(src/qp_scenarios.qp_monte_carlo_aero). Also run: the variants without the aero model.

Two sets:
  * "Healthy, mismatch": 40 fault-free flights (random V 0-12 m/s, random mismatch and wind):
    how far do the estimates drift with no fault (healthy_dev), and does adaptation cost tilt?
  * "Monte Carlo, mismatch": 100 faulty flights (the Stage 3 MC with mismatch and wind on top).
Heading is not judged: in a crosswind the aircraft without a rudder crabs (heading != track) even
when healthy, so 'recovered' is reported but is not meaningful here; tilt and loss are.

    python analysis/qp_aero_mismatch.py  ->  results/qp_aero_mismatch_runs.csv, _summary.txt  (~60 min on 2 cores)
"""
import csv
import sys
from dataclasses import replace
import numpy as np
sys.path.insert(0, ".")
from src.qp_controllers import run_qp_many
from src.qp_scenarios import qp_monte_carlo_aero
from run_qp_benchmark import set_table, KEYS

NAMES = ["baseline", "oracle", "rls", "rls_eta0", "rls_noaero", "mrac", "l1"]   # mrac_noaero: no admissible setting


def healthy_set(n=40):
    return [replace(s, name=f"hm{i:02d}", fault=None) for i, s in enumerate(qp_monte_carlo_aero(n, seed=2029))]


if __name__ == "__main__":
    sets = {"Healthy, mismatch": healthy_set(), "Monte Carlo, mismatch": qp_monte_carlo_aero(100)}
    rows, lines = [], []
    for title, scs in sets.items():
        out = run_qp_many(NAMES, scs)
        for r in out:
            r["set"] = title
        rows += out
        lines += set_table(out, NAMES, title) + [""]
        hd = {n: [r["healthy_dev"] for r in out if r["controller"] == n and r.get("healthy_dev") is not None]
              for n in NAMES}
        lines.append("healthy_dev (estimate drift before the fault / without a fault): median, 90th pct, max, "
                     "share > 0.15")
        for n, v in hd.items():
            if v:
                v = np.array(v)
                lines.append(f"  {n:<12} {np.median(v):6.3f} {np.percentile(v, 90):6.3f} {v.max():6.3f}"
                             f"   {np.mean(v > 0.15) * 100:5.1f} %")
        lines.append("")
        print("\n".join(lines), flush=True)
    with open("results/qp_aero_mismatch_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    open("results/qp_aero_mismatch_summary.txt", "w").write("\n".join(lines) + "\n")
