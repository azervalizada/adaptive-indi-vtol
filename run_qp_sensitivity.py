"""Sensitivity variants on the Stage 3 grid and speed sweep (NOT the main results; reported next to them).

Sets: Stage 3 grid, speed sweep and the 100-run Monte Carlo.
Variants (src/qp_variants.py, each tuned with the same rule and procedure as the main controller):
  rls_eta0                 RLS with eta_hat lower bound 0 (main: 0.1, decision of 28 Oct 2026). With 0, an
                           abrupt fault can drive the estimates to 0 within ~50 ms; the allocation then
                           stops using those rotors, their thrust no longer changes, RLS gets no new
                           information and the estimates stay locked at 0.
  mrac_eta01               MRAC with lower bound 0.1 (main MRAC: 0).
  rls_noaero, mrac_noaero  estimators without the nominal aerodynamic model (review M3): how much of
                           their result is owed to model knowledge L1 does not use.

    python run_qp_sensitivity.py   ->  results/qp_sens_runs.csv, results/qp_sens_summary.txt  (~85 min on 2 cores)
    python run_qp_sensitivity.py --from-csv
"""
import csv
import sys

from src.qp_controllers import run_qp_many
from src.qp_scenarios import qp_grid, qp_speed_sweep, qp_monte_carlo
from src import qp_variants
from run_qp_benchmark import load_csv, set_table, by_speed_table, min_speed_table, KEYS, METRICS

OUT = "results/qp_sens_runs.csv"


def main():
    variants = qp_variants.RUN
    sets = {"Grid": qp_grid(), "Speed sweep": qp_speed_sweep(), "Monte Carlo": qp_monte_carlo(100)}
    if "--from-csv" in sys.argv:
        new = []
        for r in csv.DictReader(open(OUT)):
            for k in METRICS + ["V", "eta", "rotor"]:
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
            r["lost"], r["recovered"] = r["lost"] == "True", r["recovered"] == "True"
            if r.get("error"):
                r["lost"], r["recovered"] = True, False
            new.append(r)
    else:
        new = []
        for title, scs in sets.items():
            out = run_qp_many(variants, scs)
            for r in out:
                r["set"] = title
            new += out
            print(f"{title}: {len(out)} runs", flush=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(new)
    lines = ["Sensitivity variants (not the main results). Tuned settings: "
             + "; ".join(f"{n}: {qp_variants.tuned(*qp_variants._SPEC[n][:2])}" for n in variants), ""]
    main_rows = [r for r in load_csv() if r["set"] in sets]
    names = ["baseline", "oracle", "rls", "rls_eta0", "rls_noaero", "mrac", "mrac_eta01", "l1"]
    for title in sets:
        rr = [r for r in main_rows + new if r["set"] == title]
        lines += set_table(rr, names, title) + [""]
        if title == "Grid":
            lines += by_speed_table(rr, names) + [""]
        elif title == "Speed sweep":
            lines += min_speed_table(rr, names) + [""]
    text = "\n".join(lines)
    print(text)
    open("results/qp_sens_summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
