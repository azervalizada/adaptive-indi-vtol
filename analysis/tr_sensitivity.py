"""Stage 4 with the sensitivity variants of src/qp_variants.py (eta_min 0.1, no aero model).
    python analysis/tr_sensitivity.py   ->  results/tr_sens_runs.csv, results/tr_sens_summary.txt  (~70 min on 2 cores)"""
import csv
import sys
from multiprocessing import Pool
sys.path.insert(0, ".")
from src.tr_scenarios import tr_grid
from src import qp_variants
from run_tr_benchmark import load, gap_closed, KEYS, _job


if __name__ == "__main__":
    names = qp_variants.RUN
    with Pool() as p:
        new = p.map(_job, [(n, s) for s in tr_grid() for n in names])
    for r in new:
        r["set"] = "Transition grid"
    with open("results/tr_sens_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore"); w.writeheader(); w.writerows(new)
    rows = [r for r in load() if r["set"] == "Transition grid"] + new
    lines = ["Stage 4 transition grid, sensitivity variants next to the main controllers", ""]
    for n in ("rls", "rls_eta0", "rls_noaero", "mrac", "mrac_eta01", "l1"):
        x = [r for r in rows if r["controller"] == n]
        g = gap_closed(rows, n)
        gs = "-" if g is None else f"{g[0]:5.0f}% [{g[1]:.0f}, {g[2]:.0f}] (n={g[3]})"
        lines.append(f"{n:12s} completed {sum(r['completed'] for r in x):3d}  lost {sum(r['lost'] for r in x):2d}"
                     f"  errors {sum(1 for r in x if r.get('error')):d}  gap closed {gs}")
    text = "\n".join(lines)
    print(text)
    open("results/tr_sens_summary.txt", "w").write(text + "\n")
