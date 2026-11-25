"""Stage 4 benchmark: rotor faults during the forward transition (src/tr_scenarios.py).

    python run_tr_benchmark.py            # 3 nominal + 84 faulty transitions x 5 controllers, ~45 min on 2 cores
    python run_tr_benchmark.py --from-csv

Outputs: results/tr_runs.csv, results/tr_summary.txt
Tables:
  * overall: completed / survived / lost counts, mean tilt over the scenarios every controller survived,
    gap closed (as in Stage 3) with bootstrap CI
  * outcome map per eta and acceleration: for each fault speed V_f, the outcome for the front and the
    rear rotor (C completed, s survived but not recovered, L lost) and the mean peak tilt
  * "outrun" table: lowest fault speed from which every faster fault speed is completed, per
    acceleration, next to the Stage 3 constant-speed minimum (results/qp_runs.csv)
"""
import argparse
import csv
import os
import time
import numpy as np

from src.qp_controllers import QP_CONTROLLERS, run_tr, error_row
from src.tr_scenarios import tr_grid, tr_nominal, ACCELS, FAULT_SPEEDS, ETAS, ROTORS
from run_qp_benchmark import gap_closed, by_scenario, lowest_from_which_all

OUT = "results"
METRICS = ["tilt_peak_deg", "tilt_final_deg", "heading_final_deg", "alt_dev_m", "rotor_sat_frac",
           "surf_sat_frac", "final_speed_err", "healthy_dev", "eta_err_end"]
KEYS = (["set", "controller", "scenario", "accel", "V_f", "rotor", "eta", "lost", "recovered", "completed"]
        + METRICS + ["error"])


def _job(job):
    try:
        return run_tr(*job)
    except Exception as e:                 # one failing run must not lose all others; reported below
        name, sc = job
        out = error_row(name, sc, e)      # counted as lost (review M4: was recorded as survived)
        out.update(accel=sc.accel, V_f=sc.V_f, rotor=None if sc.fault is None else sc.rotor + 1,
                   eta=None if sc.fault is None else sc.eta)
        return out


def load():
    rows = []
    for r in csv.DictReader(open(f"{OUT}/tr_runs.csv")):
        for k in METRICS + ["accel", "V_f", "rotor", "eta"]:
            r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
        for k in ("lost", "recovered", "completed"):
            r[k] = r[k] == "True"
        if r.get("error"):                 # review M4
            r["lost"], r["recovered"], r["completed"] = True, False, False
        rows.append(r)
    return rows


def outcome(r):
    return "L" if r["lost"] else ("C" if r["completed"] else "s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    a = ap.parse_args()
    names = list(QP_CONTROLLERS)
    if a.from_csv:
        rows = load()
    else:
        from multiprocessing import Pool
        sets = {"Nominal transition": tr_nominal(), "Transition grid": tr_grid()}
        rows, t = [], time.time()
        for title, scs in sets.items():
            with Pool() as pool:
                out = pool.map(_job, [(c, s) for s in scs for c in names])
            for r in out:
                r["set"] = title
                rows.append(r)
            print(f"{title}: {len(out)} runs ({time.time() - t:.0f} s)", flush=True)
            for r in out:
                if "error" in r:
                    print("  ERROR:", r["controller"], r["scenario"], r["error"], flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(f"{OUT}/tr_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    L = []
    for title in ("Nominal transition", "Transition grid"):
        rr = [r for r in rows if r["set"] == title]
        by = by_scenario(rr)
        # a missing controller row drops the scenario from the common set and is counted (second review m6/m9:
        # it raised KeyError on a partial --from-csv)
        common = [s for s, d in by.items() if all(n in d and not d[n]["lost"] for n in names)]
        nerr = sum(1 for r in rr if r.get("error"))
        nmiss = sum(1 for d in by.values() for n in names if n not in d)
        L += [f"{title} [{len(by)} scenarios, {len(common)} survived by all]"
              + (f"   !! {nerr} runs ERRORED, counted as lost" if nerr else "")
              + (f"   !! {nmiss} runs MISSING" if nmiss else ""), "",
              f"{'controller':<10}{'compl':>7}{'surv_nc':>8}{'lost':>6}{'tilt_peak':>11}{'tilt_final':>11}{'alt_dev':>9}"
              f"{'speed_err':>10}{'healthy_dev':>12}{'eta_err':>9}   gap closed [95% CI] (own set)   | common set"]
        for n in names:
            x = [by[s][n] for s in by if n in by[s]]
            mean = lambda k: np.mean([by[s][n][k] for s in common if by[s][n].get(k) is not None] or [np.nan])
            g = gap_closed(rr, n) if n not in ("baseline", "oracle") else None
            gc = gap_closed(rr, n, among=names) if n not in ("baseline", "oracle") else None
            gs = "-" if g is None else f"{g[0]:5.0f}%  [{g[1]:.0f}, {g[2]:.0f}] (n={g[3]})"
            gcs = "" if gc is None else f"   | {gc[0]:5.0f}%  [{gc[1]:.0f}, {gc[2]:.0f}] (n={gc[3]})"
            L.append(f"{n:<10}{sum(r['completed'] for r in x):>7}{sum(not r['lost'] and not r['completed'] for r in x):>8}"
                     f"{sum(r['lost'] for r in x):>6}{mean('tilt_peak_deg'):11.3f}{mean('tilt_final_deg'):11.3f}"
                     f"{mean('alt_dev_m'):9.3f}{mean('final_speed_err'):10.3f}{mean('healthy_dev'):12.3f}"
                     f"{mean('eta_err_end'):9.3f}   {gs}{gcs}")
        L.append("")

    grid = [r for r in rows if r["set"] == "Transition grid"]
    L.append("Outcome map: per fault speed V_f, outcome for front rotor / rear rotor (C completed, s survived "
             "not recovered, L lost) and mean peak tilt [deg] of the survived runs")
    for eta in ETAS:
        for acc in ACCELS:
            L += ["", f"eta = {eta}, accel = {acc:g} m/s2",
                  f"{'controller':<10}" + "".join(f"{f'V_f {v:g}':>14}" for v in FAULT_SPEEDS)]
            for n in names:
                cells = []
                for v in FAULT_SPEEDS:
                    rr = sorted([r for r in grid if r["controller"] == n and r["eta"] == eta and r["accel"] == acc
                                 and r["V_f"] == v], key=lambda r: r["rotor"])
                    tp = [r["tilt_peak_deg"] for r in rr if not r["lost"]]
                    cells.append(f"{''.join(outcome(r) for r in rr):>4} {(np.mean(tp) if tp else float('nan')):6.1f}   ")
                L.append(f"{n:<10}" + "".join(cells))

    L += ["", "Outrun: lowest fault speed V_f from which every faster V_f COMPLETES the transition "
          "(front rotor / rear rotor; '?' = missing/errored run),",
          "next to the SAME controller's Stage 3 constant-speed minimum safe airspeed (RECOVERED in the 4 s "
          "window). Criteria differ: completed also needs the final speed within 1 m/s and its window runs to",
          "the end of the transition (review m5, 28 Oct 2026: before, every controller was compared with the "
          "baseline's Stage 3 value)."]
    st3 = {}
    if os.path.exists(f"{OUT}/qp_runs.csv"):
        from run_qp_benchmark import load_csv
        sw = [r for r in load_csv() if r["set"] == "Speed sweep"]
        for n in names:
            for eta in ETAS:
                for rot in ROTORS:
                    rr = [r for r in sw if r["controller"] == n and r["eta"] == eta and r["rotor"] == rot + 1]
                    if not rr:
                        continue
                    vmin, complete = lowest_from_which_all(rr, "V", lambda r: r["recovered"])
                    st3[(n, eta, rot)] = ("-" if vmin is None else f"{vmin:g}") + ("" if complete else "?")
    for eta in ETAS:
        L.append(f"  eta = {eta}")
        L.append("    Stage 3 const.:  " + ",  ".join(
            f"{n}: {st3.get((n, eta, 0), 'n/a')}/{st3.get((n, eta, 1), 'n/a')}" for n in names))
        for acc in ACCELS:
            cells = []
            for n in names:
                parts = []
                for rot in ROTORS:
                    rr = [r for r in grid if r["controller"] == n and r["eta"] == eta and r["accel"] == acc
                          and r["rotor"] == rot + 1]
                    vmin, complete = lowest_from_which_all(rr, "V_f", lambda r: r["completed"], FAULT_SPEEDS)
                    parts.append((">" if vmin is None else f"{vmin:g}") + ("" if complete else "?"))
                cells.append(f"{n}: {'/'.join(parts)}")
            L.append(f"    accel {acc:g}:       " + ",  ".join(cells))
    text = "\n".join(L)
    print(text)
    open(f"{OUT}/tr_summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
