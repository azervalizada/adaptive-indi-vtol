"""Stage 3 benchmark: every quadplane controller on the fixed scenarios of src/qp_scenarios.py.

    python run_qp_benchmark.py                 # nominal + gusts + grid (72), ~10 min on 2 cores
    python run_qp_benchmark.py --sweep         # ... plus the minimum-safe-airspeed sweep, +~20 min
    python run_qp_benchmark.py --mc 100        # ... plus 100 Monte Carlo runs
    python run_qp_benchmark.py -c baseline oracle
    python run_qp_benchmark.py --from-csv      # rebuild the tables without simulating

Outputs: results/qp_runs.csv, results/qp_summary.txt

Tables
  * per set: lost / recovered counts, and means over the scenarios that EVERY controller survived
    (otherwise a controller that loses the hard cases would look better on the means)
  * gap closed on tilt_peak_deg: sum(baseline - method) / sum(baseline - oracle) over scenarios
    survived by all and with baseline - oracle >= 0.1 deg, bootstrap 95 % CI (as in Stage 2)
  * grid by airspeed: mean tilt_peak_deg per speed
  * sweep: minimum safe airspeed per (rotor, eta) = lowest V from which every faster V in the
    sweep is "recovered" (tilt_final < 1 deg, heading_final < 5 deg)
"""
import argparse
import csv
import os
import time
import numpy as np

from src.qp_controllers import QP_CONTROLLERS, run_qp_many
from src.qp_scenarios import LOST_ALT_M, qp_grid, qp_speed_sweep, qp_monte_carlo, qp_nominal_check, qp_gust_check

OUT = "results"
METRICS = ["tilt_peak_deg", "tilt_final_deg", "heading_peak_deg", "heading_final_deg", "alt_dev_m",
           "lat_dev_m", "rotor_sat_frac", "surf_sat_frac", "healthy_dev", "eta_err_end"]
KEYS = ["set", "controller", "scenario", "V", "rotor", "eta", "lost", "recovered"] + METRICS + ["error"]


def by_scenario(rows):
    """{scenario: {controller: row}}, scenarios in sorted order (review m4: the bootstrap must not
    depend on row order)."""
    by = {}
    for r in sorted(rows, key=lambda r: r["scenario"]):
        by.setdefault(r["scenario"], {})[r["controller"]] = r
    return by


def gap_closed(rows, name, metric="tilt_peak_deg", n_boot=4000, among=None):
    """Gap closed of `name` over the scenarios that baseline, oracle and `name` all survive (same rule
    as Stage 2, run_benchmark.py). Changed 28 Oct 2026 (review m2): before, a scenario was dropped if
    ANY controller in `rows` lost it, so the value depended on which other variants were in the file.
    An errored run counts as lost (review M4).
    among: optional list of controllers that must ALL have survived a scenario for it to count (the
    common set; second review m5, 4 Nov 2026: methods are otherwise compared on different sets)."""
    b, o, m = [], [], []
    req = set(("baseline", "oracle", name)) | set(among or ())
    for d in by_scenario(rows).values():
        if not req <= d.keys() or any(d[k]["lost"] for k in req):
            continue
        vb, vo, vm = d["baseline"][metric], d["oracle"][metric], d[name][metric]
        if vb - vo < 0.1:
            continue
        b.append(vb), o.append(vo), m.append(vm)
    if not b:
        return None
    b, o, m = map(np.array, (b, o, m))
    rng = np.random.default_rng(0)
    idx = rng.integers(len(b), size=(n_boot, len(b)))
    boot = 100.0 * (b[idx] - m[idx]).sum(1) / (b[idx] - o[idx]).sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100.0 * (b - m).sum() / (b - o).sum(), lo, hi, len(b)


def set_table(rows, names, title):
    by = by_scenario(rows)
    common = [s for s, d in by.items() if all(n in d and not d[n]["lost"] for n in names)]
    nerr = sum(1 for r in rows if r.get("error"))
    nmiss = sum(1 for d in by.values() for n in names if n not in d)     # second review m9
    lines = [f"{title} [{len(by)} scenarios, {len(common)} survived by all]", ""]
    if nerr:
        lines.insert(1, f"!! {nerr} runs ended with an ERROR and are counted as lost")
    if nmiss:
        lines.insert(1, f"!! {nmiss} (scenario, controller) runs are MISSING; those scenarios drop out")
    cols = ["tilt_peak_deg", "tilt_final_deg", "heading_final_deg", "alt_dev_m", "rotor_sat_frac",
            "surf_sat_frac", "healthy_dev", "eta_err_end"]
    lines.append(f"{'controller':<10}{'lost':>6}{'recov':>7}" + "".join(f"{c[:13]:>15}" for c in cols)
                 + f"{'hd_max':>9}   gap closed [95% CI] (own set)   | common set")
    lines.append("-" * (23 + 15 * len(cols) + 9 + 60))
    for n in names:
        rr = [by[s][n] for s in by if n in by[s]]
        lost = sum(r["lost"] for r in rr)
        rec = sum(r["recovered"] for r in rr)
        cells = []
        for c in cols:
            v = [by[s][n].get(c) for s in common]
            v = [x for x in v if x is not None and x == x]
            cells.append(f"{np.mean(v):15.3f}" if v else f"{'-':>15}")
        # healthy_dev: the column above is the mean over the common set; hd_max the worst run (review m10)
        hds = [r.get("healthy_dev") for r in rr if r.get("healthy_dev") is not None and not r["lost"]]
        hdm = f"{max(hds):9.3f}" if hds else f"{'-':>9}"
        g = gap_closed(rows, n) if n not in ("baseline", "oracle") else None
        gc = gap_closed(rows, n, among=names) if n not in ("baseline", "oracle") else None
        gs = "-" if g is None else f"{g[0]:5.0f}%  [{g[1]:.0f}, {g[2]:.0f}] (n={g[3]})"
        gcs = "" if gc is None else f"   | {gc[0]:5.0f}%  [{gc[1]:.0f}, {gc[2]:.0f}] (n={gc[3]})"
        lines.append(f"{n:<10}{lost:>6}{rec:>7}" + "".join(cells) + hdm + "   " + gs + gcs)
    return lines


def by_speed_table(rows, names):
    """Mean tilt_peak_deg per airspeed over the scenarios at that speed that EVERY listed controller
    survived (n = their number), and the lost count per controller. Fixed 28 Oct 2026 (review C2):
    before, each controller was averaged over its OWN survivors, so a controller that survived more
    hard cases looked worse (hover: baseline 8.30 deg shown, 2.03 deg on the common set)."""
    speeds = sorted({r["V"] for r in rows})
    by = by_scenario(rows)
    commons = {v: [s for s, d in by.items() if all(n in d for n in names) and d[names[0]]["V"] == v
                   and all(not d[n]["lost"] for n in names)] for v in speeds}
    lines = ["Grid by airspeed: mean tilt_peak_deg over the scenarios at that speed survived by ALL "
             "controllers (n); L = lost of the runs at that speed", "",
             f"{'controller':<10}" + "".join(f"{f'{v:g} m/s (n={len(commons[v])})':>16}" for v in speeds)]
    for n in names:
        cells = []
        for v in speeds:
            vals = [by[s][n]["tilt_peak_deg"] for s in commons[v]]
            nl = sum(r["lost"] for r in rows if r["controller"] == n and r["V"] == v)
            cells.append(f"{(np.mean(vals) if vals else float('nan')):11.2f}" + (f" L{nl:<3d}" if nl else "     "))
        lines.append(f"{n:<10}" + "".join(cells))
    return lines


def lowest_from_which_all(rows, xkey, ok, grid=None):
    """Lowest x from which every larger x (on `grid`, default: all x in rows) satisfies ok(row).
    Returns (x or None, complete). A missing or errored x stops the scan and complete=False
    (review m3: missing speeds were skipped silently)."""
    d = {r[xkey]: r for r in rows}
    grid = sorted(grid if grid is not None else d)
    xmin = None
    for x in reversed(grid):
        r = d.get(x)
        if r is None or r.get("error"):
            return xmin, False
        if not ok(r):
            break
        xmin = x
    return xmin, True


def min_speed_table(rows, names):
    lines = ["Minimum safe airspeed [m/s] per (rotor, eta): lowest V from which every faster V is recovered "
             "('?' = a speed is missing or errored)", ""]
    keys = sorted({(r["eta"], r["rotor"]) for r in rows}, key=lambda k: (-k[0], k[1]))
    speeds = sorted({r["V"] for r in rows})
    lines.append(f"{'eta':>5}{'rotor':>7}" + "".join(f"{n:>12}" for n in names))
    for eta, rotor in keys:
        cells = []
        for n in names:
            rr = [r for r in rows if r["controller"] == n and r["eta"] == eta and r["rotor"] == rotor]
            vmin, complete = lowest_from_which_all(rr, "V", lambda r: r["recovered"], speeds)
            txt = ">max" if vmin is None else f"{vmin:g}"
            cells.append(f"{txt + ('' if complete else '?'):>12}")
        lines.append(f"{eta:5.2f}{int(rotor):7d}" + "".join(cells))
    return lines


def load_csv():
    rows = []
    for r in csv.DictReader(open(f"{OUT}/qp_runs.csv")):
        for k in METRICS + ["V", "eta", "rotor"]:
            r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
        r["lost"], r["recovered"] = r["lost"] == "True", r["recovered"] == "True"
        if r.get("error"):                                  # review M4: an errored run counts as lost
            r["lost"], r["recovered"] = True, False
        if r["alt_dev_m"] is not None and r["alt_dev_m"] > LOST_ALT_M:   # rule of 27 Oct, see qp_scenarios
            r["lost"], r["recovered"] = True, False
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--controllers", nargs="+", default=list(QP_CONTROLLERS))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--append", action="store_true",
                    help="keep the rows already in results/qp_runs.csv that this run does not redo")
    ap.add_argument("--only", nargs="+", default=None,
                    help="run only these sets, e.g. --only mc (names: nominal gust grid sweep mc)")
    a = ap.parse_args()
    names = a.controllers
    os.makedirs(OUT, exist_ok=True)
    if a.from_csv:
        rows = load_csv()
    else:
        sets = {"Fault-free check": qp_nominal_check(), "Gust check": qp_gust_check(), "Grid": qp_grid()}
        if a.sweep:
            sets["Speed sweep"] = qp_speed_sweep()
        if a.mc:
            sets["Monte Carlo"] = qp_monte_carlo(a.mc)
        if a.only:
            short = {"nominal": "Fault-free check", "gust": "Gust check", "grid": "Grid",
                     "sweep": "Speed sweep", "mc": "Monte Carlo"}
            sets = {k: v for k, v in sets.items() if k in {short[o] for o in a.only}}
        rows = ([r for r in load_csv() if not (r["controller"] in names and r["set"] in sets)]
                if a.append else [])
        t = time.time()
        for title, scs in sets.items():
            for r in run_qp_many(names, scs):
                r["set"] = title
                rows.append(r)
            print(f"{title}: {len(scs) * len(names)} runs done ({time.time() - t:.0f} s)", flush=True)
        with open(f"{OUT}/qp_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    present = {r["controller"] for r in rows}
    names = [n for n in QP_CONTROLLERS if n in present]
    lines = []
    for title in ("Fault-free check", "Gust check", "Grid", "Speed sweep", "Monte Carlo"):
        rr = [r for r in rows if r["set"] == title]
        if not rr:
            continue
        lines += set_table(rr, names, title) + [""]
        if title == "Grid":
            lines += by_speed_table(rr, names) + [""]
        if title == "Speed sweep":
            lines += min_speed_table(rr, names) + [""]
    text = "\n".join(lines)
    print("\n" + text)
    open(f"{OUT}/qp_summary.txt", "w").write(text + "\n")


if __name__ == "__main__":
    main()
