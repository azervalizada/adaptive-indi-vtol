"""Benchmark: every controller in src/controllers.py on the fixed scenarios of src/scenarios.py.

    python run_benchmark.py                 # fault-free + gusts + grid (24 faults), ~2 min
    python run_benchmark.py --mc 100        # ... plus 100 Monte Carlo runs, ~8 min
    python run_benchmark.py -c baseline oracle
    python run_benchmark.py --from-csv      # rebuild the summary from results/benchmark_runs.csv
                                            # without simulating again

Outputs (results/):
  benchmark_runs.csv      one row per (controller, scenario)
  benchmark_summary.txt   the summary table printed below

Summary metric "gap closed" (primary metric upset_deg):
  sum over scenarios of (baseline - method)  /  sum over scenarios of (baseline - oracle)
  0 % = no better than baseline INDI, 100 % = as good as INDI that knows the fault.
  Scenarios where baseline and oracle differ by < 0.1 deg are skipped (nothing to close).
  [95 % CI] = bootstrap over scenarios (4000 resamples, fixed seed).
  Until 27 Oct this was the MEAN of per-scenario ratios, which small denominators inflate
  (RLS on the grid: 160 % that way, 133 % this way); changed after the code review.
"""
import argparse
import csv
import os
import time
import numpy as np

from src.controllers import CONTROLLERS, run_many
from src.scenarios import benchmark_grid, monte_carlo, nominal_check, gust_check

OUT = "results"
METRICS = ["upset_deg", "att_rms_deg", "recovery_s", "alt_dev_m", "sat_frac", "eta_err_end", "healthy_dev"]
TITLES = {
    "Fault-free check": "Fault-free check (5.0-7.5 s)",
    "Gust check": "Gust check: healthy vehicle, 0.5 s step moment at 6.0 s (healthy_dev = false fault detection)",
    "Grid": "Grid: 4 rotors x eta {0.8, 0.65, 0.5} x {abrupt, 1 s ramp}, fault at 6.0 s",
    "Monte Carlo": "Monte Carlo: random fault + model mismatch + noise seed",
}


def gap_closed(rows, name, n_boot=4000):
    """Ratio of sums with a bootstrap 95 % CI. Returns (value, low, high) in %, or NaNs."""
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], {})[r["controller"]] = r
    b, o, m = [], [], []
    for d in by.values():
        if not {"baseline", "oracle", name} <= d.keys():
            continue
        vb, vo, vm = d["baseline"]["upset_deg"], d["oracle"]["upset_deg"], d[name]["upset_deg"]
        if np.isnan([vb, vo, vm]).any() or vb - vo < 0.1:
            continue
        b.append(vb), o.append(vo), m.append(vm)
    if not b:
        return float("nan"), float("nan"), float("nan")
    b, o, m = map(np.array, (b, o, m))
    value = 100.0 * (b - m).sum() / (b - o).sum()
    rng = np.random.default_rng(0)
    idx = rng.integers(len(b), size=(n_boot, len(b)))
    boot = 100.0 * (b[idx] - m[idx]).sum(axis=1) / (b[idx] - o[idx]).sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(value), float(lo), float(hi)


def summarise(rows, controllers, title):
    lines = [title, ""]
    head = f"{'controller':<12}" + "".join(f"{m:>13}" for m in METRICS) + f"{'crashes':>9}  gap closed [95% CI]"
    lines += [head, "-" * len(head)]
    for c in controllers:
        rr = [r for r in rows if r["controller"] == c]
        ok = [r for r in rr if not r["crashed"]]
        vals = "".join(f"{np.mean([r[m] for r in ok]):13.3f}" if ok and not np.isnan(ok[0].get(m, np.nan))
                       else f"{'-':>13}" for m in METRICS)
        g, lo, hi = gap_closed(rows, c)
        gap = "-" if np.isnan(g) else f"{g:4.0f}%  [{lo:.0f}, {hi:.0f}]"
        lines.append(f"{c:<12}{vals}{sum(r['crashed'] for r in rr):9d}  {gap}")
    return "\n".join(lines)


def read_csv(path):
    rows = []
    for r in csv.DictReader(open(path)):
        out = {"set": r["set"], "controller": r["controller"], "scenario": r["scenario"],
               "crashed": r["crashed"] == "True"}
        for m in METRICS:
            out[m] = float(r[m]) if r.get(m) not in (None, "") else np.nan
        rows.append(out)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--controllers", nargs="+", default=list(CONTROLLERS))
    ap.add_argument("--mc", type=int, default=0, help="number of Monte Carlo runs (0 = skip)")
    ap.add_argument("--from-csv", action="store_true", help="rebuild the summary from the last run's CSV")
    args = ap.parse_args()
    for c in args.controllers:
        if c not in CONTROLLERS:
            raise SystemExit(f"unknown controller '{c}', known: {list(CONTROLLERS)}")
    os.makedirs(OUT, exist_ok=True)

    if args.from_csv:
        all_rows = read_csv(os.path.join(OUT, "benchmark_runs.csv"))
        controllers = [c for c in CONTROLLERS if any(r["controller"] == c for r in all_rows)]
        report = []
        for short, title in TITLES.items():
            rows = [r for r in all_rows if r["set"] == short]
            if rows:
                n = len({r["scenario"] for r in rows})
                report.append(summarise(rows, controllers, f"{title} [{n} scenarios]"))
        report.append(f"\n(rebuilt from {OUT}/benchmark_runs.csv, {len(all_rows)} rows)")
    else:
        sets = [("Fault-free check", nominal_check()), ("Gust check", gust_check()), ("Grid", benchmark_grid())]
        if args.mc:
            sets.append(("Monte Carlo", monte_carlo(args.mc)))
        t_start = time.time()
        all_rows, report = [], []
        for short, scenarios in sets:
            rows = run_many(args.controllers, scenarios)
            for r in rows:
                r["set"] = short
            all_rows += rows
            report.append(summarise(rows, args.controllers, f"{TITLES[short]} [{len(scenarios)} scenarios]"))
        report.append(f"\n{len(all_rows)} simulations in {time.time() - t_start:.0f} s")
        keys = ["set", "controller", "scenario", *METRICS, "crashed"]
        with open(os.path.join(OUT, "benchmark_runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)

    text = "\n\n".join(report)
    print(text)
    with open(os.path.join(OUT, "benchmark_summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
