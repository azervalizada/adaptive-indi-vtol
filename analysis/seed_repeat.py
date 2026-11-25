"""Do the conclusions depend on the noise realisation? (29-30 Oct 2026, open item "multi-seed repeat")
The Stage 3 grid and the Stage 4 transition grid are rerun with noise seeds 1 and 2 (the main results use
seed 0; every controller sees the same noise for a given seed). Settings unchanged.

    python analysis/seed_repeat.py   ->  results/seed_repeat_runs.csv, results/seed_repeat.txt  (~65 min on 2 cores)
"""
import csv
import sys
from dataclasses import replace
from multiprocessing import Pool
sys.path.insert(0, ".")
from src.qp_controllers import QP_CONTROLLERS, run_qp_many
from src.qp_scenarios import qp_grid
from src.tr_scenarios import tr_grid
from run_qp_benchmark import load_csv, gap_closed, KEYS
from run_tr_benchmark import load as tr_load, _job as tr_job, KEYS as TR_KEYS

SEEDS = (1, 2)
NAMES = list(QP_CONTROLLERS)

if __name__ == "__main__":
    # each (set, seed) part is saved as soon as it is done and reused on the next call, so a stopped
    # run only loses the part that was running (4 Nov 2026)
    import os, pickle
    PARTS = "results/seed_repeat_parts"
    os.makedirs(PARTS, exist_ok=True)

    def part(name, seed, compute):
        path = f"{PARTS}/{name}_{seed}.pkl"
        if os.path.exists(path):
            print(f"reuse {path}", flush=True)
            return pickle.load(open(path, "rb"))
        out = compute()
        for r in out:
            r.update(set={"grid": "Grid", "tr": "Transition grid"}[name], seed=seed)
        pickle.dump(out, open(path + ".tmp", "wb")); os.replace(path + ".tmp", path)
        print(f"saved {path}", flush=True)
        return out

    def tr_part(seed):
        with Pool() as p:
            return p.map(tr_job, [(n, replace(s, seed=seed)) for s in tr_grid() for n in NAMES])

    rows = []
    for seed in SEEDS:
        rows += part("grid", seed, lambda: run_qp_many(NAMES, [replace(s, seed=seed) for s in qp_grid()]))
        rows += part("tr", seed, lambda: tr_part(seed))
        print(f"seed {seed} done", flush=True)
    keys = sorted(set(KEYS) | set(TR_KEYS)) + ["seed"]
    with open("results/seed_repeat_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    base = {"Grid": [dict(r, seed=0) for r in load_csv() if r["set"] == "Grid"],
            "Transition grid": [dict(r, seed=0) for r in tr_load() if r["set"] == "Transition grid"]}
    L = ["Gap closed [95 % CI] and losses per noise seed (seed 0 = main results)", ""]
    for st in ("Grid", "Transition grid"):
        L.append(st)
        for n in NAMES:
            cells = []
            for seed in (0,) + SEEDS:
                rr = base[st] if seed == 0 else [r for r in rows if r["set"] == st and r["seed"] == seed]
                lost = sum(r["lost"] for r in rr if r["controller"] == n)
                g = gap_closed(rr, n) if n not in ("baseline", "oracle") else None
                gs = "" if g is None else f"{g[0]:4.0f}% [{g[1]:.0f}, {g[2]:.0f}] "
                cells.append(f"seed {seed}: {gs}lost {lost}")
            L.append(f"  {n:<9} " + "   |   ".join(cells))
        L.append("")
    text = "\n".join(L)
    print(text)
    open("results/seed_repeat.txt", "w").write(text + "\n")
