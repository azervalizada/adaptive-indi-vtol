"""Stage 2c figure: L1 against the other controllers, without and with an unmodelled delay.

Left:  attitude error after rotor 1 drops to 65 % (abrupt, 6.0 s), no extra delay.
Right: mean upset over the 4 robustness scenarios vs. extra motor-command delay
       (from results/robustness_delay.csv; run run_robustness.py first).
       Open marker = oscillating on the healthy vehicle, line ends = loss of control.

    python run_stage2_l1.py   ->  results/stage2_l1.png
"""
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scenarios import Scenario, Fault
from src.controllers import run_scenario

C = {"baseline": "#8a8a85", "rls": "#2a78d6", "mrac": "#1baf7a", "l1": "#e87ba4",
     "l1_a300": "#a3346a", "oracle": "#eb6834"}
LABEL = {"baseline": "baseline INDI", "rls": "RLS", "mrac": "MRAC", "l1": "L1 (tuned, a = 10)",
         "l1_a300": "L1 (a = 300)", "oracle": "oracle INDI"}
INK, INK2 = "#1f1f1e", "#5f5e5a"
OSC_DEG = 0.5
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "legend.frameon": False,
})


def main():
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.suptitle("L1 recovers fastest, but the tuned setting loses stability at 20 ms of delay; "
                 "a = 300 keeps the baseline's margin", x=0.02, ha="left", fontsize=11, color=INK,
                 fontweight="bold")

    sc = Scenario("fig", Fault(0, 0.65, 6.0))
    for c in ("baseline", "oracle", "mrac", "rls", "l1"):
        tr = run_scenario(c, sc, keep_trace=True)["trace"]
        m = (tr["t"] >= 5.95) & (tr["t"] <= 6.5)
        ax[0].plot(tr["t"][m], tr["att_err_deg"][m], color=C[c], label=LABEL[c])
    ax[0].set_title("A. Rotor 1 drops to 65 % at 6.0 s (no extra delay)", loc="left", fontsize=10, color=INK)
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("attitude error [deg]")
    ax[0].set_ylim(bottom=0)
    ax[0].legend(loc="upper right", fontsize=9)

    rows = list(csv.DictReader(open("results/robustness_delay.csv")))
    delays = sorted({int(r["delay_ms"]) for r in rows})
    for c in ("baseline", "rls", "mrac", "oracle", "l1", "l1_a300"):
        xs, ys, osc = [], [], []
        for d in delays:
            rr = [r for r in rows if r["controller"] == c and int(r["delay_ms"]) == d]
            if any(r["crashed"] == "True" for r in rr):
                break
            nom = [r for r in rr if "no fault" in r["scenario"]][0]
            xs.append(d)
            ys.append(np.mean([float(r["upset_deg"]) for r in rr]))
            osc.append(float(nom["att_rms_deg"]) > OSC_DEG)
        xs, ys, osc = map(np.array, (xs, ys, osc))
        ax[1].plot(xs, ys, color=C[c], label=LABEL[c], ls="--" if c == "l1_a300" else "-")
        ax[1].plot(xs[~osc], ys[~osc], "o", ms=5, color=C[c])
        ax[1].plot(xs[osc], ys[osc], "o", ms=6, mfc="white", mec=C[c], mew=1.5)
    ax[1].set_ylim(0, 8)
    ax[1].set_title("B. Extra, unmodelled motor-command delay", loc="left", fontsize=10, color=INK)
    ax[1].set_xlabel("extra delay [ms]")
    ax[1].set_ylabel("mean upset, 4 scenarios [deg]")
    ax[1].legend(loc="upper left", fontsize=9)
    ax[1].text(delays[-1], 0.3, "open marker = oscillating when healthy\nline ends = loss of control",
               color=INK2, fontsize=8.5, ha="right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig("results/stage2_l1.png", dpi=160)


if __name__ == "__main__":
    main()
