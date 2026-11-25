"""Stage 2a figure: what the RLS estimate does after a real fault and after a gust.

Left:  rotor 1 loses 35 % (eta 0.65, abrupt) at 6.0 s.
Right: healthy vehicle, 0.1 Nm pitch gust for 0.5 s at 6.0 s.
Top:   RLS estimate of each rotor's effectiveness (dashed = truth).
Bottom: attitude error of baseline, RLS and oracle INDI.

    python run_stage2_rls.py   ->  results/stage2_rls_estimates.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scenarios import Scenario, Fault, Gust
from src.controllers import run_scenario

ROTOR_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CTRL_C = {"baseline": "#8a8a85", "rls": "#2a78d6", "oracle": "#eb6834"}
INK, INK2 = "#1f1f1e", "#5f5e5a"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "legend.frameon": False,
})

CASES = [("A. Real fault: rotor 1 at 65 %", Scenario("fig", Fault(0, 0.65, 6.0))),
         ("B. Gust on a healthy vehicle: 0.1 Nm pitch, 0.5 s", Scenario("fig", gust=Gust(6.0, (0.0, 0.1, 0.0))))]
T0, T1 = 5.9, 7.0


def main():
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    fig.suptitle("RLS finds the damaged rotor within ~50 ms, but a gust makes it report a fault "
                 "that does not exist", x=0.02, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    for col, (title, sc) in enumerate(CASES):
        tr = {c: run_scenario(c, sc, keep_trace=True)["trace"] for c in ("baseline", "rls", "oracle")}
        t = tr["rls"]["t"]
        m = (t >= T0) & (t <= T1)
        a = ax[0, col]
        for i in range(4):
            a.plot(t[m], tr["rls"]["eta_hat"][m, i], color=ROTOR_C[i], label=f"rotor {i + 1}")
            a.plot(t[m], tr["rls"]["eta"][m, i], color=ROTOR_C[i], lw=1.0, ls="--")
        a.set_ylim(0, 1.6)
        a.set_title(title, loc="left", fontsize=10, color=INK)
        a.set_ylabel("estimated effectiveness [-]")
        if sc.gust is not None:
            a.axvspan(sc.gust.t0, sc.gust.t0 + sc.gust.duration, color="#f6f5f1", zorder=0)
            a.text(sc.gust.t0 + 0.02, 1.5, "gust on", color=INK2, fontsize=9, va="top")
        b = ax[1, col]
        for c in ("baseline", "oracle", "rls"):
            b.plot(t[m], tr[c]["att_err_deg"][m], color=CTRL_C[c], label=f"{c} INDI" if c != "rls" else "RLS INDI")
        b.set_ylabel("attitude error [deg]")
        b.set_xlabel("time [s]")
        b.set_ylim(bottom=0)
        if sc.gust is not None:
            b.axvspan(sc.gust.t0, sc.gust.t0 + sc.gust.duration, color="#f6f5f1", zorder=0)
    ax[0, 0].legend(loc="upper right", ncol=2, fontsize=9)
    ax[0, 0].text(6.45, 0.3, "solid = RLS estimate\ndashed = true value", color=INK2, fontsize=9)
    ax[1, 0].legend(loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig("results/stage2_rls_estimates.png", dpi=160)


if __name__ == "__main__":
    main()
