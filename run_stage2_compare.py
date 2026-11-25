"""Stage 2 figure: RLS vs MRAC estimates after a real fault and after a gust.

Columns: A) rotor 1 at 65 % (abrupt, 6.0 s)   B) healthy vehicle, 0.1 Nm pitch gust for 0.5 s
Rows:    RLS estimate | MRAC estimate | attitude error of all controllers

    python run_stage2_compare.py   ->  results/stage2_compare.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scenarios import Scenario, Fault, Gust
from src.controllers import run_scenario

ROTOR_C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CTRL_C = {"baseline": "#8a8a85", "rls": "#2a78d6", "mrac": "#1baf7a", "oracle": "#eb6834"}
CTRL_L = {"baseline": "baseline INDI", "rls": "RLS INDI", "mrac": "MRAC INDI", "oracle": "oracle INDI"}
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
    fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.suptitle("RLS reacts fastest but also to gusts and freezes there; MRAC is accurate on average "
                 "but oscillates at ~13 Hz and adapts more slowly", x=0.02, ha="left", fontsize=11, color=INK,
                 fontweight="bold")
    for col, (title, sc) in enumerate(CASES):
        tr = {c: run_scenario(c, sc, keep_trace=True)["trace"] for c in CTRL_C}
        t = tr["rls"]["t"]
        m = (t >= T0) & (t <= T1)
        for row, method in enumerate(("rls", "mrac")):
            a = ax[row, col]
            for i in range(4):
                a.plot(t[m], tr[method]["eta_hat"][m, i], color=ROTOR_C[i], label=f"rotor {i + 1}")
                a.plot(t[m], tr[method]["eta"][m, i], color=ROTOR_C[i], lw=1.0, ls="--")
            a.set_ylim(0, 1.6)
            a.set_ylabel(f"{CTRL_L[method].split()[0]} estimate [-]")
            if row == 0:
                a.set_title(title, loc="left", fontsize=10, color=INK)
        b = ax[2, col]
        for c in ("baseline", "oracle", "mrac", "rls"):
            b.plot(t[m], tr[c]["att_err_deg"][m], color=CTRL_C[c], label=CTRL_L[c])
        b.set_ylabel("attitude error [deg]")
        b.set_xlabel("time [s]")
        b.set_ylim(bottom=0)
        if sc.gust is not None:
            for r in range(3):
                ax[r, col].axvspan(sc.gust.t0, sc.gust.t0 + sc.gust.duration, color="#f6f5f1", zorder=0)
            ax[0, col].text(sc.gust.t0 + 0.02, 1.5, "gust on", color=INK2, fontsize=9, va="top")
    ax[0, 0].legend(loc="upper right", ncol=2, fontsize=9)
    ax[1, 0].text(6.45, 0.3, "solid = estimate\ndashed = true value", color=INK2, fontsize=9)
    ax[2, 0].legend(loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig("results/stage2_compare.png", dpi=160)


if __name__ == "__main__":
    main()
