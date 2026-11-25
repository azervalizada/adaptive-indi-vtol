"""Stage 1b: a fair baseline.

Compares three controllers on the same single-rotor fault sweep:
  INDI (rpm thrust)    Stage 1 baseline: current thrust estimated from rotor speeds
  INDI (accel thrust)  new baseline: current thrust measured by the accelerometer
  oracle INDI          knows the true rotor effectiveness (upper bound for adaptation)

Outputs (results/):
  stage1b_sweep.png    left: settled altitude sag, right: attitude upset right after the fault
  stage1b_sweep.csv    numbers behind the plot
"""
import os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.params import QuadParams, INDIGains
from src.quadrotor import Quadrotor
from src.indi import INDIController
from src.oracle import OracleINDI
from src.sim import simulate

DT_CTRL = 0.002
FAULT_T = 6.0
OUT = "results"
ETAS = np.round(np.arange(1.0, 0.449, -0.05), 2)     # stop above the 0.41 feasibility limit
CONTROLLERS = {
    "INDI (rpm thrust)":   ("baseline", "model"),
    "INDI (accel thrust)": ("baseline", "accel"),
    "oracle INDI":         ("oracle", "accel"),
}
COLORS = {"INDI (rpm thrust)": "#8a8a85", "INDI (accel thrust)": "#2a78d6", "oracle INDI": "#eb6834"}
INK, INK2 = "#1f1f1e", "#5f5e5a"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "legend.frameon": False,
})


def run_one(args):
    name, eta1 = args
    kind, fb = CONTROLLERS[name]
    plant = Quadrotor(QuadParams())
    if kind == "baseline":
        ctrl = INDIController(QuadParams(), INDIGains(), DT_CTRL, thrust_feedback=fb)
    else:
        ctrl = OracleINDI(QuadParams(), INDIGains(), DT_CTRL, plant, thrust_feedback=fb)
    r = simulate(plant, ctrl, fault_time=FAULT_T, fault_eta=[eta1, 1, 1, 1])
    t = r["t"]
    settled = (t >= 8.7) & (t < 8.95)
    upset = (t >= FAULT_T) & (t < FAULT_T + 0.5)     # first half second after the fault
    return {
        "controller": name, "eta1": float(eta1),
        "sag_m": float(np.mean(r["pos"][settled, 2] - r["ref"][settled, 2])),
        "upset_deg": float(np.max(r["att_err_deg"][upset])),
        "crash_t": r["diverged_at"],
    }


def plot(rows):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.suptitle("With accelerometer feedback the altitude sag is gone; the attitude upset "
                 "right after the fault remains", x=0.02, ha="left", fontsize=11, color=INK,
                 fontweight="bold")
    for name in CONTROLLERS:
        rr = [r for r in rows if r["controller"] == name]
        x = [r["eta1"] for r in rr]
        ax[0].plot(x, [r["sag_m"] for r in rr], "-o", ms=4, color=COLORS[name], label=name)
        if name != "INDI (rpm thrust)":
            ax[1].plot(x, [r["upset_deg"] for r in rr], "-o", ms=4, color=COLORS[name], label=name)
    ax[0].set_ylabel("settled altitude sag [m]")
    ax[0].set_title("A. Altitude", loc="left", fontsize=10, color=INK)
    ax[0].legend(loc="upper left")
    ax[1].set_ylabel("peak attitude error, 0-0.5 s after fault [deg]")
    ax[1].set_title("B. Attitude upset at the fault", loc="left", fontsize=10, color=INK)
    ax[1].legend(loc="upper left")
    ax[1].set_ylim(bottom=0)
    for a in ax:
        a.invert_xaxis()
        a.set_xlabel("remaining effectiveness of rotor 1 [-]")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "stage1b_sweep.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = [(name, e) for name in CONTROLLERS for e in ETAS]
    with Pool() as pool:                 # one simulation per CPU core at a time
        rows = pool.map(run_one, jobs)
    with open(os.path.join(OUT, "stage1b_sweep.csv"), "w") as f:
        f.write("controller,eta1,sag_m,upset_deg,crash_t\n")
        for r in rows:
            f.write(f"{r['controller']},{r['eta1']:.2f},{r['sag_m']:.4f},{r['upset_deg']:.3f},{r['crash_t'] or ''}\n")
    plot(rows)
    print(f"{'eta1':>5} | {'sag [m]: rpm':>12} {'accel':>7} {'oracle':>7} | {'upset [deg]: accel':>18} {'oracle':>7}")
    for e in ETAS:
        g = {r["controller"]: r for r in rows if r["eta1"] == float(e)}
        print(f"{e:5.2f} | {g['INDI (rpm thrust)']['sag_m']:12.3f} {g['INDI (accel thrust)']['sag_m']:7.3f} "
              f"{g['oracle INDI']['sag_m']:7.3f} | {g['INDI (accel thrust)']['upset_deg']:18.2f} "
              f"{g['oracle INDI']['upset_deg']:7.2f}")
