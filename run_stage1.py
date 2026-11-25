"""Stage 1: baseline INDI vs oracle INDI on a quadrotor with a single-rotor effectiveness loss.

Outputs (results/):
  stage1_timeseries.png   time histories for a 50 % loss on rotor 1
  stage1_sweep.png        altitude sag vs remaining rotor effectiveness
  stage1_sweep.csv        numbers behind the sweep plot
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.params import QuadParams, INDIGains
from src.quadrotor import Quadrotor
from src.indi import INDIController
from src.oracle import OracleINDI
from src.sim import simulate, metrics

DT_CTRL = 0.002
FAULT_T = 6.0
OUT = "results"
C_BASE, C_ORACLE, C_REF = "#2a78d6", "#eb6834", "#8a8a85"
INK, INK2 = "#1f1f1e", "#5f5e5a"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#ecebe6",
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.8, "legend.frameon": False,
})


def run(kind, eta1):
    plant = Quadrotor(QuadParams())
    if kind == "baseline":
        ctrl = INDIController(QuadParams(), INDIGains(), DT_CTRL, thrust_feedback="model")
    else:
        ctrl = OracleINDI(QuadParams(), INDIGains(), DT_CTRL, plant, thrust_feedback="model")
    return simulate(plant, ctrl, fault_time=FAULT_T, fault_eta=[eta1, 1, 1, 1])


def hover_sag(res):
    """Settled altitude error with the fault, just before the next position step (8.7-8.95 s); + = below target."""
    m = (res["t"] >= 8.7) & (res["t"] < 8.95)
    return float(np.mean(res["pos"][m, 2] - res["ref"][m, 2]))


def sag_analytic(eta1, p=QuadParams(), k=INDIGains()):
    """Baseline believes rotor 1 produces what its RPM says. In hover every real thrust is mg/4,
    so the model over-counts by (mg/4)(1/eta - 1) and the PD altitude loop settles where
    m*kp_z*e equals that excess."""
    return (p.mass * p.g / 4.0) * (1.0 / eta1 - 1.0) / (p.mass * k.kp_pos[2])


def feasibility_limit(p):
    """Lowest eta1 for which hover with zero moments is still reachable (all real thrusts equal)."""
    return (p.mass * p.g / 4.0) / p.T_max


def plot_timeseries(eta1=0.5):
    rb, ro = run("baseline", eta1), run("oracle", eta1)
    fig, ax = plt.subplots(4, 1, figsize=(8.5, 9.0), sharex=True)
    fig.suptitle(f"Rotor 1 loses {int(round((1 - eta1) * 100))} % effectiveness at t = {FAULT_T:.0f} s: "
                 "baseline INDI sags, the oracle holds altitude", x=0.06, ha="left", fontsize=11.5,
                 color=INK, fontweight="bold")
    for a in ax:
        a.axvspan(FAULT_T, rb["t"][-1], color="#f6f5f1", zorder=0)
        a.axvline(FAULT_T, color=C_REF, lw=1, ls="--")

    ax[0].plot(rb["t"], -rb["ref"][:, 2], color=C_REF, lw=1.2, ls=":", label="reference")
    ax[0].plot(rb["t"], -rb["pos"][:, 2], color=C_BASE, label="baseline INDI")
    ax[0].plot(ro["t"], -ro["pos"][:, 2], color=C_ORACLE, label="oracle INDI")
    ax[0].set_ylabel("altitude [m]")
    ax[0].legend(loc="lower left", ncol=3)

    ax[1].plot(rb["t"], rb["ref"][:, 0], color=C_REF, lw=1.2, ls=":")
    ax[1].plot(rb["t"], rb["pos"][:, 0], color=C_BASE)
    ax[1].plot(ro["t"], ro["pos"][:, 0], color=C_ORACLE)
    ax[1].set_ylabel("north position [m]")

    ax[2].plot(rb["t"], np.degrees(rb["euler"][:, 1]), color=C_BASE)
    ax[2].plot(ro["t"], np.degrees(ro["euler"][:, 1]), color=C_ORACLE)
    ax[2].set_ylabel("pitch [deg]")

    ax[3].plot(rb["t"], rb["u_cmd"][:, 0], color=C_BASE)
    ax[3].plot(ro["t"], ro["u_cmd"][:, 0], color=C_ORACLE)
    ax[3].axhline(QuadParams().T_max, color=C_REF, lw=1, ls=":")
    ax[3].text(0.2, QuadParams().T_max - 0.45, "motor limit", color=INK2, fontsize=9)
    ax[3].set_ylabel("rotor 1 command [N]")
    ax[3].set_xlabel("time [s]")
    ax[0].text(FAULT_T + 0.12, 1.55, "fault", color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(OUT, "stage1_timeseries.png"), dpi=160)
    plt.close(fig)


def sweep():
    etas = np.round(np.arange(1.0, 0.349, -0.05), 2)
    rows = []
    for e in etas:
        rb, ro = run("baseline", e), run("oracle", e)
        mb, mo = metrics(rb, FAULT_T), metrics(ro, FAULT_T)
        rows.append({
            "eta1": e,
            "sag_base_m": hover_sag(rb) if rb["diverged_at"] is None else np.nan,
            "sag_oracle_m": hover_sag(ro) if ro["diverged_at"] is None else np.nan,
            "tilt_max_base_deg": mb["tilt_max_deg"], "tilt_max_oracle_deg": mo["tilt_max_deg"],
            "crash_base_t": rb["diverged_at"], "crash_oracle_t": ro["diverged_at"],
        })
    keys = list(rows[0].keys())
    with open(os.path.join(OUT, "stage1_sweep.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")

    lim = feasibility_limit(QuadParams())
    x = np.array([r["eta1"] for r in rows])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xa = np.linspace(1.0, 0.45, 60)
    ax.plot(xa, [sag_analytic(e) for e in xa], color=C_REF, lw=1.2, ls=":", label="analytic prediction")
    ax.plot(x, [r["sag_base_m"] for r in rows], "-o", ms=5, color=C_BASE, label="baseline INDI")
    ax.plot(x, [r["sag_oracle_m"] for r in rows], "-o", ms=5, color=C_ORACLE, label="oracle INDI")
    ax.axvspan(0.33, lim, color="#f6f5f1", zorder=0)
    ax.axvline(lim, color=C_REF, lw=1, ls="--")
    ax.text(lim - 0.008, 0.6, f"hover\ninfeasible\nbelow {lim:.2f}\n(both crash)",
            ha="left", va="center", color=INK2, fontsize=9)
    ax.invert_xaxis()
    ax.set_xlim(1.02, 0.33)
    ax.set_xlabel("remaining effectiveness of rotor 1 [-]")
    ax.set_ylabel("settled altitude sag in hover [m]")
    ax.set_title("Baseline INDI loses altitude in proportion to the fault; knowing the fault removes it",
                 loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stage1_sweep.png"), dpi=160)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_timeseries(0.5)
    rows = sweep()
    print(f"{'eta1':>5} {'analytic':>9} {'sag_base':>9} {'sag_orac':>9} {'tilt_b':>7} {'tilt_o':>7} {'crash_b':>8} {'crash_o':>8}")
    for r in rows:
        print(f"{r['eta1']:5.2f} {sag_analytic(r['eta1']):9.3f} {r['sag_base_m']:9.3f} {r['sag_oracle_m']:9.3f} "
              f"{r['tilt_max_base_deg']:7.2f} {r['tilt_max_oracle_deg']:7.2f} "
              f"{str(r['crash_base_t']):>8} {str(r['crash_oracle_t']):>8}")
