"""Paper figures for Stages 3-4 (28 Oct 2026). Reads the result CSVs and runs a few extra simulations.
Every number and every claim in a title is read from the result files or from the simulations run
here (rewritten 28 Oct 2026 after the independent review, finding M6: values were typed in and would
have gone stale after a rerun).

    python run_figures.py      ->  results/fig1_rotor_loss_speed.png ... fig6_transition_outrun.png (+ .pdf)
                                   (~3 min on 2 cores; run the benchmarks first)

Colours: one fixed colour per controller in every figure (validated palette; two slots sit in the 6-8 CVD
band, so every line also has its own dash pattern and a label). Airspeed, a magnitude, uses one blue ramp.
"""
import csv
import os
from multiprocessing import Pool
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.qp_controllers import run_qp, run_tr
from src.qp_scenarios import QPScenario
from src.tr_scenarios import TRScenario
from src.scenarios import Fault
from src.qp_params import QPParams
from run_qp_benchmark import load_csv, gap_closed as qp_gap, lowest_from_which_all
from run_tr_benchmark import load as tr_load

OUT = "results"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C = {"baseline": "#8a8a85", "oracle": "#eb6834", "rls": "#2a78d6", "mrac": "#1baf7a", "l1": "#e87ba4"}
LS = {"baseline": "-", "oracle": (0, (5, 2)), "rls": (0, (1, 1.5)), "mrac": (0, (6, 2, 1, 2)), "l1": "-"}
LABEL = {"baseline": "baseline INDI", "oracle": "oracle INDI", "rls": "RLS", "mrac": "MRAC", "l1": "L1"}
BLUES = ["#a9cdf3", "#6aa6e8", "#2a78d6", "#1a4f93"]
plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": "#c9c8c1", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
    "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "savefig.dpi": 200, "figure.dpi": 100,
})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _qp(job):
    name, V, rotor, eta = job
    r = run_qp(name, QPScenario("fig", V, Fault(rotor, eta, 5.0)), keep_trace=True)
    return job, r


def _tr(job):
    name, accel, Vf = job
    return job, run_tr(name, TRScenario("fig", accel, Vf, 0, 0.0), keep_trace=True)


# ---------------------------------------------------------------- data helpers
def sweep_min(rows, ctrl, eta, rotor):
    rr = [r for r in rows if r["set"] == "Speed sweep" and r["controller"] == ctrl and r["eta"] == eta
          and r["rotor"] == rotor]
    v, ok = lowest_from_which_all(rr, "V", lambda r: r["recovered"])
    return v if ok else None


def outcome_word(r):
    if r.get("lost"):
        return "lost"
    if r.get("completed") or (r.get("recovered") and "completed" not in r):
        return "completed" if "completed" in r else "recovered"
    return "survived, not recovered"


def stage2_gap(name):
    from run_benchmark import gap_closed, read_csv
    rows = [r for r in read_csv(f"{OUT}/benchmark_runs.csv") if r["set"] == "Grid"]
    return gap_closed(rows, name)


def margin(rows, ctrl, rms_key, thr, set_name=None):
    """Oscillation-threshold delay margin: largest delay before the healthy run's RMS > thr or a loss."""
    rr = [r for r in rows if r["controller"] == ctrl and (set_name is None or r.get("set") == set_name)]
    delays = sorted({int(r["delay_ms"]) for r in rr})
    last = None
    for d in delays:
        x = [r for r in rr if int(r["delay_ms"]) == d]
        if any(r.get("lost", r.get("crashed")) == "True" for r in x):
            return last, False
        nom = [r for r in x if "no fault" in r["scenario"]][0]
        if float(nom[rms_key]) > thr:
            return last, False
        last = d
    return last, True          # never broke within the tested range


# ---------------------------------------------------------------- figure 1
def fig1(runs, qrows):
    vmin = sweep_min(qrows, "baseline", 0.0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    title = ("Complete loss of rotor 1 (baseline INDI): not recovered at any swept speed" if vmin is None else
             f"Complete loss of rotor 1 (baseline INDI): heading lost below {vmin:g} m/s, recovered from {vmin:g} m/s")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    for k, V in enumerate((0.0, 4.0, 6.0, 8.0)):
        r = runs[("baseline", V, 0, 0.0)]
        tr = r["trace"]
        t = tr["t"] - 5.0
        m = (t >= -0.2) & (t <= 4.0)
        lab = f"{V:g} m/s ({outcome_word(r)})"
        ax[0].plot(t[m], tr["tilt_err_deg"][m], color=BLUES[k], label=lab)
        ax[1].plot(t[m], tr["heading_err_deg"][m], color=BLUES[k], label=lab)
    ax[0].set_title("A. Tilt error")
    ax[0].set_ylabel("tilt error [deg]")
    ax[1].set_title("B. Heading error")
    ax[1].set_ylabel("heading error [deg]")
    for a in ax:
        a.set_xlabel("time after the fault [s]")
        a.axvline(0, color=INK2, lw=0.8)
    ax[1].legend(title="airspeed (outcome)", loc="upper left", fontsize=8, title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig1_rotor_loss_speed")


# ---------------------------------------------------------------- figure 2
def fig2(runs):
    p = QPParams()
    Vs = np.arange(0, 13)
    T, da, de, rec = [], [], [], []
    for V in Vs:
        r = runs[("baseline", float(V), 0, 0.3)]
        tr = r["trace"]
        tail = tr["t"] > tr["t"][-1] - 0.5
        x = tr["x_act"][tail].mean(axis=0)
        T.append(x[0:4]); da.append(np.degrees(x[5])); de.append(np.degrees(x[6]))
        rec.append(r["recovered"])
    T = np.array(T)
    rec = np.array(rec)
    vmin, _ = lowest_from_which_all([{"V": float(v), "recovered": bool(k)} for v, k in zip(Vs, rec)], "V",
                                    lambda r: r["recovered"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    fig.suptitle("Rotor 1 at 30 %: as airspeed grows, the allocation evens out the rotors and deflects the surfaces "
                 "(steady state, baseline INDI)", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    names = ["rotor 1 (damaged)", "rotor 2", "rotor 3", "rotor 4"]
    rc = ["#e34948", "#8a8a85", "#52514e", "#b5b4ad"]
    rs = ["-", (0, (5, 2)), (0, (1, 1.5)), (0, (6, 2, 1, 2))]
    for i in range(4):
        ax[0].plot(Vs, T[:, i], color=rc[i], ls=rs[i], marker="o", ms=4, label=names[i])
    ax[0].axhline(p.T_max, color=INK2, lw=0.8)
    ax[0].text(12, p.T_max - 1.2, "rotor limit", ha="right", fontsize=8, color=INK2)
    # review m8: this is the commanded thrust after the motor lag (actuator state), not the raw command
    ax[0].set_title("A. Lift-rotor thrust demand (after motor lag)")
    ax[0].set_ylabel("thrust [N]")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax[1].plot(Vs, da, color="#2a78d6", marker="o", ms=4, label="aileron")
    ax[1].plot(Vs, de, color="#eb6834", ls=(0, (5, 2)), marker="o", ms=4, label="elevator")
    ax[1].axhline(-25, color=INK2, lw=0.8)
    ax[1].text(12, -23.5, "surface limit", ha="right", fontsize=8, color=INK2)
    ax[1].set_title("B. Surface deflection")
    ax[1].set_ylabel("deflection [deg]")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    if vmin is not None and vmin > 0:
        for a in ax:
            a.axvspan(-0.5, vmin - 0.5, color="#f0efec", zorder=0)
        ax[1].text((vmin - 1) / 2, 3, "not\nrecovered", ha="center", fontsize=7.5, color=INK2)
    for a in ax:
        a.set_xlabel("airspeed [m/s]")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig2_allocation_vs_speed")


# ---------------------------------------------------------------- figure 3
def fig3(qrows):
    feas = list(csv.DictReader(open(f"{OUT}/qp_feasibility.csv")))
    titles = {0: "static, pitch 0, sideslip ≤ 5°", 1: "static, pitch 0, sideslip ≤ 20°",
              3: "static, |pitch| ≤ 1°, sideslip ≤ 5°", 2: "static, pitch free, sideslip ≤ 20°"}
    static = {titles[c]: {} for c in titles}
    for r in feas:
        static[titles[int(r["case"])]][(float(r["eta"]), int(r["rotor"]))] = float(r["v_min"]) if r["v_min"] else np.nan
    etas = (0.3, 0.15, 0.0)
    ctrls = ["baseline", "oracle", "rls", "mrac", "l1"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    fig.suptitle("Minimum safe airspeed: simulation vs the static limit of steady level flight",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    x = np.arange(3)
    styles = [("#2a78d6", "s", titles[0]), ("#6aa6e8", "v", titles[3]), ("#1baf7a", "^", titles[1]),
              ("#eb6834", "D", titles[2])]
    notes = []
    for j, (title, a, rotor) in enumerate(zip(("A. Front rotor (1)", "B. Rear rotor (2)"), ax, (1, 2))):
        sim = [sweep_min(qrows, "baseline", e, rotor) for e in etas]
        a.plot(x, sim, color=INK, marker="o", ms=9, lw=0, label="simulated, baseline INDI", zorder=3)
        for i, e in enumerate(etas):
            for c in ctrls[1:]:
                v = sweep_min(qrows, c, e, rotor)
                if v != sim[i]:
                    a.plot(x[i] - 0.12, v, marker="x", color=C[c], ms=7, mew=2, lw=0, zorder=4)
                    notes.append(f"{LABEL[c]} {v:g} (rotor {rotor}, η {e:g})")
        for k, (col, mk, lab) in enumerate(styles):
            a.plot(x + 0.1 * (k + 1), [static[lab].get((e, rotor), np.nan) for e in etas], color=col, marker=mk,
                   ms=7, lw=0, label=lab)
        a.set_xticks(x + 0.2, ["η = 0.3", "η = 0.15", "rotor lost"])
        a.set_title(title)
        a.set_ylim(0, 10)
    ax[0].set_ylabel("minimum safe airspeed [m/s]")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    if notes:
        fig.text(0.01, -0.18, "× other controllers where they differ from baseline: " + "; ".join(notes),
                 fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig3_min_safe_airspeed")


# ---------------------------------------------------------------- figure 4
def fig4():
    rows = list(csv.DictReader(open(f"{OUT}/qp_robustness_delay.csv")))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    # title from the data (second review m4): who is best at zero delay, who oscillates first
    best0, first = {}, {}
    for setname in ("hover", "8 m/s"):
        rr0 = [r for r in rows if r["set"] == setname]
        z = {c: np.mean([float(r["tilt_peak_deg"]) for r in rr0 if r["controller"] == c and int(r["delay_ms"]) == 0])
             for c in ("baseline", "oracle", "rls", "mrac", "l1")}
        best0[setname] = min(z, key=z.get)
        m = {c: margin(rows, c, "tilt_rms_deg", 0.1, setname)[0] for c in z}
        first[setname] = min(m, key=lambda c: (m[c] if m[c] is not None else -1))
    b, f = set(best0.values()), set(first.values())
    fig.suptitle(f"Quadplane under delay: best without delay {', '.join(LABEL[c] for c in sorted(b))}; "
                 f"first to oscillate {', '.join(LABEL[c] for c in sorted(f))}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    for ax, setname in zip(axs, ("hover", "8 m/s")):
        rr0 = [r for r in rows if r["set"] == setname]
        delays = sorted({int(r["delay_ms"]) for r in rr0})
        for c in ("baseline", "oracle", "rls", "mrac", "l1"):
            xs, ys, osc = [], [], []
            for d in delays:
                rr = [r for r in rr0 if r["controller"] == c and int(r["delay_ms"]) == d]
                if any(r["lost"] == "True" for r in rr):
                    break
                nom = [r for r in rr if "no fault" in r["scenario"]][0]
                xs.append(d); ys.append(np.mean([float(r["tilt_peak_deg"]) for r in rr]))
                osc.append(float(nom["tilt_rms_deg"]) > 0.1)
            xs, ys, osc = map(np.array, (xs, ys, osc))
            ax.plot(xs, ys, color=C[c], ls=LS[c], label=LABEL[c])
            ax.plot(xs[~osc], ys[~osc], "o", ms=4, color=C[c])
            ax.plot(xs[osc], ys[osc], "o", ms=5, mfc="white", mec=C[c], mew=1.5)
        ax.set_yscale("log")
        ax.set_title(f"{'A' if setname == 'hover' else 'B'}. {setname}")
        ax.set_xlabel("extra, unmodelled actuator delay [ms]")
    axs[0].set_ylabel("mean peak tilt over the set [deg]")
    axs[0].legend(fontsize=8.5, loc="upper left")
    axs[1].text(0.99, 0.02, "open marker = oscillating when healthy; line ends at a loss", transform=axs[1].transAxes,
                ha="right", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig4_delay_quadplane")


# ---------------------------------------------------------------- figure 5
def fig5(qrows, trows):
    import os.path
    groups = ["quadrotor\n(grid)", "quadplane\n(grid)", "quadplane\n(transition)"]
    grid = [r for r in qrows if r["set"] == "Grid"]
    tgrid = [r for r in trows if r["set"] == "Transition grid"]
    gap, ci = {}, {}
    for c in ("rls", "mrac", "l1"):
        s2 = stage2_gap(c)
        g3 = qp_gap(grid, c)
        g4 = qp_gap(tgrid, c)
        gap[c] = [s2[0], g3[0], g4[0]]
        ci[c] = [(s2[1], s2[2]), (g3[1], g3[2]), (g4[1], g4[2])]
    rls0 = [None, None, None]
    if os.path.exists(f"{OUT}/qp_sens_runs.csv"):
        import run_qp_sensitivity  # noqa: F401  (same CSV conventions)
        sens = [r for r in csv.DictReader(open(f"{OUT}/qp_sens_runs.csv")) if r["set"] == "Grid"]
        for r in sens:
            r["tilt_peak_deg"] = float(r["tilt_peak_deg"]) if r["tilt_peak_deg"] not in ("", "None") else None
            r["lost"] = r["lost"] == "True" or bool(r.get("error"))
        g = qp_gap(grid + sens, "rls_eta0")
        rls0[1] = None if g is None else g[0]
    if os.path.exists(f"{OUT}/tr_sens_runs.csv"):
        sens = []
        for r in csv.DictReader(open(f"{OUT}/tr_sens_runs.csv")):
            r["tilt_peak_deg"] = float(r["tilt_peak_deg"]) if r["tilt_peak_deg"] not in ("", "None") else None
            r["lost"] = r["lost"] == "True" or bool(r.get("error"))
            sens.append(r)
        g = qp_gap(tgrid + sens, "rls_eta0")
        rls0[2] = None if g is None else g[0]
    s2rows = list(csv.DictReader(open(f"{OUT}/robustness_delay.csv")))
    qrob = list(csv.DictReader(open(f"{OUT}/qp_robustness_delay.csv")))
    names = ["baseline", "rls", "mrac", "l1"]
    mg = {n: [margin(s2rows, n, "att_rms_deg", 0.5), margin(qrob, n, "tilt_rms_deg", 0.1, "hover"),
              margin(qrob, n, "tilt_rms_deg", 0.1, "8 m/s")] for n in names}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
    top = {max(("rls", "mrac", "l1"), key=lambda c: gap[c][i]) for i in range(3)}
    low = {min(names, key=lambda n: mg[n][j][0] if mg[n][j][0] is not None else -1) for j in range(3)}
    fig.suptitle(f"Across both vehicles: most gap closed {', '.join(LABEL[c] for c in sorted(top))}; "
                 f"smallest delay margin {', '.join(LABEL[c] for c in sorted(low))}",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    x = np.arange(3)
    w = 0.22
    for k, c in enumerate(("rls", "mrac", "l1")):
        lo = [g - l for g, (l, h) in zip(gap[c], ci[c])]
        hi = [h - g for g, (l, h) in zip(gap[c], ci[c])]
        ax[0].bar(x + (k - 1) * w, gap[c], w - 0.03, color=C[c], label=LABEL[c], edgecolor="white", linewidth=1)
        ax[0].errorbar(x + (k - 1) * w, gap[c], yerr=[lo, hi], fmt="none", ecolor=INK2, lw=1, capsize=2)
    xs = [x[i] - w for i in (1, 2) if rls0[i] is not None]
    ys = [rls0[i] for i in (1, 2) if rls0[i] is not None]
    if xs:
        ax[0].plot(xs, ys, "D", mfc="white", mec=INK, mew=1.5, ms=7, label="RLS, estimate bound 0 (sensitivity)")
    ax[0].axhline(100, color=C["oracle"], lw=1, ls=(0, (5, 2)))
    ax[0].text(2.45, 103, "oracle = 100 %", ha="right", fontsize=8, color=INK2)
    ax[0].axhline(0, color=INK2, lw=0.8)
    ax[0].set_xticks(x, groups)
    ax[0].set_ylabel("gap closed [%] (95 % CI)")
    ax[0].set_title("A. Gap closed between baseline and oracle INDI")
    ax[0].legend(fontsize=8, loc="upper right", ncol=2)
    hatches = [None, "////", "...."]
    vlabels = ["quadrotor", "quadplane hover", "quadplane 8 m/s"]
    xmax = 0
    for j in range(3):
        vals = [mg[n][j][0] for n in names]
        y = np.arange(len(names)) + j * 0.27
        ax[1].barh(y, vals, 0.25, color=[C[n] for n in names], edgecolor="white", linewidth=1, hatch=hatches[j],
                   label=vlabels[j])
        for yy, n in zip(y, names):
            v, open_ = mg[n][j]
            ax[1].text(v + 1, yy, f"{'≥ ' if open_ else ''}{v} ms", va="center", fontsize=7.5, color=INK2)
            xmax = max(xmax, v)
    ax[1].set_yticks(np.arange(len(names)) + 0.27, [LABEL[n] for n in names])
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, xmax + 18)
    ax[1].set_xlabel("delay margin [ms] (oscillation threshold)")
    ax[1].set_title("B. Delay margin")
    ax[1].legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig5_summary")


# ---------------------------------------------------------------- figure 6
def _smooth(y, n=50):
    yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(yp, np.ones(n) / n, mode="same")[n:-n]


def _why_lost(r, t0):
    tr = r["trace"]
    if not r["lost"]:
        return outcome_word(r)
    if tr["diverged_at"] is not None:
        return "lost: turned away"
    if np.any(tr["tilt_err_deg"][tr["t"] >= t0] > 90):        # review m9: only after the fault
        return "lost: tilt"
    return "lost: altitude"


def fig6(trans, runs, qrows, Vf):
    front, rear = sweep_min(qrows, "baseline", 0.0, 1), sweep_min(qrows, "baseline", 0.0, 2)
    rc = runs[("baseline", Vf, 0, 0.0)]
    tr_out = {acc: trans[("baseline", acc, Vf)] for acc in (1.0, 2.0, 3.0)}
    done = [f"{a:g}" for a, r in tr_out.items() if r["completed"]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    fig.suptitle(f"Rotor 1 lost at {Vf:g} m/s (baseline INDI): {_why_lost(rc, 5.0)} at constant speed, "
                 f"transition completed at {', '.join(done) if done else 'no'} m/s²",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    curves = [(f"constant {Vf:g} m/s ({_why_lost(rc, 5.0)})", rc["trace"], 5.0, INK2)]
    for k, acc in enumerate((1.0, 2.0, 3.0)):
        r = tr_out[acc]
        t0 = 3.0 + Vf / acc
        curves.append((f"transition {acc:g} m/s² ({_why_lost(r, t0)})", r["trace"], t0, BLUES[k + 1]))
    for lab, tr, t0, col in curves:
        t = tr["t"] - t0
        m = (t >= -0.2) & (t <= 10.0)
        ax[0].plot(t[m], np.maximum(tr["tilt_err_deg"][m], 1e-2), color=col, label=lab)
        ax[1].plot(t[m], tr["heading_err_deg"][m], color=col, label=lab)
        ax[2].plot(t[m], _smooth(tr["Va"])[m], color=col, label=lab)
    ax[0].set_title("A. Tilt error")
    ax[0].set_ylabel("tilt error [deg]")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-2, 200)
    ax[1].set_title("B. Heading error")
    ax[1].set_ylabel("heading error [deg]")
    ax[2].set_title("C. Airspeed (0.1 s mean)")
    ax[2].set_ylabel("airspeed [m/s]")
    lims = [v for v in (front, rear) if v is not None]
    if lims:
        lo, hi = min(lims), max(lims)
        ax[2].axhspan(lo, hi, color="#f0efec", zorder=0)
        ax[2].text(9.8, (lo + hi) / 2, f"constant-speed limit {lo:g}–{hi:g} m/s", ha="right", fontsize=8,
                   color=INK2, va="center")
    for a in ax:
        a.set_xlabel("time after the fault [s]")
        a.axvline(0, color=INK2, lw=0.8)
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig6_transition_outrun")


def main():
    os.makedirs(OUT, exist_ok=True)
    qrows = load_csv()
    trows = tr_load()
    # fault speed for fig6: the baseline's lowest fault speed from which the 2 m/s^2 transition completes
    # with the front rotor lost (rotor 1, eta 0)
    tg = [r for r in trows if r["set"] == "Transition grid" and r["controller"] == "baseline" and r["eta"] == 0.0
          and r["rotor"] == 1 and r["accel"] == 2.0]
    Vf, _ = lowest_from_which_all(tg, "V_f", lambda r: r["completed"])
    if Vf is None:                       # second review m4: no completing fault speed -> use the slowest
        Vf = 0.0
    Vf = float(Vf)
    qp_jobs = ([("baseline", V, 0, 0.0) for V in (0.0, 4.0, 6.0, 8.0)] + [("baseline", Vf, 0, 0.0)]
               + [("baseline", float(V), 0, 0.3) for V in range(13)])
    tr_jobs = [("baseline", a, Vf) for a in (1.0, 2.0, 3.0)]
    with Pool() as pool:
        runs = dict(pool.map(_qp, list(dict.fromkeys(qp_jobs))))
        trans = dict(pool.map(_tr, tr_jobs))
    fig1(runs, qrows)
    fig2(runs)
    fig3(qrows)
    fig4()
    fig5(qrows, trows)
    fig6(trans, runs, qrows, Vf)
    print("fig6 fault speed", Vf, "outcomes:", [(k[1], v["lost"], v["completed"]) for k, v in trans.items()],
          f" constant {Vf:g} m/s:", outcome_word(runs[("baseline", Vf, 0, 0.0)]))


if __name__ == "__main__":
    main()
