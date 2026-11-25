"""Linear delay-margin analysis of one axis of the INDI loop, with and without L1 (open item, 28 Oct).

The model repeats, step by step, what the code does on one rotational axis at the control rate
Ts = 2 ms (quadrotor roll axis by default):
  plant      theta' = w,  w' = a,  a' = (u_delayed - a) / tau          (a = achieved angular accel.)
             discretised exactly (zero-order hold), states sampled every Ts
  sensors    gyro w, actuator a (rpm-based), no noise
  INDI       second-order filters (semi-implicit Euler, as SecondOrderFilter) on w and a;
             nu = k_rate (w_cmd - w), w_cmd = -k_att theta;  u = a_f + (nu_a - wdot_f)   (G = 1 here)
  L1         the three blocks of l1.py (predictor with motor lag, piecewise-constant law, C(s))
  delay      n steps of pure transport delay on u (a shift register)
The step map x_{k+1} = M x_k is linear; the loop is stable iff all eigenvalues of M lie inside
the unit circle. The delay margin is the largest n (x Ts) that keeps it stable.
"""
import numpy as np
from scipy.linalg import expm


def step_matrix(n_delay, Ts=0.002, tau=0.02, k_rate=25.0, k_att=8.0, wn=50.0, zeta=0.55, l1=None):
    # state: [theta, w, a | yg, ydg, ya, yda | (L1) u_ad, nu_pred, w_hat | delay buffer (n)]
    Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / tau]])
    Bc = np.array([0, 0, 1 / tau])
    E = expm(np.block([[Ac, Bc[:, None]], [np.zeros((1, 4))]]) * Ts)
    Ad, Bd = E[:3, :3], E[:3, 3]
    nl1 = 3 if l1 else 0
    N = 3 + 4 + nl1 + n_delay
    M = np.zeros((N, N))
    for j in range(N):
        x = np.zeros(N); x[j] = 1.0
        M[:, j] = _step(x, n_delay, Ts, tau, k_rate, k_att, wn, zeta, l1, Ad, Bd)
    return M


def _step(x, n, Ts, tau, k_rate, k_att, wn, zeta, l1, Ad, Bd):
    th, w, a = x[0:3]
    yg, ydg, ya, yda = x[3:7]
    i = 7
    if l1:
        u_ad, nu_pred, w_hat = x[i:i + 3]; i += 3
    buf = x[i:i + n]
    # controller (same order as indi.py / l1.py)
    ydg = ydg + Ts * (wn ** 2 * (w - yg) - 2 * zeta * wn * ydg); yg = yg + Ts * ydg
    yda = yda + Ts * (wn ** 2 * (a - ya) - 2 * zeta * wn * yda); ya = ya + Ts * yda
    nu = k_rate * (-k_att * th - w)
    if l1:
        A_, wc = l1
        ead = np.exp(-A_ * Ts); phi = (1 - ead) / A_; lp = 1 - np.exp(-wc * Ts); act = 1 - np.exp(-Ts / tau)
        e = w_hat - w
        sig = -ead / phi * e
        u_ad = u_ad + lp * (sig - u_ad)
        nu_a = nu - u_ad
        nu_pred = nu_pred + act * (nu_a - nu_pred)
        w_hat = w + ead * e + phi * (nu_pred + sig)
    else:
        nu_a = nu
    u = ya + (nu_a - ydg)
    # transport delay: the plant gets the command of n steps ago
    if n > 0:
        u_app = buf[-1]
        buf = np.concatenate([[u], buf[:-1]])
    else:
        u_app = u
    p = Ad @ np.array([th, w, a]) + Bd * u_app
    out = list(p) + [yg, ydg, ya, yda]
    if l1:
        out += [u_ad, nu_pred, w_hat]
    return np.array(out + list(buf))


def margin_ms(l1=None, max_ms=80, **kw):
    Ts = kw.get("Ts", 0.002)
    last = None
    for n in range(0, int(max_ms / 1000 / Ts) + 1):
        rho = max(abs(np.linalg.eigvals(step_matrix(n, l1=l1, **kw))))
        if rho >= 1.0:
            return last
        last = n * Ts * 1000
    return last


def loop_response(l1=None, freqs_hz=None, **kw):
    """Open-loop frequency response L = -u_out / u_in with the loop broken at the actuator command
    (no delay). Crossover where |L| = 1; delay margin ~ phase margin / crossover frequency."""
    Ts = kw.get("Ts", 0.002)
    M0 = step_matrix(0, l1=l1, **kw)          # closed loop without delay, used to read A, B, C, D
    N = M0.shape[0]
    # open-loop: replace u by an external input. Build by finite differences on _step with n = 1:
    # with one delay slot, the plant input is the buffer state and the controller output goes into it.
    M1 = step_matrix(1, l1=l1, **kw)
    A = M1[:N, :N]                              # states without the buffer
    B = M1[:N, N]                               # effect of the buffer (= plant input) on the states
    C = M1[N, :N]                               # controller output written into the buffer
    D = M1[N, N]
    freqs_hz = np.logspace(-1, np.log10(240), 600) if freqs_hz is None else freqs_hz
    out = []
    for f in freqs_hz:
        z = np.exp(1j * 2 * np.pi * f * Ts)
        G = C @ np.linalg.solve(z * np.eye(N) - A, B) + D
        out.append(-G)
    return freqs_hz, np.array(out)


def crossover(l1=None, **kw):
    Ts = kw.get("Ts", 0.002)
    f, L = loop_response(l1=l1, **kw)
    mag = np.abs(L)
    idx = np.nonzero((mag[:-1] >= 1) & (mag[1:] < 1))[0]
    i = idx[-1]
    pm = 180 + np.degrees(np.angle(L[i]))
    pm = (pm + 180) % 360 - 180
    return f[i], pm, pm / 360 / f[i] * 1000


if __name__ == "__main__":
    # writes results/delay_margin_linear.txt (made reproducible 28 Oct 2026; rows for the re-tuned
    # L1, wc = 1280, added)
    L1S = [(0.1, 1280), (10, 1280), (10, 320), (30, 320), (100, 320), (300, 320), (10, 40), (300, 40), (30, 640), (30, 1280),
           (100, 1280), (300, 1280)]
    lines = ["Linear one-axis analysis (analysis/delay_margin.py). Delay margin = largest pure delay with all "
             "closed-loop", "eigenvalues inside the unit circle (2 ms steps); crossover / phase margin from the "
             "loop broken at the actuator command.", ""]
    for title, kw in (("Quadrotor roll axis (k_rate 25, k_att 8)", {}),
                      ("Quadplane roll axis IN HOVER (k_rate 20, k_att 6; no surfaces, aerodynamic damping or "
                       "servo model: valid only in hover, review m7)", dict(k_rate=20.0, k_att=6.0))):
        lines.append(title)
        for lab, l1 in [("baseline", None)] + [(f"L1 a={a} wc={wc}", (a, wc)) for a, wc in L1S]:
            f, pm, d = crossover(l1=l1, **kw)
            lines.append(f"  {lab:<17} crossover {f:5.2f} Hz  phase margin {pm:5.1f} deg  PM/omega_c {d:5.1f} ms"
                         f"  eigenvalue margin {margin_ms(l1=l1, **kw):5.1f} ms")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    open("results/delay_margin_linear.txt", "w").write(text + "\n")
