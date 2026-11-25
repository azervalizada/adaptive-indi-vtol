r"""Weighted-least-squares control allocation with actuator limits (active-set method).

Problem (Haerkegaard 2002, "Efficient active set algorithms for solving constrained least squares
problems in aircraft control allocation"; the form used in INDI by Smeur et al. and Paparazzi):

    minimise   gamma * || W_v (G du - v) ||^2  +  || W_u (du - du_pref) ||^2
    subject to du_min <= du <= du_max

* first term  (primary):   achieve the requested increments v (angular accelerations, thrust)
                           as well as possible; W_v sets the priority between axes when not
                           everything is achievable (e.g. roll/pitch before yaw).
* second term (secondary): among all du that achieve v, prefer the one closest to du_pref;
                           W_u sets which actuators are "cheap". gamma >> 1 keeps it secondary.
* limits: each actuator can move only between its current position and its saturation.

Both terms together are one stacked least-squares problem  min || A du - b ||^2  with
    A = [sqrt(gamma) W_v G ; W_u],   b = [sqrt(gamma) W_v v ; W_u du_pref].

Active-set idea: start from a feasible du; repeatedly solve the UNCONSTRAINED problem for the
actuators that are not at a limit ("free"). If that step would push a free actuator past its
limit, walk only as far as the first limit and fix that actuator there. If no limit is hit,
check the fixed actuators: if the gradient says the cost would fall by releasing one of them,
release it; otherwise the current du is optimal (KKT conditions hold).
"""
import numpy as np


def wls_alloc(G, v, du_min, du_max, du_pref, W_v, W_u, gamma=1000.0, imax=100):
    n = G.shape[1]
    sg = np.sqrt(gamma)
    A = np.vstack([sg * (W_v[:, None] * G), np.diag(W_u)])
    b = np.concatenate([sg * W_v * v, W_u * du_pref])
    du = np.clip(du_pref, du_min, du_max)         # feasible start
    W = np.zeros(n, dtype=int)                     # 0 free, -1 at lower limit, +1 at upper limit
    W[du <= du_min] = -1
    W[du >= du_max] = +1
    for _ in range(imax):
        free = W == 0
        d = b - A @ du
        p = np.zeros(n)
        if free.any():
            p[free] = np.linalg.lstsq(A[:, free], d, rcond=None)[0]
        target = du + p
        viol = free & ((target < du_min - 1e-12) | (target > du_max + 1e-12))
        if not viol.any():
            du = target
            g = A.T @ (A @ du - b)                 # gradient of 0.5 ||A du - b||^2
            lam = -W * g                           # >= 0 for every fixed actuator at optimum
            if np.all(lam[~free] >= -1e-10):
                return du
            i = np.argmin(np.where(free, np.inf, lam))
            W[i] = 0                               # release the most "wrongly" fixed actuator
        else:
            # walk along p until the first free actuator reaches a limit
            alpha = 1.0
            hit = -1
            for i in np.nonzero(viol)[0]:
                lim = du_min[i] if p[i] < 0 else du_max[i]
                a = (lim - du[i]) / p[i]
                if a < alpha:
                    alpha, hit = a, i
            du = du + alpha * p
            if hit >= 0:
                W[hit] = -1 if p[hit] < 0 else 1
                du[hit] = du_min[hit] if p[hit] < 0 else du_max[hit]
    return np.clip(du, du_min, du_max)             # iteration cap (not reached in practice)
