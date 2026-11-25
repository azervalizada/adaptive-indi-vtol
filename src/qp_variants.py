"""Sensitivity variants of the quadplane adaptive controllers (NOT the main results).

Each variant is tuned with the same rule and procedure as the main controller (tune_qp.py) and its
settings are read from that tuning's CSV, so nothing is hand-copied:
    rls_eta0                  RLS with eta_hat lower bound 0 (the main RLS uses 0.1 since 28 Oct 2026)
                              (results/tuning_qp_rls_etamin0.csv;   python tune_qp.py rls --eta-min 0)
    mrac_eta01                MRAC with lower bound 0.1 instead of its main 0
                              (results/tuning_qp_mrac_etamin0.1.csv; python tune_qp.py mrac --eta-min 0.1)
    rls_noaero, mrac_noaero   estimator WITHOUT the nominal aerodynamic model (review M3, 28 Oct 2026)
                              (results/tuning_qp_<m>_noaero.csv;    python tune_qp.py <m> --no-aero)
"""
import csv
import os
from dataclasses import replace

from .qp_adaptive import QPRLSINDI, QPRLSSettings, QPMRACINDI, QPMRACSettings

_SPEC = {
    "rls_eta0": ("rls", "_etamin0", dict(eta_min=0.0)),
    "mrac_eta01": ("mrac", "_etamin0.1", dict(eta_min=0.1)),
    "rls_noaero": ("rls", "_noaero", dict(aero_model=False)),
    "mrac_noaero": ("mrac", "_noaero", dict(aero_model=False)),
}
_CLS = {"rls": (QPRLSSettings, QPRLSINDI), "mrac": (QPMRACSettings, QPMRACINDI)}
_ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def tuned(method, suffix=""):
    """Winner of results/tuning_qp_<method><suffix>.csv (lowest score among the admissible)."""
    from tune_qp import LADDER
    rows = [r for r in csv.DictReader(open(os.path.join(_ROOT, f"tuning_qp_{method}{suffix}.csv")))
            if r["ok"] == "True"]
    best = min(rows, key=lambda r: float(r["score"]))
    return {k: (int(float(best[k])) if isinstance(v[0], int) else float(best[k])) for k, v in LADDER[method].items()}


def settings(name):
    method, suffix, extra = _SPEC[name]
    S, _ = _CLS[method]
    return replace(S(), **tuned(method, suffix), **extra)


def make(name):
    """Factory with the QP_CONTROLLERS signature."""
    method = _SPEC[name][0]
    C = _CLS[method][1]
    st = settings(name)
    return lambda pn, g, dt, plant: C(pn, g, dt, st)


NAMES = list(_SPEC)
# variants that are run: mrac_noaero is not, because NO candidate passes the tuning rule (28 Oct 2026:
# without the aero model MRAC's estimates drift >= 0.5 in healthy flight, rule (c) fails for all 48)
RUN = [n for n in NAMES if n != "mrac_noaero"]
