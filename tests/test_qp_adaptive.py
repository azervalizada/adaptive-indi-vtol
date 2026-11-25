"""Quadplane adaptive controllers (added 28 Oct 2026 after the independent review: there was no test
of src/qp_adaptive.py, and finding C1 would have been caught by the first test below)."""
import numpy as np
import pytest

from src.qp_params import QPParams
from src.quadplane import Quadplane
from src.qp_indi import QPGains
from src.qp_adaptive import QPRLSINDI, QPRLSSettings, QPMRACINDI
from src.qp_sim import simulate_qp
from src.scenarios import Fault


class _Collect(QPRLSINDI):
    """RLS that never adapts (eta_hat stays 1) and records its regression rows."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rows = []

    def observe(self, u_f, wdot_f, acc_f, meas):
        r = self.regression(u_f, wdot_f, acc_f, meas)
        if r is not None:
            self.rows.append((self._k * self.dt, *r))


def _batch_ls(V, aero_model=True):
    plant = Quadplane(QPParams(), seed=0, gyro_noise_std=0.0, acc_noise_std=0.0, airspeed_noise_std=0.0)
    c = _Collect(QPParams(), QPGains(), 0.002, QPRLSSettings(lag=1, threshold=0.0, aero_model=aero_model))
    simulate_qp(plant, c, V, t_end=8.0, fault=Fault(0, 0.6, 5.0))
    rows = [(y, P) for t, y, P in c.rows if t > 5.5]
    Y = np.concatenate([y for y, _ in rows])
    Phi = np.vstack([P for _, P in rows])
    return np.linalg.lstsq(Phi, Y, rcond=None)[0]


@pytest.mark.parametrize("V", [0.0, 8.0, 12.0])
def test_rls_regression_is_unbiased_on_noise_free_data(V):
    """y = Phi eta must hold: batch LS on noise-free data returns the true eta at every speed.
    (Before the 28 Oct fix it gave [-0.13, -3.57, 0.46, 0.54] at 8 m/s.)"""
    est = _batch_ls(V)
    assert np.allclose(est, [0.6, 1.0, 1.0, 1.0], atol=0.05), est


def test_rls_without_aero_model_is_biased_at_speed():
    """Documents why the aero model is needed (review C1): the sensitivity variant is biased at speed."""
    est = _batch_ls(8.0, aero_model=False)
    assert np.max(np.abs(est - [0.6, 1.0, 1.0, 1.0])) > 0.2, est


def test_mrac_predictor_matches_healthy_flight():
    """No fault, no noise: the MRAC estimate must stay at 1 (its predictor matches the plant)."""
    from src.qp_controllers import run_qp
    from src.qp_scenarios import QPScenario
    m = run_qp("mrac", QPScenario("t", 8.0, seed=1, noise_scale=0.0))
    assert m["healthy_dev"] < 0.05


def test_main_settings_are_the_tuning_winners():
    """The defaults in qp_adaptive.py must be the winners in results/tuning_qp_<m>.csv (no hand edits)."""
    from src.qp_variants import tuned
    from src.qp_adaptive import QPRLSSettings, QPMRACSettings, QPL1Settings
    for m, S in (("rls", QPRLSSettings), ("mrac", QPMRACSettings), ("l1", QPL1Settings)):
        s = S()
        for k, v in tuned(m).items():
            assert getattr(s, k) == v, (m, k, getattr(s, k), v)


def test_rls_estimate_freezes_after_an_abrupt_fault_known_limitation():
    """Second review M3 (4 Nov 2026), documented, not fixed (decision: report it). For ~0.2 s after an
    abrupt fault the incremental model y = Phi eta does not hold (the synchronisation filters span the
    jump in eta); with the tuned settings (forget 1.0, dead-zone 0.05 N) RLS learns from those rows and
    then stops updating, so its estimate freezes at a wrong value although control stays good. If this
    test fails, the behaviour changed: update the paper's statement that RLS does not identify the fault."""
    from src.qp_controllers import run_qp
    from src.qp_scenarios import QPScenario
    m = run_qp("rls", QPScenario("t", 8.0, Fault(0, 0.6, 5.0)), keep_trace=True)
    tr = m["trace"]
    i = np.searchsorted(tr["t"], 5.25)
    assert np.allclose(tr["eta_hat"][i], tr["eta_hat"][-1], atol=1e-3)     # frozen after 0.25 s
    assert m["eta_err_end"] > 0.1                                           # and wrong
    assert m["recovered"] and m["tilt_peak_deg"] < 1.0                      # control is still good
