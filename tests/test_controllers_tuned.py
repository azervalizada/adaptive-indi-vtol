"""The quadrotor defaults must be the tuning winners in results/tuning_<m>.csv (no hand edits)."""
import csv
from tune import LADDER
from src.rls import RLSSettings
from src.mrac import MRACSettings
from src.l1 import L1Settings


def test_quadrotor_settings_are_the_tuning_winners():
    for m, S in (("rls", RLSSettings), ("mrac", MRACSettings), ("l1", L1Settings)):
        rows = [r for r in csv.DictReader(open(f"results/tuning_{m}.csv")) if r["ok"] == "True"]
        best = min(rows, key=lambda r: float(r["score"]))
        s = S()
        for k, v in LADDER[m].items():
            want = int(float(best[k])) if isinstance(v[0], int) else float(best[k])
            assert getattr(s, k) == want, (m, k, getattr(s, k), want)
