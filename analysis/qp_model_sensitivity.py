"""Model sensitivity of the Stage 3 grid results (open item, 28 Oct): do the conclusions survive
(a) twice the sensor noise and (b) a 50 % slower true motor than the controller assumes?
Controllers unchanged (Stage 3 settings).
    python analysis/qp_model_sensitivity.py   ->  results/qp_model_sensitivity.txt   (~30 min on 2 cores)"""
import sys
from dataclasses import replace
sys.path.insert(0, ".")
from src.qp_controllers import QP_CONTROLLERS, run_qp_many
from src.qp_scenarios import qp_grid
from run_qp_benchmark import set_table, by_speed_table

VARIANTS = {"noise x2": dict(noise_scale=2.0), "motor 50 % slower than modelled": dict(tau_scale=1.5)}

if __name__ == "__main__":
    names = list(QP_CONTROLLERS) + ["rls_eta0"]   # variant added 28 Oct (eta_min decision)
    lines = []
    for title, kw in VARIANTS.items():
        rows = run_qp_many(names, [replace(s, **kw) for s in qp_grid()])
        lines += set_table(rows, names, f"Grid, {title}") + [""] + by_speed_table(rows, names) + [""]
        print("\n".join(lines[-20:]), flush=True)
    open("results/qp_model_sensitivity.txt", "w").write("\n".join(lines) + "\n")
