# Adaptive INDI under Rotor Faults: Quadrotor and Quadplane VTOL

Simulation code and paper for comparing three adaptive augmentations of Incremental Nonlinear
Dynamic Inversion (INDI), namely online RLS estimation, predictor-based MRAC and L1 adaptive
augmentation, under rotor faults. The vehicles are a quadrotor and a quadplane VTOL, flown in
hover, slow forward flight and during the forward transition. Every method is compared against
a baseline INDI and an "oracle" INDI that knows the true fault.

Author: Azer Valizada (Technical University of Munich).

The paper is in `paper/` (`main.tex`, `main.pdf`); its figures are read from `results/`.

## Structure

```
src/                  models and controllers
  params.py           quadrotor parameters, rotor layout, control-effectiveness matrix, gains
  rotations.py        quaternion helpers (scalar-first, body -> NED)
  quadrotor.py        "true" quadrotor plant: RK4, first-order motors, noisy sensors, faults (eta_i)
  indi.py             baseline INDI: position -> attitude -> INDI angular-acceleration loop
  rls.py, mrac.py     INDI + RLS / predictor-based MRAC estimate of each rotor's effectiveness
  l1.py               INDI + L1 augmentation (lumped uncertainty, piecewise-constant law)
  oracle.py           INDI that knows the true fault
  sim.py, scenarios.py, controllers.py   quadrotor simulation loop, fixed scenarios and metrics, registry
  qp_params.py        quadplane: PX4 standard_vtol + Beard & McLain aerodynamics (sources and deviations listed)
  quadplane.py        "true" quadplane plant: lift rotors, pusher, wing, servos, airspeed sensor
  qp_indi.py, wls.py  quadplane INDI with weighted-least-squares control allocation
  qp_adaptive.py      RLS, MRAC and L1 on the quadplane
  qp_variants.py      sensitivity variants (not the main results)
  qp_sim.py, qp_scenarios.py, qp_controllers.py   quadplane simulation, scenarios, registry
  tr_scenarios.py     faults during the forward transition
tests/                automatic checks (pytest)
analysis/             delay-margin model, model sensitivity, aerodynamic mismatch, noise seeds
run_*.py              experiments (one per result set) -> results/
tune.py, tune_qp.py   tuning on separate tuning sets, never on the benchmark scenarios
scripts/pipeline.sh   resumable pipeline: quadplane tuning and every quadplane result
results/              result tables (.txt), per-run data (.csv), tuning logs, figures
paper/                manuscript
```

The plant and the controller are kept strictly apart: faults live only in the plant, and the
controller works from its own nominal model and the sensor outputs.

## Conventions

NED inertial frame, FRD body frame. Rotors 1-4 are front-right, rear-right, rear-left and
front-left (X configuration). A fault `eta_i < 1` scales the real thrust of rotor i at a given
speed, as a damaged propeller would; the RPM-based thrust measurement does not see it.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                             # all checks, 2-6 min
python run_benchmark.py --mc 100   # quadrotor: fault-free, gusts, fault grid, 100 Monte Carlo runs
python tune.py rls                 # re-tune a quadrotor method (rls | mrac | l1)
python run_robustness.py           # quadrotor: unmodelled motor-command delay
bash scripts/pipeline.sh tune      # quadplane tuning, ~1.5-2 h on 2 cores
bash scripts/pipeline.sh rerun     # every quadplane and transition result, ~6-8 h on 2 cores (resumable)
python run_figures.py              # paper figures from the result files
```

Times are from a 2-core machine and vary by a factor of 2-5 between machines. The pipeline
writes logs to `results/logs/` and commits each finished step.

To add a controller, subclass `INDIController`, override `effectiveness()` and add one line to
the registry (`src/controllers.py` or `src/qp_controllers.py`). The tests and benchmarks pick it up.

## License

Code: MIT (see `LICENSE`). The manuscript in `paper/` is under review and is not covered by the MIT license.
