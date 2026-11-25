#!/bin/bash
# Resumable pipeline (4 Nov 2026): every step leaves results/logs/<name>.done when it succeeds and is
# skipped on the next call, so a container restart only loses the step that was running.
#   bash scripts/pipeline.sh tune      tuning of the quadplane methods and variants (~80 min)
#   bash scripts/pipeline.sh rerun     every Stage 3-4 result, seeds, figures (~6 h)
cd "$(dirname "$0")/.."
mkdir -p results/logs
step() {   # step <name> <python args...>
  local name=$1; shift
  [ -f "results/logs/$name.done" ] && { echo "skip $name"; return 0; }
  # ONLY="a b c": run just these steps (keeps each batch under a 2 h job limit)
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && { echo "not in batch: $name"; return 0; }
  echo "=== $(date +%H:%M) $name"
  if ( time python "$@" ) > "results/logs/$name.log" 2>&1; then
    touch "results/logs/$name.done"; git add -A results > /dev/null 2>&1; git commit -qm "pipeline: $name" > /dev/null 2>&1
  else echo "FAILED $name"; tail -5 "results/logs/$name.log"; exit 1; fi
}
if [ "$1" = tune ]; then
  step tune_rls tune_qp.py rls
  step tune_mrac tune_qp.py mrac
  step tune_l1 tune_qp.py l1
  step tune_rls_eta0 tune_qp.py rls --eta-min 0
  step tune_mrac_eta01 tune_qp.py mrac --eta-min 0.1
  step tune_rls_noaero tune_qp.py rls --no-aero
  step tune_mrac_noaero tune_qp.py mrac --no-aero
fi
if [ "$1" = rerun ]; then
  step bench run_qp_benchmark.py --sweep --mc 100
  step robust run_qp_robustness.py
  step tr run_tr_benchmark.py
  step qp_sens run_qp_sensitivity.py
  step tr_sens analysis/tr_sensitivity.py
  step model_sens analysis/qp_model_sensitivity.py
  step aero analysis/qp_aero_mismatch.py
  step feas run_qp_feasibility.py
  step linear analysis/delay_margin.py
  step seeds analysis/seed_repeat.py
  step figures run_figures.py
fi
echo "=== $(date +%H:%M) pipeline $1 done"
