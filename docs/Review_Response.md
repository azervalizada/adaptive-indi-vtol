# Response to the independent code review (28 Oct 2026)

The review was a blind review of the repository at commit `4b0d63e`, run by Azer with Claude Code
(Opus). Its report is `REVIEW.md` in the review copy, with 17 check scripts. This file lists every
finding, what was changed, and what the change did to the results.

Code state after the fixes: commits `7655753` (model), `24c3dac` (code), `1237111` (tuning) and
later. Every Stage 3/4 result was rerun with `scripts/rerun_stage34.sh`. The Stage 1–2 (quadrotor)
results do not use any changed code and are unchanged. The only exception is `tune.py` m1, a print
statement.

## Summary of the effect on the results

| | before (4b0d63e) | after |
|---|---|---|
| Stage 3 grid: lost (baseline / oracle / RLS / MRAC / L1) | 2 / 2 / 3 / 2 / 4 | 0 / 0 / 0 / 0 / 4 |
| Stage 3 grid gap closed: RLS | 42 % [−5, 68] | η_min 0.1 (new main): 108 % [102, 116]; η_min 0: 78 % [48, 96] |
| Stage 3 grid gap closed: MRAC | 86 % [81, 91] | 74 % [64, 87] |
| Stage 3 grid gap closed: L1 | 140 % [134, 149] | 146 % [140, 153] |
| Stage 3 Monte Carlo gap closed: RLS / MRAC / L1 | 191 / 139 / 346 % (n = 58, very wide CIs) | 111 / 79 / 158 % (RLS η_min 0: −42 %) |
| Stage 4 gap closed: RLS / MRAC / L1 | 38 / 81 / 127 % | 86 / 73 / 129 % (RLS η_min 0: 23 %) |
| Minimum safe airspeed, baseline, front/rear (η 0.3 / 0.15 / 0) | 4/5, 6/7, 6/8 m/s | unchanged |
| Outrun, complete front-rotor loss | completes from 4 m/s (1 m/s²), 2 m/s (2–3 m/s²) | completes from 4 m/s at every acceleration |
| Delay margin, baseline | "≥ 60 ms" (8 m/s only) | 45 ms in hover, ≥ 60 ms at 8 m/s |
| Delay margin, L1 | 20 ms | 20 ms (hover and 8 m/s) |

Why the Monte Carlo moved so much: its gap closed was a ratio of small sums with intervals hundreds
of percent wide. The model fix (M1) changed the low-speed cases, and the fixed RLS (C1) exposed its
lock-in failure.

## Critical

**C1. The quadplane RLS regression left the wing, pusher and gyroscopic terms in the measurement.**
Confirmed. RLS and MRAC now share one model of the known non-rotor part (`KnownPart` in
`src/qp_adaptive.py`): the nominal aerodynamics, the pusher torque and ω × Jω. It is passed through
the same synchronisation filter as the measurement. Batch least squares on noise-free data now
returns the true η at 0, 8 and 12 m/s (8 m/s: [0.601, 1.004, 0.999, 1.000], before
[−0.13, −3.57, 0.46, 0.54]). This is a test now: `tests/test_qp_adaptive.py`.

Effect: once RLS is correct and re-tuned, its remaining weakness is the η_min = 0 lock-in. An abrupt
fault drives all four estimates to 0 within 20–50 ms. The allocation then stops moving the rotors,
so no new information arrives. This dominates the Monte Carlo (mc088: 33° peak tilt with bound 0,
0.38° with 0.1) and the transitions. **Decision (28 Oct 2026, Azer):** the main RLS uses
η_min = 0.1, as in Stage 2, tuned with the same rule and procedure (`results/tuning_qp_rls.csv`).
η_min = 0 is reported as the sensitivity variant `rls_eta0`. This choice was changed after seeing
results, and the paper says so. MRAC keeps η_min = 0, because 0.1 gives it no clear gain.

**C2. The by-speed table averaged each controller over its own survivors.** Confirmed and fixed:
the table now uses the common survived set per speed and prints n. Run on the old data it reproduces
the reviewer's numbers exactly (hover 2.03 / 0.93 / 1.13 / 1.25 / 0.50). This is a test now:
`tests/test_summaries.py`.

## Major

**M1. Moments and side force were not stall-blended, and C_m jumped at α = ±180°.** Confirmed and
fixed in `src/quadplane.py`. The pitch moment now uses the same stall blend as lift. Beyond stall it
is the flat-plate normal force 2 sin α acting at the linear static margin, so it is continuous and
periodic. The sideslip terms saturate as β_s tanh(β/β_s) with β_s = 30°. Nothing changes at small
angles (trim and G(V) are identical).

Effect: the hover "losses" disappear. With complete rotor loss in hover, the controllers hold
altitude on the two diagonal rotors and give up yaw (the known spinning-quadrotor mode). A 25 s run
shows this is not a recovery either: yaw rate keeps growing (front rotor), or the aircraft sinks
17 m in 20 s (rear rotor). So "survived" versus "lost" for complete loss in hover depends on the
window, and the paper says so. The minimum safe airspeed and the outrun results are unchanged or
nearly so, as the reviewer expected.

**M2. The baseline delay margin "≥ 60 ms" held only at 8 m/s.** Confirmed. `run_qp_robustness.py`
now has a hover set and a 5 ms grid. Baseline, oracle, RLS and MRAC have 45 ms in hover and
≥ 60 ms at 8 m/s; L1 has 20 ms in both. The linear model gives 44 ms and 22 ms. Both are now
labelled as oscillation-threshold margins (m7).

**M3. MRAC had the exact aero model and perfect air data.** Addressed in three ways:
1. RLS now has exactly the same model knowledge as MRAC (C1).
2. Variants without the aero model, tuned with the same rule. MRAC without it: **no setting passes
   the rule** (healthy-flight drift ≥ 0.5 for all 48 candidates). RLS without it: passes, but tunes
   itself towards barely adapting.
3. `analysis/qp_aero_mismatch.py`: true aero coefficients ±30 % and unknown wind up to 3 m/s. The
   wrong coefficients alone are harmless. Unknown wind breaks MRAC: in 40 healthy flights its
   estimates drift > 0.15 in 75 % (RLS 10 %), because it takes ground velocity as air velocity.

**M4. Errored transition runs counted as survived.** Fixed. An errored run is now lost, flagged in
every table and kept in the CSV (`error` column). No run errored in the reruns.

**M5. Unequal tuning budgets, and winners on grid edges.** Fixed with one procedure for every
method (`tune_qp.py`): a 48-candidate coarse grid, then a local pattern search of up to 48 more
candidates on ordered ladders. It stops when no neighbour on any axis is better. Every winner is now
interior with all neighbours evaluated, or at a physical bound: RLS lag = 1, MRAC γ_d = 0, and L1
ωc = 1280 rad/s at the Nyquist side of the 500 Hz loop. L1's delay margin is about 20 ms at
ωc = 320, 640 and 1280 alike (it is set by the predictor pole), so its winner on that bound costs no
robustness.

**M6. Figures had numbers typed in.** Fixed: `run_figures.py` reads every value, and every claim in a
title, from the result files.

**M7. RLS reads a gust as a fault on all four rotors.** Still true for the corrected RLS
(healthy_dev 0.43 in the gust check). Stated as a limitation: RLS does not *identify* faults, it
fits an effectiveness to whatever the rotors did not deliver.

## Minor

| | finding | action |
|---|---|---|
| m1 | `tune.py` print raised TypeError | fixed (missing `+`) |
| m2 | gap closed depended on the other controllers in the file | now uses only baseline, oracle and the method (Stage 2 rule) |
| m3 | minimum-speed scan skipped missing speeds | a missing or errored speed now stops the scan and is marked `?` |
| m4 | bootstrap CI depended on row order | scenarios sorted before resampling |
| m5 | outrun compared every controller with the baseline's Stage 3 value | each controller is compared with its own; the different criteria are stated |
| m6 | feasibility equilibria were not level | vertical velocity = 0 added. A case with \|pitch\| ≤ 1° matches the simulation's tolerance |
| m7 | linear margin valid only in hover, and the sim margin is an oscillation threshold | labelled as such |
| m8 | fig2 label said "commanded" | now "thrust demand (after motor lag)" |
| m9 | `_why_lost` tested tilt before the fault | only after the fault |
| m10 | transition healthy window up to 12 s | kept (same for every controller), noted |
| m11 | tuning set near two sweep cases | kept and noted. It does not touch the grid or the Monte Carlo |
| m12 | MRAC compared unfiltered aero with filtered acceleration | both filtered now (`KnownPart`) |
| m13 | L1 predictor models everything with the motor lag | kept as designed (L1 has no vehicle model), noted |
| m14 | RLS estimates hit 0 even in hover | the lock-in of C1; main RLS now uses η_min = 0.1 |
| m15 | `u_hist` grew without delay | fixed |
| m16 | airspeed noise is clipped at 0, biased about +0.16 m/s in hover | kept, noted (negligible effect on G) |
| m17 | `run_tr` wrote rotor = 1 without a fault | fixed |
| style | C_D,δe·\|δe\|, column `surv`, `sat_frac` 5.999 | documented / renamed to `surv_nc` / uses T_max |

## Tests added

- `tests/test_qp_adaptive.py`:
  - RLS regression unbiased on noise-free data at 0, 8 and 12 m/s;
  - the variant without the aero model is biased;
  - MRAC healthy flight;
  - main settings equal the tuning winners.
- `tests/test_summaries.py`: common-set speed table, gap closed independent of other controllers,
  bootstrap order, missing or errored speeds, sorting.
- `tests/test_quadplane.py`: static moments bounded and continuous all around the aircraft.
- `tests/test_qp_control.py`: the hover complete-loss test now expects "not recovered" (it was
  "lost").

## Follow-up after the response (30 Oct 2026)

- **M5 on the quadrotor too.** `tune.py` now uses the same procedure as `tune_qp.py`. Where a first
  search ended on a ladder edge that is not a physical bound (MRAC gamma_T, L1 a), the ladder was
  extended and the method re-tuned. Stage 2 rerun:
  - grid gap closed: RLS 153 %, MRAC 87 %, L1 183 % (before 133 / 82 / 177 %);
  - Monte Carlo: 183 / 92 / 200 %;
  - ranking and delay margins unchanged; all pairwise differences significant.
- **Noise seeds** (`analysis/seed_repeat.py`). The Stage 3 grid and the Stage 4 transitions were
  rerun with two more noise seeds. Each gap closed changed by at most 6 points, and the ranking did
  not change.
- **Stage 4 tests** (`tests/test_transition.py`): the reference, the fault time and window, and the
  `completed` / lost logic.
- **Bibliography** checked against publisher and repository pages.

# Response to the second independent review (4 Nov 2026)

The second review was blind as well (`scripts/REVIEW2_PROMPT.md`): a copy with code, tests and README
only, no results, docs, paper or git history. It found no critical problems and four major ones.
Decisions (Azer, 4 Nov 2026): fix the aero model and rerun everything (M1, M2); report the RLS
estimate error honestly without changing the design (M3); describe the tuning asymmetry openly
without adding a new rule (M4).

## Effect on the results

Re-tune and rerun of every Stage 3/4 result with the corrected aero model (`scripts/pipeline.sh`).
The re-tune returned the same settings for all three methods.

| | before (28 Oct rerun) | after (4 Nov) |
|---|---|---|
| Stage 3 grid: lost (baseline / oracle / RLS / MRAC / L1) | 0 / 0 / 0 / 0 / 4 | 2 / 2 / 2 / 2 / 6 |
| Stage 3 grid gap closed, own set: RLS / MRAC / L1 | 108 / 74 / 146 % | 106 / 63 / 145 % |
| Stage 3 grid gap closed, common set (new): RLS / MRAC / L1 | 112 / 81 / 146 % | 112 / 81 / 145 % |
| Stage 3 Monte Carlo gap closed: RLS / MRAC / L1 | 111 / 79 / 158 % | 109 / 79 / 147 % |
| Stage 3 sweep gap closed: RLS / MRAC / L1 | 100 / 58 / 148 % | 96 / 48 / 126 % |
| Minimum safe airspeed, complete loss, front / rear | 6 / 8 m/s | 6-7 / 8 m/s (by rotor and controller) |
| "L1 gains most at 4 m/s" | 0.7 deg vs 8.2 deg baseline | withdrawn: L1 is lost in 2 complete-loss runs at 4 m/s that the others survive |
| Stage 4 gap closed: RLS / MRAC / L1 | 86 / 73 / 129 % | 86 / 78 / 134 % (common set 86 / 83 / 134 %) |
| Stage 4 losses of 84: baseline / RLS / MRAC / L1 | 6 / 6 / 6 / 6 | 7 / 8 / 4 / 8 |
| Outrun, complete loss, rear rotor at 1 m/s² | 4 m/s | 2 m/s (L1: 4) |
| Delay margins | 45 / >= 60 ms, L1 20 ms | unchanged |
| Aero/wind mismatch MC: RLS / MRAC / L1 | 102 / 60 / 182 % | 106 / 58 / 171 % |
| MRAC healthy drift > 0.15 with wind | 75 % | 75 % |
| Static feasibility, controller's constraints, front / rear | 4 / 5, 5.5 / 6.5, 6 / 8 m/s | unchanged |
| Noise seeds: largest change in gap closed | 6 points | 8 points (RLS grid 106 / 107 / 114 %); ranking unchanged |

The losses that appeared are complete rotor losses in hover (rear rotors, all controllers) and at
4 m/s (L1). These runs pass through large sideslip and angle of attack, where the model is least
trustworthy and which the fix changed; the README and paper now report them as "no controller
recovers", not as a ranking.

## Major

**M1. The aero model was discontinuous at large sideslip.** Confirmed: in pure sideways flow the
wing force jumped between −61 N and +61 N as u and w changed sign, because α = atan2(w, u) is
undefined there but the full dynamic pressure was still applied. Fixed in `src/quadplane.py`: lift,
drag and pitch moment now use the dynamic pressure of the flow in the aircraft's x–z plane,
0.5 ρ (u² + w²). Sideways flow now gives the same force for every sign of u and w
(`test_sideways_flow_is_continuous`). Straight flight is unchanged to the last digit, so trim and
G(V) are unchanged. The yaw-spin runs (complete loss in hover) are the cases that fly through this
region; their outcomes are model-limited and the paper says so.

**M2. The post-stall pitch moment did not match the post-stall force.** Fixed: beyond stall the
moment is the flat-plate normal force 2 sin α |sin α| acting at the linear static margin
(C_mα / C_Lα), the same normal force the lift and drag use
(`test_post_stall_moment_matches_post_stall_force`).

**M3. RLS does not identify the fault.** Confirmed and kept as a reported limitation. After an
abrupt fault the dead-zone stops the update within about 0.2 s, because the allocation stops
moving the rotors and no new information arrives. The estimate then stays at a wrong value
(error 0.18–0.21 in the reviewer's runs; for example 8 m/s, rotor 1 to 0.6:
η̂ = [0.32, 0.93, 1.21, 0.83]). Control is still good, because the controller only needs an
effectiveness that explains what the rotors did not deliver, not the true one. Changes:
- the Stage 4 table reports the final estimate error (`eta_err`); the Stage 3 tables report the
  worst healthy-flight drift (`hd_max`) next to the mean;
- the README and paper no longer say RLS "identifies" or "estimates" the fault; they say it fits an
  effectiveness that keeps the loop working;
- a test pins the behaviour (`test_rls_estimate_freezes_after_an_abrupt_fault_known_limitation`):
  the estimate freezes, is wrong by more than 0.1, and the run still recovers. If a later change
  makes RLS identify the fault, the test fails and the text must be updated.

**M4. The tuning is equal in procedure and budget, but not in resolution or constraint.** Confirmed;
stated in the README and the paper, no new rule:
- L1 has two parameters, so its 48 coarse candidates form a 4 × 12 grid; RLS and MRAC share theirs
  over 3–4 parameters (3 × 4 × 2 × 2). L1's grid is finer per axis.
- Rule (c) (healthy-flight drift ≤ 0.15) binds only on the estimators. L1 has no estimate to drift,
  so it is constrained by rules (a) and (b) only. The unconstrained best of RLS and MRAC (rule (a)
  only) would score better but drifts by 0.9–1.0 in healthy flight.
- L1's winner ωc = 1280 rad/s is the upper ladder end: the filter gain per step is 0.923, so this is
  practically the unfiltered limit. L1 then works as a predictor-based disturbance observer. Its
  delay margin does not depend on ωc (about 20 ms at 320, 640 and 1280 rad/s).
- Some decisions are ties: RLS forgetting 1.0 beats 0.999 by 0.07 % of the score, MRAC γ_d = 0
  beats 3 by 0.02 %. The local search takes any strictly better candidate, however small the
  difference, so these choices should be read as ties, not as findings.
- The tuning set is small (no hover, no complete loss, no wind) and each method got the η bound
  that suits it better (RLS 0.1, MRAC 0).

The re-tune with the corrected aero model (4 Nov 2026) returned the same winners for all three main
methods: RLS lag 1, threshold 0.05, forgetting 1.0, p0 0.01; MRAC a 30, Γ_η 16, γ_d 0, γ_T 1; L1
a 30, ωc 1280. MRAC without the aero model still has no admissible setting (0 of 48).

## Minor

| | finding | action |
|---|---|---|
| m1 | the reason for MRAC's wind sensitivity was incomplete | stated: RLS differences its data (Δy), so a constant bias from the wrong airspeed cancels; MRAC uses the known part in absolute terms. `KnownPart` uses ground velocity and ignores the pitot. Not changed |
| m2 | half-sample misalignment between the backward-difference ω̇_f and u_f | noted (hover residual 1.8 %; at speed the coded alignment is best). Test tolerance 0.05 kept |
| m3 | parasitic drag vanished post-stall; control and damping derivatives not stall-blended | drag fixed (C_D,p kept post-stall). Derivatives kept, noted |
| m4 | `run_figures.py` crashed on missing values and had hard-coded title claims | fixed: None handled; titles built from the data |
| m5 | gap closed on per-method sets | both reported: own set and common set |
| m6 | `run_tr_benchmark.py` KeyError on a missing row | fixed; missing rows are counted |
| m7 | trim ignores the wind, so wind runs start out of trim | checked: at 8 m/s in crosswind the aircraft is still turning into the wind at 3.5 s (tilt 0.27°, heading 8°). MRAC's drift persists in 20 s healthy flights (1.0 crosswind, 0.28 headwind at 18–20 s; RLS ≈ 0), so the conclusion holds. Noted |
| m8 | perfect position, velocity and attitude | stated in README and paper |
| m9 | missing runs dropped silently | every table prints a missing-run counter |
| m10 | gust "false detection" is a mean | worst case (0.71) reported next to the mean |
| m11 | runtimes in the README too short | README gives ranges and says they depend on the machine |
