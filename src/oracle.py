"""Oracle INDI: knows the true rotor effectiveness instantly.

Not a realistic controller. It is the REFERENCE for "perfect fault knowledge": the gap between
baseline INDI and the oracle is what correct effectiveness information is worth.
It knows eta only: in the Monte Carlo set it still uses the nominal mass and inertia.

It is NOT a performance upper bound. A controller that over-reacts to a fault (e.g. an
estimator that briefly underestimates the damaged rotor and drives it to full thrust) can
recover faster than the oracle, at the price of more motor saturation and of reacting the
same way to things that are not faults (gusts). Seen with RLS on 26 Oct 2026: "gap closed" > 100 %.
"""
import numpy as np
from .indi import INDIController


class OracleINDI(INDIController):
    def __init__(self, params_nominal, gains, dt, plant, thrust_feedback="accel"):
        super().__init__(params_nominal, gains, dt, thrust_feedback)
        self.plant = plant

    def effectiveness(self):
        return super().effectiveness() * self.plant.eta[None, :]
