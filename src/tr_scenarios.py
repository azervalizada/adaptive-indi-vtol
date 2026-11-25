"""Stage 4: rotor faults DURING the forward transition (hover -> cruise). Scenarios and metrics fixed
on 28 Oct 2026 before any controller was run on them; the controllers keep their Stage 3 settings
(no re-tuning: the transition tests whether those settings carry over).

Manoeuvre: hover at 20 m until T_START = 3 s, then accelerate north at a constant `accel` until
V_CRUISE = 16 m/s, then hold it. Altitude and heading held, pitch held at 0 (as in Stage 3); the
pusher provides the acceleration. The reference carries the acceleration as feedforward.

Fault: one lift rotor, at the moment the reference speed passes V_f (t0 = T_START + V_f / accel).
Question: can the aircraft outrun a fault that would be fatal at constant speed, i.e. reach the
Stage 3 minimum safe airspeed before it loses control?

Metrics: those of Stage 3 (qp_scenarios.qp_metrics), with the window extended to cover the whole
remaining transition: [t0, max(t0 + 4 s, end of acceleration + 2 s)]. Added for Stage 4:
  final_speed_err   |mean airspeed over the last 0.5 s - V_CRUISE|
  completed         recovered (tilt < 1 deg, heading < 5 deg at the end) AND final_speed_err < 1 m/s
"""
from dataclasses import dataclass
import numpy as np

from .scenarios import Fault
from .qp_scenarios import WINDOW_S, TAIL_S, qp_metrics

T_START = 3.0
V_CRUISE = 16.0
Z_REF = -20.0
ACCELS = (1.0, 2.0, 3.0)
FAULT_SPEEDS = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
ROTORS = (0, 1)          # 1 front-right, 2 rear-right; front/rear pairs behaved alike in Stage 3
ETAS = (0.3, 0.0)


def transition_reference(accel):
    t_acc = V_CRUISE / accel

    def ref(t):
        s = t - T_START
        if s <= 0:
            x, v, a = 0.0, 0.0, 0.0
        elif s < t_acc:
            x, v, a = 0.5 * accel * s * s, accel * s, accel
        else:
            x, v, a = 0.5 * accel * t_acc ** 2 + V_CRUISE * (s - t_acc), V_CRUISE, 0.0
        return {"pos": np.array([x, 0.0, Z_REF]), "vel": np.array([v, 0.0, 0.0]),
                "acc": np.array([a, 0.0, 0.0]), "yaw": 0.0}
    return ref


@dataclass(frozen=True)
class TRScenario:
    name: str
    accel: float
    V_f: float = None              # reference speed at the fault (None: no fault)
    rotor: int = 0
    eta: float = 1.0
    ramp: float = 0.0
    seed: int = 0
    mass_scale: float = 1.0
    J_scale: tuple = (1.0, 1.0, 1.0)
    surf_scale: tuple = (1.0, 1.0)
    delay_s: float = 0.0
    gust = None
    V = 0.0                        # starts in hover

    @property
    def fault(self):
        if self.V_f is None:
            return None
        return Fault(self.rotor, self.eta, T_START + self.V_f / self.accel, self.ramp)

    @property
    def t0(self):
        return self.fault.t0 if self.fault is not None else T_START

    @property
    def t_end(self):
        end_acc = T_START + V_CRUISE / self.accel
        ramp = self.ramp if self.fault is not None else 0.0
        return max(self.t0 + ramp + WINDOW_S, end_acc + 2.0) + 0.01

    def label(self):
        s = f"{self.name}: accel {self.accel:g} m/s2"
        if self.fault is None:
            return s + ", no fault"
        return s + f", fault at {self.V_f:g} m/s, rotor {self.rotor + 1}, eta {self.eta:.2f}"


def tr_grid():
    """3 accelerations x 7 fault speeds x 2 rotors (front, rear) x eta {0.3, 0}: 84 scenarios."""
    return [TRScenario("tr", a, Vf, r, e) for e in ETAS for a in ACCELS for Vf in FAULT_SPEEDS for r in ROTORS]


def tr_nominal():
    return [TRScenario("tr-nominal", a) for a in ACCELS]


def tr_metrics(res, sc):
    m = qp_metrics(res, sc)
    if m["lost"]:
        m.update(final_speed_err=None, completed=False)
        return m
    t = res["t"]
    t1 = sc.t_end - 0.01
    tail = (t >= t1 - TAIL_S) & (t < t1)
    m["final_speed_err"] = float(abs(np.mean(res["Va"][tail]) - V_CRUISE))
    m["completed"] = bool(m["recovered"] and m["final_speed_err"] < 1.0)
    return m
