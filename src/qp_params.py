"""Quadplane parameters (Stage 3). Every number has a source; deviations are listed below.

Sources
-------
[PX4]  PX4 Gazebo model `standard_vtol` (github.com/PX4/PX4-gazebo-models, models/standard_vtol/
       model.sdf, cloned 27 Oct 2026): mass, inertia, rotor positions and spin, pusher position,
       wing area and span, surface limits.
[BM]   Beard & McLain, "Small Unmanned Aircraft", Aerosonde coefficients as published in
       github.com/byu-magicc/mavsim_public, mavsim_python/parameters/aerosonde_parameters.py
       (commit 10363e0, 12 Jun 2026): all NON-DIMENSIONAL aerodynamic coefficients.

Gazebo uses FLU axes (y left, z up); everything here is converted to FRD (y right, z down).

Deviations from PX4 (decided 27 Oct 2026, see decision log), each with its reason
-------------------------------------------------------------------------------
* Lift-rotor max thrust 24.5 N instead of 45 N (2e-5 x 1500^2): PX4 gives thrust/weight 3.7, at
  which a rotor fault is trivially absorbed. Real quadplanes size the lift system near T/W = 2
  (it is dead weight in cruise). Hover then uses 50 % of a rotor.
* Yaw torque per thrust k_m = 0.025 m instead of 0.06: estimated for a ~15 inch propeller,
  k_m = D * C_Q / C_T = 0.38 m * (0.045 / 2 pi) / 0.11 = 0.025 m (typical C_T = 0.11, C_P = 0.045).
* Pusher max thrust 20 N instead of 105 N: cruise drag at 15 m/s is ~7 N here; 20 N is a
  realistic pusher for a 5 kg airframe.
* Motor time constant: one value 0.02 s (PX4: 0.0125 s up, 0.025 s down).
* Surface limit +-25 deg instead of +-45 deg: [BM] coefficients are linear, which is not credible
  beyond ~25 deg. Servo dynamics (not in PX4): first order, 0.03 s.
* Aerodynamics: the PX4 lift-drag plugin (one CL slope per half wing) has no control or damping
  derivatives. We use the full [BM] coefficient set on the PX4 geometry (S = 1.0 m^2 = 2 x 0.5,
  b = 2.144 m from the collision box, c = S / b). The [BM] drag polar is replaced by a flat-plate
  blend beyond stall (see quadplane.py), because at alpha ~ 90 deg (vertical descent) the [BM]
  formula gives an unphysical C_D ~ 6.
Known simplifications: no rotor-wing interaction, rotor thrust independent of airspeed, no
rotor drag, no rudder (PX4 standard_vtol has none), constant air density.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class QPParams:
    # ---- rigid body [PX4] ----
    mass: float = 5.0
    J: np.ndarray = field(default_factory=lambda: np.diag([0.477708, 0.341667, 0.811042]))
    g: float = 9.81
    # ---- lift rotors: FRD positions, numbering as in Stage 1-2 ----
    # 1 front-right [PX4 rotor_0, ccw], 2 rear-right [rotor_3, cw],
    # 3 rear-left [rotor_1, ccw], 4 front-left [rotor_2, cw]
    rotor_xy: np.ndarray = field(default_factory=lambda: np.array(
        [[0.35, 0.35], [-0.35, 0.35], [-0.35, -0.35], [0.35, -0.35]]))
    # spin = +1: reaction torque on the body is +z (nose right) = rotor turns CCW seen from above
    spin: np.ndarray = field(default_factory=lambda: np.array([1.0, -1.0, 1.0, -1.0]))
    k_m: float = 0.025                  # m, yaw torque per N of thrust (see deviations)
    T_max: float = 24.5                 # N per lift rotor (see deviations)
    tau_motor: float = 0.02             # s
    # ---- pusher [PX4 rotor_puller at x = -0.22 m on the axis, thrust along +x_b] ----
    T_push_max: float = 20.0            # N (see deviations)
    k_m_push: float = 0.01              # m, roll torque per N [PX4 momentConstant], cw seen from behind
    # ---- control surfaces ----
    delta_max: float = np.radians(25.0) # rad, aileron (antisymmetric) and elevator
    tau_servo: float = 0.03             # s
    # ---- air and wing geometry [PX4] ----
    rho: float = 1.225
    S: float = 1.0                      # m^2
    b: float = 2.144                    # m
    # ---- non-dimensional aerodynamics [BM] ----
    e_oswald: float = 0.9
    C_L_0: float = 0.23
    C_L_alpha: float = 5.61
    C_L_q: float = 7.95
    C_L_delta_e: float = 0.13
    C_D_p: float = 0.043
    C_D_q: float = 0.0
    C_D_delta_e: float = 0.0135
    C_m_0: float = 0.0135
    C_m_alpha: float = -2.74
    C_m_q: float = -38.21
    C_m_delta_e: float = -0.99
    M_blend: float = 50.0
    alpha0: float = 0.47
    beta_sat: float = 0.52              # rad (30 deg): sideslip terms saturate (see quadplane.aero)
    C_Y_0: float = 0.0
    C_Y_beta: float = -0.98
    C_Y_p: float = 0.0
    C_Y_r: float = 0.0
    C_Y_delta_a: float = 0.075
    C_ell_0: float = 0.0
    C_ell_beta: float = -0.13
    C_ell_p: float = -0.51
    C_ell_r: float = 0.25
    C_ell_delta_a: float = 0.17
    C_n_0: float = 0.0
    C_n_beta: float = 0.073
    C_n_p: float = 0.069
    C_n_r: float = -0.095
    C_n_delta_a: float = -0.011

    @property
    def c(self):
        return self.S / self.b          # mean chord, 0.466 m

    @property
    def AR(self):
        return self.b ** 2 / self.S

    def B_rotor_moments(self):
        """3x4: lift-rotor thrusts [N] -> body moments [Nm] (same physics as params.py)."""
        xy = self.rotor_xy
        return np.vstack([-xy[:, 1], xy[:, 0], self.k_m * self.spin])

    def surface_moment_per_rad(self, Va):
        """3x2: d(moment)/d(delta_a, delta_e) [Nm/rad] at airspeed Va (alpha, beta ~ 0).
        Grows with dynamic pressure 0.5 rho Va^2: zero in hover."""
        qS = 0.5 * self.rho * Va ** 2 * self.S
        return np.array([[qS * self.b * self.C_ell_delta_a, 0.0],
                         [0.0, qS * self.c * self.C_m_delta_e],
                         [qS * self.b * self.C_n_delta_a, 0.0]])
