"""Vehicle and controller parameters.

Conventions
-----------
* Inertial frame: NED (x north, y east, z down).
* Body frame: FRD (x forward, y right, z down). Rotor thrust acts along -z_b.
* Rotor layout (X configuration, seen from above):

        4 (front-left, CCW)     1 (front-right, CW)
                        \\      /
                          body
                        /      \\
        3 (rear-left, CW)       2 (rear-right, CCW)

  Spin direction as seen from above. A rotor's reaction torque on the body is opposite
  to its spin: a CCW rotor pushes the body CW, i.e. +z yaw (nose right) in FRD.
  spin = +1 means the reaction torque on the body is +z (yaw right).
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class QuadParams:
    mass: float = 1.0                    # kg
    J: np.ndarray = field(default_factory=lambda: np.diag([0.0089, 0.0089, 0.0160]))  # kg m^2
    arm: float = 0.17                    # m, centre to rotor
    k_m: float = 0.016                   # m, yaw torque per unit thrust
    T_max: float = 6.0                   # N, max thrust per rotor
    tau_motor: float = 0.02              # s, first-order motor time constant
    g: float = 9.81

    @property
    def rotor_xy(self):
        a = self.arm / np.sqrt(2.0)
        return np.array([[+a, +a],   # 1 front-right
                         [-a, +a],   # 2 rear-right
                         [-a, -a],   # 3 rear-left
                         [+a, -a]])  # 4 front-left

    @property
    def spin(self):
        # diagonal pairs (1,3) and (2,4) spin the same way
        return np.array([-1.0, +1.0, -1.0, +1.0])

    def B_moments(self):
        """3x4 map from rotor thrusts T_i [N] to body moments [L, M, N] [Nm].

        Force F = (0, 0, -T) at r = (x, y, 0): r x F = (-y T, x T, 0).
        """
        xy = self.rotor_xy
        L = -xy[:, 1]
        M = +xy[:, 0]
        N = self.k_m * self.spin
        return np.vstack([L, M, N])

    def B_full(self):
        """4x4 map from rotor thrusts to [L, M, N, total thrust]."""
        return np.vstack([self.B_moments(), np.ones(4)])


@dataclass
class INDIGains:
    # outer position loop (PD on NED position)
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.2, 2.5]))
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([1.8, 1.8, 3.0]))
    a_max_xy: float = 4.0                 # m/s^2, horizontal accel limit
    # attitude loop: quaternion error -> rate command
    k_att: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 3.0]))
    # rate loop: rate error -> desired angular acceleration
    k_rate: np.ndarray = field(default_factory=lambda: np.array([25.0, 25.0, 12.0]))
    # second-order synchronisation filter applied to gyro and actuator signals
    filt_wn: float = 50.0                 # rad/s
    filt_zeta: float = 0.55
