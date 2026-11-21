"""
A small quadrotor (drone) simulator, the size of a Crazyflie 2.x, written with plain NumPy.

It can simulate MANY drones at the same time (one row per drone). That is useful because the planner
tries hundreds of possible command sequences at once.

The 12 numbers that describe the drone (the "state"), in this order:
    0-2   position      x, y, z          (metres)
    3-5   velocity      vx, vy, vz       (metres per second)
    6-8   angles        roll, pitch, yaw (radians)
    9-11  turning rates p, q, r          (radians per second, in the drone's own frame)

The 4 commands, in this order:
    0     total thrust of the four propellers (newton), between 0 and 0.6
    1-3   wanted turning rates (radians per second), between -6 and +6

A fast inner loop (500 times per second) turns the wanted turning rates into twisting forces (torques),
like the "rate controller" inside every real drone's flight computer.
"""

import numpy as np


# ======================================================================================================
# Part 1: the drone's numbers
# ======================================================================================================
GRAVITY = 9.81                      # metres per second squared
NORMAL_MASS = 0.027                 # kg (Crazyflie 2.x)
NORMAL_INERTIA = np.array([1.4e-5, 1.4e-5, 2.17e-5])   # kg m^2 around the x, y, z axes

CONTROL_STEP = 0.02                 # the planner decides every 0.02 s (50 times per second)
SUB_STEP = 0.002                    # the physics is computed every 0.002 s (500 times per second)
SUB_STEPS_PER_CONTROL_STEP = 10     # 0.02 / 0.002

MOTOR_TIME_CONSTANT = 0.03         # the propellers need about 0.03 s to reach a new thrust (real motors are not instant)
RATE_LOOP_GAIN = 20.0               # how quickly the inner loop follows the wanted turning rates (1/s)
MAX_TORQUE = np.array([3e-3, 3e-3, 1e-3])     # largest twisting force the motors can make (N m)

THRUST_MIN = 0.0
THRUST_MAX = 0.6
RATE_LIMIT = 6.0

# Air drag, per second, along the drone's own x, y, z axes. The textbook equations ignore this.
REAL_DRAG = np.array([0.35, 0.35, 0.10])

# Sensor noise (standard deviation) for the 12 measured numbers
SENSOR_NOISE = np.array([0.002, 0.002, 0.002,
                         0.01, 0.01, 0.01,
                         0.003, 0.003, 0.003,
                         0.02, 0.02, 0.02])

HOVER_THRUST = NORMAL_MASS * GRAVITY   # 0.265 N: the thrust that exactly holds the normal drone in the air


def clip_commands(commands):
    """Keep the commands inside what the drone can do. commands: [N, 4]."""
    clipped = commands.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], THRUST_MIN, THRUST_MAX)
    clipped[:, 1:4] = np.clip(clipped[:, 1:4], -RATE_LIMIT, RATE_LIMIT)
    return clipped


# ======================================================================================================
# Part 2: one tiny physics step (0.002 s) for N drones at once
# ======================================================================================================
def physics_sub_step(state, motor_thrust, commands, mass, inertia, drag, dt, motor_time_constant):
    """
    state        [N, 12]
    motor_thrust [N]      the thrust the propellers make RIGHT NOW (it lags behind the command)
    commands     [N, 4]   (already clipped)
    mass     [N]      mass of each drone (changes when a payload is attached)
    inertia  [N, 3]   inertia of each drone
    drag     [3]      air drag per second (zeros = the textbook equations)
    dt       the length of this tiny step in seconds
    motor_time_constant  0 = instant motors (the textbook equations)
    Returns the new state [N, 12] and the new motor thrust [N].
    """

    # Step 1: take the state apart
    velocity = state[:, 3:6]
    roll = state[:, 6]
    pitch = state[:, 7]
    yaw = state[:, 8]
    rates = state[:, 9:12]

    # Step 2: sines and cosines of the three angles
    sin_roll = np.sin(roll)
    cos_roll = np.cos(roll)
    sin_pitch = np.sin(pitch)
    cos_pitch = np.cos(pitch)
    sin_yaw = np.sin(yaw)
    cos_yaw = np.cos(yaw)

    # Step 3: the direction the propellers push (the drone's own "up" axis), in the world frame
    up_x = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    up_y = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    up_z = cos_pitch * cos_roll

    # Step 4: acceleration from the thrust:  thrust / mass, along the "up" axis
    # 4a: the motors move a little towards the commanded thrust (first-order lag)
    if motor_time_constant > 0:
        thrust = motor_thrust + (dt / motor_time_constant) * (commands[:, 0] - motor_thrust)
    else:
        thrust = commands[:, 0]
    thrust_acceleration = thrust / mass
    acc_x = thrust_acceleration * up_x
    acc_y = thrust_acceleration * up_y
    acc_z = thrust_acceleration * up_z - GRAVITY

    # Step 5: air drag (only if drag is not zero)
    if np.any(drag > 0):
        # 5a: the rotation matrix from the drone's frame to the world frame, built column by column
        forward_x = cos_yaw * cos_pitch
        forward_y = sin_yaw * cos_pitch
        forward_z = -sin_pitch
        side_x = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
        side_y = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
        side_z = cos_pitch * sin_roll

        # 5b: the velocity seen from the drone (dot product with each of its axes)
        v_forward = velocity[:, 0] * forward_x + velocity[:, 1] * forward_y + velocity[:, 2] * forward_z
        v_side = velocity[:, 0] * side_x + velocity[:, 1] * side_y + velocity[:, 2] * side_z
        v_up = velocity[:, 0] * up_x + velocity[:, 1] * up_y + velocity[:, 2] * up_z

        # 5c: drag pushes against that velocity, then we turn it back into the world frame
        drag_forward = -drag[0] * v_forward
        drag_side = -drag[1] * v_side
        drag_up = -drag[2] * v_up
        acc_x = acc_x + drag_forward * forward_x + drag_side * side_x + drag_up * up_x
        acc_y = acc_y + drag_forward * forward_y + drag_side * side_y + drag_up * up_y
        acc_z = acc_z + drag_forward * forward_z + drag_side * side_z + drag_up * up_z

    # Step 6: the inner turning-rate loop: torque = normal inertia x gain x (wanted rate - rate)
    # (the flight computer uses the NORMAL inertia; with a payload the drone turns more slowly)
    wanted_rates = commands[:, 1:4]
    rate_error = wanted_rates - rates
    torque = NORMAL_INERTIA * RATE_LOOP_GAIN * rate_error
    torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)

    # Step 7: how the turning rates change (Euler's equation for a spinning body)
    p = rates[:, 0]
    q = rates[:, 1]
    r = rates[:, 2]
    Ix = inertia[:, 0]
    Iy = inertia[:, 1]
    Iz = inertia[:, 2]
    p_dot = (torque[:, 0] - (Iz - Iy) * q * r) / Ix
    q_dot = (torque[:, 1] - (Ix - Iz) * p * r) / Iy
    r_dot = (torque[:, 2] - (Iy - Ix) * p * q) / Iz

    # Step 8: how the three angles change, from the turning rates
    tan_pitch = sin_pitch / cos_pitch
    roll_dot = p + sin_roll * tan_pitch * q + cos_roll * tan_pitch * r
    pitch_dot = cos_roll * q - sin_roll * r
    yaw_dot = (sin_roll * q + cos_roll * r) / cos_pitch

    # Step 9: move everything forward by dt (simple Euler step; dt is tiny)
    new_state = state.copy()
    new_state[:, 0:3] = state[:, 0:3] + dt * velocity
    new_state[:, 3] = state[:, 3] + dt * acc_x
    new_state[:, 4] = state[:, 4] + dt * acc_y
    new_state[:, 5] = state[:, 5] + dt * acc_z
    new_state[:, 6] = roll + dt * roll_dot
    new_state[:, 7] = pitch + dt * pitch_dot
    new_state[:, 8] = yaw + dt * yaw_dot
    new_state[:, 9] = p + dt * p_dot
    new_state[:, 10] = q + dt * q_dot
    new_state[:, 11] = r + dt * r_dot
    return new_state, thrust


# ======================================================================================================
# Part 3: one control step (0.02 s) = several tiny physics steps with the same command
# ======================================================================================================
def control_step(state, motor_thrust, commands, mass, inertia, drag=REAL_DRAG,
                 motor_time_constant=MOTOR_TIME_CONSTANT, sub_steps=SUB_STEPS_PER_CONTROL_STEP):
    """Move N drones forward by one control step (0.02 s). Returns the new state [N, 12] and motor thrust [N]."""

    # Step 1: keep the commands inside the limits
    safe_commands = clip_commands(commands)

    # Step 2: the length of one tiny step
    dt = CONTROL_STEP / sub_steps

    # Step 3: repeat the tiny step
    new_state = state
    new_thrust = motor_thrust
    for _ in range(sub_steps):
        new_state, new_thrust = physics_sub_step(new_state, new_thrust, safe_commands, mass, inertia, drag,
                                                 dt, motor_time_constant)
    return new_state, new_thrust


# ======================================================================================================
# Part 4: the drone controller that small drones use today
#         (cascaded proportional-integral-derivative controller)
# ======================================================================================================
class CascadedPID:
    """
    Outer loop: position error -> wanted acceleration (with an integral term that removes slow offsets).
    Middle:     wanted acceleration -> total thrust and wanted roll / pitch angles.
    Inner:      angle error -> wanted turning rates (the simulator's rate loop does the rest).
    Works on N drones at once.
    """

    def __init__(self, n_drones, kp=50.0, kd=5.0, ki=32.0, k_angle=20.0, integral_limit=3.0):
        # Gains tuned on a CIRCLE path with the payload, sensor noise, motor lag and command delay
        # (grid search over kp, kd, ki and the angle gain), then frozen. The figure-eight test paths were not used.
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.k_angle = k_angle
        self.integral_limit = integral_limit
        self.integral = np.zeros((n_drones, 3))

    def command(self, state, ref_position, ref_velocity, ref_acceleration):
        """All inputs are [N, 3] (state is [N, 12]). Returns commands [N, 4]."""

        # Step 1: position and velocity errors
        position_error = ref_position - state[:, 0:3]
        velocity_error = ref_velocity - state[:, 3:6]

        # Step 2: the integral term (sum of past position errors), kept inside a limit
        self.integral = self.integral + CONTROL_STEP * position_error
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        # Step 3: the wanted acceleration (plus gravity, which must always be cancelled)
        wanted_acc = (self.kp * position_error
                      + self.kd * velocity_error
                      + self.ki * self.integral
                      + ref_acceleration)
        wanted_acc[:, 2] = wanted_acc[:, 2] + GRAVITY

        # Step 4: total thrust = normal mass x (wanted acceleration along the drone's up axis)
        roll = state[:, 6]
        pitch = state[:, 7]
        up_z = np.cos(pitch) * np.cos(roll)
        up_x = np.sin(pitch) * np.cos(roll)
        up_y = -np.sin(roll)
        along_up = wanted_acc[:, 0] * up_x + wanted_acc[:, 1] * up_y + wanted_acc[:, 2] * up_z
        thrust = NORMAL_MASS * along_up

        # Step 5: wanted angles that point the up axis along the wanted acceleration (yaw kept at 0)
        size = np.sqrt(np.sum(wanted_acc ** 2, axis=1)) + 1e-6
        wanted_roll = np.arcsin(np.clip(-wanted_acc[:, 1] / size, -0.6, 0.6))
        wanted_pitch = np.arctan2(wanted_acc[:, 0], wanted_acc[:, 2])
        wanted_pitch = np.clip(wanted_pitch, -0.6, 0.6)

        # Step 6: wanted turning rates from the angle errors
        roll_rate = self.k_angle * (wanted_roll - roll)
        pitch_rate = self.k_angle * (wanted_pitch - pitch)
        yaw_rate = -self.k_angle * state[:, 8]

        # Step 7: put the 4 commands together
        commands = np.zeros((state.shape[0], 4))
        commands[:, 0] = thrust
        commands[:, 1] = roll_rate
        commands[:, 2] = pitch_rate
        commands[:, 3] = yaw_rate
        return clip_commands(commands)


# ======================================================================================================
# Part 5: the figure-eight path used in the test flights
# ======================================================================================================
def figure_eight(t, size, lap_time, height=1.0):
    """
    Position, velocity and acceleration of the reference path at time t (seconds), for ONE drone.
        x = size * sin(w t)
        y = size * sin(2 w t) / 2
        z = height
    Returns three arrays of shape [3].
    """
    w = 2.0 * np.pi / lap_time

    position = np.array([size * np.sin(w * t),
                         0.5 * size * np.sin(2 * w * t),
                         height])
    velocity = np.array([size * w * np.cos(w * t),
                         size * w * np.cos(2 * w * t),
                         0.0])
    acceleration = np.array([-size * w * w * np.sin(w * t),
                             -2.0 * size * w * w * np.sin(2 * w * t),
                             0.0])
    return position, velocity, acceleration
