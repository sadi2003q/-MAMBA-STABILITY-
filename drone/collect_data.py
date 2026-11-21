"""
Collect training flights for the learned drone models (normal mass only, no payload).

The cascaded proportional-integral-derivative controller flies random smooth paths, and random
extra noise is added to its commands so the data covers many different situations.

Real-drone details that are simulated:
    - sensor noise on every measured number
    - the motors need about 0.03 s to reach a new thrust
    - a command decided at step k is only applied during step k+1 (0.02 s delay)

Saved file: drone/data/flights.npz with
    measured   [flights, steps + 1, 12]   measured states
    applied    [flights, steps, 4]        the command that was really applied during each step
    previous   [flights, steps, 4]        the command applied during the step before (needed because of the
                                          motor lag: the thrust now still depends on the last command)

Run:  python drone/collect_data.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone import quadrotor_sim as sim


N_FLIGHTS = 300
FLIGHT_SECONDS = 15.0
SEED = 0


def random_smooth_path(rng, n_flights):
    """Random paths made of a few slow sine waves. Returns a function t -> (position, velocity, acc) [N, 3]."""

    # Step 1: for every flight and every axis, 3 sine waves with random size, speed and phase
    sizes = rng.uniform(0.0, 0.5, size=(n_flights, 3, 3))      # metres
    speeds = rng.uniform(0.6, 2.5, size=(n_flights, 3, 3))     # radians per second
    phases = rng.uniform(0.0, 2 * np.pi, size=(n_flights, 3, 3))
    sizes[:, 2, :] = sizes[:, 2, :] * 0.5                       # smaller up-down movement

    # Step 2: the centre of each path
    centre = np.zeros((n_flights, 3))
    centre[:, 2] = rng.uniform(0.8, 1.3, size=n_flights)

    def path(t):
        position = centre.copy()
        velocity = np.zeros((n_flights, 3))
        acceleration = np.zeros((n_flights, 3))
        for wave in range(3):
            angle = speeds[:, :, wave] * t + phases[:, :, wave]
            position = position + sizes[:, :, wave] * np.sin(angle)
            velocity = velocity + sizes[:, :, wave] * speeds[:, :, wave] * np.cos(angle)
            acceleration = acceleration - sizes[:, :, wave] * speeds[:, :, wave] ** 2 * np.sin(angle)
        return position, velocity, acceleration

    return path


def main():
    rng = np.random.default_rng(SEED)
    n_steps = int(round(FLIGHT_SECONDS / sim.CONTROL_STEP))

    # Step 1: random paths and a controller for all flights at once
    path = random_smooth_path(rng, N_FLIGHTS)
    controller = sim.CascadedPID(N_FLIGHTS)

    # Step 2: some flights get a softer controller, so the data is not all from one style of flying
    softness = rng.uniform(0.4, 1.0, size=(N_FLIGHTS, 1))   # one number per flight, used on x, y and z
    controller.kp = controller.kp * softness
    controller.ki = controller.ki * softness

    # Step 3: every flight starts on its path, hovering, with the normal mass
    state = np.zeros((N_FLIGHTS, 12))
    start_position, start_velocity, _ = path(0.0)
    state[:, 0:3] = start_position
    state[:, 3:6] = start_velocity
    motor_thrust = np.full(N_FLIGHTS, sim.HOVER_THRUST)
    mass = np.full(N_FLIGHTS, sim.NORMAL_MASS)
    inertia = np.tile(sim.NORMAL_INERTIA, (N_FLIGHTS, 1))

    # Step 4: random command noise that changes smoothly (correlated over about 0.1 s)
    noise_size = np.array([0.03, 1.0, 1.0, 0.5])
    noise_size = noise_size * rng.uniform(0.3, 1.5, size=(N_FLIGHTS, 1))
    command_noise = np.zeros((N_FLIGHTS, 4))
    keep = 0.8

    # Step 5: the command applied during the first step, and the one before it (hover)
    hover_command = np.zeros((N_FLIGHTS, 4))
    hover_command[:, 0] = sim.HOVER_THRUST
    pending = hover_command.copy()
    previous_applied = hover_command.copy()

    # Step 6: storage
    measured = np.zeros((N_FLIGHTS, n_steps + 1, 12))
    applied = np.zeros((N_FLIGHTS, n_steps, 4))
    previous = np.zeros((N_FLIGHTS, n_steps, 4))

    # Step 7: fly
    for k in range(n_steps):
        t = k * sim.CONTROL_STEP

        # 7a: measure the state (with sensor noise)
        sensor_noise = rng.normal(0.0, 1.0, size=(N_FLIGHTS, 12)) * sim.SENSOR_NOISE
        measurement = state + sensor_noise
        measured[:, k] = measurement

        # 7b: the controller decides the NEXT command (it will act during the next step)
        ref_position, ref_velocity, ref_acceleration = path(t + sim.CONTROL_STEP)
        decided = controller.command(measurement, ref_position, ref_velocity, ref_acceleration)

        # 7c: add the smooth random noise to the decided command
        fresh = rng.normal(0.0, 1.0, size=(N_FLIGHTS, 4)) * noise_size
        command_noise = keep * command_noise + np.sqrt(1 - keep * keep) * fresh
        decided = sim.clip_commands(decided + command_noise)

        # 7d: the drone moves for one step with the command decided one step earlier
        this_command = sim.clip_commands(pending)
        applied[:, k] = this_command
        previous[:, k] = previous_applied
        state, motor_thrust = sim.control_step(state, motor_thrust, this_command, mass, inertia)

        # 7e: shift the commands along
        previous_applied = this_command
        pending = decided

    # Step 8: the last measurement
    sensor_noise = rng.normal(0.0, 1.0, size=(N_FLIGHTS, 12)) * sim.SENSOR_NOISE
    measured[:, n_steps] = state + sensor_noise

    # Step 9: drop flights that went wild (should be none)
    largest_tilt = np.abs(measured[:, :, 6:8]).max(axis=(1, 2))
    lowest_height = measured[:, :, 2].min(axis=1)
    highest = measured[:, :, 2].max(axis=1)
    furthest_sideways = np.abs(measured[:, :, 0:2]).max(axis=(1, 2))
    fastest = np.sqrt((measured[:, :, 3:6] ** 2).sum(axis=2)).max(axis=1)
    good = np.isfinite(measured).all(axis=(1, 2))
    good = good & (largest_tilt < 1.2)              # never tilted more than about 69 degrees
    good = good & (lowest_height > 0.05)            # never touched the ground
    good = good & (highest < 2.5)                   # stayed below 2.5 m
    good = good & (furthest_sideways < 2.5)         # stayed within 2.5 m sideways
    good = good & (fastest < 3.5)                   # never faster than 3.5 m/s (test paths reach about 1.6 m/s)
    print(f"good flights: {int(good.sum())} of {N_FLIGHTS}")

    # Step 10: save
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
    out_path = os.path.join(os.path.dirname(__file__), "data", "flights.npz")
    np.savez_compressed(out_path,
                        measured=measured[good],
                        applied=applied[good],
                        previous=previous[good])
    speed = np.sqrt((measured[good][:, :, 3:6] ** 2).sum(-1))
    print(f"saved {out_path}")
    print(f"speed: median {np.median(speed):.2f} m/s, largest {speed.max():.2f} m/s;"
          f" tilt: largest {np.degrees(largest_tilt[good].max()):.1f} degrees")


if __name__ == "__main__":
    main()
