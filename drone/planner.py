"""
The planner (model predictive control) used by every model-based drone controller, and the predictors
it can use.

Planner: model predictive path integral control (a sampling planner, common on drones).
    At every step (50 times per second):
    1. Start from the plan of the last step, moved one step along (warm start).
    2. Make 256 noisy copies of it (one copy has no noise).
    3. Let the predictor imagine the next 26 steps for every copy.
    4. Give each copy a cost (distance from the path, shaking, big command jumps).
    5. Average the copies, giving much more weight to the cheap ones  -> the new plan.
    6. Send the first command of the new plan.

The command delay: the command decided now is only applied during the NEXT step, so the planner first
predicts one step with the command that is already on its way, then plans from there.

Offset-free correction (the mpc01c rule): every step, compare where we are with where the predictor said
we would be (prediction WITHOUT correction).
    correction = 0.5 x old correction + 0.5 x (measured state - predicted state)
The correction is added after every predicted step. This is what can absorb a payload without retraining.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone import drone_models as dm
from drone import quadrotor_sim as sim


# ======================================================================================================
# Part 1: predictors. Each one answers: "starting at this state, with these commands, where do we go?"
#   predict(x_now [12], commands [K, S, 4], previous [K, S, 4], disturbance [12] or None) -> [K, S, 12]
# ======================================================================================================
class LearnedPredictor:
    def __init__(self, model, numbers):
        self.model = model
        self.state_mean = torch.tensor(numbers["state_mean"], dtype=torch.float32)
        self.state_std = torch.tensor(numbers["state_std"], dtype=torch.float32)
        self.command_mean = torch.tensor(numbers["command_mean"], dtype=torch.float32)
        self.command_std = torch.tensor(numbers["command_std"], dtype=torch.float32)

    @torch.no_grad()
    def predict(self, x_now, commands, previous, disturbance=None):
        n_samples = commands.shape[0]

        # Step 1: normalise the state (the same start for every sample)
        x = torch.tensor(x_now, dtype=torch.float32)
        x = (x - self.state_mean) / self.state_std
        x = x.unsqueeze(0).repeat(n_samples, 1)

        # Step 2: normalise the commands: [applied now, applied the step before]
        both = np.concatenate([commands, previous], axis=2)
        u = torch.tensor(both, dtype=torch.float32)
        u = (u - self.command_mean) / self.command_std

        # Step 3: the correction, in normalised units (a change, so no average is subtracted)
        if disturbance is None:
            d = None
        else:
            d = torch.tensor(disturbance, dtype=torch.float32) / self.state_std

        # Step 4: chained prediction, then back to physical units
        predicted = dm.roll_out(self.model, x, u, d)
        predicted = predicted * self.state_std + self.state_mean
        return predicted.numpy()


class PhysicsPredictor:
    """
    The textbook drone equations: normal mass, no air drag, instant motors (2 physics steps per control step).
    With oracle=True: the TRUE simulator, which also knows the current payload and the motors' lag.
    """

    def __init__(self, oracle=False):
        self.oracle = oracle
        self.mass = sim.NORMAL_MASS
        self.inertia = sim.NORMAL_INERTIA.copy()
        self.motor_thrust = sim.HOVER_THRUST      # only used by the oracle

    def predict(self, x_now, commands, previous, disturbance=None):
        n_samples = commands.shape[0]
        n_steps = commands.shape[1]

        # Step 1: the same start for every sample
        x = np.tile(x_now, (n_samples, 1))
        mass = np.full(n_samples, self.mass)
        inertia = np.tile(self.inertia, (n_samples, 1))
        thrust = np.full(n_samples, self.motor_thrust)

        # Step 2: step forward
        predicted = np.zeros((n_samples, n_steps, 12))
        for k in range(n_steps):
            if self.oracle:
                x, thrust = sim.control_step(x, thrust, commands[:, k], mass, inertia)
            else:
                x, thrust = sim.control_step(x, thrust, commands[:, k], mass, inertia,
                                             drag=np.zeros(3), motor_time_constant=0.0, sub_steps=2)
            if disturbance is not None:
                x = x + disturbance
            predicted[:, k] = x
        return predicted


# ======================================================================================================
# Part 2: the planner
# ======================================================================================================
class SamplingPlanner:
    def __init__(self, predictor, offset_free, horizon=25, n_samples=256, seed=0,
                 noise=(0.04, 1.5, 1.5, 0.5), temperature=1.0,
                 weight_position=200.0, weight_velocity=5.0, weight_tilt=1.0, weight_yaw=5.0,
                 weight_rates=0.05, weight_command_change=0.02, disturbance_filter=0.5,
                 noise_correlation=0.8):
        self.predictor = predictor
        self.offset_free = offset_free
        self.horizon = horizon
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.noise = np.array(noise)
        self.temperature = temperature
        self.weight_position = weight_position
        self.weight_velocity = weight_velocity
        self.weight_tilt = weight_tilt
        self.weight_yaw = weight_yaw
        self.weight_rates = weight_rates
        self.weight_command_change = weight_command_change
        self.disturbance_filter = disturbance_filter
        self.noise_correlation = noise_correlation
        self.reset()

    def reset(self):
        hover = np.array([sim.HOVER_THRUST, 0.0, 0.0, 0.0])
        self.plan = np.tile(hover, (self.horizon, 1))       # [H, 4]
        self.disturbance = None
        self.last_measurement = None
        self.last_applied = None
        self.last_previous = None
        self.planning_times = []

    def _update_correction(self, measurement):
        """Offset-free correction: how wrong was the last one-step prediction (made WITHOUT correction)?"""
        if self.last_measurement is None:
            return
        commands = self.last_applied.reshape(1, 1, 4)
        previous = self.last_previous.reshape(1, 1, 4)
        predicted_now = self.predictor.predict(self.last_measurement, commands, previous, None)[0, 0]
        mismatch = measurement - predicted_now
        if self.disturbance is None:
            self.disturbance = self.disturbance_filter * mismatch
        else:
            keep = 1.0 - self.disturbance_filter
            self.disturbance = keep * self.disturbance + self.disturbance_filter * mismatch

    def decide(self, measurement, applied_now, applied_before, reference):
        """
        measurement     [12]    measured state now
        applied_now     [4]     command being applied during this step (decided one step ago)
        applied_before  [4]     command applied during the previous step
        reference       (positions [H+1, 3], velocities [H+1, 3]) for the next H+1 steps
        Returns the command to apply during the NEXT step [4].
        """
        start = time.perf_counter()

        # Step 1: offset-free correction (only if switched on)
        if self.offset_free:
            self._update_correction(measurement)
        disturbance = self.disturbance if self.offset_free else None

        # Step 2: noisy copies of the plan (copy 0 has no noise)
        # smooth noise: each step keeps 80% of the last step's noise (jumpy noise makes poor plans)
        fresh = self.rng.normal(0.0, 1.0, size=(self.n_samples, self.horizon, 4))
        noise = np.zeros((self.n_samples, self.horizon, 4))
        new_part = np.sqrt(1.0 - self.noise_correlation ** 2)
        noise[:, 0] = fresh[:, 0]
        for k in range(1, self.horizon):
            noise[:, k] = self.noise_correlation * noise[:, k - 1] + new_part * fresh[:, k]
        noise = noise * self.noise
        noise[0] = 0.0
        candidates = self.plan[None, :, :] + noise
        candidates = sim.clip_commands(candidates.reshape(-1, 4)).reshape(self.n_samples, self.horizon, 4)
        noise = candidates - self.plan[None, :, :]

        # Step 3: the full command sequence: the command already on its way, then the candidate plan
        on_its_way = np.tile(applied_now, (self.n_samples, 1, 1))                  # [K, 1, 4]
        commands = np.concatenate([on_its_way, candidates], axis=1)                # [K, H+1, 4]
        before = np.tile(applied_before, (self.n_samples, 1, 1))
        previous = np.concatenate([before, commands[:, :-1]], axis=1)              # [K, H+1, 4]

        # Step 4: imagine the future
        predicted = self.predictor.predict(measurement, commands, previous, disturbance)   # [K, H+1, 12]

        # Step 5: cost of every candidate
        ref_position, ref_velocity = reference
        position_error = predicted[:, :, 0:3] - ref_position[None]
        velocity_error = predicted[:, :, 3:6] - ref_velocity[None]
        cost = self.weight_position * np.sum(position_error ** 2, axis=(1, 2))
        cost = cost + self.weight_velocity * np.sum(velocity_error ** 2, axis=(1, 2))
        cost = cost + self.weight_tilt * np.sum(predicted[:, :, 6:8] ** 2, axis=(1, 2))
        cost = cost + self.weight_yaw * np.sum(predicted[:, :, 8] ** 2, axis=1)
        cost = cost + self.weight_rates * np.sum(predicted[:, :, 9:12] ** 2, axis=(1, 2))
        jumps = (commands[:, 1:] - commands[:, :-1]) / self.noise
        cost = cost + self.weight_command_change * np.sum(jumps ** 2, axis=(1, 2))
        cost = np.nan_to_num(cost, nan=1e12, posinf=1e12)

        # Step 6: weights: cheap candidates count much more (exp of minus cost / temperature)
        # (the temperature is measured relative to the spread of the costs, so it does not depend on units)
        cheapest = cost.min()
        spread = np.std(cost) + 1e-9
        weights = np.exp(-(cost - cheapest) / (self.temperature * spread))
        weights = weights / weights.sum()

        # Step 7: the new plan = old plan + weighted average of the noise
        weighted_noise = np.sum(weights[:, None, None] * noise, axis=0)
        new_plan = sim.clip_commands(self.plan + weighted_noise)
        decided = new_plan[0].copy()

        # Step 8: warm start for next time: move the plan one step along, repeat the last command
        self.plan = np.concatenate([new_plan[1:], new_plan[-1:]], axis=0)

        # Step 9: remember what is needed for the correction next step
        self.last_measurement = measurement.copy()
        self.last_applied = applied_now.copy()
        self.last_previous = applied_before.copy()

        self.planning_times.append(time.perf_counter() - start)
        return decided
