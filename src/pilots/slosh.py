"""
Space pilot C helpers: data, controller cost, the standard attitude law, and the closed-loop test.

Everything here works on many spacecraft at once (batch dimension S).
"""

from __future__ import annotations

import math
import time

import torch

from src.systems.slosh_spacecraft import SloshSpacecraft


def build_system(cfg: dict) -> SloshSpacecraft:
    s = cfg["system"]
    return SloshSpacecraft(dt=s["dt"], substeps=s["substeps"], u_max=s["u_max"], dry_inertia=s["dry_inertia"],
                           hinge_offset=s["hinge_offset"], thrust_acceleration=s["thrust_acceleration"],
                           tank_fuel_mass=s["tank_fuel_mass"], damping_ratio=s["damping_ratio"])


def add_sensor_noise(measured: torch.Tensor, cfg: dict, generator: torch.Generator) -> torch.Tensor:
    """Small sensor noise on angle and turn rate."""
    angle_noise = math.radians(float(cfg["system"]["angle_noise_degrees"]))
    rate_noise = math.radians(float(cfg["system"]["rate_noise_degrees"]))
    noise = torch.randn(measured.shape, generator=generator)
    noise[:, 0] = noise[:, 0] * angle_noise
    noise[:, 1] = noise[:, 1] * rate_noise
    return measured + noise.to(measured.device)


# ==========================================================================================================
# Training data: what a real mission would log during many burns at many fill levels
# ==========================================================================================================
def make_trajectories(system: SloshSpacecraft, cfg: dict, number: int, seed: int):
    """
    Returns measured states [N, T + 1, 2], pushes [N, T, 1], fills [N].
    Pushes: a randomly tuned attitude law towards random targets that change now and then, plus random
    held kicks and small dither, so the data covers both big turns and quiet pointing.
    """
    data_cfg = cfg["data"]
    steps = int(data_cfg["trajectory_steps"])
    generator = torch.Generator().manual_seed(seed)

    def uniform(low, high, size):
        return low + (high - low) * torch.rand(size, generator=generator)

    # Step 1: fill level and starting state of every trajectory
    fills = uniform(float(data_cfg["fill_low"]), float(data_cfg["fill_high"]), (number,))
    start_angle = math.radians(float(data_cfg["start_angle_degrees"]))
    start_slosh = math.radians(float(data_cfg["start_slosh_degrees"]))
    state = torch.zeros(number, 4)
    state[:, 0] = uniform(-start_angle, start_angle, (number,))
    state[:, 1] = uniform(-0.01, 0.01, (number,))
    state[:, 2] = uniform(-start_slosh, start_slosh, (number,))
    state[:, 3] = uniform(-0.05, 0.05, (number,))

    # Step 2: settings of the random attitude law of each trajectory
    stiffness = uniform(0.3, 6.0, (number,))          # N m per rad
    damping = uniform(1.0, 15.0, (number,))           # N m per rad/s
    kick_share = uniform(0.0, 1.0, (number,))         # how much random kicking
    target = uniform(-start_angle, start_angle, (number,))
    kick = torch.zeros(number)
    kick_left = torch.zeros(number)

    measured_list = [add_sensor_noise(system.measure(state), cfg, generator)]
    push_list = []
    for step in range(steps):
        # 2a: new target now and then (about every 15 s)
        change = torch.rand(number, generator=generator) < 1.0 / 75.0
        target = torch.where(change, uniform(-start_angle, start_angle, (number,)), target)

        # 2b: random held kick: a new level when the old one runs out
        renew = kick_left <= 0
        kick = torch.where(renew, uniform(-system.u_max, system.u_max, (number,)), kick)
        kick_left = torch.where(renew, torch.randint(1, 25, (number,), generator=generator).float(), kick_left - 1)

        # 2c: the push = attitude law + kick + dither, clipped to the limit
        law = -stiffness * (state[:, 0] - target) - damping * state[:, 1]
        dither = 0.05 * system.u_max * torch.randn(number, generator=generator)
        push = torch.clamp(law + kick_share * kick + dither, -system.u_max, system.u_max).unsqueeze(1)

        state = system.step(state, push, fills)
        measured_list.append(add_sensor_noise(system.measure(state), cfg, generator))
        push_list.append(push)

    measured = torch.stack(measured_list, dim=1)
    pushes = torch.stack(push_list, dim=1)
    return measured, pushes, fills


# ==========================================================================================================
# Controller cost and the standard attitude law
# ==========================================================================================================
def pseudo_huber(z: torch.Tensor) -> torch.Tensor:
    """Quadratic for small z, linear for large z (keeps the solver sharp near the target and calm far away)."""
    return torch.sqrt(1.0 + z * z) - 1.0


def make_cost(cfg: dict, targets: torch.Tensor):
    """targets [S] (rad). Only the first two predicted numbers (angle, turn rate) are scored."""
    c = cfg["controller"]
    angle_scale = math.radians(float(c["angle_scale_degrees"]))
    rate_scale = math.radians(float(c["rate_scale_degrees"]))
    q_angle, q_rate = float(c["q_angle"]), float(c["q_rate"])
    p_angle, p_rate = float(c["p_angle"]), float(c["p_rate"])
    r_push, r_change = float(c["r_push"]), float(c["r_change"])

    def cost(predicted, plan_scaled, previous_scaled, reference):
        angle_error = (predicted[:, :, 0] - targets.unsqueeze(1)) / angle_scale
        rate = predicted[:, :, 1] / rate_scale
        stage = q_angle * pseudo_huber(angle_error).mean(dim=1) + q_rate * pseudo_huber(rate).mean(dim=1)
        final = p_angle * pseudo_huber(angle_error[:, -1]) + p_rate * pseudo_huber(rate[:, -1])
        effort = r_push * (plan_scaled ** 2).mean(dim=(1, 2))
        all_pushes = torch.cat([previous_scaled.unsqueeze(1), plan_scaled], dim=1)
        smooth = r_change * ((all_pushes[:, 1:] - all_pushes[:, :-1]) ** 2).mean(dim=(1, 2))
        return stage + final + effort + smooth

    return cost


class AttitudeLaw:
    """Low-bandwidth proportional-derivative law (the paper's controller type), tuned for the nominal fill."""

    def __init__(self, system: SloshSpacecraft, cfg: dict, targets: torch.Tensor):
        c = cfg["controller"]
        p = system.parameters(float(cfg["system"]["nominal_fill"]))
        inertia = p["J"] + p["m"] * (p["b"] + p["l"]) ** 2          # all fuel frozen
        natural = 2.0 * math.pi * float(c["pd_bandwidth_hz"])
        self.stiffness = inertia * natural ** 2
        self.damping = 2.0 * float(c["pd_damping"]) * natural * inertia
        self.u_max = system.u_max
        self.targets = targets
        self.planning_seconds = []

    def reset(self):
        self.planning_seconds = []

    def solve(self, x_now, u_previous, past_states, past_pushes, reference=None):
        started = time.perf_counter()
        push = -self.stiffness * (x_now[:, 0] - self.targets) - self.damping * x_now[:, 1]
        self.planning_seconds.append(time.perf_counter() - started)
        return torch.clamp(push, -self.u_max, self.u_max).unsqueeze(1)


# ==========================================================================================================
# Closed-loop test
# ==========================================================================================================
def make_test_cases(cfg: dict):
    """Every combination of test fill, target and starting slosh. Returns fills [S], targets [S], starts [S, 4]."""
    t = cfg["test"]
    fills, targets, starts = [], [], []
    for fill in t["fills"]:
        for target in t["targets_degrees"]:
            for slosh in t["start_slosh_degrees"]:
                fills.append(float(fill))
                targets.append(math.radians(float(target)))
                starts.append([0.0, 0.0, math.radians(float(slosh)), 0.0])
    return torch.tensor(fills), torch.tensor(targets), torch.tensor(starts)


def run_closed_loop(controller, system: SloshSpacecraft, fills, targets, starts, cfg: dict, full_state: bool,
                    context_len: int, noise_seed: int, bar=None) -> dict:
    """
    full_state = True: the controller is TOLD the fuel state (true-model and pendulum-model reference controllers).
    full_state = False: it sees only the measured angle and turn rate (with sensor noise), like a real spacecraft.
    """
    t = cfg["test"]
    steps = int(t["steps"])
    number = starts.shape[0]
    settle = math.radians(float(t["settle_angle_degrees"]))
    lost_rate = math.radians(float(t["lost_rate_degrees"]))
    lost_slosh = math.radians(float(t["lost_slosh_degrees"]))
    generator = torch.Generator().manual_seed(noise_seed)

    state = starts.clone()
    controller.reset()
    previous_push = torch.zeros(number, 1)
    history_states, history_pushes = [], []
    angle_errors, slosh_angles = [], []
    effort = torch.zeros(number)
    lost = torch.zeros(number, dtype=torch.bool)

    for step in range(steps):
        # Step 1: what the controller sees
        if full_state:
            seen = state.clone()
        else:
            seen = add_sensor_noise(system.measure(state), cfg, generator)

        # Step 2: recent history for predictors with memory
        window = min(step, context_len)
        if window > 0:
            past_states = torch.stack(history_states[-window:], dim=1)
            past_pushes = torch.stack(history_pushes[-window:], dim=1)
        else:
            past_states, past_pushes = None, None

        # Step 3: plan, apply the first push, the real spacecraft moves
        push = controller.solve(seen, previous_push, past_states, past_pushes).detach()
        history_states.append(seen)
        history_pushes.append(push.clone())
        new_state = system.step(state, push, fills)
        new_state = torch.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
        lost = lost | (new_state[:, 1].abs() > lost_rate) | (new_state[:, 2].abs() > lost_slosh)
        state = torch.where(lost.unsqueeze(1), state, new_state)

        angle_errors.append((state[:, 0] - targets).abs())
        slosh_angles.append(state[:, 2].abs())
        effort = effort + (push[:, 0] / system.u_max) ** 2 * system.dt
        previous_push = push
        if bar is not None:
            bar.update(1)

    # Step 4: metrics per spacecraft
    errors = torch.stack(angle_errors, dim=1)                   # [S, steps]
    slosh = torch.stack(slosh_angles, dim=1)
    last_part = int(round(steps * 0.75))
    settle_time = []
    for s in range(number):
        above = (errors[s] >= settle).nonzero()
        if lost[s]:
            settle_time.append(float("inf"))
        elif above.numel() == 0:
            settle_time.append(0.0)
        elif int(above[-1]) == steps - 1:
            settle_time.append(float("inf"))                    # never settled
        else:
            settle_time.append((int(above[-1]) + 1) * system.dt)

    return {
        "settle_time": settle_time,
        "final_error_degrees": [math.degrees(float(v)) for v in errors[:, last_part:].mean(dim=1)],
        "final_slosh_degrees": [math.degrees(float(v)) for v in slosh[:, last_part:].max(dim=1).values],
        "effort": [float(v) for v in effort],
        "lost": [bool(v) for v in lost],
        "planning_ms": 1000.0 * sum(controller.planning_seconds) / max(1, len(controller.planning_seconds)),
    }


# ==========================================================================================================
# Prediction check with warm-up (memory filled from measured history first)
# ==========================================================================================================
@torch.no_grad()
def chained_errors_with_warmup(predictor, measured, pushes, context_len: int, horizon: int, every: int):
    """
    Root-mean-square error of the predicted angle (degrees) after 1..horizon chained steps, each prediction
    started after `context_len` measured steps were shown to the predictor.
    measured [N, T + 1, 2], pushes [N, T, 1].
    """
    starts, plans, truths, past_s, past_u = [], [], [], [], []
    total = pushes.shape[1]
    start = context_len
    while start + horizon <= total:
        past_s.append(measured[:, start - context_len:start])
        past_u.append(pushes[:, start - context_len:start])
        starts.append(measured[:, start])
        plans.append(pushes[:, start:start + horizon])
        truths.append(measured[:, start + 1:start + 1 + horizon])
        start = start + every
    memory, _ = predictor.prepare(torch.cat(past_s), torch.cat(past_u))
    predicted = predictor.predict(torch.cat(starts), torch.cat(plans), memory, None)
    truth = torch.cat(truths)
    difference = torch.rad2deg(predicted[:, :, 0] - truth[:, :, 0])
    return [float(v) for v in torch.sqrt((difference ** 2).mean(dim=0))]
