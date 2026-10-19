"""Stage 2 calculations: what makes the chained predictions go wrong (docs/stage2_plan.md).

Used by tools/diagnose_errors.py. Every function works on models that are already trained; nothing
here trains anything.

Words used below
  timing error  - the prediction is on the right path but ahead of or behind the real system
                  (the part of the error along the direction the real system is moving)
  shape error   - the prediction is off the path (the part across that direction)
  chained       - each prediction is fed back in as the next input, as a controller does
  one-step      - prediction made from the TRUE history (teacher forcing), so nothing is fed back
  window        - one test example: true states x_0 ... x_{C+H} and inputs u_0 ... u_{C+H-1}
                  (C = context length 10, H = horizon 50)

Shapes: W = number of windows, H = horizon, 2 = (position, velocity).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.autograd.functional import jacobian

from src.data.datasets import Split


# =================================================================================================
# Small helpers
# =================================================================================================
def keep_first_windows(split: Split, n: int) -> Split:
    """The same split with only its first n windows (used by --quick and max_windows)."""
    return Split(x=split.x, u=split.u, xw=split.xw[:n], uw=split.uw[:n], starts=split.starts[:n])


def predict(model, regime, x_ctx, u, horizon, start=None, state=None):
    """Chained or direct prediction for one batch. Returns (predictions, hit_clamp), normalised units."""
    if regime == "direct":
        preds, hit = model.direct(x_ctx, u, horizon, start=start)
    else:
        preds, hit = model.chained(x_ctx, u, horizon, state=state, start=start)
    return preds, hit


def root_mean_square(values):
    """Square root of the mean of the squares of all numbers in the array."""
    squares = np.square(values)
    mean_square = float(np.mean(squares))
    return float(np.sqrt(mean_square))


# =================================================================================================
# Predictions on every test window
# =================================================================================================
@torch.no_grad()
def predictions_on_windows(model, regime, split, norm, context_len, horizon, device, batch_size):
    """Chained (or direct) predictions and teacher-forced one-step predictions for every window.

    Returns a dictionary of NumPy arrays in PHYSICAL units:
      true        [W, C+H+1, 2]   true states of each window
      predicted   [W, H, 2]       chained (or direct) predictions of x_{C+1} ... x_{C+H}
      one_step    [W, H, 2]       one-step predictions of x_{C+1} ... x_{C+H} from the true history
      hit_clamp   [W]             True if the prediction ran away (hit the safety clamp)
    """
    model.eval()
    C = context_len
    H = horizon
    predicted_parts = []
    one_step_parts = []
    hit_parts = []
    n_windows = split.xw.shape[0]

    for first in range(0, n_windows, batch_size):
        # Step 1: one batch of windows (normalised units)
        x = split.xw[first:first + batch_size].to(device)
        u = split.uw[first:first + batch_size].to(device)

        # Step 2: chained (or direct) prediction, exactly as Stage 1 evaluated it
        preds, target, hit = model.rollout(regime, x, u, C, H)

        # Step 3: one-step predictions from the true history.
        #   teacher_forced returns predictions of x_1 ... x_{C+H};
        #   position C predicts x_{C+1}, so positions C ... C+H-1 are the 50 we need.
        all_one_step = model.teacher_forced(x[:, :-1], u)
        one_step = all_one_step[:, C:C + H]

        predicted_parts.append(preds.cpu().numpy())
        one_step_parts.append(one_step.cpu().numpy())
        hit_parts.append(hit.cpu().numpy())

    # Step 4: stick the batches together and convert to physical units
    predicted = norm.x_inverse(np.concatenate(predicted_parts, axis=0).astype(np.float64))
    one_step = norm.x_inverse(np.concatenate(one_step_parts, axis=0).astype(np.float64))
    true = norm.x_inverse(split.xw.numpy().astype(np.float64))
    hit_clamp = np.concatenate(hit_parts, axis=0)
    return {"true": true, "predicted": predicted, "one_step": one_step, "hit_clamp": hit_clamp}


def true_velocity(split, context_len, horizon, dt):
    """Velocity of the true system at x_{C+1} ... x_{C+H} of every window, physical units per time.

    Central difference on the long true trajectory: (x[p+1] - x[p-1]) / (2 dt). At the very end of
    the trajectory (no x[p+1]) it falls back to the backward difference.
    Returns [W, H, 2].
    """
    x = split.x
    last_index = len(x) - 1
    steps = np.arange(1, horizon + 1)                                   # k = 1 ... H

    # Step 1: index of x_{C+k} in the long trajectory, for every window and step: [W, H]
    position = split.starts[:, None] + context_len + steps[None, :]

    # Step 2: neighbours before and after
    before = position - 1
    after = np.minimum(position + 1, last_index)

    # Step 3: difference divided by the time between the two neighbours
    gap_in_steps = (after - before)[..., None]
    velocity = (x[after] - x[before]) / (gap_in_steps * dt)
    return velocity


def unit_vectors(vectors):
    """Divide each 2-vector by its length (zero vectors stay zero). Also returns the lengths."""
    lengths = np.linalg.norm(vectors, axis=-1)
    safe_lengths = np.where(lengths > 0, lengths, 1.0)
    units = vectors / safe_lengths[..., None]
    return units, lengths


# =================================================================================================
# Q1 — how much of the error is timing and how much is shape?
# =================================================================================================
def q1_timing_and_shape(error, direction):
    """Split each error vector into its part along the direction of motion and the part across it.

    error      [W, H, 2]  prediction minus truth (physical units)
    direction  [W, H, 2]  unit vector along the true direction of motion
    """
    # Step 1: timing part = projection onto the direction of motion (a signed number)
    timing_part = np.sum(error * direction, axis=-1)                   # [W, H]

    # Step 2: squared sizes. Squared error = timing^2 + shape^2 exactly (Pythagoras).
    squared_error = np.sum(error * error, axis=-1)                     # [W, H]
    squared_timing = timing_part * timing_part
    squared_shape = np.maximum(squared_error - squared_timing, 0.0)

    # Step 3: timing share per window over all 50 steps, then the median over windows
    per_window_error = squared_error.sum(axis=1)
    per_window_timing = squared_timing.sum(axis=1)
    usable = per_window_error > 0
    per_window_share = per_window_timing[usable] / per_window_error[usable]
    median_share = float(np.median(per_window_share))

    # Step 4: two more views of the same split (recorded, not used by the rule)
    pooled_share = float(squared_timing.sum() / squared_error.sum())
    step50_share = float(squared_timing[:, -1].sum() / squared_error[:, -1].sum())

    # Step 5: root-mean-square size of each part at every step (for the log and Q3)
    timing_curve = np.sqrt(np.mean(squared_timing, axis=0))           # [H]
    shape_curve = np.sqrt(np.mean(squared_shape, axis=0))             # [H]

    return {"timing_part": timing_part,
            "median_timing_share": median_share,
            "pooled_timing_share": pooled_share,
            "step50_timing_share": step50_share,
            "timing_curve": timing_curve,
            "shape_curve": shape_curve}


def q1_alignment_cross_check(split, predicted, context_len, horizon, dt, search_steps):
    """Cross-check of Q1 that also works for large errors: slide each prediction along the true path.

    For every prediction x_hat_k, find the closest point on the true trajectory within
    search_steps steps before or after x_{C+k} (straight lines between the stored points).
      lag    = how far back in time that closest point is (time units; positive = behind)
      shape  = the distance that remains (the part no time shift can remove)
    Timing share = 1 - sum(shape^2) / sum(error^2), per window, median over windows.
    """
    x = split.x
    last_index = len(x) - 1
    n_windows = predicted.shape[0]
    lag = np.zeros((n_windows, horizon))
    shape_distance = np.zeros((n_windows, horizon))
    at_edge = np.zeros((n_windows, horizon), dtype=bool)
    squared_error = np.zeros((n_windows, horizon))

    offsets = np.arange(-search_steps, search_steps)                    # segment j joins j and j + 1

    for k in range(1, horizon + 1):
        # Step 1: index of x_{C+k} in the long trajectory, one per window: [W]
        centre = split.starts + context_len + k

        # Step 2: the line segments of the true path around it: [W, segments]
        index_a = np.clip(centre[:, None] + offsets[None, :], 0, last_index)
        index_b = np.clip(centre[:, None] + offsets[None, :] + 1, 0, last_index)
        point_a = x[index_a]                                            # [W, segments, 2]
        point_b = x[index_b]
        segment = point_b - point_a

        # Step 3: closest point on each segment to the prediction
        p = predicted[:, k - 1][:, None, :]                             # [W, 1, 2]
        segment_length_squared = np.sum(segment * segment, axis=-1)
        safe_length = np.where(segment_length_squared > 0, segment_length_squared, 1.0)
        fraction = np.sum((p - point_a) * segment, axis=-1) / safe_length
        fraction = np.clip(fraction, 0.0, 1.0)
        closest = point_a + fraction[..., None] * segment
        distance_squared = np.sum((p - closest) ** 2, axis=-1)          # [W, segments]

        # Step 4: the best segment for each window
        best = np.argmin(distance_squared, axis=1)
        rows = np.arange(n_windows)
        position_in_steps = offsets[best] + fraction[rows, best]       # 0 = exactly at x_{C+k}

        # Step 5: store lag (positive = the prediction sits where the truth was EARLIER = behind)
        lag[:, k - 1] = -position_in_steps * dt
        shape_distance[:, k - 1] = np.sqrt(distance_squared[rows, best])
        at_edge[:, k - 1] = (position_in_steps <= -search_steps + 0.01) | (position_in_steps >= search_steps - 0.01)

        true_point = x[np.minimum(centre, last_index)]
        squared_error[:, k - 1] = np.sum((predicted[:, k - 1] - true_point) ** 2, axis=-1)

    per_window_error = squared_error.sum(axis=1)
    per_window_shape = (shape_distance ** 2).sum(axis=1)
    usable = per_window_error > 0
    share = 1.0 - per_window_shape[usable] / per_window_error[usable]
    return {"median_timing_share": float(np.median(share)),
            "fraction_at_search_edge": float(at_edge.mean()),
            "mean_lag_curve": lag.mean(axis=0),                         # [H], time units
            "mean_lag_at_step50": float(lag[:, -1].mean())}


# =================================================================================================
# Q2 — do errors just add up, or does feedback amplify them?
# =================================================================================================
def q2_feedback(chained_error, one_step_error, true_states_before, system):
    """Compare the chained error with the one-step errors (a) just added up and (b) carried by the
    true system's own Jacobians.

    chained_error       [W, H, 2]  chained prediction minus truth
    one_step_error      [W, H, 2]  one-step prediction (true history) minus truth, same positions
    true_states_before  [W, H, 2]  x_C ... x_{C+H-1}: the true state BEFORE each predicted step
    """
    # Step 1: (a) just added up: error at step k = e_1 + e_2 + ... + e_k
    added_up = np.cumsum(one_step_error, axis=1)

    # Step 2: (b) carried by the true system: E_k = J_true(x_{C+k-1}) E_{k-1} + e_k
    #   (how a PERFECT feedback loop would pass the same one-step errors on)
    true_jacobians = system.jacobian(true_states_before)               # [W, H, 2, 2]
    carried = np.zeros_like(one_step_error)
    previous = np.zeros_like(one_step_error[:, 0])                     # [W, 2]
    for k in range(one_step_error.shape[1]):
        passed_on = np.einsum("wij,wj->wi", true_jacobians[:, k], previous)
        current = passed_on + one_step_error[:, k]
        carried[:, k] = current
        previous = current

    # Step 3: ratios of root-mean-square sizes over all windows and steps
    chained_size = root_mean_square(chained_error)
    added_up_size = root_mean_square(added_up)
    carried_size = root_mean_square(carried)
    one_step_size = root_mean_square(one_step_error)

    # Step 4: sanity check — at step 1 nothing has been fed back yet, so chained = one-step
    step1_difference = float(np.max(np.abs(chained_error[:, 0] - one_step_error[:, 0])))

    return {"ratio_to_added_up": chained_size / added_up_size,
            "ratio_to_true_feedback": chained_size / carried_size,
            "chained_rms": chained_size,
            "added_up_rms": added_up_size,
            "true_feedback_rms": carried_size,
            "one_step_rms": one_step_size,
            "step1_chained_minus_one_step_max": step1_difference}


# =================================================================================================
# Q3 — slow clock or amplifier?
# =================================================================================================
def _exponential(k, scale, rate):
    return scale * np.exp(rate * k)


def q3_growth_shape(timing_curve):
    """Fit a straight line and an exponential to the timing-error curve; compare the misfits."""
    from scipy.optimize import curve_fit

    steps = np.arange(1, len(timing_curve) + 1, dtype=float)

    # Step 1: straight line a + b k
    line = np.polyfit(steps, timing_curve, 1)
    line_values = np.polyval(line, steps)
    misfit_line = float(np.sum((timing_curve - line_values) ** 2))

    # Step 2: exponential A exp(g k). Start from a straight-line fit of the logarithm, then refine.
    positive = np.maximum(timing_curve, 1e-12)
    log_line = np.polyfit(steps, np.log(positive), 1)
    start_rate = float(log_line[0])
    start_scale = float(np.exp(log_line[1]))
    try:
        best, _ = curve_fit(_exponential, steps, timing_curve, p0=[start_scale, start_rate], maxfev=20000)
        scale = float(best[0])
        rate = float(best[1])
    except Exception:          # if the refinement fails, keep the starting fit
        scale = start_scale
        rate = start_rate
    exp_values = _exponential(steps, scale, rate)
    misfit_exp = float(np.sum((timing_curve - exp_values) ** 2))

    # Step 3: is growth speeding up (> 1), steady (about 1) or slowing down (< 1)?
    #   growth over steps 26-50 divided by growth over steps 1-25
    middle = len(timing_curve) // 2
    early_growth = timing_curve[middle - 1] - timing_curve[0]
    late_growth = timing_curve[-1] - timing_curve[middle - 1]
    if abs(early_growth) > 0:
        late_over_early = float(late_growth / early_growth)
    else:
        late_over_early = None

    return {"misfit_line": misfit_line, "misfit_exponential": misfit_exp,
            "line_slope_per_step": float(line[0]), "line_intercept": float(line[1]),
            "exponential_rate_per_step": rate, "exponential_scale": scale,
            "late_over_early_growth": late_over_early}


def q3_one_step_sign(one_step_error, velocity, dt, min_speed_fraction):
    """Do the one-step errors consistently point behind (or ahead), or at random?

    consistency = |mean of timing part| / root-mean-square of timing part   (physical units; the rule)
    Also in time units: lag = -(timing part) / speed, positive = behind. Points slower than
    min_speed_fraction of the median speed are skipped for the time-unit numbers (division by ~0).
    """
    direction, speed = unit_vectors(velocity)

    # Step 1: timing part of every one-step error (physical units) — used by the rule
    timing_part = np.sum(one_step_error * direction, axis=-1)          # [W, H]
    mean_part = float(np.mean(timing_part))
    rms_part = root_mean_square(timing_part)
    consistency = abs(mean_part) / rms_part if rms_part > 0 else 0.0

    # Step 2: the same in time units (how much time the model loses or gains per step)
    fast_enough = speed > min_speed_fraction * np.median(speed)
    lag = -timing_part[fast_enough] / speed[fast_enough]
    mean_lag = float(np.mean(lag))
    rms_lag = root_mean_square(lag)
    consistency_time = abs(mean_lag) / rms_lag if rms_lag > 0 else 0.0

    # Step 3: implied clock speed. Losing mean_lag of time every dt means the clock runs at
    #   1 - mean_lag / dt of the true speed (1.0 = right, 0.8 = 20% slow, 1.2 = 20% fast).
    clock_speed = 1.0 - mean_lag / dt

    return {"consistency": consistency, "mean_timing_part": mean_part, "rms_timing_part": rms_part,
            "consistency_time_units": consistency_time, "mean_one_step_lag": mean_lag,
            "implied_clock_speed": clock_speed,
            "fraction_points_used_time_units": float(fast_enough.mean())}


# =================================================================================================
# Q4 — does the model correct a push like the true system does?
# =================================================================================================
def true_limit_cycle_points(system, start, settle_steps, n_points, context_len):
    """Points spread evenly around one loop of the true system with zero input.

    Returns (trajectory [N, 2], list of indices into it, steps per loop).
    """
    # Step 1: simulate long enough to settle and then go round a few more times
    extra_steps = 400
    inputs = np.zeros(settle_steps + extra_steps)
    trajectory = system.simulate(np.asarray(start, dtype=float), inputs)

    # Step 2: find two upward zero crossings of the position after settling = one loop
    settled = trajectory[settle_steps:, 0]
    crossings = np.where((settled[:-1] < 0) & (settled[1:] >= 0))[0]
    if len(crossings) < 2:
        raise RuntimeError("The true system did not complete a loop; increase loop_settle_steps.")
    first = settle_steps + int(crossings[0]) + 1
    steps_per_loop = int(crossings[1] - crossings[0])

    # Step 3: evenly spaced points on that loop (each needs context_len points before it)
    indices = []
    for i in range(n_points):
        index = first + int(round(i * steps_per_loop / n_points))
        index = max(index, context_len)
        indices.append(index)
    return trajectory, indices, steps_per_loop


def _along_and_across(difference, path_end, path_before_end):
    """Split a difference vector into its part along / across the direction of motion at the end."""
    motion = path_end - path_before_end
    length = np.linalg.norm(motion)
    if length == 0:
        return float("nan"), float("nan")
    along_direction = motion / length
    across_direction = np.array([-along_direction[1], along_direction[0]])
    along = abs(float(np.dot(difference, along_direction)))
    across = abs(float(np.dot(difference, across_direction)))
    return along, across


@torch.no_grad()
def q4_push_response(model, regime, norm, system, trajectory, indices, context_len, horizon,
                     push_size, device):
    """Push the starting state along / across the direction of motion; follow it for H steps."""
    model.eval()
    C = context_len
    H = horizon
    dt = system.dt
    zero_inputs_physical = np.zeros((C + H, 1))
    u = torch.tensor(norm.u(zero_inputs_physical)[None], dtype=torch.float32, device=device)

    rows = []
    for index in indices:
        # Step 1: context = true states on the loop up to the point; the point itself is x_C
        context_physical = trajectory[index - C:index + 1]
        x_ctx = torch.tensor(norm.x(context_physical)[None], dtype=torch.float32, device=device)
        point = trajectory[index]

        # Step 2: direction of motion at the point and the direction across it
        motion = (trajectory[index + 1] - trajectory[index - 1]) / (2.0 * dt)
        along_direction = motion / np.linalg.norm(motion)
        across_direction = np.array([-along_direction[1], along_direction[0]])

        # Step 3: the model's prediction without any push
        plain, plain_hit = predict(model, regime, x_ctx, u, H)
        plain_physical = norm.x_inverse(plain[0].cpu().numpy().astype(np.float64))      # [H, 2]
        if bool(plain_hit.any()) or not np.all(np.isfinite(plain_physical)):
            continue

        # Step 4: the true system without any push (index k = step k)
        true_plain = system.simulate(point, np.zeros(H))                                  # [H + 1, 2]

        row = {"loop_index": int(index)}
        pushes = {"timing": along_direction, "shape": across_direction}
        for push_name, push_direction in pushes.items():
            pushed_point = point + push_size * push_direction

            # Step 5: the model, started from the pushed point (same context before it)
            start = torch.tensor(norm.x(pushed_point)[None], dtype=torch.float32, device=device)
            pushed, pushed_hit = predict(model, regime, x_ctx, u, H, start=start)
            pushed_physical = norm.x_inverse(pushed[0].cpu().numpy().astype(np.float64))
            if bool(pushed_hit.any()) or not np.all(np.isfinite(pushed_physical)):
                row = None
                break
            model_difference = pushed_physical[H - 1] - plain_physical[H - 1]
            model_along, model_across = _along_and_across(model_difference, plain_physical[H - 1],
                                                          plain_physical[H - 2])

            # Step 6: the true system, started from the same pushed point
            true_pushed = system.simulate(pushed_point, np.zeros(H))
            true_difference = true_pushed[H] - true_plain[H]
            true_along, true_across = _along_and_across(true_difference, true_plain[H], true_plain[H - 1])

            # Step 7: everything divided by the push size (1.0 = the push kept its size)
            row[f"{push_name}_push_model_along"] = model_along / push_size
            row[f"{push_name}_push_model_across"] = model_across / push_size
            row[f"{push_name}_push_true_along"] = true_along / push_size
            row[f"{push_name}_push_true_across"] = true_across / push_size
        if row is not None:
            rows.append(row)

    if not rows:
        return {"points_used": 0}

    # Step 8: ratios model / true, median over the points
    timing_ratios = [r["timing_push_model_along"] / r["timing_push_true_along"] for r in rows]
    shape_ratios = [r["shape_push_model_across"] / r["shape_push_true_across"] for r in rows]
    return {"points_used": len(rows),
            "timing_push_ratio": float(np.median(timing_ratios)),
            "shape_push_ratio": float(np.median(shape_ratios)),
            "model_shape_survival": float(np.median([r["shape_push_model_across"] for r in rows])),
            "true_shape_survival": float(np.median([r["shape_push_true_across"] for r in rows])),
            "model_timing_keep": float(np.median([r["timing_push_model_along"] for r in rows])),
            "true_timing_keep": float(np.median([r["timing_push_true_along"] for r in rows])),
            "per_point": rows}


# =================================================================================================
# Recorded only — Pervez and Locatello (2026)
# =================================================================================================
def largest_magnification(model, regime, split, norm, system, context_len, horizon, n_windows, device):
    """max over steps k of the largest singular value of d x_hat(C+k) / d x(C), model and true system."""
    C = context_len
    H = horizon
    std = np.asarray(norm.x_std, dtype=np.float64)
    count = min(n_windows, split.xw.shape[0])
    chosen = np.linspace(0, split.xw.shape[0] - 1, count).astype(int)

    was_training = model.training
    model.train()      # recurrent-network backward on graphics cards needs training mode; no dropout is used
    model_values = []
    true_values = []
    try:
        for i in chosen:
            x = split.xw[i:i + 1].to(device)
            u = split.uw[i:i + 1].to(device)
            x_ctx = x[:, :C + 1]
            state = None
            if regime != "direct":
                with torch.no_grad():
                    state = model.warmup(x_ctx, u)

            # Step 1: function from the starting state x_C to all H predictions
            def all_predictions(start, x_ctx=x_ctx, u=u, state=state):
                preds, _ = predict(model, regime, x_ctx, u, H, start=start.unsqueeze(0), state=state)
                return preds[0]                                             # [H, 2]

            # Step 2: its Jacobian [H, 2, 2], converted from normalised to physical units
            J = jacobian(all_predictions, x_ctx[0, C].detach().clone())
            J = J.detach().cpu().numpy().astype(np.float64)
            J_physical = std[None, :, None] * J / std[None, None, :]
            if not np.all(np.isfinite(J_physical)):
                continue
            model_largest = max(np.linalg.svd(J_physical[k], compute_uv=False)[0] for k in range(H))

            # Step 3: the true system's product of one-step Jacobians along the same window
            true_states = norm.x_inverse(x[0, C:C + H].cpu().numpy().astype(np.float64))
            product = np.eye(2)
            true_largest = 0.0
            for J_step in system.jacobian(true_states):
                product = J_step @ product
                true_largest = max(true_largest, np.linalg.svd(product, compute_uv=False)[0])

            model_values.append(float(model_largest))
            true_values.append(float(true_largest))
    finally:
        model.train(was_training)

    if not model_values:
        return {"windows": 0}
    ratios = [m / t for m, t in zip(model_values, true_values)]
    return {"windows": len(model_values),
            "model_median": float(np.median(model_values)),
            "true_median": float(np.median(true_values)),
            "ratio_median": float(np.median(ratios))}


def _collect_tensors(state, found):
    """Put every tensor inside a (possibly nested) state into the list `found`, in a fixed order."""
    if isinstance(state, torch.Tensor):
        found.append(state)
        return
    for part in state:
        _collect_tensors(part, found)


def _rebuild_like(template, pieces):
    """Rebuild a nested state with the same structure as `template` from an iterator of tensors."""
    if isinstance(template, torch.Tensor):
        return next(pieces)
    rebuilt = [_rebuild_like(part, pieces) for part in template]
    if isinstance(template, tuple):
        return tuple(rebuilt)
    return rebuilt


def flatten_state(state):
    """All numbers of a memory state (batch size 1) as one long vector."""
    tensors = []
    _collect_tensors(state, tensors)
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_state(vector, template):
    """Inverse of flatten_state, using `template` for the shapes."""
    tensors = []
    _collect_tensors(template, tensors)
    pieces = []
    position = 0
    for t in tensors:
        size = t.numel()
        pieces.append(vector[position:position + size].reshape(t.shape))
        position = position + size
    return _rebuild_like(template, iter(pieces))


def step_jacobian_shape(model, split, context_len, horizon, n_windows, steps, device):
    """Full one-step Jacobian of (prediction, memory) during chained prediction; how non-normal it is.

    Normalised units (the model's own). Non-normality measures depend on the units; compare models
    with each other, not with the true system.
    """
    C = context_len
    H = horizon
    count = min(n_windows, split.xw.shape[0])
    chosen = np.linspace(0, split.xw.shape[0] - 1, count).astype(int)
    radius_values = []
    henrici_values = []
    defect_values = []
    size = None

    was_training = model.training
    model.train()      # see largest_magnification
    try:
        for i in chosen:
            x = split.xw[i:i + 1].to(device)
            u = split.uw[i:i + 1].to(device)
            x_ctx = x[:, :C + 1]

            # Step 1: run the chained prediction once and remember (prediction, memory) before each step
            remembered = {}
            with torch.no_grad():
                state = model.warmup(x_ctx, u)
                x_hat = x_ctx[:, C]
                for k in range(H):
                    if (k + 1) in steps:
                        remembered[k] = (x_hat, state)
                    inp = model._inp(x_hat, u[:, C + k], 0.0)
                    y, state = model.core.step(inp, state)
                    x_hat = x_hat + model.head(y)

            for k, (x_before, state_before) in remembered.items():
                n_x = x_before.shape[-1]

                # Step 2: one step of the closed loop as a function of one long vector
                def one_step(z, k=k, template=state_before, n_x=n_x, u=u):
                    x_part = z[:n_x].reshape(1, n_x)
                    memory = unflatten_state(z[n_x:], template)
                    inp = model._inp(x_part, u[:, C + k], 0.0)
                    y, new_memory = model.core.step(inp, memory)
                    x_next = x_part + model.head(y)
                    return torch.cat([x_next.reshape(-1), flatten_state(new_memory)])

                z0 = torch.cat([x_before.reshape(-1), flatten_state(state_before)]).detach().clone()
                J = jacobian(one_step, z0).detach().cpu().numpy().astype(np.float64)
                if not np.all(np.isfinite(J)):
                    continue
                size = J.shape[0]

                # Step 3: spectral radius, Henrici's departure from normality, Pervez-Locatello defect
                eigenvalues = np.linalg.eigvals(J)
                spectral_radius = float(np.max(np.abs(eigenvalues)))
                frobenius_squared = float(np.sum(J * J))
                eigen_squared = float(np.sum(np.abs(eigenvalues) ** 2))
                departure = np.sqrt(max(frobenius_squared - eigen_squared, 0.0))
                henrici = departure / np.sqrt(frobenius_squared) if frobenius_squared > 0 else 0.0
                defect = float(np.linalg.norm(J.T @ J - J @ J.T))

                radius_values.append(spectral_radius)
                henrici_values.append(float(henrici))
                defect_values.append(defect)
    finally:
        model.train(was_training)

    if not radius_values:
        return {"jacobians": 0}
    return {"jacobians": len(radius_values), "state_size": size,
            "spectral_radius_median": float(np.median(radius_values)),
            "spectral_radius_max": float(np.max(radius_values)),
            "henrici_median": float(np.median(henrici_values)),
            "pervez_defect_median": float(np.median(defect_values))}
