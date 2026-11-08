"""
Training losses for the three training regimes.

All losses are computed in normalised units, using mean squared error
(the average of the squared differences between prediction and truth).

The three regimes are:
    - "teacher_forcing": the model always sees the true previous state
    - "direct":          the model predicts the whole horizon in one forward pass
    - anything else:     chained prediction (the model is fed its own predictions)
"""

from __future__ import annotations

import torch

from src.models.dynamics import DynamicsModel


def mean_squared_error(prediction, target):
    # Step 1: difference between prediction and truth
    difference = prediction - target

    # Step 2: square every difference (so negative and positive errors both count)
    squared_difference = difference ** 2

    # Step 3: average over everything (batch, time, state dimensions)
    loss = torch.mean(squared_difference)

    return loss


def teacher_forcing_loss(model: DynamicsModel, x, u):
    """
    Teacher forcing: at every step the model gets the TRUE current state
    and must predict the next state.
    """

    # Step 1: split the true trajectory into inputs and targets
    # inputs  = states 0, 1, ..., T-1   (everything except the last state)
    # targets = states 1, 2, ..., T     (everything except the first state)
    true_inputs = x[:, :-1]
    true_targets = x[:, 1:]

    # Step 2: ask the model for its one-step predictions
    raw_prediction = model.teacher_forced(true_inputs, u)

    # Step 3: clamp any exploding values so the loss stays finite
    # "is_bad" marks which values had to be clamped
    safe_prediction, is_bad = model._safe(raw_prediction)

    # Step 4: compute the loss
    loss = mean_squared_error(safe_prediction, true_targets)

    # Step 5: count how many rows (trajectories) hit the clamp at least once
    # .any(dim=1) -> True for a row if ANY time step in that row was bad
    row_was_bad = is_bad.any(dim=1)
    number_of_bad_rows = int(row_was_bad.sum())

    return loss, number_of_bad_rows


def direct_loss(model: DynamicsModel, x, u, context_len: int, horizon: int):
    """
    Direct multi-step prediction: the model predicts the whole horizon in one forward pass.
    Nothing is fed back.
    """

    # Step 1: run the direct prediction
    # predictions : what the model predicted for the horizon
    # targets     : the true states for the same horizon
    # hit_clamp   : which rows had to be clamped because values exploded
    predictions, targets, hit_clamp = model.rollout(
        "direct",
        x,
        u,
        context_len,
        horizon,
    )

    # Step 2: compute the loss
    loss = mean_squared_error(predictions, targets)

    # Step 3: count how many rows hit the clamp
    number_of_bad_rows = int(hit_clamp.sum())

    return loss, number_of_bad_rows


def chained_loss(model: DynamicsModel, x, u, context_len: int, horizon: int,
                 noise_std: float = 0.0, train_horizon:int = None, teacher_weight:float = 0.0):
    """
    Chained prediction: the model is fed its OWN predictions, one step after another.
    RAW version: always chain the full horizon, and use only the chained loss.
    """

    # Step 1: how many steps to chain: always the full horizon (50)
    if train_horizon is None: steps_to_chain = horizon
    else:
        step_to_chain = int(train_horizon)
        step_to_chain = min(step_to_chain, horizon)
        step_to_chain = max(1, step_to_chain)

    # Step 2: the true starting states (context), and the true future to compare against
    true_context = x[:, :context_len + 1]
    future_start = context_len + 1
    future_end = context_len + 1 + steps_to_chain
    true_future = x[:, future_start:future_end]

    # Step 3: let the model chain its own predictions
    predictions, hit_clamp = model.chained(true_context, u, steps_to_chain, noise_std=noise_std)

    # Step 4: the chained loss
    loss = mean_squared_error(predictions, true_future)
        
    # Step 5 (RECIPE): add the one-step loss, so each single step also stays accurate
    if teacher_weight > 0:
        one_step_loss, _ = teacher_forcing_loss(model, x, u)
        loss = loss + teacher_weight*one_step_loss



    # Step 5: count how many rows hit the clamp
    number_of_bad_rows = int(hit_clamp.sum())

    return loss, number_of_bad_rows


def regime_loss(model: DynamicsModel, x, u, regime: str, context_len: int, horizon: int,
                noise_std: float = 0.0, train_horizon = None, teacher_weight: float = 0.0):
    """Main entry point used by the trainer. Returns (loss, number of rows that hit the clamp)."""

    if regime == "teacher_forcing":
        loss, number_of_bad_rows = teacher_forcing_loss(model, x, u)
    elif regime == "direct":
        loss, number_of_bad_rows = direct_loss(model, x, u, context_len, horizon)
    else:
        loss, number_of_bad_rows = chained_loss(
            model,
            x,
            u,
            context_len,
            horizon,
            noise_std=noise_std,
            train_horizon=train_horizon,
            teacher_weight=teacher_weight
        )

    return loss, number_of_bad_rows
