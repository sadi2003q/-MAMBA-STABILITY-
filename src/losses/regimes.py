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


def rollout_loss(model: DynamicsModel, x, u, regime: str, context_len: int, horizon: int):
    """
    Rollout training: the model predicts several steps into the future.
    Either all at once ("direct") or by feeding its own predictions back ("chained").
    """

    # Step 1: decide which kind of rollout to use
    # Only the exact name "direct" means direct multi-step prediction.
    # Every other regime name falls back to chained prediction.
    if regime == "direct":
        rollout_type = "direct"
    else:
        rollout_type = "chained"

    # Step 2: run the rollout
    # predictions : what the model predicted for the horizon
    # targets     : the true states for the same horizon
    # hit_clamp   : which rows had to be clamped because values exploded
    predictions, targets, hit_clamp = model.rollout(
        rollout_type,
        x,
        u,
        context_len,
        horizon,
    )

    # Step 3: compute the loss
    loss = mean_squared_error(predictions, targets)

    # Step 4: count how many rows hit the clamp
    number_of_bad_rows = int(hit_clamp.sum())

    return loss, number_of_bad_rows


def regime_loss(model: DynamicsModel, x, u, regime: str, context_len: int, horizon: int):
    """
    Main entry point used by the trainer.

    Shapes:
        x : [batch, context_len + horizon + 1, number_of_state_variables]
        u : [batch, context_len + horizon,     number_of_control_inputs]

    Returns:
        (loss, number of rows that hit the clamp)
    """

    if regime == "teacher_forcing":
        loss, number_of_bad_rows = teacher_forcing_loss(model, x, u)
    else:
        loss, number_of_bad_rows = rollout_loss(model, x, u, regime, context_len, horizon)

    return loss, number_of_bad_rows