import torch


# ----------------------------------------------------------------------------------------
# The loss: how wrong is the model on one batch of windows?
# ----------------------------------------------------------------------------------------
def teacher_forcing_loss(model, states, push_window):
    # Step 1: inputs = true states 0..59, targets = true states 1..60
    current_state = states[:, :-1, :]        # [batch, 60, 2]
    true_next_state = states[:, 1:, :]       # [batch, 60, 2]

    # Step 2: predict all 60 next states at once, each from a TRUE current state
    prediction = model(current_state, push_window)

    # Step 3: mean squared error (normalised units)
    difference = prediction - true_next_state
    squared = difference ** 2
    loss = torch.mean(squared)

    return loss


# ----------------------------------------------------------------------------------------
# Validation: measure the loss on windows the model does not learn from
# ----------------------------------------------------------------------------------------
def validation_loss(model, window_state, window_push, loss_function, batch_size):
    # Step 1: switch to measuring mode
    model.eval()

    total = 0.0
    number_of_windows = window_state.shape[0]

    # Step 2: no gradients needed (we only measure, we do not learn)
    with torch.no_grad():
        for start in range(0, number_of_windows, batch_size):
            # Step 2a: cut out one batch
            end = start + batch_size
            batch_states = window_state[start:end]
            batch_pushes = window_push[start:end]

            # Step 2b: the average loss of this batch
            loss = loss_function(model, batch_states, batch_pushes)

            # Step 2c: turn the batch average back into a batch total
            windows_in_batch = batch_states.shape[0]
            total = total + float(loss) * windows_in_batch

    # Step 3: the fair average over all windows
    average = total / number_of_windows
    return average


# ----------------------------------------------------------------------------------------
# Training: nudge the weights, batch after batch, epoch after epoch
# ----------------------------------------------------------------------------------------
def train(model, batch_size, data, epochs, learning_rate, loss_function=teacher_forcing_loss):
    # Step 1: the data
    train_state, train_push = data["windows"]["train"]
    validation_state, validation_push = data["windows"]["validation"]
    number_of_windows = train_state.shape[0]

    # Step 2: the optimiser: the rule that changes the weights
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        # Step 3: switch to learning mode
        model.train()

        loss_sum = 0.0
        number_of_batches = 0

        # Step 4: a new random order of windows every epoch
        shuffled_order = torch.randperm(number_of_windows)

        for start in range(0, number_of_windows, batch_size):
            # Step 5a: pick this batch's windows
            end = start + batch_size
            batch_indices = shuffled_order[start:end]
            batch_states = train_state[batch_indices]
            batch_pushes = train_push[batch_indices]

            # Step 5b: forget the gradients of the previous batch
            optimiser.zero_grad()

            # Step 5c: how wrong is the model on this batch?
            loss = loss_function(model, batch_states, batch_pushes)

            # Step 5d: for every weight, which direction makes the loss smaller?
            loss.backward()

            # Step 5e: move every weight a small step in that direction
            optimiser.step()

            # Step 5f: keep a plain number (not the tensor) for the report
            loss_sum = loss_sum + float(loss)
            number_of_batches = number_of_batches + 1

        # Step 6: report this epoch
        average_train_loss = loss_sum / number_of_batches
        average_validation_loss = validation_loss(model, validation_state, validation_push,
                                                  loss_function, batch_size)

        print(f"epoch {epoch + 1:3d}/{epochs}   "
              f"training loss {average_train_loss:.6f}   "
              f"validation loss {average_validation_loss:.6f}")

    return model