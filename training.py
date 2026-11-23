

import torch
import numpy as np


def teacher_forcing_loss(model, states, push_window):
    current_state = states[:, :-1, :]
    true_next_state = states[:, 1:, :]

    prediction = model(current_state, push_window)
    diff = prediction-true_next_state
    square = diff**2
    return torch.mean(square)        


def validation_loss(model, window_state, window_push, loss_function, batch_size):
    model.eval()
    total = 0.0
    num_of_window = window_state.shape[0]

    with torch.no_grad():
        for start in range(0, num_of_window, batch_size):
            batch_states = window_state[start: start + batch_size]
            batch_pushes = window_push[start: start + batch_size]

            loss = loss_function(model, batch_states, batch_pushes)

            windows_in_batch = batch_states.shape[0]
            total = total + float(loss) * windows_in_batch
        return total / num_of_window





def train(
        model,  
        batch_size, 
        data, 
        epoch, 
        learning_rate,

        loss_function=teacher_forcing_loss
    ):
    train_state, train_push = data["windows"]["train"]
    validation_state, validation_push = data["windows"]["validation"]
    number_of_windows = train_state.shape[0]

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch):
        model.train()
        
        loss_sum = 0
        num_of_batch = 0
        shuffled_order = torch.randperm(number_of_windows)


        for start in range(0, number_of_windows, batch_size):
            
            end = start + batch_size
            batch_indices = shuffled_order[start:end]
            train_batch = train_state[batch_indices]
            push_batch  = train_push[batch_indices]

            
            optimiser.zero_grad()
            loss = loss_function(model, train_batch, push_batch)
            loss.backward()
            optimiser.step()


            loss_sum = loss_sum + float(loss)
            num_of_batch = num_of_batch + 1

        total_loss = loss/num_of_batch
        val_loss = validation_loss(model, validation_state, validation_push, loss_function, batch_size)

        print(f"epoch {epoch + 1}/{epoch} : training loss {total_loss:.6f} || validation loss {val_loss:.6f}")

    return model
    

