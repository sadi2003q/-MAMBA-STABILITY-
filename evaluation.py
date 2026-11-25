
import torch



def real_unite_error(states, standard_deviation) :
    normaliser = torch.tensor(standard_deviation, dtype=torch.float32)
    return states * normaliser


def one_step_error(
        model,
        state_window,
        push_window,
        normaliser
):
    model.eval()
    with torch.no_grad():
        input_state = state_window[:, :-1, :]
        gt_state    = state_window[:, 1:, :]

        prediction = model(input_state, push_window)
        error = real_unite_error(prediction - gt_state, normaliser["state_std"])

    return float(torch.sqrt(torch.mean(error**2)))


def chain_prediction(
        model, 
        state_window, 
        push_window,
        context_length,
        horizon
):
    input_state = state_window[:, context_length, :]

    all_prediction = []
    for k in range(horizon):
        input_push = push_window[:, context_length+k, :]
        predicted_state = model(input_state, input_push)

        
        
        next_states = torch.nan_to_num(predicted_state, nan=0.0, posinf=50, neginf=-50)
        next_states = torch.clamp(predicted_state, -50, 50)

        input_state = next_states
        all_prediction.append(predicted_state)

    prediction = torch.stack(all_prediction, dim=1)
    return prediction


def chained_error(
    model,
    state_window,
    push_window,
    context_length,
    horizon,
    normaliser 
):

    model.eval()
    with torch.no_grad():
        
        push_window
        prediction = chain_prediction(
            model,
            state_window,
            push_window,
            context_length,
            horizon,
            
        )

        first = context_length+1
        last = context_length+1+horizon
        gt_states = state_window[:, first:last, :]
        error = real_unite_error(prediction - gt_states, normaliser["state_std"])
        squared = error ** 2

        # Step 4: error at each step (average over windows and components, then root)
        per_step = torch.sqrt(torch.mean(squared, dim=(0, 2)))     # [horizon]

        # Step 5: ONE number over all 50 steps: the 50-step chained error
        overall = torch.sqrt(torch.mean(squared))

        # Step 6: how many windows hit the clamp (blew up)
        hit_clamp = (prediction.abs() >= 50).any(dim=2).any(dim=1)
        number_blown_up = int(hit_clamp.sum())

    return  {
        "overall": float(overall),
        "per_step": per_step.numpy(),
        "blown_up": number_blown_up,
    }
    