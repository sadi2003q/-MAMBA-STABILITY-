


import numpy as np


MU = 1.0
DT = 0.1


def per_step_change(state, push):
    
    position = state[0]
    velocity = state[1]

    position_rate = velocity
    velocity_rate = MU * (1-position**2)*velocity - position + push


    return np.array([position_rate, velocity_rate])

def eular_step(state, push):
    rates = per_step_change(state, push)
    changes = DT * rates

    next_state = state + changes
    return next_state


def simulate(start_state, pushes:any):

    all_states = [np.array(start_state, dtype=float)]
    current_state = start_state
    for k in range(len(pushes)):
        upcomming_state = eular_step(state=current_state, push=pushes[k])
        if not np.all(np.isfinite(upcomming_state)):
            raise ValueError(f"Infinite at step {k}")
        all_states.append(upcomming_state)
        current_state = upcomming_state
    states = np.stack(all_states)

    return states

