

import numpy as np
import torch
from vanderpol import DT, simulate



SLOWEST_FREQUENCY = 0.0049     # hertz
FASTEST_FREQUENCY = 4.88       # hertz
NUMBER_OF_WAVES = 30
PEAK_PUSH = 15.0

# Start-state range
START_LOW = [-2.5, -2.0]       # [position, velocity]
START_HIGH = [2.5, 2.0]

# How long each trajectory is
TRAIN_STEPS = 40000
VALIDATION_STEPS = 10000
TEST_STEPS = 10000

DATA_SEED = 12345



def make_multisine(num_of_steps, random_generator):
    
    step_number = np.arange(num_of_steps)
    times = step_number * DT

    frequencies = np.linspace(
        SLOWEST_FREQUENCY, 
        FASTEST_FREQUENCY, 
        NUMBER_OF_WAVES
    )
    phases = random_generator.uniform(
        0.0, 
        2.0*np.pi, 
        NUMBER_OF_WAVES
    )
    all_waves = []
    for i in range(NUMBER_OF_WAVES):
        angle = 2.0 * np.pi * frequencies[i] * times + phases[i]
        one_wave = np.sin(angle)
        all_waves.append(one_wave)

    wave_table = np.stack(all_waves, axis = 1)
    pushes     = np.sum(wave_table,  axis=1)
    largest    = np.max(np.abs(pushes))
    pushes     = pushes / largest
    pushes     = pushes * PEAK_PUSH

    return pushes


def make_trajectory(num_of_steps, random_generator):
    pushes = make_multisine(
        num_of_steps=num_of_steps, 
        random_generator=random_generator
    )

    start_state = random_generator.uniform(START_LOW, START_HIGH)
    states      = simulate(start_state=start_state, pushes=pushes)

    return states, pushes

def make_all_trajectories():
    random_generator = np.random.default_rng(DATA_SEED)
    train_states , train_pushes = make_trajectory(TRAIN_STEPS, random_generator)
    validation_states, validation_pushes = make_trajectory(VALIDATION_STEPS, random_generator)
    test_states, test_pushes     = make_trajectory(TEST_STEPS, random_generator)
    

    trajectories = {
        "train": (train_states, train_pushes),
        "validation": (validation_states, validation_pushes),
        "test": (test_states, test_pushes),
    }
    return trajectories



def compute_normaliser(train_states, train_pushes):
    state_mean = np.mean(train_states, axis=0)
    state_std  = np.std(train_states, axis=0)

    push_mean = np.mean(train_pushes, axis=0)
    push_std  = np.std(train_pushes, axis=0)

    normaliser = {
        "state_mean": state_mean,
        "state_std": state_std,
        "push_mean": push_mean,
        "push_std": push_std,
    }
    return normaliser


def normalise_states(states, normaliser):
    shifted = states - normaliser["state_mean"]
    scaled = shifted / normaliser["state_std"]

    return scaled

def normalise_push(push, normaliser):
    shifted = push - normaliser["push_mean"]
    scaled = shifted / normaliser["push_std"]

    return scaled




def unnormalise_state(normalised_state, normaliser):
    scaled_back = normalised_state * normaliser["state_std"]
    shifted_back = scaled_back + normaliser["state_mean"]

    return shifted_back



# -------  WINDOW MANAGEMENT -------

CONTEXT_LENGTH = 10    # true steps given before chaining starts
HORIZON = 50           # steps the model must predict on its own
STRIDE = 5             # gap between the starts of two windows

def cut_windws(normalised_state, normalised_push):
    
    window_length = CONTEXT_LENGTH + HORIZON
    number_of_pushes = len(normalised_push)
    
    last_possible_start = number_of_pushes - window_length
    starts = np.arange(0, last_possible_start+1, STRIDE)

    states_window = []
    pushs_window = []
    
    for start in starts:
        state_end = start + window_length + 1
        push_end = start + window_length

        state_window = normalised_state[start : state_end]
        push_window  = normalised_push[start : push_end]

        states_window.append(state_window)
        pushs_window.append(push_window)


    state_window = np.stack(states_window)
    push_window  = np.stack(pushs_window)

    push_window = push_window[:, :, np.newaxis]

    state_tensor = torch.tensor(state_window, dtype=torch.float32)
    push_tensor = torch.tensor(push_window, dtype=torch.float32)
    return state_tensor, push_tensor


def build_data():
    trajectories = make_all_trajectories()
    train_state, train_push = trajectories["train"]

    normaliser = compute_normaliser(
        train_state,
        train_push
    )

    window = {}

    for split_name in ["train", "validation", "test"]:
        states, pushes = trajectories[split_name]
        normalised_states = normalise_states(
            states, 
            normaliser=normaliser
        )

        normalised_pushes = normalise_push(
            pushes,
            normaliser
        )

        state_windows, push_windows = cut_windws(normalised_states, normalised_pushes)
        window[split_name] = (
            state_windows, 
            push_windows
        )

    data = {
        "trajectories" : trajectories, 
        "normaliser"   : normaliser,
        "windows"      : window
    }

    return data