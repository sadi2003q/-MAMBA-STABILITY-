import numpy as np

from data import build_data

# Step 1: build everything
data = build_data()

# Step 2: the raw training trajectory
print("\n\n\n\n\n\n------------ TRAINING ------------")
train_states, train_pushes = data["trajectories"]["train"]
print("training start state:", np.round(train_states[0], 2))
print("first three pushes: ", np.round(train_pushes[:3], 2))
print("largest absolute push:", np.round(np.max(np.abs(train_pushes)), 2))

# Step 3: the normalisation numbers
print("\n\n\n\n\n\n------------ NORMALISER ------------")
normaliser = data["normaliser"]
print("state mean  :", np.round(normaliser["state_mean"], 2))
print("state spread:", np.round(normaliser["state_std"], 2))
print("push mean.  :", np.round(normaliser["push_mean"], 2), " push spread:", np.round(normaliser["push_std"], 2))

# Step 4: the windows
print("\n\n\n\n\n\n------------ SPLITTING SYSTEM ------------")
for split_name in ["train", "validation", "test"]:
    state_windows, push_windows = data["windows"][split_name]
    print(split_name, "states:", tuple(state_windows.shape), " pushes:", tuple(push_windows.shape))

# Step 5: normalised training windows should average about 0 with spread about 1
train_state_windows, _ = data["windows"]["train"]
print("normalised training mean:", np.round(train_state_windows.mean(dim=(0, 1)), 2))
print("normalised training spread:", np.round(train_state_windows.std(dim=(0, 1)), 2))
