
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


from seed_helper import set_seed
from vanderpol import simulate


set_seed(0)
if not os.path.exists("outputs"):
    os.mkdir("outputs")

num_of_steps = 1000
num_of_push = np.zeros(num_of_steps)

inside_start = np.array([0.5, 0.0])
outside_start = np.array([3.0, 3.0])

from_inside = simulate(start_state=inside_start, pushes=num_of_push)
from_outside = simulate(start_state=outside_start, pushes=num_of_push)

settle_from_inside = from_inside[500:, :]
settled_position = settle_from_inside[..., 0]

plt.plot(settled_position)
plt.xlabel("Index of position")
plt.ylabel("Value of position")
plt.savefig("outputs/IndexPosition.png", dpi=120)

print(f"first Position : {round(settled_position[1], 2)}")
print(f"Max Position : {round(settled_position.max(), 2)}")
print(f"Minimunm Positino : {round(settled_position.min(), 2)}")

upward_crossings = []
for k in range(1, len(settled_position)):
    was_below = settled_position[k-1] < 0
    now_above = settled_position[k] >= 0

    if was_below and now_above:
        upward_crossings.append(k)
steps_per_loop = np.diff(upward_crossings)
print(f"step per loop : {steps_per_loop}")



plt.figure(figsize=(6, 6))
plt.plot(from_inside[:, 0], from_inside[:, 1], label="start inside (0.5, 0)")
plt.plot(from_outside[:, 0], from_outside[:, 1], label="start outside (3, 3)")
plt.xlabel("position")
plt.ylabel("velocity")
plt.title("Van der Pol: both starts end on the same loop")
plt.legend()
plt.savefig("outputs/stage1_limit_cycle.png", dpi=120)
print("picture saved to outputs/stage1_limit_cycle.png")

