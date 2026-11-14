import torch

import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))



from src.models.registry import build_model
from src.data.datasets import build_data
from src.control.predictors import TrueModelPredictor, LearnedModelPredictor

# load the trained memory-removed model
path = "outputs/exp04_memory_removed/runs/chain_mamba_no_memory_seed0/checkpoint.pt"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
model = build_model(checkpoint["config"])
model.load_state_dict(checkpoint["model"])

# the same normalisation numbers and test data as in training
data = build_data(checkpoint["config"])
test = data["splits"]["test"]
device = torch.device("cpu")

learned = LearnedModelPredictor(model, data["norm"], device)
perfect = TrueModelPredictor(mu=1.0, dt=0.1)

# 5 starting states from the test data and their true next 10 pushes
x_now = torch.tensor(test.x[0:5], dtype=torch.float32)                         # [5, 2]
u_plan = torch.tensor(test.u[0:14, 0], dtype=torch.float32).unfold(0, 10, 1)   # [5, 10]

with torch.no_grad():
    a = learned.predict(x_now, u_plan)
    b = perfect.predict(x_now, u_plan)

print(a.shape)                          # torch.Size([5, 10, 2])
print((a - b).abs().max())              # should be small, around 0.01–0.1