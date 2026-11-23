

from config_loader import load_config
from seed_helper import set_seed
from data import build_data
from plain_predictor import PlainPredictor
from training import train

import os
import torch

config = load_config()
set_seed(config["model_seed"])


data = build_data()


# Variable Declaration : 
hidden_dim = config["plain_predictor"]["hidden_dim"]
batch_size = config["teacher_forcing_training"]["batch_size"]
epoch = config["teacher_forcing_training"]["epochs"]
learning_rate = config["teacher_forcing_training"]["learning_rate"]


model = PlainPredictor(hidden_dim=hidden_dim)

train(
    model=model,
    batch_size=batch_size,
    data=data,
    epoch=epoch,
    learning_rate=learning_rate
)

os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), "outputs/stage3_plain.pt")
print("saved outputs/stage3_plain.pt")





