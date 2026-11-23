

import torch
import torch.nn as nn
from seed_helper import set_seed

set_seed(0)


class PlainPredictor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(3, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, state, push):
        x = torch.cat([state, push], dim=-1)

        feature = self.layer1(x)
        feature = self.activation(feature)

        feature = self.layer2(feature)
        feature = self.activation(feature)

        output = self.output(feature)

        return output

def count_parameter(model):
    total = 0
    for parameter in model.parameters():
        total = total + parameter.numel()
    return total


if __name__ == "__main__":
    model = PlainPredictor( hidden_dim=64 )
    print(model)
    print(f"Parameter Count : {count_parameter(model)}")

    one_state = torch.tensor([1.0, 1.0])
    one_push  = torch.tensor([0.0])

    predicted = model(one_state, one_push)
    print(f"Untrained Prediction : {predicted}")

    many_states = torch.zeros(5, 60, 2)
    many_pushes = torch.zeros(5, 60, 1)

    predicted = model(many_states, many_pushes)
    print(f"Output Shape for {predicted.shape}")

