import torch
import torch.nn as nn


class ReadoutPredictor(nn.Module):
    def __init__(self, hidden_size, predict_change, use_scaling=False, change_scale=None,
                 use_anchor=False):
        super().__init__()

        # Step 1: the SAME body as Stage 3's PlainPredictor
        self.layer_1 = nn.Linear(3, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 2)
        self.activation = nn.SiLU()

        # Step 2: predict the change instead of the state (Part 4.1)
        self.predict_change = predict_change

        # Step 3: output scaling (Part 4.2)
        self.use_scaling = use_scaling
        if use_scaling:
            scale = torch.tensor(change_scale, dtype=torch.float32)
        else:
            scale = torch.ones(2)
        self.register_buffer("change_scale", scale) # not understood

        # Step 4: the anchor (Part 4.3)
        self.use_anchor = use_anchor
        if use_anchor:
            self.anchor = nn.Linear(3, 2, bias=False)
            nn.init.zeros_(self.anchor.weight)

    def forward(self, current_states, pushes):
        # Step 1: state and push side by side -> [..., 3]
        inputs = torch.cat([current_states, pushes], dim=-1)

        # Step 2: the body
        features = self.layer_1(inputs)
        features = self.activation(features)
        features = self.layer_2(features)
        features = self.activation(features)
        network_output = self.output_layer(features)

        # Step 3: the anchor adds its straight-line guess
        if self.use_anchor:
            anchor_output = self.anchor(inputs)
            network_output = network_output + anchor_output

        # Step 4: output scaling
        scaled_output = network_output * self.change_scale

        # Step 5: the readout
        if self.predict_change:
            next_states = current_states + scaled_output
        else:
            next_states = scaled_output

        return next_states