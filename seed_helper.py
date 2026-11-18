import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    set_seed(0)
    first_draw = torch.rand(3)

    set_seed(0)
    second_draw = torch.rand(3)

    print(f"First draw : {first_draw}")
    print(f"Second draw {second_draw}")
    print(f"Same ? ---> {torch.equal(first_draw, second_draw)}")