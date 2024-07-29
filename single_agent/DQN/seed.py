import numpy as np
import torch
def seed_all(seed: int, env: object) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)