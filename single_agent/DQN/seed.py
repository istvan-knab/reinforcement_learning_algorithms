import numpy as np
import torch
def seed_all(seed: int, env: object) -> None:

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)