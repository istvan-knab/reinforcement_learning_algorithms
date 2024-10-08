import gymnasium as gym
import torch
class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    def reset(self):
        state = self.env.reset()[0]
        return torch.from_numpy(state).unsqueeze_(dim=0).float()

    def step(self, action):
        action = action.item()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze_(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        terminated = torch.tensor(terminated).view(1, -1).float()
        truncated = torch.tensor(truncated).view(1, -1).float()

        return next_state, reward, terminated, truncated, info
