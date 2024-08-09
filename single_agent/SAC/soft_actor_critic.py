import yaml
import mujoco
import torch
from torch.li
import gymnasium as gym

from single_agent.DQN.experience_replay import ReplayMemory
from single_agent.DQN.env_wrapper import EnvWrapper

class SAC(Li)
def soft_actor_critic(config):
    env = gym.make(config['ENVIRONMENT'], render_mode="human")
    env = EnvWrapper(env)
    print(env.action_space)
    memory = ReplayMemory(config["BUFFER"], config["BATCH"])
    for episode in range(config["EPISODES"]):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        while not done:
            env.render()
if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    soft_actor_critic(config)