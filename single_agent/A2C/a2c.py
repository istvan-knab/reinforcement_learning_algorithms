import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm

from single_agent.DQN.seed import seed_all
from evaluation.log_values import Logger
from single_agent.DQN.env_wrapper import EnvWrapper
from single_agent.DQN.neural_network import NeuralNetwork
from single_agent.A2C.actor_critic_network import ActorNetwork, CriticNetwork

def a2c(config):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.seed()
    # setup environment
    env = gym.make(config["ENVIRONMENT"])
    env = EnvWrapper(env)
    # setup logger
    logger = Logger("A2C", config["ENVIRONMENT"], config["ALPHA"], config["GAMMA"])
    logger.start_training(config)
    seed_all(config["SEED"], env)
    policy_network = ActorNetwork(env)
    value_network = CriticNetwork(env)

    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        while not done:
            pass

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    a2c(config)