import random
import copy
import yaml
import gymnasium as gym
import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm


from single_agent.DQN.seed import seed_all
from evaluation.log_values import Logger
from single_agent.DQN.env_wrapper import EnvWrapper

def deep_sarsa_training(config: dict) -> None:

    env = gym.make(config["ENVIRONMENT"])
    env = EnvWrapper(env)
    logger = Logger("Deep SARSA", "MountainCar-v0", config["ALPHA"], config["GAMMA"])
    seed_all(config["SEED"], env)



def deep_sarsa_test(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"])


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    deep_sarsa_training(config)
    deep_sarsa_test(config)

