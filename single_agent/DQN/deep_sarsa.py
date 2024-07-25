import random
import copy
import yaml
import gymnasium as gym
import torch

from single_agent.DQN.seed import seed_all
from evaluation.log_values import Logger
from single_agent.DQN.env_wrapper import EnvWrapper
from single_agent.DQN.neural_network import NeuralNetwork


def epsilon_greedy_policy(network, state, env, epsilon):
    compare = random.random()
    if compare < epsilon:
        return torch.tensor(env.action_space.sample()).unsqueeze(0).unsqueeze(0)
    else:
        return torch.argmax(network(state).detach(), dim=1, keepdim=True)

def deep_sarsa_training(config: dict) -> None:

    env = gym.make(config["ENVIRONMENT"])
    env = EnvWrapper(env)
    logger = Logger("Deep SARSA", "MountainCar-v0", config["ALPHA"], config["GAMMA"])
    seed_all(config["SEED"], env)
    q_network = NeuralNetwork(env)
    state = env.reset()
    target_network = copy.deepcopy(q_network)
    target_network = target_network.eval()
    action = epsilon_greedy_policy(q_network, state, env, config["EPSILON"])
    print(action)




def deep_sarsa_test(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"])


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    deep_sarsa_training(config)
    deep_sarsa_test(config)

