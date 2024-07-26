import random
import copy
import yaml
import gymnasium as gym
import torch
from torch.optim import AdamW
from tqdm import tqdm
import time

from single_agent.DQN.seed import seed_all
from evaluation.log_values import Logger
from single_agent.DQN.env_wrapper import EnvWrapper
from single_agent.DQN.neural_network import NeuralNetwork
from single_agent.DQN.experience_replay import ReplayMemory


def epsilon_greedy_policy(network, state, env, epsilon):
    compare = random.random()
    if compare < epsilon:
        return torch.tensor(env.action_space.sample()).unsqueeze(0).unsqueeze(0)
    else:
        return torch.argmax(network(state).detach(), dim=1, keepdim=True)
def set_tqdm():
    WHITE = '\033[97m'
    RESET = '\033[0m'
    tqdm_format = f'{WHITE}{{l_bar}}{{bar}}{{r_bar}}{RESET}'
    return tqdm_format

def deep_sarsa_training(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"], 500, render_mode="rgb_array")
    env = EnvWrapper(env)
    logger = Logger("Deep SARSA", "MountainCar-v0", config["ALPHA"], config["GAMMA"])
    seed_all(config["SEED"], env)
    q_network = NeuralNetwork(env)
    state = env.reset()
    target_network = copy.deepcopy(q_network)
    target_network = target_network.eval()
    oprimizer = AdamW(q_network.parameters(), lr=config["ALPHA"])
    memory = ReplayMemory(config["BUFFER"], config["BATCH"])
    epsilon = 1

    for episode in tqdm(range(config["EPISODES"]),desc='Training Process',
                        bar_format=set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        epsilon *= config["EPSILON"]
        episode_step = 0

        while not done :
            episode_step +=1
            action = epsilon_greedy_policy(q_network, state, env, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            memory.add_element(state, action, next_state, reward, done)
            if memory.__len__() >= config["BATCH"]:
                batch = memory.sample()






def deep_sarsa_test(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"])


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    deep_sarsa_training(config)
    deep_sarsa_test(config)

