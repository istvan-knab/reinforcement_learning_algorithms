import random
import copy
import yaml
import gymnasium as gym
import torch
from torch.optim import AdamW
from torch import nn as nn
from tqdm import tqdm
import numpy as np

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


def deep_q_training(config):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.seed()
    #setup environment
    env = gym.make(config["ENVIRONMENT"], render_mode="rgb_array")
    env = EnvWrapper(env)
    #setup logger
    logger = Logger("DQN", config["ENVIRONMENT"], config["ALPHA"], config["GAMMA"])
    logger.start_training(config)
    seed_all(config["SEED"], env)
    #setup environments
    q_network = NeuralNetwork(env)
    target_network = NeuralNetwork(env)
    #create optimization tools
    criterion = nn.MSELoss()
    optimizer = AdamW(q_network.parameters(), lr=config["ALPHA"])
    epsilon = 1
    #Create Experience Replay
    memory = ReplayMemory(config["BUFFER"], config["BATCH"])

    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        if memory.__len__() >= 5 * config["BATCH"]:
            epsilon *= config["EPSILON"]

        while not done:
            action = epsilon_greedy_policy(q_network, state, env, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            env.render()
            episode_reward += reward
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            memory.add_element(state, action, next_state, reward, done)
            if memory.__len__() >= config["BATCH"]:
                state_batch, action_batch, next_state_batch, reward_batch, done_batch = memory.sample()
                qsa_batch = q_network(state_batch).gather(1, action_batch)
                next_qsa_batch = target_network(next_state_batch)
                next_qsa_batch = torch.max(next_qsa_batch, dim=-1, keepdim=True)[0]
                target = reward_batch + ~done_batch * config["GAMMA"] * next_qsa_batch
                loss = criterion(qsa_batch, target)
                q_network.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()

        if episode % config["UPDATE_FREQ"] == 0:
            target_network = q_network
        logger.step(episode_reward, epsilon, episode, episode_loss)


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    deep_q_training(config)
