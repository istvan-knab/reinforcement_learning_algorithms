import random
import copy
import yaml
import gymnasium as gym
import torch
from torch.optim import Adam
from torch.functional import F
from tqdm import tqdm

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

def deep_sarsa_training(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"], 500, render_mode="rgb_array")
    env = EnvWrapper(env)
    logger = Logger("Deep SARSA", config["ENVIRONMENT"], config["ALPHA"], config["GAMMA"])
    logger.start_training(config)
    seed_all(config["SEED"], env)
    q_network = NeuralNetwork(env)
    target_network = copy.deepcopy(q_network)
    target_network = target_network.eval()
    oprimizer = Adam(q_network.parameters(), lr=config["ALPHA"])
    memory = ReplayMemory(config["BUFFER"], config["BATCH"])
    epsilon = 1


    for episode in tqdm(range(config["EPISODES"]),desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        epsilon *= config["EPSILON"]

        while not done :
            action = epsilon_greedy_policy(q_network, state, env, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            env.render()
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            memory.add_element(state, action, next_state, reward, done)
            if memory.__len__() >= 5 * config["BATCH"]:
                state_batch, action_batch, next_state_batch, reward_batch, done_batch = memory.sample()
                qsa_batch = q_network(state_batch).gather(1, action_batch)
                next_action_batch = epsilon_greedy_policy(q_network, next_state_batch, env, epsilon)
                next_qsa_batch = target_network(next_state_batch).gather(1, next_action_batch)
                target_batch = reward_batch + ~done_batch * config["GAMMA"] * next_qsa_batch
                loss = F.mse_loss(qsa_batch, target_batch)
                q_network.zero_grad()
                loss.backward()
                oprimizer.step()
                episode_loss += loss.item()
        if episode % config["UPDATE_FREQ"] == 0:
            target_network.load_state_dict(q_network.state_dict())
        logger.step(episode_reward, epsilon, episode, episode_loss)






def deep_sarsa_test(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"])


if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    deep_sarsa_training(config)
    deep_sarsa_test(config)

