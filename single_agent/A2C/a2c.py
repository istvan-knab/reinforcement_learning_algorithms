import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.optim import AdamW
import torch.nn.functional as F
from torch import nn

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
    env = gym.make(config["ENVIRONMENT"], render_mode=config["RENDER"])
    env = EnvWrapper(env)
    # setup logger
    logger = Logger("A2C", config["ENVIRONMENT"], config["ALPHA"], config["GAMMA"])
    logger.start_training(config)
    seed_all(config["SEED"], env)
    policy_network = ActorNetwork(env)
    value_network = CriticNetwork(env)
    policy_optim = AdamW(policy_network.parameters(), lr=config["ALPHA"])
    value_optim = AdamW(value_network.parameters(), lr=config["ALPHA"])

    for episode in tqdm(range(config["EPISODES"]), desc='Training Process',
                        bar_format=logger.set_tqdm(), colour='white'):
        state = env.reset()
        done = torch.tensor(False).unsqueeze(0).unsqueeze(0)
        episode_reward = 0
        episode_loss = 0
        I = 1.
        while not done:
            action = policy_network(state).multinomial(1).detach()
            next_state, reward, terminated, truncated, _ = env.step(action)
            if config["RENDER"] == "human":
                env.render()
            if terminated or truncated:
                done = torch.tensor(True).unsqueeze(0).unsqueeze(0)
            value = value_network(next_state)
            #Update value net
            target = reward + ~done * config["GAMMA"] * value_network(next_state).detach()
            critic_loss = F.mse_loss(value, target)
            value_network.zero_grad()
            critic_loss.backward()
            value_optim.step()

            #Update policy
            advantage = (target - value).detach()
            probs = policy_network(state)
            log_probs = torch.log(probs + 1e-6)
            action_log_prob = log_probs.gather(1, action)

            #Calculate entropy
            entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
            actor_loss = -I * action_log_prob * advantage -0.1 * entropy
            policy_network.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(policy_network.parameters(), 0.5)
            policy_optim.step()

            episode_reward += reward
            episode_loss += actor_loss
            state = next_state
            I *= config["GAMMA"]
        logger.step(episode_reward, 0, episode, episode_loss)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    a2c(config)