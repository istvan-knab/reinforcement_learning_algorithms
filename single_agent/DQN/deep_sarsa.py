import random
import copy
import yaml
import gymnasium as gym



from single_agent.DQN.seed import seed_all
from evaluation.log_values import Logger
from single_agent.DQN.env_wrapper import EnvWrapper
from single_agent.DQN.neural_network import NeuralNetwork

def deep_sarsa_training(config: dict) -> None:

    env = gym.make(config["ENVIRONMENT"])
    env = EnvWrapper(env)
    logger = Logger("Deep SARSA", "MountainCar-v0", config["ALPHA"], config["GAMMA"])
    seed_all(config["SEED"], env)
    q_network = NeuralNetwork(env)
    target_network = copy.deepcopy(q_network)
    target_network = target_network.eval()



def deep_sarsa_test(config: dict) -> None:
    env = gym.make(config["ENVIRONMENT"])


if __name__ == "__main__":

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    deep_sarsa_training(config)
    deep_sarsa_test(config)

