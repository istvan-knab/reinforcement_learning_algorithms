import yaml
import mujoco
import gymnasium as gym

from single_agent.DQN.experience_replay import ReplayMemory
def soft_actor_critic(config):
    env = gym.make(config['ENVIRONMENT'])
    memory = ReplayMemory(config["BUFFER"], config["BATCH"])
if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    soft_actor_critic(config)