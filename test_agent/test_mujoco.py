import gym
import mujoco_py as mujoco

env = gym.make("HalfCheetah-v2")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())