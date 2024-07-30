import gymnasium as gym
import time
if __name__ == "__main__":
    env = gym.make("HalfCheetah-v4", render_mode = "human")
    state = env.reset()
    for episode in range(5):
        env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            time.sleep(0.2)
