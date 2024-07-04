import time
from typing import Tuple, Dict, Optional, Iterable

from environments.maze import Maze
from matplotlib import pyplot as plt


if __name__ == "__main__":
     env = Maze()
     state = env.reset()
     done = False
     print(f"The agent starts from state: {state}")
     trajectory = []

     while not done:
          action = env.action_space.sample()
          next_state, reward, done, info = env.step(action)
          env.render(mode='human')
          trajectory.append([state, action, reward, done, next_state])
          state = next_state
          time.sleep(0.2)
     env.close()








