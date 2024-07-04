import time
from typing import Tuple, Dict, Optional, Iterable

from environments.maze import Maze


if __name__ == "__main__":
     env = Maze()
     state = env.reset()
     print(f"The agent starts from state: {state}")


     frame = env.render(mode="human")





