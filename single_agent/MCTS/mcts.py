import gymnasium as gym
class Node:
    def __init__(self, id, parent):
        self.parent = parent
        self.value = 0
        self.id = id

class Tree:
    def __init__(self):
        self.root = None


class MCTS:
    def __init__(self):
        pass

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')