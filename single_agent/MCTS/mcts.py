import gymnasium as gym
class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}

class MCTS:
    def __init__(self):
        pass

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1')