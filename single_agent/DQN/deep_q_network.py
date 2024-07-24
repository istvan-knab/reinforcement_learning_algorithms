from environments.maze import Maze
from evaluation.log_values import Logger

def deep_q_training():
    env = Maze("human")
    logger = Logger("Deep SARSA", "Maze")

def deep_q_test():
    env = Maze("human")


if __name__ == "__main__":
    deep_q_training()
    deep_q_test()
