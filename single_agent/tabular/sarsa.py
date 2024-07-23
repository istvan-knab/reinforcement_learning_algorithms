import numpy as np
import time
from environments.maze import Maze
from evaluation.log_values import Logger

def egredy_policy(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        Q_values = Q[state]
        best_action = np.random.choice(np.flatnonzero(Q_values == Q_values.max()))
        return best_action


def sarsa(env, EPISODES = 400, alpha = 0.1, gamma = 0.99, epsilon = 1, epsilon_decay = 0.995):

    np.random.seed(0)
    Q = np.zeros((5, 5, 4))
    logger = Logger("SARSA", "Maze", alpha, gamma)
    for episode in range(EPISODES):
        state = env.reset()
        action = egredy_policy(state, Q, epsilon)
        done = False
        epsilon *= epsilon_decay
        episode_reward = 0
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
            next_action = egredy_policy(next_state, Q, epsilon)
            Qsa = Q[state][action]
            next_Qsa = Q[next_state][next_action]
            Q[state][action] = Qsa + alpha * (reward + gamma * next_Qsa - Qsa)
            state = next_state
            action = next_action
            episode_reward += reward

        logger.step(episode_reward, epsilon, episode)

    #TEST
    done = False
    state = env.reset()
    while not done:
        next_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
        next_action = egredy_policy(next_state, Q, epsilon)
        action = next_action
        env.render()
        time.sleep(0.3)

if __name__=="__main__":

    env = Maze("human")
    sarsa(env)