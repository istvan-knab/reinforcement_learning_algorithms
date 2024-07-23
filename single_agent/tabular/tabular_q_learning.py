import numpy as np
import time
from environments.maze import Maze
from evaluation.log_values import Logger

def target_policy(state, Q):
    Q_values = Q[state]
    best_action = np.random.choice(np.flatnonzero(Q_values == Q_values.max()))
    return best_action

def exploratory_policy():
    return np.random.randint(4)

def tabular_q_learning(env, EPISODES = 400, alpha = 0.1, gamma = 0.99, epsilon = 1):
    Q = np.zeros((5, 5, 4))
    logger = Logger("Tabular Q Learning Simple", "Maze", alpha, gamma)
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = exploratory_policy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                done = True
            next_action = target_policy(next_state, Q)

            Qsa = Q[state][action]
            next_Qsa = Q[next_state][next_action]
            Q[state][action] = Qsa + alpha * (reward + gamma * next_Qsa - Qsa)

            episode_reward += reward
            state = next_state
        logger.step(episode_reward, epsilon, episode)

    state = env.reset()
    done = False
    while not done:
        action = target_policy(state, Q)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
        env.render()
        time.sleep(0.4)
        state = next_state

def egreedy_q_learning(env, EPISODES = 400, alpha = 0.1, gamma = 0.99, epsilon = 1, epsilon_decay = 0.995):
    Q = np.zeros((5, 5, 4))
    logger = Logger("Tabular Q Learning Epsilon greedy", "Maze", alpha, gamma)
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        epsilon *= epsilon_decay
        while not done:
            if np.random.random() < epsilon:
                action = exploratory_policy()
            else:
                action = target_policy(state, Q)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated :
                done = True
            next_action = target_policy(next_state, Q)

            Qsa = Q[state][action]
            next_Qsa = Q[next_state][next_action]
            Q[state][action] = Qsa + alpha * (reward + gamma * next_Qsa - Qsa)

            episode_reward += reward
            state = next_state
        logger.step(episode_reward, epsilon, episode)

    state = env.reset()
    done = False
    while not done:
        action = target_policy(state, Q)
        next_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
        env.render()
        time.sleep(0.4)
        state = next_state

if __name__=="__main__":

    env = Maze('human')
    tabular_q_learning(env)