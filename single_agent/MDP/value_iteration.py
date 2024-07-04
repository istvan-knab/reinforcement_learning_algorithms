import numpy as np
import time

from environments.maze import Maze

def policy(state, policy) -> np.ndarray:
    return policy[state]

def value_iteration(policy, state_values, theta=1e-3, gamma=0.99):
    delta = float("inf")
    while delta > theta:
        print(delta)
        delta = 0
        for row in range(5):
            for column in range(5):
                v = state_values[(row, column)]
                action_probabilities = None
                max_q_sa = float("-inf")
                
                for action in range(4):
                    next_state, reward, _, _ = env.simulate_step((row, column), action)
                    q_sa = reward + gamma * state_values[next_state]

                    if q_sa > max_q_sa:
                        max_q_sa = q_sa
                        action_probabilities = np.zeros(4)
                        action_probabilities[action] = 1.0

                    state_values[(row, column)] = max_q_sa
                    policy[(row, column)] = action_probabilities

                    delta = max(delta, abs(max_q_sa - v))

    print(state_values)
if __name__ == "__main__":
    env = Maze()
    state = env.reset()
    done = False
    print(f"The agent starts from state: {state}")
    trajectory = []
    G_t = 0
    gamma = 0.9

    #5x5 states, 4 actions every transition same probability
    policy_probabilities = np.full((5, 5, 4), 0.25, dtype=float)
    state_values = np.zeros((5, 5), dtype=float)

    value_iteration(policy_probabilities, state_values)

    while not done:
        action_probabilities = policy(state, policy_probabilities)
        action = np.random.choice(range(4), 1, p=action_probabilities)
        next_state, reward, done, info = env.step(action)
        env.render(mode='human')
        G_t = reward + gamma * G_t
        trajectory.append([state, action, reward, done, next_state])
        state = next_state
        print(state_values)
        print(f'Actual return: {G_t}')
        time.sleep(0.2)
    env.close()
