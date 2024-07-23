import numpy as np
import time
import gymnasium as gym

from environments.maze import Maze
from single_agent.policy_introduction.policy_visualization import PolicyVisualization

def policy(state, policy) -> np.ndarray:
    return policy[state]

def value_iteration(env, policy, state_values, theta=1e-6, gamma=0.99):
    delta = float('inf')
    secondary_env = Maze()
    secondary_env.reset()

    while delta > theta:
        delta = 0
        for row in range(5):
            for col in range(5):
                old_value = state_values[(row, col)]
                action_probs = None
                max_qsa = float('-inf')

                for action in range(4):
                    secondary_env.state = (row, col)
                    next_state, reward, _, _ = secondary_env.step(action)
                    qsa = reward + gamma * state_values[next_state]
                    if qsa > max_qsa:
                        max_qsa = qsa
                        action_probs = np.zeros(4)
                        action_probs[action] = 1.

                state_values[(row, col)] = max_qsa
                policy[(row, col)] = action_probs

                delta = max(delta, abs(max_qsa - old_value))

    print(state_values)
if __name__ == "__main__":
    env = Maze()
    visualization = PolicyVisualization()
    state = env.reset()
    done = False
    print(f"The agent starts from state: {state}")
    trajectory = []
    G_t = 0
    gamma = 0.9

    #5x5 states, 4 actions every transition same probability
    policy_probabilities = np.full((5, 5, 4), 0.25, dtype=float)
    state_values = np.zeros((5, 5), dtype=float)
    action_probabilities = policy((0,0), policy_probabilities)
    value_iteration(env, policy_probabilities, state_values)

    while not done:
        action_probabilities = policy(state, policy_probabilities)
        action = np.random.choice(range(4), 1, p=action_probabilities)
        state,_,_,_ = env.step(action)
        frame = env.render(mode='human')
        time.sleep(0.2)
    env.close()
