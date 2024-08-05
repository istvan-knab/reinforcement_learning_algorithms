from environments.maze import Maze
import numpy as np
import time

def policy(state, policy) -> np.ndarray:
    return policy[state]

def policy_evaluation(sec_env, policy_probabilities, state_values):
    threshold = 1e-4
    delta = 10000
    action_probabilities = policy((0, 0), policy_probabilities)
    while delta > threshold:
        delta = 0
        for row in range(5):
            for column in range(5):
                v = state_values[(row, column)]
                new_value = 0.
                action_probabilities = policy_probabilities[(row, column)]
                for action in range(len(action_probabilities)):
                    next_state, reward, terminated, truncated, _ = sec_env.step(action)
                    new_value += action_probabilities[action] * (reward + gamma * state_values[next_state])
                    print(v)
                delta = max(delta, abs(new_value - v))
                print(delta)
                state_values[(row, column)] = new_value

def policy_improvement(secondary_env, policy_probabilities, state_values):
    policy_stable = True
    for row in range(5):
        for column in range(5):
            old_action = policy_probabilities[(row, column)]. argmax()
            new_action = None
            max_qsa = float("-inf")
            for action in range(4):
                next_state, reward, terminated, truncated,_ = secondary_env.step(action)
                qsa = reward + gamma * state_values[next_state]
                if qsa > max_qsa:
                    max_qsa = qsa
                    new_action = action
            action_probabilities = np.zeros(4)
            action_probabilities[old_action] = 1.
            policy_probabilities[(row, column)] = action_probabilities
            if new_action != old_action:
                policy_stable = False

    return policy_stable
def policy_iteration(policy_probabilities, state_values):
    policy_stable = False
    sec_env = Maze('rgb_array')
    while not policy_stable:
        policy_evaluation(sec_env, policy_probabilities, state_values)
        policy_stable = policy_improvement(sec_env, policy_probabilities, state_values)



if __name__ == "__main__":
    env = Maze('human')
    state = env.reset()
    done = False
    trajectory = []
    G_t = 0
    gamma = 0.9
    action = 0
    policy_probabilities = np.full((5, 5, 4), 0.25, dtype=float)
    state_values = np.zeros((5, 5), dtype=float)
    action_probabilities = policy((0, 0), policy_probabilities)

    policy_iteration(policy_probabilities, state_values)

    while not done:
        action_probabilities = policy(state, policy_probabilities)
        action = np.random.choice(range(4), 1, p=action_probabilities)
        state,_,_, _, _  = env.step(action)
        frame = env.render()
        time.sleep(0.2)
    env.close()