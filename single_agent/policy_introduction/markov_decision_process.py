from environments.maze import Maze
import time

if __name__ == "__main__":
    env = Maze('human')
    state = env.reset()
    done = False
    print(f"The agent starts from state: {state}")
    trajectory = []
    G_t = 0
    gamma = 0.9

    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            done = True
        env.render()
        G_t = reward + gamma * G_t
        trajectory.append([state, action, reward, done, next_state])
        state = next_state
        print(f'Actual return: {G_t}')
        time.sleep(0.1)
    env.close()
