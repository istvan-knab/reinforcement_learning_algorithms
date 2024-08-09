import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the policy network (actor)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, action_dim)
        self.fc3_log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        log_std = torch.clamp(self.fc3_log_std(x), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std


# Define the Q-network (critic)
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the value network (target Q network)
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def update_critic(state, action, reward, next_state, done, gamma, alpha):
    with torch.no_grad():
        next_action, next_log_prob = actor(next_state)
        target_q_value = reward + (1 - done) * gamma * (torch.min(critic1(next_state, next_action), critic2(next_state,
                                                                                                            next_action)) - alpha * next_log_prob)

    q_value1 = critic1(state, action)
    q_value2 = critic2(state, action)

    critic1_loss = nn.MSELoss()(q_value1, target_q_value)
    critic2_loss = nn.MSELoss()(q_value2, target_q_value)

    critic1_optim.zero_grad()
    critic1_loss.backward()
    critic1_optim.step()

    critic2_optim.zero_grad()
    critic2_loss.backward()
    critic2_optim.step()


def update_value(state, alpha):
    action, log_prob = actor(state)
    q_value1 = critic1(state, action)
    q_value2 = critic2(state, action)
    value_loss = nn.MSELoss()(value(state), torch.min(q_value1, q_value2) - alpha * log_prob)

    value_optim.zero_grad()
    value_loss.backward()
    value_optim.step()


def update_actor(state, alpha):
    action, log_prob = actor(state)
    q_value1 = critic1(state, action)
    q_value2 = critic2(state, action)
    actor_loss = (alpha * log_prob - torch.min(q_value1, q_value2)).mean()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()
state_dim = 4  # Example state dimension
action_dim = 2  # Example action dimension
hidden_dim = 256

actor = Actor(state_dim, action_dim, hidden_dim)
critic1 = QNetwork(state_dim, action_dim, hidden_dim)
critic2 = QNetwork(state_dim, action_dim, hidden_dim)
value = ValueNetwork(state_dim, hidden_dim)

actor_optim = optim.Adam(actor.parameters(), lr=3e-4)
critic1_optim = optim.Adam(critic1.parameters(), lr=3e-4)
critic2_optim = optim.Adam(critic2.parameters(), lr=3e-4)
value_optim = optim.Adam(value.parameters(), lr=3e-4)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action, _ = actor(torch.FloatTensor(state).unsqueeze(0))
        next_state, reward, done, _ = env.step(action.numpy())

        # Convert rewards and states to PyTorch tensors
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action)
        reward_tensor = torch.FloatTensor([reward])
        next_state_tensor = torch.FloatTensor(next_state)
        done_tensor = torch.FloatTensor([done])

        update_critic(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor, gamma, alpha)
        update_value(state_tensor, alpha)
        update_actor(state_tensor, alpha)

        state = next_state
