import torch
import torch.nn.functional as F
from torch import nn as nn
import random

class ActorNetwork(nn.Module):
    def __init__(self, env):
        super(ActorNetwork, self).__init__()
        self.layer1 = nn.Linear(env.observation_space.shape[0], 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, env.action_space.n)


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = F.softmax(self.layer3(x), dim=-1)
        return out

class CriticNetwork(nn.Module):
    def __init__(self, env):
        super(CriticNetwork, self).__init__()
        self.layer1 = nn.Linear(env.observation_space.shape[0], 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class SACActorNetwork(nn.Module):
    def __init__(self, env):
        super(SACActorNetwork, self).__init__()
        self.input_layer = nn.Linear(env.observation_space.shape[0], 128)
        self.hidden_layer = nn.Linear(128, 128)
        self.mu_layer = nn.Linear(128, env.action_space.n)
        self.log_std_layer = nn.Linear(128, env.action_space.n)

    def forward(self, state):
        # Forward pass through the network
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hidden_layer(x))
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = F.softplus(log_std) + 1e-3


        return mu, log_std