import torch
import torch.nn.functional as F
from torch import nn as nn
import random

class NeuralNetwork(nn.Module):
    def __init__(self, env):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(env.observation_space.shape[0], 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, env.action_space.n)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

