"""Policy für den vollständigen Zustand des CartPole mit zwei Stäben."""

import math

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def beobachtung(zustand):
    x, a, b, vx, va, vb = zustand
    a = (a + math.pi) % (2 * math.pi) - math.pi
    b = (b + math.pi) % (2 * math.pi) - math.pi
    return np.array([x, vx, a, va, b, vb], dtype=np.float32)


class GelenkPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(6, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, observation):
        return Categorical(logits=self.network(torch.as_tensor(observation, dtype=torch.float32)))
