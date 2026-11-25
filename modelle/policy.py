import torch
from torch import nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """Lernbare Strategie: vier Beobachtungswerte → zwei Aktionswahrscheinlichkeiten."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )

    def forward(self, observation):
        """Erstellt eine Verteilung, aus der links oder rechts gezogen werden kann."""
        observation = torch.as_tensor(observation, dtype=torch.float32)
        return Categorical(logits=self.network(observation))
