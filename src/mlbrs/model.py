import torch
import torch.nn.functional as F
from typing import Any
from abc import ABC, abstractmethod
from .configurable import Configurable


class BaseModel(Configurable, torch.nn.Module):
    """Abstract base class for machine learning models."""

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Model(BaseModel):
    """A simple feedforward neural network model."""

    def __init__(
        self,
        conv1_args: list[int],
        conv2_args: list[int],
        fc1_args: list[int],
        fc2_args: list[int],
        dropout_rate1: float = 0.25,
        dropout_rate2: float = 0.5,
        conv1_kwargs: dict[str, Any] | None = None,
        conv2_kwargs: dict[str, Any] | None = None,
        fc1_kwargs: dict[str, Any] | None = None,
        fc2_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(*conv1_args, **(conv1_kwargs or {}))
        self.conv2 = torch.nn.Conv2d(*conv2_args, **(conv2_kwargs or {}))
        self.dropout1 = torch.nn.Dropout(dropout_rate1)
        self.dropout2 = torch.nn.Dropout(dropout_rate2)
        self.fc1 = torch.nn.Linear(*fc1_args, **(fc1_kwargs or {}))
        self.fc2 = torch.nn.Linear(*fc2_args, **(fc2_kwargs or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    @classmethod
    def from_config(cls, config: dict) -> "Model":
        return cls(
            conv1_args=config["conv1_args"],
            conv2_args=config["conv2_args"],
            fc1_args=config["fc1_args"],
            fc2_args=config["fc2_args"],
            dropout_rate1=config.get("dropout_rate1", 0.25),
            dropout_rate2=config.get("dropout_rate2", 0.5),
        )
