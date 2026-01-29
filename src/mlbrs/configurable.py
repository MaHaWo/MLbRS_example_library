from abc import ABC, abstractmethod

class Configurable(ABC):
    """Abstract base class for configurable components."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "Configurable":
        pass