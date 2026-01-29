import torch
import torchvision

from .configurable import Configurable

from pathlib import Path

from typing import Any, Callable
from torchvision import transforms


class Dataset(Configurable, torch.utils.data.Dataset):
    """Custom wrapper dataset that composes a target PyTorch dataset with sequential transforms.

    Inherits from both Configurable and torch.utils.data.Dataset to provide
    a flexible, configurable dataset that wraps existing PyTorch datasets
    (e.g., MNIST, CIFAR10) and applies a chain of transforms.
    """

    def __init__(
        self,
        root: str | Path,
        target_dataset: type[torch.utils.data.Dataset] | str,
        train: bool = True,
        download: bool = False,
        transform: list[Callable] | list[str] | None = None,
        size= None,
    ):
        """Initialize the Dataset.

        Args:
            root (str | Path): Root directory where the dataset is stored or will be downloaded to.
            target_dataset (type[torch.utils.data.Dataset]): The PyTorch dataset class to use (e.g., torchvision.datasets.MNIST).
            train (bool, optional): If True, load the training split; else load the test split. Defaults to True.
            download (bool, optional): If True, download the dataset if not found at root. Defaults to False.
            transform (list[Callable] | None, optional): List of transforms to apply sequentially to each sample. Defaults to None.
            size (int | None, optional): If specified, limits the dataset to the first 'size' samples. Defaults to None.
        """
        self.root = Path(root)
        self.train = train
        self.target_dataset = target_dataset
        self.download = download

        if transform and all(isinstance(t, str) for t in transform):
            transform = [
                getattr(transforms, t)()
                for t in transform  # type: ignore
            ]

        self.transform = transforms.Compose(transform) if transform else None

        if isinstance(target_dataset, str):
            target_dataset = getattr(torchvision.datasets, target_dataset)

        self._data = target_dataset(
            root=self.root,
            train=self.train,
            download=self.download,
            transform=None,
        )

        if size is not None:
            # randomly select subset of data
            self._data = torch.utils.data.Subset(self._data,
                                                 torch.randperm(len(self._data))[:size])

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the underlying dataset.
        """
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieve and process a single sample from the dataset.

        Fetches the sample from the underlying dataset and applies all
        transforms in the order they were specified.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Tuple of (transformed_image, label).
        """
        image, label = self._data[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Dataset":
        """Create a Dataset instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the Dataset.

        Returns:
            Dataset: An instance of the Dataset class created from the configuration.
        """
        return cls(
            root=config["root"],
            target_dataset=config["target_dataset"],
            train=config.get("train", True),
            download=config.get("download", False),
            transform=config.get("transform", None),
            size = config.get("size", None),
        )
