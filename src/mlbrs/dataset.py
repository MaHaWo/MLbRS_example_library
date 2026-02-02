import torch
import torchvision

from .configurable import Configurable

from pathlib import Path
import random

from typing import Any, Callable
from datasets import load_from_disk


class BaseDataset(Configurable, torch.utils.data.Dataset):

    def __init__(self, path: str | Path, transform: list[Callable | str] | None = None, size: int | None = None, shuffle: bool = False):
        """Base dataset class.

        Args:
            path (str | Path): Path to dataset.
            transform (list[Callable] | None, optional): List of transform callables to apply. Defaults to None.
        """
        super().__init__()
        self.path = Path(path)
        self.transform = self._build_transforms(transform=transform)
        self.size = size
        self.shuffle = shuffle


    def _build_transforms(self, transform: list[Callable | str] | None) -> Callable | None:
        """Compose a list of transforms into a single transform callable.

        Args:
            transform (list[Callable] | None): List of transform callables to compose.

        Returns:
            Callable | None: Composed transform callable or None if no transforms provided.
        """
        transformlist = []
        if transform:
            for t in transform:
                if isinstance(t, Callable):
                    transformlist.append(t)
                elif isinstance(t, str):
                    transformlist.append(getattr(torchvision.transforms, t)())
                else: 
                    raise ValueError(f"Transform {t} is neither a callable nor a valid torchvision transform name.")

        return torchvision.transforms.Compose(transformlist) if transformlist else None

class TorchDataset(BaseDataset):
    """Custom wrapper torchdataset that composes a target PyTorch torchdataset with sequential transforms.

    Inherits from both Configurable and torch.utils.data.Dataset to provide
    a flexible, configurable torchdataset that wraps existing PyTorch datasets
    (e.g., MNIST, CIFAR10) and applies a chain of transforms.
    """

    def __init__(
        self,
        root: str | Path,
        target_dataset: type[torch.utils.data.Dataset] | str,
        train: bool = True,
        download: bool = False,
        transform: list[Callable | str] | None = None,
        size: int | None = None,
        shuffle: bool = False,
    ):
        """Initialize the TorchDataset.

        Args:
            root (str | Path): Root directory where the torchdataset is stored or will be downloaded to.
            target_dataset (type[torch.utils.data.Dataset]): The PyTorch torchdataset class to use (e.g., torchvision.datasets.MNIST).
            train (bool, optional): If True, load the training split; else load the test split. Defaults to True.
            download (bool, optional): If True, download the torchdataset if not found at root. Defaults to False.
            transform (list[Callable] | None, optional): List of transforms to apply sequentially to each sample. Defaults to None.
            size (int | None, optional): If specified, limits the torchdataset to the first 'size' samples. Defaults to None.
            shuffle (bool, optional): If True, shuffle the dataset samples. Defaults to False.
        """
        super().__init__(path=root, transform=transform, size=size, shuffle = shuffle)
        self.train = train
        self.target_dataset = target_dataset
        self.download = download
        if isinstance(target_dataset, str):
            target_dataset = getattr(torchvision.datasets, target_dataset)

        self._data = target_dataset(
            root=self.path,
            train=self.train,
            download=self.download)

        if size is not None:
            # randomly select subset of data
            self._data = torch.utils.data.Subset(self._data, range(self.size))
        
        self.indices = list(range(len(self._data)))
        if shuffle:
            # shuffle the dataset
            random.shuffle(self.indices)
            self._data = torch.utils.data.Subset(self._data, self.indices)

    def __len__(self) -> int:
        """Return the total number of samples in the torchdataset.

        Returns:
            int: Number of samples in the underlying torchdataset.
        """
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Retrieve and process a single sample from the torchdataset.

        Fetches the sample from the underlying torchdataset and applies all
        transforms in the order they were specified.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int]: Tuple of (transformed_image, label).
        """
        image, label = self._data[self.indices[idx]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TorchDataset":
        """Create a TorchDataset instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the TorchDataset.

        Returns:
            TorchDataset: An instance of the TorchDataset class created from the configuration.
        """
        return cls(
            root=config["root"],
            target_dataset=config["target_dataset"],
            train=config.get("train", True),
            download=config.get("download", False),
            transform=config.get("transform", None),
            size = config.get("size", None),
        )


class Dataset(BaseDataset):
    """dataset wrapper reading .pt files from disk."""

    def __init__(
        self,
        path: str | Path | None = None,
        transform: list[Callable | str] | None = None,
        size: int | None = None,
        shuffle: bool = False,
    ):
        """Initialize a basic dataset that reads .pt files from disk.

        Args:
            path (str | Path | None, optional): Path to directory containing .pt files. Defaults to None.
            transform (Callable | list[Callable] | None, optional): Transform to apply to each sample. Defaults to None.
            size (int | None, optional): If specified, limits the dataset to the first 'size' samples. Defaults to None.
            shuffle (bool, optional): If True, shuffle the dataset samples. Defaults to False.
        """

        super().__init__(path=path, transform=transform, size=size, shuffle = shuffle)
        if isinstance(path, (str, Path)):
            self.file_paths = sorted(Path(self.path).resolve().absolute().glob("*.pt"))
            self.num_samples = len(self.file_paths)

        if size is not None:
            self.num_samples = min(size, self.num_samples)

        self.indices = list(range(self.num_samples))
        if shuffle:
            random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the underlying Huggingface dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Retrieve and process a single sample from the dataset.

        Fetches the sample from the underlying Huggingface dataset and applies
        the specified transform if provided.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Any, Any]: Tuple of (transformed_sample, label).
        """
        sample = torch.load(self.file_paths[self.indices[idx]], weights_only=False)
        data, label = sample["image"], sample["label"]
        if self.transform is not None: 
            data = self.transform(data)
        return data, label

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Dataset":
        """Create a Dataset instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary containing parameters for the Dataset.
                Must include 'hf_dataset' key with a path or dataset object.
                Optionally includes 'transform' key.
        Returns:
            Dataset: An instance of the Dataset class created from the configuration.
        """

        return cls(
            path = config["path"],
            transform=config["transform"],
            shuffle=config.get("shuffle", True),
            size = config.get("size", None),
        )