import torch
from pathlib import Path
from torchvision import transforms
from mlbrs.dataset import TorchDataset
import numpy as np

class mock_target_dataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        self.data = [(np.random.randn(1, 28, 28), i) for i in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_dataset_init():
    """Test that initialization stores parameters correctly."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        train=False,
        download=True,
    )
    assert torchdataset.root == Path("/tmp/data")
    assert torchdataset.train is False
    assert torchdataset.download is True


def test_dataset_len():
    """Test __len__ returns correct length."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
    )
    assert len(torchdataset) == 10


def test_dataset_getitem_returns_data():
    """Test __getitem__ returns image and label tuple."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
    )
    image, label = torchdataset[0]
    assert isinstance(image, np.ndarray)
    assert isinstance(label, int)


def test_dataset_getitem_without_transform():
    """Test __getitem__ without transform returns raw data."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        transform=None,
    )
    image, label = torchdataset[0]
    assert image.shape == (1, 28, 28)
    assert label == 0


def test_dataset_transform_applied_when_present():
    """Test that transform is applied to data."""

    transform_list = [transforms.ToTensor()]
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        transform=transform_list,
    )

    image, label = torchdataset[0]
    assert isinstance(image, torch.Tensor)
    assert label == 0


def test_dataset_from_config_all_parameters():
    """Test from_config with all parameters."""
    transform_list = [transforms.ToTensor()]
    config = {
        "root": "/tmp/data",
        "target_dataset": mock_target_dataset,
        "train": False,
        "download": True,
        "transform": transform_list,
    }
    torchdataset = TorchDataset.from_config(config)
    assert torchdataset.train is False
    assert torchdataset.download is True
    assert torchdataset.transform is not None
