import torch
from pathlib import Path
from torchvision import transforms
from mlbrs.dataset import TorchDataset, HFDataset
import numpy as np
from PIL import Image

class mock_target_dataset(torch.utils.data.Dataset  ):
    def __init__(self, *args, **kwargs):
        self.data = [(np.random.randn(1, 28, 28), i) for i in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class mock_hf_dataset_split:
    """Mock HF dataset split with select method."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def select(self, indices):
        """Select subset of data like HF datasets."""
        selected_data = [self.data[i] for i in indices]
        return mock_hf_dataset_split(selected_data)


class mock_hf_dataset:
    """Mock Hugging Face DatasetDict for testing."""
    def __init__(self):
        all_data = [
            {"image": Image.new("L", (28, 28)), "label": i} for i in range(10)
        ]
        self.splits = {
            "train": mock_hf_dataset_split(all_data),
            "test": mock_hf_dataset_split(all_data[8:])
        }

    def __getitem__(self, split_name):
        return self.splits[split_name]

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


# HFDataset Tests


def test_hf_dataset_init():
    """Test that HFDataset initialization stores parameters correctly."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds, transform=None)
    assert len(hf_dataset.hf_dataset) == 10
    assert hf_dataset.transform is None


def test_hf_dataset_len():
    """Test __len__ returns correct length for HFDataset."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds)
    assert len(hf_dataset) == 10


def test_hf_dataset_getitem_without_transform():
    """Test __getitem__ without transform returns raw data."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds, transform=None)
    image, label = hf_dataset[0]
    assert isinstance(image, Image.Image)
    assert label == 0


def test_hf_dataset_getitem_with_transform():
    """Test that transform is applied to HFDataset data."""
    mock_ds = mock_hf_dataset()
    transform = transforms.ToTensor()
    hf_dataset = HFDataset(dataset=mock_ds, transform=transform)
    
    image, label = hf_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert label == 0
    assert image.shape == (1, 28, 28)


def test_hf_dataset_getitem_returns_correct_index():
    """Test __getitem__ returns correct sample for given index."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds)
    
    for i in range(5):
        _, label = hf_dataset[i]
        assert label == i


def test_hf_dataset_from_config():
    """Test HFDataset.from_config with dataset object."""
    mock_ds = mock_hf_dataset()
    config = {
        "dataset": mock_ds,
        "transform": transforms.ToTensor(),
        "root": "/tmp",
        "train": True,
    }
    hf_dataset = HFDataset.from_config(config)
    assert len(hf_dataset) == 10
    assert hf_dataset.transform is not None


def test_hf_dataset_with_size_limit():
    """Test HFDataset with size parameter limits dataset."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds, size=5, root="/tmp")
    assert len(hf_dataset) == 5


def test_hf_dataset_with_size_larger_than_dataset():
    """Test HFDataset with size larger than actual dataset returns full dataset."""
    mock_ds = mock_hf_dataset()
    hf_dataset = HFDataset(dataset=mock_ds, size=20, root="/tmp")
    assert len(hf_dataset) == 10

