import pytest
import torch
from pathlib import Path
from torchvision import transforms
from mlbrs.dataset import TorchDataset, Dataset
import numpy as np
from PIL import Image

class mock_target_dataset(torch.utils.data.Dataset  ):
    def __init__(self, *args, **kwargs):
        self.data = [(np.random.randn(1, 28, 28), i) for i in range(10)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

@pytest.fixture
def make_mock_data(tmp_path):
    for i in range(10):
        arr = Image.fromarray(np.random.randn(28, 28))
        label = i%3 
        torch.save({"image":arr, "label": label}, tmp_path / f"data_{i}.pt")
    return tmp_path

def test_torchdataset_init():
    """Test that initialization stores parameters correctly."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        train=False,
        download=True,
    )
    assert torchdataset.path == Path("/tmp/data")
    assert torchdataset.train is False
    assert torchdataset.download is True


def test_torchdataset_len():
    """Test __len__ returns correct length."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
    )
    assert len(torchdataset) == 10


def test_torchdataset_getitem_returns_data():
    """Test __getitem__ returns image and label tuple."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
    )
    image, label = torchdataset[0]
    assert isinstance(image, np.ndarray)
    assert isinstance(label, int)


def test_torchdataset_getitem_without_transform():
    """Test __getitem__ without transform returns raw data."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        transform=None,
    )
    image, label = torchdataset[0]
    assert image.shape == (1, 28, 28)
    assert label == 0

def test_torchdataset_getitem_with_shuffle():
    """Test __getitem__ without transform returns raw data."""
    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        transform=None,
        shuffle=True,
    )
    image, label = torchdataset[3]
    assert image.shape == (1, 28, 28)
    assert label != 3



def test_torchdataset_transform_applied_when_present():
    """Test that transform is applied to data."""

    torchdataset = TorchDataset(
        root="/tmp/data",
        target_dataset=mock_target_dataset,
        transform= [transforms.ToTensor(),],
    )
    
    image, label = torchdataset[0]
    assert isinstance(image, torch.Tensor)
    assert label == 0


def test_torchdataset_from_config_all_parameters():
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


# Dataset Tests


def test_dataset_init(make_mock_data):
    """Test that Dataset initialization stores parameters correctly."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=None)
    assert len(dataset) == 10
    assert dataset.transform is None

def test_dataset_init_with_size(make_mock_data):
    """Test that Dataset initialization stores parameters correctly."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=None, size = 4)
    assert len(dataset) == 4
    assert dataset.transform is None

def test_dataset_get(make_mock_data):
    """Test __getitem__ returns image and label tuple."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=None)
    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert image.size == (28, 28)


def test_dataset_get_with_transform(make_mock_data):
    """Test __getitem__ returns image and label tuple."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=[transforms.Resize((14,14)),])
    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert image.size == (14, 14)

def test_dataset_getitem_returns_correct_index(make_mock_data):
    """Test __getitem__ returns correct sample for given index."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=[transforms.Resize((14,14)),],shuffle = False)
    for i in range(5):
        _, label = dataset[i]
        assert label == i % 3

def test_shuffled_dataset_getitem_returns_randomized_index(make_mock_data):
    """Test __getitem__ returns correct sample for given index."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, transform=[transforms.Resize((14,14)),], shuffle = False)
    idxs = []
    for i in range(5):
        _, label = dataset[i]
        idxs.append(label)
    assert idxs != list(range(5))

def test_dataset_from_config(make_mock_data):
    """Test Dataset.from_config with dataset object."""
    tmp_dir = make_mock_data
    mock_ds = Dataset(path=tmp_dir)
    config = {
        "transform": ["ToTensor",],
        "path": tmp_dir,
        "shuffle": True,       
    }
    dataset = Dataset.from_config(config)
    assert len(dataset) == 10
    assert dataset.transform is not None


def test_dataset_with_size_larger_than_dataset(make_mock_data):
    """Test Dataset with size larger than actual dataset returns full dataset."""
    tmp_dir = make_mock_data
    dataset = Dataset(path=tmp_dir, size=20)
    assert len(dataset) == 10

