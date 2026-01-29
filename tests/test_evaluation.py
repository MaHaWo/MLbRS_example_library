import torch
import torch.nn as nn
import numpy as np
from mlbrs.evaluation import evaluate_f1_score


class SimpleModel(nn.Module):
    """Simple toy model for testing."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        results = torch.tensor([[0.1, 0.8], [0.2, 0.7]], dtype=torch.float32)
        return results

class ToyDataset(torch.utils.data.Dataset):
    """Toy dataset that returns perfect predictions."""
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.labels = torch.tensor([i % 2 for i in range(num_samples)])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(1, 28, 28)
        return image, self.labels[idx]


def test_evaluate_f1_score():
    """Test that evaluate_f1_score computes F1 score."""
    model = SimpleModel()
    torchdataset = ToyDataset(num_samples=4)

    # Model should produce reasonable output
    f1 = evaluate_f1_score(model, torchdataset, batch_size=2)

    # F1 should be between 0 and 1
    assert np.isclose(f1, 0.33, atol= 0.01)

