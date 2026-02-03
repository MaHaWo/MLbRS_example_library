import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from typing import Callable, Tuple
import numpy as np


def evaluate_f1_score(
    model: torch.nn.Module,
    torchdataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    lossfunc: Callable = torch.nn.CrossEntropyLoss(),
    device: str = "cpu",
) -> Tuple:
    """Compute F1 score on a torchdataset and return f1 and loss statistics.

    Args:
        model: The trained model to evaluate.
        torchdataset: The test torchdataset to evaluate on.
        batch_size: Batch size for evaluation.
        lossfunc: Loss function to compute loss statistics.
        device: Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        F1 score (macro-averaged).
    """
    model.eval()
    model.to(device)

    dataloader = DataLoader(torchdataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []
    all_loss = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            loss = lossfunc(outputs, labels.to(device))
            all_loss.append(loss.item())

    mean_loss = np.mean(all_loss)
    min_loss = np.min(all_loss)
    max_loss = np.max(all_loss)
    q25_loss = np.percentile(all_loss, 25)
    q75_loss = np.percentile(all_loss, 75)
    return (
        mean_loss,
        min_loss,
        max_loss,
        q25_loss,
        q75_loss,
        f1_score(all_labels, all_preds, average="macro"),
    )
