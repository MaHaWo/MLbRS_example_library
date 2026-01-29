import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def evaluate_f1_score(
    model: torch.nn.Module,
    torchdataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    device: str = "cpu",
) -> float:
    """Compute F1 score on a torchdataset.

    Args:
        model: The trained model to evaluate.
        torchdataset: The test torchdataset to evaluate on.
        batch_size: Batch size for evaluation.
        device: Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        F1 score (macro-averaged).
    """
    model.eval()
    model.to(device)

    dataloader = DataLoader(torchdataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return f1_score(all_labels, all_preds, average="macro")
