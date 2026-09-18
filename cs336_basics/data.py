import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random language-modeling windows and their one-token targets."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must contain more tokens than context_length")

    starts = np.random.randint(0, num_starts, size=batch_size)
    offsets = np.arange(context_length)
    input_ids = dataset[starts[:, None] + offsets]
    target_ids = dataset[starts[:, None] + offsets + 1]

    return (
        torch.as_tensor(input_ids, dtype=torch.long, device=device),
        torch.as_tensor(target_ids, dtype=torch.long, device=device),
    )
