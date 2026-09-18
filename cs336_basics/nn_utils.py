from collections.abc import Iterable

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply a numerically stable softmax along ``dim``."""
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute mean cross-entropy from unnormalized logits."""
    log_normalizer = torch.logsumexp(inputs, dim=-1)
    correct_logit = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - correct_logit).mean()


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    """Clip all available gradients by their combined L2 norm in-place."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return

    with torch.no_grad():
        gradient_norms = torch.stack(
            [torch.linalg.vector_norm(parameter.grad.detach()) for parameter in parameters]
        )
        total_norm = torch.linalg.vector_norm(gradient_norms)
        clip_coefficient = max_l2_norm / (total_norm + 1e-6)
        clip_coefficient = torch.clamp(clip_coefficient, max=1.0)
        for parameter in parameters:
            parameter.grad.mul_(clip_coefficient)
