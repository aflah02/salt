import torch
from torch import Tensor, nn


class BetaNLLLoss(nn.Module):
    """Beta-NLL loss from Seitzer et al. (2022).
    Modifies the standard Gaussian NLL weighting by sigma^(2*beta),
    reducing the dominance of low uncertainty predictions in the gradient.

    Parameters
    ----------
    beta : float
        Value for the exponent of sigma^2, between 0 and 1.
        beta=0 corresponds to standard Gaussian NLL loss.
    reduction : str
        Reduction method, one of 'none', 'mean', 'sum'.
    """

    def __init__(self, beta: float = 0.0, reduction: str = "none"):
        super().__init__()
        assert 0.0 <= beta <= 1.0, "beta must be in [0, 1]"
        assert reduction in {"none", "mean", "sum"}
        self.beta = beta
        self.clamp = 1e-6
        self.reduction = reduction

    def forward(self, means: Tensor, targets: Tensor, var: Tensor) -> Tensor:
        """Compute the Beta-NLL loss.

        Parameters
        ----------
        means : Tensor
            Predicted means, shape [B, R].
        targets : Tensor
            Target values, shape [B, R].
        var : Tensor
            Predicted variances, shape [B, R].

        Returns
        -------
        Tensor
            Loss tensor, shape [B, R] if reduction='none', scalar otherwise.

        Raises
        ------
        ValueError
            If the loss contains NaN or Inf values before reduction.
        """
        # Clamp the variance without torch.no_grad()
        var = var.clamp(min=self.clamp)
        # Standard Gaussian NLL
        loss = 0.5 * (torch.log(var) + (targets - means) ** 2 / var)

        # Weighting by sigma^(2*beta) = var^beta
        if self.beta > 0:
            loss = loss * (var.detach() ** self.beta)

        # Check for NaN or Inf values in the loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise ValueError(f"Regression loss is NaN or Inf before reduction: {loss}")

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
