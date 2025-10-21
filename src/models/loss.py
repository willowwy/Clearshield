import argparse
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn


class HuberLoss(nn.Module):
    """Huber / Pseudo-Huber loss.

    Args:
        delta: Transition point δ; if None and auto_delta_p is not None, adapt via p-quantile of |e|.
        pseudo: True to use Pseudo-Huber; False for standard Huber.
        reduction: "mean" | "sum" | "none".
        apply_sigmoid: If True, apply sigmoid to predictions before residual (binary prob. regression).
        auto_delta_p: If set (e.g., 0.9), use that p-quantile of |e| as δ (prefer from validation residuals).
    Notes:
        - Standard Huber:
          L_δ(e) = 0.5 e^2, |e| <= δ;  δ(|e| - 0.5δ), otherwise
        - Pseudo-Huber:
          L_δ(e) = δ^2 ( sqrt(1 + (e/δ)^2) - 1 )
    """

    def __init__(
        self,
        delta: Optional[float] = 1.0,
        *,
        pseudo: bool = False,
        reduction: str = "mean",
        apply_sigmoid: bool = True,
        auto_delta_p: Optional[float] = None,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.delta = None if delta is None else float(delta)
        self.pseudo = bool(pseudo)
        self.reduction = reduction
        self.apply_sigmoid = bool(apply_sigmoid)
        self.auto_delta_p = None if auto_delta_p is None else float(auto_delta_p)

    @staticmethod
    def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
        q = float(q)
        k = max(1, int(round((q * (x.numel() - 1)))))
        # Use torch.kthvalue to approximate percentile/quantile
        values, _ = torch.kthvalue(x.view(-1), k)
        return values

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)
        e = preds - target
        abs_e = e.abs()

        if self.auto_delta_p is not None:
            # In-batch adaptive δ (recommended to compute on validation residuals)
            with torch.no_grad():
                delta_t = self._percentile(abs_e.detach(), self.auto_delta_p).clamp(min=1e-6)
            delta = delta_t
        else:
            if self.delta is None:
                raise ValueError("delta is None and auto_delta_p is None; one must be provided")
            delta = torch.as_tensor(self.delta, device=preds.device, dtype=preds.dtype)

        if self.pseudo:
            # δ^2 ( sqrt(1 + (e/δ)^2) - 1 )
            loss = (delta ** 2) * (torch.sqrt(1 + (e / delta) ** 2) - 1)
        else:
            # Standard Huber piecewise form
            quad = 0.5 * (e ** 2)
            lin = delta * (abs_e - 0.5 * delta)
            loss = torch.where(abs_e <= delta, quad, lin)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class QuantileLoss(nn.Module):
    """Quantile/Pinball loss with multi-quantile and anti-crossing regularization.

    Args:
        quantiles: List of quantiles, e.g., [0.1, 0.5, 0.9].
        reduction: "mean" | "sum" | "none".
        crossing_lambda: Anti-crossing regularization λ; if >0, add ReLU(ŷ^qi - ŷ^qj) for i<j.
        apply_sigmoid: If True, apply sigmoid to predictions (for probability regression).

    Shapes:
        - Single quantile: preds [B], target [B]
        - Multi-quantile: preds [B, Q] with len(quantiles)=Q
    """

    def __init__(
        self,
        quantiles: Sequence[float],
        *,
        reduction: str = "mean",
        crossing_lambda: float = 0.0,
        apply_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        if not quantiles:
            raise ValueError("quantiles must be non-empty")
        for q in quantiles:
            if not (0.0 < q < 1.0):
                raise ValueError("each quantile q must be in (0,1)")
        self.quantiles = [float(q) for q in quantiles]
        self.reduction = reduction
        self.crossing_lambda = float(crossing_lambda)
        self.apply_sigmoid = bool(apply_sigmoid)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            preds = torch.sigmoid(preds)

        q_list = self.quantiles
        if preds.dim() == 1 and len(q_list) == 1:
            preds_q = preds.unsqueeze(-1)
        else:
            preds_q = preds

        if preds_q.size(-1) != len(q_list):
            raise ValueError(
                f"preds last dim {preds_q.size(-1)} must match number of quantiles {len(q_list)}"
            )

        target = target.unsqueeze(-1).expand_as(preds_q)
        diff = target - preds_q  # y - y_hat

        losses = []
        for i, q in enumerate(q_list):
            qi = torch.as_tensor(q, device=preds_q.device, dtype=preds_q.dtype)
            # max{ q*(y-ŷ), (q-1)*(y-ŷ) }
            lhs = qi * diff[..., i]
            rhs = (qi - 1.0) * diff[..., i]
            loss_q = torch.maximum(lhs, rhs)
            losses.append(loss_q)
        loss = torch.stack(losses, dim=-1).sum(dim=-1)  # sum over Q

        # Anti-crossing penalty: sum_{i<j} ReLU(ŷ^qi - ŷ^qj)
        if self.crossing_lambda > 0.0 and preds_q.size(-1) > 1:
            penalty = 0.0
            for i in range(len(q_list)):
                for j in range(i + 1, len(q_list)):
                    penalty = penalty + torch.relu(preds_q[..., i] - preds_q[..., j])
            loss = loss + self.crossing_lambda * penalty

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def parse_quantiles(text: Optional[str]) -> Optional[Iterable[float]]:
    if text is None or text == "":
        return None
    parts = [p.strip() for p in text.split(",")]
    return [float(p) for p in parts if p]


def build_loss_from_args(args: argparse.Namespace) -> nn.Module:
    """Build loss function from args.

    Supported:
        - bce: Binary BCEWithLogitsLoss (as in current training)
        - huber: Huber
        - pseudohuber: Pseudo-Huber
        - quantile: Quantile / multi-quantile with anti-crossing
    Extra args:
        --loss_type {bce,huber,pseudohuber,quantile}
        --delta float
        --auto_delta_p float (e.g., 0.9)
        --quantiles "0.1,0.5,0.9"
        --crossing_lambda float
        --apply_sigmoid_in_regression (bool) apply sigmoid for probability vs 0/1 labels
    """
    loss_type = getattr(args, "loss_type", "bce").lower()

    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_type in {"huber", "pseudohuber"}:
        return HuberLoss(
            delta=getattr(args, "delta", 1.0),
            pseudo=(loss_type == "pseudohuber"),
            reduction="mean",
            apply_sigmoid=getattr(args, "apply_sigmoid_in_regression", True),
            auto_delta_p=getattr(args, "auto_delta_p", None),
        )

    if loss_type == "quantile":
        quantiles = getattr(args, "quantiles", None)
        if isinstance(quantiles, str):
            quant_list = list(parse_quantiles(quantiles) or [])
        elif isinstance(quantiles, (list, tuple)):
            quant_list = [float(q) for q in quantiles]
        else:
            # default 3 quantiles
            quant_list = [0.1, 0.5, 0.9]

        return QuantileLoss(
            quantiles=quant_list,
            reduction="mean",
            crossing_lambda=float(getattr(args, "crossing_lambda", 0.0)),
            apply_sigmoid=getattr(args, "apply_sigmoid_in_regression", True),
        )

    raise ValueError(f"Unknown loss_type: {loss_type}")


