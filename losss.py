import argparse
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """Huber / Pseudo-Huber loss with optional column weights.

    Args:
        delta: Transition point δ; if None and auto_delta_p is not None, adapt via p-quantile of |e|.
        pseudo: True to use Pseudo-Huber; False for standard Huber.
        reduction: "mean" | "sum" | "none".
        apply_sigmoid: If True, apply sigmoid to predictions before residual (binary prob. regression).
        auto_delta_p: If set (e.g., 0.9), use that p-quantile of |e| as δ (prefer from validation residuals).
        column_weights: Optional tensor or list of weights for each column/dimension. Shape: [D] or None.
    Notes:
        - Standard Huber:
          L_δ(e) = 0.5 e^2, |e| <= δ;  δ(|e| - 0.5δ), otherwise
        - Pseudo-Huber:
          L_δ(e) = δ^2 ( sqrt(1 + (e/δ)^2) - 1 )
        - With column weights: loss = sum_d (w_d * L_δ(e_d))
    """

    def __init__(
        self,
        delta: Optional[float] = 1.0,
        *,
        pseudo: bool = False,
        reduction: str = "mean",
        apply_sigmoid: bool = True,
        auto_delta_p: Optional[float] = None,
        column_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean','sum','none'}")
        self.delta = None if delta is None else float(delta)
        self.pseudo = bool(pseudo)
        self.reduction = reduction
        self.apply_sigmoid = bool(apply_sigmoid)
        self.auto_delta_p = None if auto_delta_p is None else float(auto_delta_p)
        
        # Register column weights as buffer if provided
        if column_weights is not None:
            if isinstance(column_weights, (list, tuple)):
                column_weights = torch.tensor(column_weights, dtype=torch.float32)
            elif not isinstance(column_weights, torch.Tensor):
                raise ValueError("column_weights must be a tensor, list, or None")
            self.register_buffer('column_weights', column_weights.float())
        else:
            self.register_buffer('column_weights', None)

    @staticmethod
    def _percentile(x: torch.Tensor, q: float) -> torch.Tensor:
        q = float(q)
        k = max(1, int(round((q * (x.numel() - 1)))))
        # Use torch.kthvalue to approximate percentile/quantile
        values, _ = torch.kthvalue(x.view(-1), k)
        return values

    def forward(self, preds: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            preds: [B, T, D] or [B, D] predictions
            target: [B, T, D] or [B, D] targets
            mask: Optional [B, T] mask for valid time steps (for sequence data)
        """
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

        # Apply column weights if provided
        if self.column_weights is not None:
            # Ensure column_weights shape matches the last dimension of loss
            if loss.dim() == 3:  # [B, T, D]
                weights = self.column_weights.view(1, 1, -1).to(loss.device)
            elif loss.dim() == 2:  # [B, D]
                weights = self.column_weights.view(1, -1).to(loss.device)
            else:
                weights = self.column_weights.to(loss.device)
            loss = loss * weights

        # Apply mask if provided (for sequence data)
        if mask is not None:
            if loss.dim() == 3:  # [B, T, D]
                mask_expanded = mask.unsqueeze(-1).to(loss.dtype)  # [B, T, 1]
                loss = loss * mask_expanded
                if self.reduction == "mean":
                    # denom should be the total number of valid elements (B*T*D), not just B*T
                    # mask_expanded.sum() gives B*T (number of valid time steps), need to multiply by feature dim D
                    denom = (mask_expanded.sum() * preds.shape[-1]).clamp_min(1.0)
                    return loss.sum() / denom
                elif self.reduction == "sum":
                    return loss.sum()
                else:
                    return loss
            else:
                # For 2D case, mask should be [B] if needed
                if mask.dim() == 1 and mask.size(0) == loss.size(0):
                    mask_expanded = mask.unsqueeze(-1).to(loss.dtype)
                    loss = loss * mask_expanded
                    if self.reduction == "mean":
                        return loss.sum() / mask_expanded.sum().clamp_min(1.0)
                    elif self.reduction == "sum":
                        return loss.sum()
                    else:
                        return loss

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
        - cross_entropy: Standard CrossEntropyLoss (default)
        - bce: Binary BCEWithLogitsLoss (legacy)
        - huber: Huber
        - pseudohuber: Pseudo-Huber
        - quantile: Quantile / multi-quantile with anti-crossing
        - focal: Focal Loss for class imbalance
        - weighted: Weighted Cross Entropy Loss
        - recall_focused: Recall-focused custom loss
        - adaptive: Adaptive loss based on recall performance
        - hinge: Hinge Loss for binary classification
    Extra args:
        --loss_type {cross_entropy,bce,huber,pseudohuber,quantile,focal,weighted,recall_focused,adaptive,hinge}
        --delta float
        --auto_delta_p float (e.g., 0.9)
        --quantiles "0.1,0.5,0.9"
        --crossing_lambda float
        --apply_sigmoid_in_regression (bool) apply sigmoid for probability vs 0/1 labels
        --focal_alpha float (default: 0.25)
        --focal_gamma float (default: 2.0)
        --class_weights "1.0,5.0" (weights for normal,fraud classes)
        --recall_weight float (default: 1.0)
        --target_recall float (default: 0.8)
        --adaptation_rate float (default: 0.1)
        --hinge_margin float (default: 1.0)
    """
    loss_type = getattr(args, "loss_type", "cross_entropy").lower()

    if loss_type in {"bce", "cross_entropy"}:
        return nn.CrossEntropyLoss()

    if loss_type in {"huber", "pseudohuber"}:
        # Parse column weights if provided
        column_weights = getattr(args, "column_weights", None)
        if column_weights is not None:
            if isinstance(column_weights, str):
                # Parse from comma-separated string
                weights_list = [float(w.strip()) for w in column_weights.split(",")]
                column_weights = torch.tensor(weights_list, dtype=torch.float32)
            elif isinstance(column_weights, (list, tuple)):
                column_weights = torch.tensor(column_weights, dtype=torch.float32)
            # If already a tensor, use as is
        
        return HuberLoss(
            delta=getattr(args, "delta", 1.0),
            pseudo=(loss_type == "pseudohuber"),
            reduction="mean",
            apply_sigmoid=getattr(args, "apply_sigmoid_in_regression", True),
            auto_delta_p=getattr(args, "auto_delta_p", None),
            column_weights=column_weights,
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
    
    if loss_type == "focal":
        return FocalLoss(
            alpha=getattr(args, "focal_alpha", 0.25),
            gamma=getattr(args, "focal_gamma", 2.0),
            reduction="mean"
        )
    
    if loss_type == "weighted":
        class_weights = getattr(args, "class_weights", "1.0,5.0")
        if isinstance(class_weights, str):
            weights = [float(w.strip()) for w in class_weights.split(",")]
        else:
            weights = [1.0, 5.0]  # default: normal=1.0, fraud=5.0
        return WeightedCrossEntropyLoss(class_weights=weights, reduction="mean")
    
    if loss_type == "recall_focused":
        return RecallFocusedLoss(
            recall_weight=getattr(args, "recall_weight", 1.0),
            reduction="mean"
        )
    
    if loss_type == "adaptive":
        return AdaptiveLoss(
            target_recall=getattr(args, "target_recall", 0.8),
            adaptation_rate=getattr(args, "adaptation_rate", 0.1),
            reduction="mean"
        )
    
    if loss_type == "hinge":
        return HingeLoss(
            margin=getattr(args, "hinge_margin", 1.0),
            reduction="mean"
        )

    raise ValueError(f"Unknown loss_type: {loss_type}")


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in fraud detection.
    
    Focal Loss = -α(1-p_t)^γ * log(p_t)
    where p_t is the predicted probability for the true class.
    
    Args:
        alpha: Weighting factor for rare class (fraud). Default: 0.25
        gamma: Focusing parameter. Higher gamma focuses more on hard examples. Default: 2.0
        reduction: "mean" | "sum" | "none"
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply softmax to get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    
    Args:
        class_weights: List of weights for each class [weight_normal, weight_fraud]
        reduction: "mean" | "sum" | "none"
    """
    
    def __init__(self, class_weights: list = [1.0, 5.0], reduction: str = "mean"):
        super().__init__()
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move weights to same device as inputs
        weights = self.class_weights.to(inputs.device)
        
        # Calculate weighted cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        weighted_loss = ce_loss * weights[targets]
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class RecallFocusedLoss(nn.Module):
    """
    Custom loss function that directly optimizes for recall.
    
    Combines cross-entropy with a recall-focused penalty term.
    
    Args:
        recall_weight: Weight for recall penalty term. Default: 1.0
        reduction: "mean" | "sum" | "none"
    """
    
    def __init__(self, recall_weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.recall_weight = recall_weight
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        fraud_probs = probs[:, 1]  # Probability of fraud class
        
        # Recall penalty: penalize low fraud probabilities for fraud samples
        fraud_mask = targets == 1
        recall_penalty = torch.zeros_like(ce_loss)
        if fraud_mask.any():
            # For fraud samples, penalize low fraud probability
            recall_penalty[fraud_mask] = (1 - fraud_probs[fraud_mask]) ** 2
        
        # Combine losses
        total_loss = ce_loss + self.recall_weight * recall_penalty
        
        if self.reduction == "mean":
            return total_loss.mean()
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts based on current recall performance.
    
    Args:
        target_recall: Target recall rate. Default: 0.8
        adaptation_rate: How quickly to adapt weights. Default: 0.1
        reduction: "mean" | "sum" | "none"
    """
    
    def __init__(self, target_recall: float = 0.8, adaptation_rate: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.target_recall = target_recall
        self.adaptation_rate = adaptation_rate
        self.reduction = reduction
        self.register_buffer('fraud_weight', torch.tensor(1.0))
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate current recall (approximate)
        probs = F.softmax(inputs, dim=1)
        fraud_mask = targets == 1
        if fraud_mask.any():
            current_recall = (probs[fraud_mask, 1] > 0.5).float().mean()
            # Adapt fraud weight based on current recall
            if current_recall < self.target_recall:
                self.fraud_weight *= (1 + self.adaptation_rate)
            else:
                self.fraud_weight *= (1 - self.adaptation_rate * 0.1)
        
        # Weighted cross entropy
        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[targets == 1] = self.fraud_weight
        
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        weighted_loss = ce_loss * weights
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


class HingeLoss(nn.Module):
    """
    Hinge Loss for binary classification.
    
    Hinge Loss = max(0, 1 - y * f(x))
    where y is the true label (-1 or 1) and f(x) is the predicted score.
    
    For multi-class classification, we use one-vs-rest approach:
    - For class 0: hinge_loss = max(0, 1 - (score_0 - score_1))
    - For class 1: hinge_loss = max(0, 1 - (score_1 - score_0))
    
    Args:
        margin: Margin parameter. Default: 1.0
        reduction: "mean" | "sum" | "none"
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, 2] logits for binary classification
            targets: [B] class labels (0 or 1)
        """
        # Convert targets to -1/1 format for hinge loss
        targets_hinge = 2 * targets - 1  # Convert 0->-1, 1->1
        
        # For binary classification, we use the difference between class scores
        # inputs[:, 1] - inputs[:, 0] gives the score difference (class 1 vs class 0)
        scores = inputs[:, 1] - inputs[:, 0]
        
        # Calculate hinge loss: max(0, margin - y * f(x))
        hinge_loss = torch.clamp(self.margin - targets_hinge * scores, min=0.0)
        
        if self.reduction == "mean":
            return hinge_loss.mean()
        elif self.reduction == "sum":
            return hinge_loss.sum()
        else:
            return hinge_loss
