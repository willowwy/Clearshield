import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from typing import Tuple, Optional
import json
from datetime import datetime


class FraudJudgeDNN(nn.Module):
    """
    DNN-based fraud judgment model
    Input: comparison features between main model predictions and ground truth
    Output: binary classification result (0=normal, 1=fraud)
    """
    
    def __init__(self, 
                 pred_dim: int,
                 hidden_dims: list = [64, 32, 16],
                 dropout: float = 0.2,
                 use_attention: bool = False,
                 use_statistical_features: bool = True,
                 use_basic_features: bool = True,
                 hidden_dim: int = None,
                 use_pred: bool = False):
        super().__init__()
        
        self.pred_dim = pred_dim
        self.use_attention = use_attention
        self.use_statistical_features = use_statistical_features
        self.use_basic_features = use_basic_features
        self.hidden_dim = hidden_dim  # Dimension of hidden representation
        self.use_pred = use_pred  # Whether to use predictions in addition to hidden representation
        
        # Calculate input dimension based on feature configuration
        input_dim = 0
        
        # Hidden representation is always used (default) or when use_pred is False
        if hidden_dim is not None:
            input_dim += hidden_dim  # Hidden representation
        
        # Predictions are used only when use_pred is True
        if use_pred and use_basic_features:
            input_dim += pred_dim * 3  # Predictions, targets, errors
        
        # Statistical features
        if use_statistical_features:
            input_dim += 6  # MSE, MAE, max_error, min_error, std_error, mean_error
        
        if input_dim == 0:
            raise ValueError("At least one feature type must be enabled")
        
        # Build DNN layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer: binary classification
        layers.append(nn.Linear(prev_dim, 2))
        
        self.dnn = nn.Sequential(*layers)
        
        # Optional attention mechanism (for sequence data)
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=pred_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
    
    def extract_features(self, predictions: Optional[torch.Tensor] = None, 
                        targets: Optional[torch.Tensor] = None,
                        hidden_representation: Optional[torch.Tensor] = None,
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract comparison features from predictions, ground truth, and hidden representation
        
        Args:
            predictions: [B, T, pred_dim] or [B, pred_dim] (optional, used only when use_pred=True)
            targets: [B, T, pred_dim] or [B, pred_dim] ground truth (required for statistical features)
            hidden_representation: [B, hidden_dim] or [B, T, hidden_dim] or tuple for LSTM hidden state
            mask: [B, T] or None, valid step markers for sequence data
            
        Returns:
            features: [B, feature_dim] extracted features
        """
        # Determine batch size from available inputs
        if hidden_representation is not None:
            if isinstance(hidden_representation, tuple):
                batch_size = hidden_representation[0].shape[1]  # LSTM: (h_n, c_n), h_n: [layers, B, hidden]
            elif len(hidden_representation.shape) == 3:
                batch_size = hidden_representation.shape[0]
            else:
                batch_size = hidden_representation.shape[0]
        elif targets is not None:
            batch_size = targets.shape[0]
        elif predictions is not None:
            batch_size = predictions.shape[0]
        else:
            raise ValueError("At least one of predictions, targets, or hidden_representation must be provided")
        
        feature_list = []
        
        # Process hidden representation (always used when available)
        if hidden_representation is not None:
            if isinstance(hidden_representation, tuple):
                # LSTM hidden state: (h_n, c_n) where each is [num_layers * num_directions, B, hidden_size]
                h_n, c_n = hidden_representation
                # Take the last layer's hidden state
                # h_n: [num_layers * num_directions, B, hidden_size]
                # We take the last layer: h_n[-1] -> [B, hidden_size]
                hidden_agg = h_n[-1]  # [B, hidden_size]
            elif len(hidden_representation.shape) == 3:  # [B, T, hidden_dim]
                if mask is not None:
                    # Use mask for weighted average
                    valid_steps = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
                    hidden_agg = (hidden_representation * mask.unsqueeze(-1)).sum(dim=1) / valid_steps  # [B, hidden_dim]
                else:
                    # Simple average
                    hidden_agg = hidden_representation.mean(dim=1)  # [B, hidden_dim]
            else:  # [B, hidden_dim]
                hidden_agg = hidden_representation  # [B, hidden_dim]
            
            feature_list.append(hidden_agg)  # [B, hidden_dim]
        
        # Process predictions and targets (only when use_pred=True)
        if self.use_pred and predictions is not None:
            # Handle sequence data: if input is 3D, need aggregation
            if len(predictions.shape) == 3:  # [B, T, pred_dim]
                if mask is not None:
                    # Use mask for weighted average
                    valid_steps = mask.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
                    pred_agg = (predictions * mask.unsqueeze(-1)).sum(dim=1) / valid_steps  # [B, pred_dim]
                    target_agg = (targets * mask.unsqueeze(-1)).sum(dim=1) / valid_steps  # [B, pred_dim]
                else:
                    # Simple average
                    pred_agg = predictions.mean(dim=1)  # [B, pred_dim]
                    target_agg = targets.mean(dim=1)  # [B, pred_dim]
            else:  # [B, pred_dim]
                pred_agg = predictions
                target_agg = targets
            
            # Calculate error features
            error = pred_agg - target_agg  # [B, pred_dim]
            
            # Add basic features
            if self.use_basic_features:
                feature_list.extend([
                    pred_agg,  # [B, pred_dim] predictions
                    target_agg,  # [B, pred_dim] ground truth
                    error,  # [B, pred_dim] error
                ])
        
        # Add statistical features if enabled (based on predictions and targets)
        if self.use_statistical_features:
            if self.use_pred and predictions is not None and targets is not None:
                # Calculate error from predictions and targets
                if len(predictions.shape) == 3:  # [B, T, pred_dim]
                    if mask is not None:
                        valid_steps = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                        pred_agg = (predictions * mask.unsqueeze(-1)).sum(dim=1) / valid_steps
                        target_agg = (targets * mask.unsqueeze(-1)).sum(dim=1) / valid_steps
                    else:
                        pred_agg = predictions.mean(dim=1)
                        target_agg = targets.mean(dim=1)
                else:
                    pred_agg = predictions
                    target_agg = targets
                error = pred_agg - target_agg
                
                abs_error = torch.abs(error)  # [B, pred_dim]
                
                # Statistical features
                mse = (error ** 2).mean(dim=1, keepdim=True)  # [B, 1]
                mae = abs_error.mean(dim=1, keepdim=True)  # [B, 1]
                max_error = abs_error.max(dim=1, keepdim=True)[0]  # [B, 1]
                min_error = abs_error.min(dim=1, keepdim=True)[0]  # [B, 1]
                std_error = abs_error.std(dim=1, keepdim=True)  # [B, 1]
                mean_error = abs_error.mean(dim=1, keepdim=True)  # [B, 1]
                
                feature_list.extend([
                    mse,  # [B, 1]
                    mae,  # [B, 1]
                    max_error,  # [B, 1]
                    min_error,  # [B, 1]
                    std_error,  # [B, 1]
                    mean_error,  # [B, 1]
                ])
            elif targets is not None:
                # If not using predictions, calculate statistics based on targets only
                # (e.g., variance, range, etc. of targets)
                if len(targets.shape) == 3:  # [B, T, pred_dim]
                    if mask is not None:
                        valid_steps = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                        target_agg = (targets * mask.unsqueeze(-1)).sum(dim=1) / valid_steps
                    else:
                        target_agg = targets.mean(dim=1)
                else:
                    target_agg = targets
                
                # Statistical features based on targets
                target_abs = torch.abs(target_agg)  # [B, pred_dim]
                mse = (target_agg ** 2).mean(dim=1, keepdim=True)  # [B, 1]
                mae = target_abs.mean(dim=1, keepdim=True)  # [B, 1]
                max_error = target_abs.max(dim=1, keepdim=True)[0]  # [B, 1]
                min_error = target_abs.min(dim=1, keepdim=True)[0]  # [B, 1]
                std_error = target_abs.std(dim=1, keepdim=True)  # [B, 1]
                mean_error = target_abs.mean(dim=1, keepdim=True)  # [B, 1]
                
                feature_list.extend([
                    mse,  # [B, 1]
                    mae,  # [B, 1]
                    max_error,  # [B, 1]
                    min_error,  # [B, 1]
                    std_error,  # [B, 1]
                    mean_error,  # [B, 1]
                ])
        
        # Combine all features
        if feature_list:
            features = torch.cat(feature_list, dim=1)
        else:
            raise ValueError("No features enabled - at least one feature type must be used")
        
        return features
    
    def forward(self, predictions: Optional[torch.Tensor] = None, 
                targets: Optional[torch.Tensor] = None,
                hidden_representation: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            predictions: [B, T, pred_dim] or [B, pred_dim] main model predictions (optional, used only when use_pred=True)
            targets: [B, T, pred_dim] or [B, pred_dim] ground truth (required when use_pred=True)
            hidden_representation: [B, hidden_dim] or [B, T, hidden_dim] or tuple for LSTM hidden state
            mask: [B, T] or None, valid step markers for sequence data
            
        Returns:
            logits: [B, 2] binary classification logits
        """
        # Extract features
        features = self.extract_features(predictions, targets, hidden_representation, mask)
        
        # Pass through DNN
        logits = self.dnn(features)
        
        return logits
    
    def predict_proba(self, predictions: Optional[torch.Tensor] = None, 
                     targets: Optional[torch.Tensor] = None,
                     hidden_representation: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict probabilities"""
        with torch.no_grad():
            logits = self.forward(predictions, targets, hidden_representation, mask)
            return F.softmax(logits, dim=1)
    
    def predict(self, predictions: Optional[torch.Tensor] = None, 
               targets: Optional[torch.Tensor] = None,
               hidden_representation: Optional[torch.Tensor] = None,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict classes"""
        with torch.no_grad():
            logits = self.forward(predictions, targets, hidden_representation, mask)
            return torch.argmax(logits, dim=1)


class FraudJudgeTrainer:
    """Fraud judgment model trainer"""
    
    def __init__(self, model: FraudJudgeDNN, device: torch.device, criterion=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    
    def setup_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4):
        """Setup optimizer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    
    def setup_scheduler(self, scheduler_type: str = 'step', **kwargs):
        """
        Setup learning rate scheduler
        
        Args:
            scheduler_type: Type of scheduler ('step', 'plateau', 'cosine', 'exponential', None)
            **kwargs: Additional arguments for scheduler
                - For 'step': step_size, gamma (default: step_size=30, gamma=0.1)
                - For 'plateau': mode, factor, patience, verbose (default: mode='max', factor=0.5, patience=10)
                - For 'cosine': T_max, eta_min (default: T_max=100, eta_min=0)
                - For 'exponential': gamma (default: gamma=0.95)
        """
        if self.optimizer is None:
            raise ValueError("Optimizer must be setup before scheduler. Call setup_optimizer() first.")
        
        if scheduler_type is None or scheduler_type == 'none':
            self.scheduler = None
            return
        
        scheduler_type = scheduler_type.lower()
        
        if scheduler_type == 'step':
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == 'plateau':
            mode = kwargs.get('mode', 'max')
            factor = kwargs.get('factor', 0.5)
            patience = kwargs.get('patience', 10)
            verbose = kwargs.get('verbose', False)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
            )
        elif scheduler_type == 'cosine':
            T_max = kwargs.get('T_max', 100)
            eta_min = kwargs.get('eta_min', 0)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_type == 'exponential':
            gamma = kwargs.get('gamma', 0.95)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}. "
                           f"Supported types: 'step', 'plateau', 'cosine', 'exponential', None")
    
    def step_scheduler(self, metric: Optional[float] = None):
        """
        Step the learning rate scheduler
        
        Args:
            metric: Metric value for ReduceLROnPlateau scheduler (e.g., validation accuracy)
        """
        if self.scheduler is None:
            return
        
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is None:
                raise ValueError("Metric value is required for ReduceLROnPlateau scheduler")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        if self.optimizer is None:
            return 0.0
        return self.optimizer.param_groups[0]['lr']
    
    def train_step(self, predictions: Optional[torch.Tensor] = None, 
                   targets: Optional[torch.Tensor] = None,
                   labels: torch.Tensor = None,
                   hidden_representation: Optional[torch.Tensor] = None,
                   mask: Optional[torch.Tensor] = None) -> float:
        """Single training step"""
        self.model.train()  # Enable dropout and batch norm training mode
        self.optimizer.zero_grad()
        
        # Forward propagation
        logits = self.model(predictions, targets, hidden_representation, mask)
        loss = self.criterion(logits, labels)
        
        # Backward propagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, predictions: Optional[torch.Tensor] = None, 
                 targets: Optional[torch.Tensor] = None,
                 labels: torch.Tensor = None,
                 hidden_representation: Optional[torch.Tensor] = None,
                 mask: Optional[torch.Tensor] = None) -> dict:
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(predictions, targets, hidden_representation, mask)
            loss = self.criterion(logits, labels)
            
            # Calculate accuracy
            pred_labels = torch.argmax(logits, dim=1)
            accuracy = (pred_labels == labels).float().mean().item()
            
            # Calculate prediction probabilities
            probs = F.softmax(logits, dim=1)
            
            return {
                'loss': loss.item(),
                'accuracy': accuracy,
                'predictions': pred_labels.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'logits': logits.cpu().numpy()
            }


def build_judge_model(pred_dim: int, hidden_dims: list = [64, 32, 16], 
                     dropout: float = 0.2, use_attention: bool = False,
                     use_statistical_features: bool = True,
                     use_basic_features: bool = True,
                     hidden_dim: int = None,
                     use_pred: bool = False) -> FraudJudgeDNN:
    """Build judgment model"""
    return FraudJudgeDNN(
        pred_dim=pred_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_attention=use_attention,
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features,
        hidden_dim=hidden_dim,
        use_pred=use_pred
    )

