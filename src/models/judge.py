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
                 use_basic_features: bool = True):
        super().__init__()
        
        self.pred_dim = pred_dim
        self.use_attention = use_attention
        self.use_statistical_features = use_statistical_features
        self.use_basic_features = use_basic_features
        
        # Calculate input dimension based on feature configuration
        if use_basic_features and use_statistical_features:
            input_dim = pred_dim * 3 + 6  # Basic features + statistical features
        elif use_basic_features and not use_statistical_features:
            input_dim = pred_dim * 3  # Only basic features (predictions, targets, errors)
        elif not use_basic_features and use_statistical_features:
            input_dim = 6  # Only statistical features
        else:
            raise ValueError("At least one of use_basic_features or use_statistical_features must be True")
        
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
    
    def extract_features(self, predictions: torch.Tensor, targets: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract comparison features from predictions and ground truth
        
        Args:
            predictions: [B, T, pred_dim] or [B, pred_dim]
            targets: [B, T, pred_dim] or [B, pred_dim]  
            mask: [B, T] or None, valid step markers for sequence data
            
        Returns:
            features: [B, feature_dim] extracted features
        """
        batch_size = predictions.shape[0]
        
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
        
        # Build feature list based on configuration
        feature_list = []
        
        # Add basic features if enabled
        if self.use_basic_features:
            feature_list.extend([
                pred_agg,  # [B, pred_dim] predictions
                target_agg,  # [B, pred_dim] ground truth
                error,  # [B, pred_dim] error
            ])
        
        # Add statistical features if enabled
        if self.use_statistical_features:
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
        
        # Combine all features
        if feature_list:
            features = torch.cat(feature_list, dim=1)
        else:
            raise ValueError("No features enabled - at least one feature type must be used")
        
        return features
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward propagation
        
        Args:
            predictions: [B, T, pred_dim] or [B, pred_dim] main model predictions
            targets: [B, T, pred_dim] or [B, pred_dim] ground truth
            mask: [B, T] or None, valid step markers for sequence data
            
        Returns:
            logits: [B, 2] binary classification logits
        """
        # Extract features
        features = self.extract_features(predictions, targets, mask)
        
        # Pass through DNN
        logits = self.dnn(features)
        
        return logits
    
    def predict_proba(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict probabilities"""
        with torch.no_grad():
            logits = self.forward(predictions, targets, mask)
            return F.softmax(logits, dim=1)
    
    def predict(self, predictions: torch.Tensor, targets: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict classes"""
        with torch.no_grad():
            logits = self.forward(predictions, targets, mask)
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
    
    def train_step(self, predictions: torch.Tensor, targets: torch.Tensor, 
                   labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Single training step"""
        self.model.train()  # Enable dropout and batch norm training mode
        self.optimizer.zero_grad()
        
        # Forward propagation
        logits = self.model(predictions, targets, mask)
        loss = self.criterion(logits, labels)
        
        # Backward propagation
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """Evaluate model"""
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(predictions, targets, mask)
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
                     use_basic_features: bool = True) -> FraudJudgeDNN:
    """Build judgment model"""
    return FraudJudgeDNN(
        pred_dim=pred_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_attention=use_attention,
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features
    )

