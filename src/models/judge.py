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
    DNN-based fraud judgment model with 1D CNN for sequence processing
    Input: 
        - Hidden sequence: [B, max_len, hidden_dim] processed by 1D CNN
        - Predictions and targets: [B, pred_dim] (optional)
        - Statistical features: [B, 6] (optional)
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
                 use_pred: bool = False,
                 max_len: int = 50,
                 cnn_out_channels: int = 64,
                 cnn_kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        
        self.pred_dim = pred_dim
        self.use_attention = use_attention
        self.use_statistical_features = use_statistical_features
        self.use_basic_features = use_basic_features
        self.hidden_dim = hidden_dim  # Dimension of hidden representation
        self.use_pred = use_pred  # Whether to use predictions in addition to hidden representation
        self.max_len = max_len  # Maximum sequence length for hidden representation
        self.cnn_out_channels = cnn_out_channels  # Store for debugging
        self.cnn_kernel_sizes = cnn_kernel_sizes  # Store for debugging
        
        # Build 1D CNN for processing hidden sequence
        cnn_features_dim = 0
        if hidden_dim is not None:
            self.cnn_layers = nn.ModuleList()
            for kernel_size in cnn_kernel_sizes:
                # Each CNN branch: Conv1d -> BatchNorm -> ReLU -> GlobalMaxPool
                conv = nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=cnn_out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2  # Same padding
                )
                bn = nn.BatchNorm1d(cnn_out_channels)
                self.cnn_layers.append(nn.Sequential(conv, bn, nn.ReLU()))
                cnn_features_dim += cnn_out_channels  # Each branch contributes cnn_out_channels features
        else:
            self.cnn_layers = None
            cnn_features_dim = 0
        
        # Calculate input dimension based on feature configuration
        input_dim = 0
        
        # CNN features from hidden sequence
        if hidden_dim is not None:
            input_dim += cnn_features_dim  # CNN output features
        
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
            hidden_representation: [B, max_len, hidden_dim] or [B, T, hidden_dim] or [B, hidden_dim] or tuple for LSTM
                - For sequence: will be padded/truncated to max_len and processed by 1D CNN
                - For single time step: will be expanded to [B, max_len, hidden_dim]
                - For LSTM tuple: will use last hidden state and expand to sequence
            mask: [B, T] or [B, max_len] or None, valid step markers for sequence data
            
        Returns:
            features: [B, feature_dim] extracted features
                - CNN features: [B, cnn_out_channels * len(cnn_kernel_sizes)]
                - Basic features (if enabled): [B, pred_dim * 3]
                - Statistical features (if enabled): [B, 6]
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
        
        # Process hidden representation using 1D CNN
        # Input: [B, max_len, hidden_dim] or [B, T, hidden_dim]
        if hidden_representation is not None and self.cnn_layers is not None:
            if isinstance(hidden_representation, tuple):
                # LSTM hidden state: (h_n, c_n) - cannot process with CNN, use last hidden state
                h_n, c_n = hidden_representation
                hidden_last = h_n[-1]  # [B, hidden_size]
                # Expand to sequence format: [B, 1, hidden_size] -> [B, max_len, hidden_size]
                # Repeat the last hidden state to match max_len
                hidden_seq = hidden_last.unsqueeze(1).expand(-1, self.max_len, -1)  # [B, max_len, hidden_dim]
            elif len(hidden_representation.shape) == 3:  # [B, T, hidden_dim]
                # Sequence data: pad or truncate to max_len
                seq_len = hidden_representation.shape[1]
                if seq_len < self.max_len:
                    # Pad with zeros at the beginning
                    pad_len = self.max_len - seq_len
                    padding = torch.zeros(
                        hidden_representation.shape[0], 
                        pad_len, 
                        hidden_representation.shape[2],
                        device=hidden_representation.device,
                        dtype=hidden_representation.dtype
                    )  # [B, pad_len, hidden_dim]
                    hidden_seq = torch.cat([padding, hidden_representation], dim=1)  # [B, max_len, hidden_dim]
                elif seq_len > self.max_len:
                    # Truncate to max_len (take last max_len steps)
                    hidden_seq = hidden_representation[:, -self.max_len:, :]  # [B, max_len, hidden_dim]
                else:
                    hidden_seq = hidden_representation  # [B, max_len, hidden_dim]
            else:  # [B, hidden_dim] - single time step
                # Expand to sequence format
                hidden_seq = hidden_representation.unsqueeze(1).expand(-1, self.max_len, -1)  # [B, max_len, hidden_dim]
            
            # Apply mask if provided (zero out padding positions)
            if mask is not None:
                # mask: [B, T] or [B, max_len]
                if mask.shape[1] != self.max_len:
                    # Pad or truncate mask to max_len
                    if mask.shape[1] < self.max_len:
                        pad_len = self.max_len - mask.shape[1]
                        mask_padding = torch.zeros(
                            mask.shape[0], pad_len,
                            device=mask.device, dtype=mask.dtype
                        )
                        mask = torch.cat([mask_padding, mask], dim=1)
                    else:
                        mask = mask[:, -self.max_len:]
                hidden_seq = hidden_seq * mask.unsqueeze(-1)  # [B, max_len, hidden_dim]
            
            # Apply 1D CNN: [B, max_len, hidden_dim] -> [B, hidden_dim, max_len] (transpose for Conv1d)
            hidden_seq_transposed = hidden_seq.transpose(1, 2)  # [B, hidden_dim, max_len]
            
            # Process through multiple CNN branches with different kernel sizes
            cnn_outputs = []
            for cnn_layer in self.cnn_layers:
                cnn_out = cnn_layer(hidden_seq_transposed)  # [B, cnn_out_channels, max_len]
                # Global max pooling over sequence dimension
                cnn_pooled = F.adaptive_max_pool1d(cnn_out, output_size=1).squeeze(-1)  # [B, cnn_out_channels]
                cnn_outputs.append(cnn_pooled)
            
            # Concatenate outputs from all CNN branches
            cnn_features = torch.cat(cnn_outputs, dim=1)  # [B, cnn_out_channels * len(cnn_kernel_sizes)]
            feature_list.append(cnn_features)
        
        # Process predictions and targets (only when use_pred=True)
        if self.use_pred:
            if predictions is not None:
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
            else:
                # predictions is None but use_pred=True
                # This can happen during testing when predictions weren't extracted
                # In this case, we can't extract basic features (pred, target, error)
                # but we can still use targets for statistical features
                if self.use_basic_features:
                    # If basic features are required but predictions are None, we need to handle this
                    # Option 1: Use zeros for predictions (not ideal)
                    # Option 2: Only use targets (but this changes feature dimension)
                    # Option 3: Raise an error to indicate missing predictions
                    # For now, we'll use targets as both pred and target (error will be zero)
                    if targets is not None:
                        if len(targets.shape) == 3:  # [B, T, pred_dim]
                            if mask is not None:
                                valid_steps = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                                target_agg = (targets * mask.unsqueeze(-1)).sum(dim=1) / valid_steps
                            else:
                                target_agg = targets.mean(dim=1)
                        else:
                            target_agg = targets
                        
                        # Use targets as both predictions and targets (error = 0)
                        pred_agg = target_agg  # Use target as prediction (error will be 0)
                        error = torch.zeros_like(target_agg)  # Zero error
                        
                        feature_list.extend([
                            pred_agg,  # [B, pred_dim] (using target as prediction)
                            target_agg,  # [B, pred_dim] ground truth
                            error,  # [B, pred_dim] (zero error)
                        ])
                    else:
                        raise ValueError("When use_pred=True and use_basic_features=True, either predictions or targets must be provided")
        
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
        
        # Debug: check feature dimension
        if features.shape[1] != self.dnn[0].in_features:
            raise RuntimeError(
                f"Feature dimension mismatch: extracted features have shape {features.shape}, "
                f"but DNN first layer expects input_dim={self.dnn[0].in_features}. "
                f"Model config: hidden_dim={self.hidden_dim}, use_pred={self.use_pred}, "
                f"use_basic_features={self.use_basic_features}, use_statistical_features={self.use_statistical_features}, "
                f"max_len={self.max_len}, cnn_out_channels={getattr(self, 'cnn_out_channels', 'N/A')}, "
                f"cnn_kernel_sizes={getattr(self, 'cnn_kernel_sizes', 'N/A')}"
            )
        
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
                     use_pred: bool = False,
                     max_len: int = 50,
                     cnn_out_channels: int = 64,
                     cnn_kernel_sizes: list = [3, 5, 7]) -> FraudJudgeDNN:
    """Build judgment model"""
    return FraudJudgeDNN(
        pred_dim=pred_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        use_attention=use_attention,
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features,
        hidden_dim=hidden_dim,
        use_pred=use_pred,
        max_len=max_len,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_sizes=cnn_kernel_sizes
    )

