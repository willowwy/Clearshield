'''
User guide: 

run

python eval.py \
--model_path models/best_model_enc.pth \
--data_dir {data_dir_with_matched_fraud}  \
--save_dir {.}

after putting precessed csv files in data/precessed/ and model in checkpoints

'''

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns  # Not used in current implementation
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.models.backbone_model import build_arg_parser, build_seq_model
from src.models.datasets import create_dataloader
from src.models.load_model import count_parameters


def print_model_hyperparameters(args):
    """Print model hyperparameters in a structured format"""
    print("\n" + "="*60)
    print("Model Hyperparameters")
    print("="*60)
    
    # Convert args to dict if it's Namespace
    if hasattr(args, '__dict__'):
        args_dict = vars(args)
    elif isinstance(args, dict):
        args_dict = args
    else:
        args_dict = {}
    
    # Model architecture parameters
    print("\n[Model Architecture]")
    model_type = args_dict.get('model_type', 'unknown')
    print(f"  Model Type: {model_type}")
    
    if model_type in ['transformer', 'fraudenc', 'fraudftenc']:
        print(f"  Transformer Dimension: {args_dict.get('transformer_dim', 'N/A')}")
        print(f"  Number of Attention Heads: {args_dict.get('nhead', 'N/A')}")
        print(f"  Number of Encoder Layers: {args_dict.get('num_encoder_layers', 'N/A')}")
        dim_ff = args_dict.get('dim_feedforward', 0)
        if dim_ff == 0:
            transformer_dim = args_dict.get('transformer_dim', 128)
            dim_ff = transformer_dim * 2
            print(f"  Feedforward Dimension: {dim_ff} (auto: 2 * transformer_dim)")
        else:
            print(f"  Feedforward Dimension: {dim_ff}")
        print(f"  Activation Function: {args_dict.get('activation', 'N/A')}")
        print(f"  Pre-LN (norm_first): {args_dict.get('norm_first', False)}")
        print(f"  Max Sequence Length: {args_dict.get('max_seq_len', 'N/A')}")
        if model_type == 'fraudftenc':
            print(f"  Feature-level Token Encoder: Each feature is an independent token")
    else:
        print(f"  LSTM Hidden Dimension: {args_dict.get('lstm_hidden', 'N/A')}")
        print(f"  LSTM Layers: {args_dict.get('lstm_layers', 'N/A')}")
        print(f"  Bidirectional: {args_dict.get('bidirectional', False)}")
    
    print(f"  Dropout Rate: {args_dict.get('dropout', 'N/A')}")
    print(f"  Embedding Dropout: {args_dict.get('embedding_dropout', 0.0)}")
    
    # Embedding parameters
    print("\n[Embedding Parameters]")
    print(f"  Day Vocabulary Size: {args_dict.get('day_vocab', 'N/A')}")
    print(f"  Day Embedding Dim: {args_dict.get('day_emb_dim', 'N/A')}")
    print(f"  Hour Vocabulary Size: {args_dict.get('hour_vocab', 'N/A')}")
    print(f"  Hour Embedding Dim: {args_dict.get('hour_emb_dim', 'N/A')}")
    print(f"  Minute Vocabulary Size: {args_dict.get('minute_vocab', 'N/A')}")
    print(f"  Minute Embedding Dim: {args_dict.get('minute_emb_dim', 'N/A')}")
    print(f"  Account Open Date Vocabulary Size: {args_dict.get('aod_day_vocab', 'N/A')}")
    print(f"  Account Open Date Embedding Dim: {args_dict.get('aod_day_emb_dim', 'N/A')}")
    print(f"  Continuous Feature Dimension: {args_dict.get('cont_dim', 'N/A')}")
    
    # Training parameters
    print("\n[Training Parameters]")
    print(f"  Learning Rate: {args_dict.get('lr', 'N/A')}")
    print(f"  Weight Decay: {args_dict.get('weight_decay', 'N/A')}")
    print(f"  Batch Size: {args_dict.get('batch_size', 'N/A')}")
    print(f"  Max Sequence Length: {args_dict.get('max_len', 'N/A')}")
    print(f"  Random Seed: {args_dict.get('seed', 'N/A')}")
    
    # Loss function parameters
    print("\n[Loss Function Parameters]")
    loss_type = args_dict.get('loss_type', 'N/A')
    print(f"  Loss Type: {loss_type}")
    if loss_type in ['huber', 'pseudohuber']:
        print(f"  Delta (δ): {args_dict.get('delta', 'N/A')}")
        auto_delta_p = args_dict.get('auto_delta_p', None)
        if auto_delta_p is not None:
            print(f"  Auto Delta P: {auto_delta_p}")
        column_weights = args_dict.get('column_weights', None)
        if column_weights:
            print(f"  Column Weights: {column_weights}")
        print(f"  Apply Sigmoid: {args_dict.get('apply_sigmoid_in_regression', True)}")
    elif loss_type == 'quantile':
        print(f"  Quantiles: {args_dict.get('quantiles', 'N/A')}")
        print(f"  Crossing Lambda: {args_dict.get('crossing_lambda', 'N/A')}")
    
    # Data split parameters
    print("\n[Data Split Parameters]")
    print(f"  Train Ratio: {args_dict.get('train_ratio', 'N/A')}")
    print(f"  Validation Ratio: {args_dict.get('val_ratio', 'N/A')}")
    print(f"  Test Ratio: {args_dict.get('test_ratio', 'N/A')}")
    
    # Target features
    target_names = args_dict.get('target_names', None)
    if target_names:
        print("\n[Target Features]")
        print(f"  Number of Target Features: {len(target_names)}")
        print(f"  Target Features: {target_names}")
    
    # Feature names
    feature_names = args_dict.get('feature_names', None)
    if feature_names:
        print(f"\n  Total Features: {len(feature_names)}")
    
    print("="*60 + "\n")


def extract_hidden_representation(hidden_state, mask, device):
    """
    Extract hidden representation from model output.
    
    Handles different model types:
    - LSTM: hidden_state is (h_n, c_n) tuple, extract last layer hidden state
    - Transformer: hidden_state is [B, T, hidden_dim], extract last valid time step
    
    Args:
        hidden_state: Model hidden state output (varies by model type)
        mask: [B, T] mask indicating valid time steps
        device: torch device
        
    Returns:
        hidden_repr: [B, hidden_dim] or None if extraction fails
    """
    if hidden_state is None:
        return None
    
    # Handle LSTM hidden state (tuple of (h_n, c_n))
    if isinstance(hidden_state, tuple):
        h_n, c_n = hidden_state
        # h_n shape: [num_layers * num_directions, B, hidden_dim]
        # For bidirectional LSTM: num_directions = 2
        # For unidirectional LSTM: num_directions = 1
        if h_n.dim() == 3:
            num_layers_times_directions, B, hidden_dim = h_n.shape
            # Determine if bidirectional (assuming even number means bidirectional)
            # Common cases: 1 layer unidirectional = 1, 1 layer bidirectional = 2
            #               2 layers unidirectional = 2, 2 layers bidirectional = 4
            if num_layers_times_directions == 1:
                # Single layer, unidirectional
                hidden_repr = h_n[0]  # [B, hidden_dim]
            elif num_layers_times_directions == 2:
                # Could be: 1 layer bidirectional OR 2 layers unidirectional
                # Try to detect: if both are similar, likely bidirectional
                # For simplicity, assume it's bidirectional if we have 2
                hidden_repr = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, hidden_dim * 2]
            else:
                # Multiple layers: take the last layer
                # If bidirectional, last two elements are forward and backward of last layer
                if num_layers_times_directions % 2 == 0:
                    # Bidirectional: concatenate last two
                    hidden_repr = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B, hidden_dim * 2]
                else:
                    # Unidirectional: take last
                    hidden_repr = h_n[-1]  # [B, hidden_dim]
        else:
            hidden_repr = h_n
        return hidden_repr
    
    # Handle Transformer hidden state [B, T, hidden_dim]
    elif isinstance(hidden_state, torch.Tensor):
        if hidden_state.dim() == 3:
            # Extract last valid time step for each sequence
            B, T, D = hidden_state.shape
            hidden_repr = []
            for b in range(B):
                # Find last valid time step
                valid_steps = mask[b].nonzero(as_tuple=True)[0]
                if len(valid_steps) > 0:
                    last_idx = valid_steps[-1].item()
                    hidden_repr.append(hidden_state[b, last_idx])
                else:
                    # If no valid steps, use first time step
                    hidden_repr.append(hidden_state[b, 0])
            hidden_repr = torch.stack(hidden_repr, dim=0)  # [B, hidden_dim]
            return hidden_repr
        elif hidden_state.dim() == 2:
            # Already [B, hidden_dim]
            return hidden_state
        else:
            print(f"Warning: Unexpected hidden_state shape: {hidden_state.shape}")
            return None
    
    else:
        print(f"Warning: Unknown hidden_state type: {type(hidden_state)}")
        return None


def plot_pca_visualization(results, save_dir, n_components=2):
    """
    Perform PCA on hidden states and visualize with fraud/non-fraud labels.
    
    Args:
        results: Dictionary containing 'hidden_states' and 'labels'
        save_dir: Directory to save the plot
        n_components: Number of PCA components (default 2 for 2D visualization)
    """
    if len(results.get('hidden_states', [])) == 0:
        print("No hidden states available for PCA visualization.")
        return
    
    # Concatenate all hidden states
    all_hidden = np.concatenate(results['hidden_states'], axis=0)  # [N, hidden_dim]
    
    # Get corresponding labels
    # Hidden states are extracted per sequence (one per batch item)
    # Labels are [B, T] shape, we need to match each sequence's hidden state with its label
    all_labels_flat = []
    masks = results.get('masks', [])
    
    for batch_idx, labels_batch in enumerate(results.get('labels', [])):
        # labels_batch shape: [B, T]
        # Get corresponding mask for this batch
        mask_batch = masks[batch_idx] if batch_idx < len(masks) else None
        
        for seq_idx, seq_labels in enumerate(labels_batch):
            # Find last valid time step for this sequence
            if mask_batch is not None and seq_idx < len(mask_batch):
                valid_mask = mask_batch[seq_idx] == 1
                valid_indices = np.where(valid_mask)[0]
                if len(valid_indices) > 0:
                    # Use label from last valid time step
                    last_valid_idx = valid_indices[-1]
                    all_labels_flat.append(seq_labels[last_valid_idx])
                else:
                    # No valid steps, use first label
                    all_labels_flat.append(seq_labels[0] if len(seq_labels) > 0 else 0)
            else:
                # No mask available, use last non-zero label or last element
                valid_labels = seq_labels[seq_labels != 0] if len(seq_labels) > 0 else seq_labels
                if len(valid_labels) > 0:
                    all_labels_flat.append(valid_labels[-1])
                else:
                    all_labels_flat.append(seq_labels[-1] if len(seq_labels) > 0 else 0)
    
    # Ensure we have the same number of labels as hidden states
    if len(all_labels_flat) != all_hidden.shape[0]:
        print(f"Warning: Label count ({len(all_labels_flat)}) doesn't match hidden state count ({all_hidden.shape[0]}). Adjusting...")
        # Truncate or pad to match
        if len(all_labels_flat) > all_hidden.shape[0]:
            all_labels_flat = all_labels_flat[:all_hidden.shape[0]]
        elif len(all_labels_flat) < all_hidden.shape[0]:
            all_labels_flat.extend([0] * (all_hidden.shape[0] - len(all_labels_flat)))
    
    all_labels_flat = np.array(all_labels_flat)
    
    print(f"\nPerforming PCA on hidden states...")
    print(f"  Hidden state shape: {all_hidden.shape}")
    print(f"  Number of samples: {len(all_labels_flat)}")
    print(f"  Fraud samples (label=1): {np.sum(all_labels_flat == 1)}")
    print(f"  Non-fraud samples (label=0): {np.sum(all_labels_flat == 0)}")
    
    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    hidden_pca = pca.fit_transform(all_hidden)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"  Explained variance ratio: {explained_variance}")
    print(f"  Total explained variance: {np.sum(explained_variance):.4f}")
    
    # Plot PCA visualization
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate fraud and non-fraud samples
    fraud_mask = all_labels_flat == 1
    non_fraud_mask = all_labels_flat == 0
    
    if n_components == 2:
        # 2D scatter plot
        if np.any(non_fraud_mask):
            ax.scatter(hidden_pca[non_fraud_mask, 0], hidden_pca[non_fraud_mask, 1], 
                      c='blue', label='Non-Fraud (label=0)', alpha=0.6, s=20)
        if np.any(fraud_mask):
            ax.scatter(hidden_pca[fraud_mask, 0], hidden_pca[fraud_mask, 1], 
                      c='red', label='Fraud (label=1)', alpha=0.6, s=20)
        
        ax.set_xlabel(f'PC1 (Explained Variance: {explained_variance[0]:.2%})')
        ax.set_ylabel(f'PC2 (Explained Variance: {explained_variance[1]:.2%})')
        ax.set_title('PCA Visualization of Hidden States\n(Fraud vs Non-Fraud)')
    elif n_components == 3:
        # 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        
        if np.any(non_fraud_mask):
            ax.scatter(hidden_pca[non_fraud_mask, 0], 
                      hidden_pca[non_fraud_mask, 1], 
                      hidden_pca[non_fraud_mask, 2],
                      c='blue', label='Non-Fraud (label=0)', alpha=0.6, s=20)
        if np.any(fraud_mask):
            ax.scatter(hidden_pca[fraud_mask, 0], 
                      hidden_pca[fraud_mask, 1], 
                      hidden_pca[fraud_mask, 2],
                      c='red', label='Fraud (label=1)', alpha=0.6, s=20)
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
        ax.set_title('PCA Visualization of Hidden States (3D)\n(Fraud vs Non-Fraud)')
    else:
        # For n_components > 3, plot first two components
        if np.any(non_fraud_mask):
            ax.scatter(hidden_pca[non_fraud_mask, 0], hidden_pca[non_fraud_mask, 1], 
                      c='blue', label='Non-Fraud (label=0)', alpha=0.6, s=20)
        if np.any(fraud_mask):
            ax.scatter(hidden_pca[fraud_mask, 0], hidden_pca[fraud_mask, 1], 
                      c='red', label='Fraud (label=1)', alpha=0.6, s=20)
        
        ax.set_xlabel(f'PC1 (Explained Variance: {explained_variance[0]:.2%})')
        ax.set_ylabel(f'PC2 (Explained Variance: {explained_variance[1]:.2%})')
        ax.set_title(f'PCA Visualization of Hidden States (First 2 Components)\n(Fraud vs Non-Fraud)')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'hidden_state_pca_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"PCA visualization saved to: {save_path}")


def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from: {model_path}")
    
    # load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # get model parameters
    args = checkpoint['args']
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss (MSE): {checkpoint.get('val_loss', 0.0):.6f}")
    
    # rebuild model
    model = build_seq_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count and display model parameters
    total_params, trainable_params = count_parameters(model)
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    # Print model hyperparameters
    print_model_hyperparameters(args)
    
    return model, args


def evaluate_model(model, test_loader, device, feature_names, target_indices, args):
    """Evaluate model performance"""
    model.eval()
    
    # Store all predictions and true values
    all_predictions = []
    all_targets = []
    all_masks = []
    all_labels = []  # aligned fraud labels for target/pred steps (t+1)
    all_hidden_states = []  # store hidden states for PCA visualization
    
    # Regression metrics
    total_mse = 0.0
    total_mae = 0.0
    total_loss = 0.0
    num_batches = 0
    
    print("Starting model evaluation...")
    
    with torch.no_grad():
        for batch_idx, (X, y, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Forward pass
            preds, hidden_state = model(X, mask, feature_names)
            
            if mask.any():
                # Calculate regression metrics
                pred_steps = preds[:, :-1, :]
                target_steps = X[:, 1:, target_indices]
                valid_steps = mask[:, 1:]
                label_steps = y[:, 1:]  # align labels with next-step targets
                
                # Calculate MSE and MAE
                m = valid_steps.unsqueeze(-1).to(preds.dtype)
                diff = (pred_steps - target_steps) * m
                # denom should be the total number of valid elements (B*T*D), not just B*T
                # m.sum() gives B*T (number of valid time steps), need to multiply by feature dim D
                denom = (m.sum() * pred_steps.shape[-1]).clamp_min(1.0)
                mse = (diff ** 2).sum() / denom
                mae = diff.abs().sum() / denom
                loss = mse
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_loss += loss.item()
                num_batches += 1
                
                # Store prediction results for further analysis
                all_predictions.append(pred_steps.cpu().numpy())
                all_targets.append(target_steps.cpu().numpy())
                all_masks.append(valid_steps.cpu().numpy())
                all_labels.append(label_steps.cpu().numpy())
                
                # Store hidden states for PCA visualization
                # Extract hidden state representation based on model type
                hidden_repr = extract_hidden_representation(hidden_state, mask, device)
                if hidden_repr is not None:
                    all_hidden_states.append(hidden_repr.cpu().numpy())
    
    # Calculate average metrics
    avg_mse = total_mse / num_batches if num_batches > 0 else 0.0
    avg_mae = total_mae / num_batches if num_batches > 0 else 0.0
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'mse': avg_mse,
        'mae': avg_mae,
        'loss': avg_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'masks': all_masks,
        'labels': all_labels,
        'hidden_states': all_hidden_states
    }


def analyze_predictions(results, target_names):
    """Analyze detailed statistics of prediction results"""
    print("\n" + "="*60)
    print("Detailed Prediction Analysis")
    print("="*60)
    
    # Concatenate all prediction results
    all_preds = np.concatenate([pred for pred in results['predictions']], axis=0)
    all_targets = np.concatenate([target for target in results['targets']], axis=0)
    all_masks = np.concatenate([mask for mask in results['masks']], axis=0)
    all_labels = np.concatenate([lab for lab in results.get('labels', [])], axis=0) if len(results.get('labels', [])) > 0 else None
    
    print(f"Total prediction steps: {all_preds.shape[0] * all_preds.shape[1]}")
    print(f"Valid prediction steps: {all_masks.sum()}")
    
    # Analyze by feature
    print(f"\nTarget features: {target_names}")
    print(f"Prediction dimensions: {all_preds.shape[-1]}")
    print(f"Target dimensions: {all_targets.shape[-1]}")
    
    # Calculate MSE and MAE for each feature
    feature_mse = []
    feature_mae = []
    feature_mse_by_label = {0: [], 1: []}
    
    for i in range(min(all_preds.shape[-1], all_targets.shape[-1])):
        # Only consider valid steps
        valid_mask = all_masks == 1
        if valid_mask.any():
            pred_feature = all_preds[:, :, i][valid_mask]
            target_feature = all_targets[:, :, i][valid_mask]
            
            mse = np.mean((pred_feature - target_feature) ** 2)
            mae = np.mean(np.abs(pred_feature - target_feature))
            
            feature_mse.append(mse)
            feature_mae.append(mae)
            
            print(f"Feature {target_names[i] if i < len(target_names) else f'Feature_{i}'}:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")

            # Group-wise MSE by fraud label (0/1) if labels are available
            if all_labels is not None:
                labels_valid = all_labels[valid_mask]
                for lbl in [0, 1]:
                    idx = labels_valid == lbl
                    if np.any(idx):
                        mse_lbl = np.mean((pred_feature[idx] - target_feature[idx]) ** 2)
                    else:
                        mse_lbl = float('nan')
                    feature_mse_by_label[lbl].append(mse_lbl)
                # Print group-wise stats
                print(f"  MSE | fraud=0: {feature_mse_by_label[0][-1]:.6f} | fraud=1: {feature_mse_by_label[1][-1]:.6f}")
    
    return {
        'feature_mse': feature_mse,
        'feature_mae': feature_mae,
        'all_predictions': all_preds,
        'all_targets': all_targets,
        'all_masks': all_masks,
        'all_labels': all_labels,
        'feature_mse_by_label': feature_mse_by_label
    }


def plot_evaluation_results(results, analysis, target_names, save_dir):
    """Plot evaluation result charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set font for Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Feature-level MSE and MAE comparison
    ax1 = axes[0, 0]
    x = np.arange(len(target_names))
    width = 0.35
    
    ax1.bar(x - width/2, analysis['feature_mse'], width, label='MSE', alpha=0.8)
    ax1.bar(x + width/2, analysis['feature_mae'], width, label='MAE', alpha=0.8)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Error Values')
    ax1.set_title('Feature-wise Prediction Error Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(target_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted vs True values scatter plot (select first feature)
    ax2 = axes[0, 1]
    if len(analysis['all_predictions']) > 0:
        valid_mask = analysis['all_masks'] == 1
        if valid_mask.any():
            pred_sample = analysis['all_predictions'][:, :, 0][valid_mask]
            target_sample = analysis['all_targets'][:, :, 0][valid_mask]
            
            # Random sampling for display
            n_samples = min(1000, len(pred_sample))
            indices = np.random.choice(len(pred_sample), n_samples, replace=False)
            
            ax2.scatter(target_sample[indices], pred_sample[indices], alpha=0.5, s=1)
            ax2.plot([target_sample.min(), target_sample.max()], 
                    [target_sample.min(), target_sample.max()], 'r--', lw=2)
            ax2.set_xlabel(f'True Values ({target_names[0]})')
            ax2.set_ylabel(f'Predicted Values ({target_names[0]})')
            ax2.set_title('Predicted vs True Values')
            ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution histogram
    ax3 = axes[1, 0]
    if len(analysis['all_predictions']) > 0:
        valid_mask = analysis['all_masks'] == 1
        if valid_mask.any():
            errors = analysis['all_predictions'][:, :, 0][valid_mask] - analysis['all_targets'][:, :, 0][valid_mask]
            ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Prediction Error')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Prediction Error Distribution')
            ax3.grid(True, alpha=0.3)
    
    # 4. Feature-wise MSE comparison by fraud label (grouped bars)
    ax4 = axes[1, 1]
    x = np.arange(len(target_names))
    width = 0.35
    mse0 = analysis.get('feature_mse_by_label', {}).get(0, [np.nan] * len(target_names))
    mse1 = analysis.get('feature_mse_by_label', {}).get(1, [np.nan] * len(target_names))
    ax4.bar(x - width/2, mse0, width, label='fraud=0', alpha=0.8)
    ax4.bar(x + width/2, mse1, width, label='fraud=1', alpha=0.8)
    ax4.set_xlabel('Features')
    ax4.set_ylabel('MSE')
    ax4.set_title('Feature-wise MSE by Fraud Label')
    ax4.set_xticks(x)
    ax4.set_xticklabels(target_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    plot_path = os.path.join(save_dir, 'evaluation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation chart saved to: {plot_path}")


def save_evaluation_results(results, analysis, target_names, save_dir, model=None):
    """Save evaluation results to file"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get model parameters if model is provided
    model_info = {}
    if model is not None:
        total_params, trainable_params = count_parameters(model)
        model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024
        }
    
    # Save basic metrics
    basic_results = {
        'evaluation_time': datetime.now().isoformat(),
        'mse': float(results['mse']),
        'mae': float(results['mae']),
        'loss': float(results['loss']),
        'target_features': target_names,
        'feature_mse': [float(x) for x in analysis['feature_mse']],
        'feature_mae': [float(x) for x in analysis['feature_mae']],
        'feature_mse_by_label': {
            'fraud_0': [float(x) if x == x else None for x in analysis.get('feature_mse_by_label', {}).get(0, [])],
            'fraud_1': [float(x) if x == x else None for x in analysis.get('feature_mse_by_label', {}).get(1, [])]
        },
        'model_info': model_info
    }
    
    # Save to JSON file
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(basic_results, f, indent=2, ensure_ascii=False)
    
    print(f"Evaluation results saved to: {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Results Summary")
    print("="*60)
    print(f"Overall MSE: {results['mse']:.6f}")
    print(f"Overall MAE: {results['mae']:.6f}")
    print(f"Overall Loss: {results['loss']:.6f}")
    print(f"Number of target features: {len(target_names)}")
    print(f"Results saved to directory: {save_dir}")


def _load_tokenize_dict(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Tokenize dict json not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _compute_cosine_matrix(emb_matrix_np):
    # emb_matrix_np: [N, D]
    if emb_matrix_np.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    # Normalize rows
    norms = np.linalg.norm(emb_matrix_np, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb_matrix_np / norms
    return emb_norm @ emb_norm.T


def _plot_cosine_heatmap(cos_mat, labels, title, save_path, max_ticks=50):
    # Configure font for Chinese labels
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_mat, cmap='viridis', vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    # Tick density control
    n = len(labels)
    if n <= max_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        step = max(1, n // max_ticks)
        idxs = np.arange(0, n, step)
        ax.set_xticks(idxs)
        ax.set_yticks(idxs)
        ax.set_xticklabels([labels[i] for i in idxs], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([labels[i] for i in idxs], fontsize=8)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_cat_embeddings(model, json_path, save_dir, features=None, max_tokens_per_feature=50):
    """Visualize cosine heatmaps for selected categorical embeddings.

    - model: trained model with cat_embs
    - json_path: path to tokenize_dict.json
    - save_dir: directory to save heatmaps
    - features: list of base feature names in tokenize_dict.json to visualize
    - max_tokens_per_feature: cap to avoid overly large plots
    """
    tok_dict = _load_tokenize_dict(json_path)
    if not hasattr(model, 'cat_embs') or model.cat_embs is None:
        print("Model has no categorical embeddings to visualize.")
        return

    if features is None:
        features = ['Account Type', 'Action Type', 'Source Type', 'Product ID']

    os.makedirs(save_dir, exist_ok=True)

    for base_name in features:
        model_key = f"{base_name}_enc"
        if model_key not in model.cat_embs:
            print(f"Skip: embedding not found in model for {model_key}")
            continue
        if base_name not in tok_dict:
            print(f"Skip: {base_name} not found in tokenize_dict.json")
            continue

        name_to_id = tok_dict[base_name]
        # Sort tokens by id to align with embedding row indices
        items = sorted(name_to_id.items(), key=lambda kv: kv[1])

        emb_layer = model.cat_embs[model_key]
        vocab_size = emb_layer.num_embeddings
        emb_weight = emb_layer.weight.detach().cpu().numpy()

        # Filter valid ids within embedding vocab and cap count
        valid = [(name, idx) for name, idx in items if 0 <= idx < vocab_size]
        if len(valid) == 0:
            print(f"Skip: no valid tokens within vocab for {base_name}")
            continue
        if len(valid) > max_tokens_per_feature:
            valid = valid[:max_tokens_per_feature]

        labels = [name for name, idx in valid]
        rows = [idx for name, idx in valid]
        sub_emb = emb_weight[rows, :]
        cos_mat = _compute_cosine_matrix(sub_emb)

        filename = f"{base_name.replace(' ', '_')}_cosine_heatmap.png"
        save_path = os.path.join(save_dir, filename)
        _plot_cosine_heatmap(
            cos_mat,
            labels,
            title=f"{base_name} Embedding Cosine Heatmap",
            save_path=save_path,
            max_ticks=50,
        )
        print(f"Saved: {save_path}")


def visualize_all_embeddings(model, json_path, save_dir, max_tokens_per_feature=50):
    """Visualize cosine heatmaps for *all* embedding layers in model.cat_embs.

    This includes:
    - Encoded categorical features (e.g. "*_enc")
    - Quantized / discretized numeric features (e.g. "Amount", "account_age_quantized")
    - Special categorical features (e.g. "is_int", "Member Age", "cluster_id")

    Clustering is NOT performed here; we only draw plain cosine similarity heatmaps.
    """
    # Try to load tokenize dict for human-readable labels; fall back to index labels if missing.
    try:
        tok_dict = _load_tokenize_dict(json_path)
    except Exception as e:
        print(f"Failed to load tokenize dict from {json_path}: {e}")
        tok_dict = {}

    if not hasattr(model, 'cat_embs') or model.cat_embs is None or len(model.cat_embs) == 0:
        print("Model has no categorical / quantized embeddings to visualize.")
        return

    os.makedirs(save_dir, exist_ok=True)

    for emb_name, emb_layer in model.cat_embs.items():
        vocab_size = emb_layer.num_embeddings
        if vocab_size == 0:
            print(f"Skip: empty embedding layer {emb_name}")
            continue

        emb_weight = emb_layer.weight.detach().cpu().numpy()

        # Derive base feature name, e.g. "Account Type_enc" -> "Account Type"
        base_name = emb_name[:-4] if emb_name.endswith("_enc") else emb_name

        # Prefer human-readable labels from tokenize_dict.json when available
        labels = None
        rows = None
        filename_base = None
        title = None

        if base_name in tok_dict:
            name_to_id = tok_dict[base_name]
            items = sorted(name_to_id.items(), key=lambda kv: kv[1])
            valid = [(name, idx) for name, idx in items if 0 <= idx < vocab_size]
            if len(valid) == 0:
                print(f"Skip: no valid tokens within vocab for {base_name}")
                continue
            if len(valid) > max_tokens_per_feature:
                valid = valid[:max_tokens_per_feature]
            labels = [name for name, idx in valid]
            rows = [idx for _, idx in valid]
            filename_base = base_name.replace(" ", "_")
            title = f"{base_name} Embedding Cosine Heatmap (all tokens)"
        else:
            # Fall back to index-based labels for embeddings without tokenize_dict entries
            n_tokens = min(vocab_size, max_tokens_per_feature)
            rows = list(range(n_tokens))
            labels = [f"{emb_name}_{i}" for i in rows]
            filename_base = emb_name.replace(" ", "_")
            title = f"{emb_name} Embedding Cosine Heatmap (index labels)"

        sub_emb = emb_weight[rows, :]
        cos_mat = _compute_cosine_matrix(sub_emb)

        # Use different filename suffix to avoid overwriting existing clustered / manual plots.
        filename = f"{filename_base}_all_cosine_heatmap.png"
        save_path = os.path.join(save_dir, filename)
        _plot_cosine_heatmap(
            cos_mat,
            labels,
            title=title,
            save_path=save_path,
            max_ticks=50,
        )
        print(f"Saved: {save_path}")


def plot_clustered_product_id_embedding(
    model, json_path, save_dir, n_clusters=8, max_tokens=50
):
    """
    Perform KMeans clustering on Product ID embedding, and create a heatmap divided by cluster labels, then save the image.
    """
    tok_dict = _load_tokenize_dict(json_path)
    base_name = 'Product ID'
    model_key = f'{base_name}_enc'
    if model_key not in model.cat_embs or base_name not in tok_dict:
        print(f"Cannot find Product ID embedding or this feature is not in tokenize_dict.json.")
        return
    name_to_id = tok_dict[base_name]
    items = sorted(name_to_id.items(), key=lambda kv: kv[1])
    emb_layer = model.cat_embs[model_key]
    vocab_size = emb_layer.num_embeddings
    emb_weight = emb_layer.weight.detach().cpu().numpy()
    valid = [(name, idx) for name, idx in items if 0 <= idx < vocab_size]
    if len(valid) == 0:
        print(f"No valid Product ID tokens")
        return
    if len(valid) > max_tokens:
        valid = valid[:max_tokens]
    labels = [name for name, idx in valid]
    rows = [idx for name, idx in valid]
    sub_emb = emb_weight[rows, :]
    cos_mat = _compute_cosine_matrix(sub_emb)
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(sub_emb)
    # Sort by cluster groups
    sort_idx = np.argsort(clusters)
    cos_sorted = cos_mat[sort_idx][:, sort_idx]
    labels_sorted = [labels[i] for i in sort_idx]
    clusters_sorted = clusters[sort_idx]
    # Plot clustered heatmap
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_sorted, cmap='viridis', vmin=-1.0, vmax=1.0)
    n = len(labels_sorted)
    max_ticks = 50
    if n <= max_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_sorted, fontsize=7)
    else:
        step = max(1, n // max_ticks)
        idxs = np.arange(0, n, step)
        ax.set_xticks(idxs)
        ax.set_yticks(idxs)
        ax.set_xticklabels([labels_sorted[i] for i in idxs], rotation=90, fontsize=7)
        ax.set_yticklabels([labels_sorted[i] for i in idxs], fontsize=7)
    ax.set_title(f'Product ID Embedding Cosine Heatmap (Clustered, K={n_clusters})')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')
    # Draw borders to separate each cluster boundary
    for j in range(1, n):
        if clusters_sorted[j] != clusters_sorted[j-1]:
            ax.axhline(j-0.5, color='red', linewidth=0.5, alpha=0.5)
            ax.axvline(j-0.5, color='red', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'Product_ID_cosine_heatmap_clustered_{n_clusters}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Product ID clustered heatmap saved: {save_path}')


def plot_cluster_id_embedding(model, save_dir, max_tokens=60):
    """
    Plot cosine similarity heatmap for cluster_id embedding.
    
    - model: Trained model containing cluster_id embedding
    - save_dir: Directory to save images
    - max_tokens: Maximum number of tokens to display (default 60, corresponding to vocab_size)
    """
    model_key = 'cluster_id'
    
    if not hasattr(model, 'cat_embs') or model.cat_embs is None:
        print("Model has no categorical embedding layers.")
        return
    
    if model_key not in model.cat_embs:
        print(f"{model_key} embedding not found in model.")
        return
    
    emb_layer = model.cat_embs[model_key]
    vocab_size = emb_layer.num_embeddings
    emb_weight = emb_layer.weight.detach().cpu().numpy()
    
    # Use all valid embeddings (0 to vocab_size-1)
    n_tokens = min(vocab_size, max_tokens)
    rows = list(range(n_tokens))
    labels = [f'cluster_{i}' for i in rows]
    sub_emb = emb_weight[rows, :]
    cos_mat = _compute_cosine_matrix(sub_emb)
    
    # Plot heatmap
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_mat, cmap='viridis', vmin=-1.0, vmax=1.0)
    
    n = len(labels)
    max_ticks = 50
    if n <= max_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    else:
        step = max(1, n // max_ticks)
        idxs = np.arange(0, n, step)
        ax.set_xticks(idxs)
        ax.set_yticks(idxs)
        ax.set_xticklabels([labels[i] for i in idxs], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([labels[i] for i in idxs], fontsize=8)
    
    ax.set_title('Cluster ID Embedding Cosine Heatmap')
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cluster_id_cosine_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Cluster ID heatmap saved: {save_path}')


def plot_clustered_cluster_id_embedding(
    model, save_dir, n_clusters=8, max_tokens=60
):
    """
    Perform KMeans clustering on cluster_id embedding, and create a heatmap divided by cluster labels, then save the image.
    """
    model_key = 'cluster_id'
    
    if not hasattr(model, 'cat_embs') or model.cat_embs is None:
        print("Model has no categorical embedding layers.")
        return
    
    if model_key not in model.cat_embs:
        print(f"{model_key} embedding not found in model.")
        return
    
    emb_layer = model.cat_embs[model_key]
    vocab_size = emb_layer.num_embeddings
    emb_weight = emb_layer.weight.detach().cpu().numpy()
    
    # Use all valid embeddings (0 to vocab_size-1)
    n_tokens = min(vocab_size, max_tokens)
    rows = list(range(n_tokens))
    labels = [f'cluster_{i}' for i in rows]
    sub_emb = emb_weight[rows, :]
    cos_mat = _compute_cosine_matrix(sub_emb)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(sub_emb)
    
    # Sort by cluster groups
    sort_idx = np.argsort(clusters)
    cos_sorted = cos_mat[sort_idx][:, sort_idx]
    labels_sorted = [labels[i] for i in sort_idx]
    clusters_sorted = clusters[sort_idx]
    
    # Plot clustered heatmap
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cos_sorted, cmap='viridis', vmin=-1.0, vmax=1.0)
    
    n = len(labels_sorted)
    max_ticks = 50
    if n <= max_ticks:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels_sorted, rotation=90, fontsize=7)
        ax.set_yticklabels(labels_sorted, fontsize=7)
    else:
        step = max(1, n // max_ticks)
        idxs = np.arange(0, n, step)
        ax.set_xticks(idxs)
        ax.set_yticks(idxs)
        ax.set_xticklabels([labels_sorted[i] for i in idxs], rotation=90, fontsize=7)
        ax.set_yticklabels([labels_sorted[i] for i in idxs], fontsize=7)
    
    ax.set_title(f'Cluster ID Embedding Cosine Heatmap (Clustered, K={n_clusters})')
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')
    
    # Draw borders to separate each cluster boundary
    for j in range(1, n):
        if clusters_sorted[j] != clusters_sorted[j-1]:
            ax.axhline(j-0.5, color='red', linewidth=0.5, alpha=0.5)
            ax.axvline(j-0.5, color='red', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cluster_id_cosine_heatmap_clustered_{n_clusters}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Cluster ID clustered heatmap saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Model file path (e.g., checkpoints/best_model.pth)")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="final/matched", 
                       help="Test data directory")
    parser.add_argument("--max_len", type=int, default=50, 
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size")
    
    # Output parameters
    parser.add_argument("--save_dir", type=str, default="evaluation_results", 
                       help="Results save directory")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    parser.add_argument('--product_id_clusters', type=int, default=8, help='Number of KMeans clusters for Product ID embedding')
    parser.add_argument('--cluster_id_clusters', type=int, default=8, help='Number of KMeans clusters for Cluster ID embedding')
    
    # Visualization parameters
    parser.add_argument('--plot_hidden', action='store_true', 
                       help='Whether to plot PCA visualization of hidden states')
    parser.add_argument('--plot_emb', action='store_true', 
                       help='Whether to plot embedding visualizations')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")
    
    # Load model
    model, model_args = load_model(args.model_path, device)

    # Embedding visualization
    if args.plot_emb:
        try:
            emb_save_dir = os.path.join(args.save_dir, 'embeddings')
            # Existing visualizations (with clustering where applicable)
            visualize_cat_embeddings(model, json_path='config/tokenize_dict.json', save_dir=emb_save_dir)
            plot_clustered_product_id_embedding(
                model, 'config/tokenize_dict.json', emb_save_dir, n_clusters=args.product_id_clusters
            )
            plot_cluster_id_embedding(model, save_dir=emb_save_dir)
            plot_clustered_cluster_id_embedding(
                model, emb_save_dir, n_clusters=args.cluster_id_clusters
            )
            # New: visualize ALL embedding layers without clustering.
            # This will create additional heatmaps for every embedding in model.cat_embs,
            # including discretized numeric features such as "Amount" and "account_age_quantized".
            visualize_all_embeddings(model, json_path='config/tokenize_dict.json', save_dir=emb_save_dir)
        except Exception as e:
            print(f"Embedding visualization failed: {e}")
    else:
        print("Skipping embedding visualization (use --plot_emb to enable)")
    
    # Create test data loader
    print("Loading test data...")
    test_loader, feature_names = create_dataloader(
        matched_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=False,
        mode="train",
        split=(0.8, 0.1, 0.1),  # Use same split ratio as training
        seed=args.seed,
        drop_all_zero_batches=False
    )
    
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of features: {len(feature_names)}")
    
    # Get target feature indices
    target_names = getattr(model_args, 'target_names', [
        'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
        'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
    ])
    
    # Ensure target features are in feature list
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_indices = [name_to_idx[n] for n in target_names if n in name_to_idx]
    resolved_target_names = [feature_names[i] for i in target_indices]
    
    print(f"Target features: {resolved_target_names}")
    print(f"Target feature indices: {target_indices}")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, feature_names, target_indices, model_args)
    
    # Analyze prediction results
    analysis = analyze_predictions(results, resolved_target_names)
    
    # Plot evaluation results
    plot_evaluation_results(results, analysis, resolved_target_names, args.save_dir)
    
    # PCA visualization of hidden states
    if args.plot_hidden:
        try:
            plot_pca_visualization(results, args.save_dir, n_components=2)
        except Exception as e:
            print(f"PCA visualization failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping hidden state PCA visualization (use --plot_hidden to enable)")
    
    # Save evaluation results
    save_evaluation_results(results, analysis, resolved_target_names, args.save_dir, model)
    
    print("\nModel evaluation completed!")


if __name__ == "__main__":
    main()