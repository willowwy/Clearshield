'''
User guide: 

run

python train.py train_judge.py --sequence_model_path checkpoints/best_model_enc.pth                                 

after running train.py

then trained model will be stored in /checkpoints/best_judge_model.pth
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
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)

from src.models.backbone_model import build_arg_parser, build_seq_model
from src.models.datasets import create_dataloader
from src.models.judge import FraudJudgeDNN, FraudJudgeTrainer, build_judge_model
from src.models.loss import build_loss_from_args
from src.models.load_model import load_seq_model, load_judge_model, count_parameters





def resolve_target_indices(feature_names, model_args):
    """Resolve target feature indices"""
    # Custom target feature list - can be modified as needed
    # target_names = [
    #     'Product ID_enc', 'Amount', 'Action Type_enc', 'Source T',
    #     'is_int',
    #     'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
    # ]
    target_names = ['Amount', 'Action Type_enc', 'Source Type_enc']
    
    print(f"Using custom target features: {target_names}")
    
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_indices = [name_to_idx[n] for n in target_names if n in name_to_idx]
    resolved_target_names = [feature_names[i] for i in target_indices]
    return target_indices, resolved_target_names, target_names


def extract_judge_training_data(sequence_model, dataloader, device, feature_names, target_indices, target_names, use_pred=False, max_len=50):
    """
    Extract training data for judge model using sequence model predictions and hidden states
    
    Args:
        use_pred: If True, also extract predictions; if False, only use hidden representation
    
    Returns:
        predictions: [N, pred_dim] - sequence model predictions (None if use_pred=False)
        targets: [N, pred_dim] - ground truth targets  
        hidden_states: [N, hidden_dim] - hidden representations
        labels: [N] - fraud labels (0/1)
    """
    sequence_model.eval()
    
    all_predictions = []
    all_targets = []
    all_hidden_states = []
    all_labels = []
    
    print("Extracting training data for judge model...")
    print(f"Using predictions: {use_pred}")
    
    with torch.no_grad():
        for batch_idx, (X, y, mask) in enumerate(tqdm(dataloader, desc="Extracting data")):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Get sequence model predictions and hidden states
            preds, hidden_state = sequence_model(X, mask, feature_names)
            
            # Align predictions with next-step targets
            pred_steps = preds[:, :-1, :]  # [B, T-1, full_pred_dim]
            target_steps = X[:, 1:, target_indices]  # [B, T-1, selected_pred_dim]
            
            # Only use selected feature dimensions for prediction
            # Assume sequence model prediction order matches target_names order
            model_target_names = getattr(sequence_model, 'target_names', None)
            if model_target_names is None:
                # If no target_names attribute, use default order
                model_target_names = [
                    'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
                    'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
                ]
            
            # Find indices of selected features in model predictions
            model_name_to_idx = {n: i for i, n in enumerate(model_target_names)}
            selected_pred_indices = [model_name_to_idx[n] for n in target_names if n in model_name_to_idx]
            
            # Only select corresponding prediction dimensions
            pred_steps = pred_steps[:, :, selected_pred_indices]  # [B, T-1, selected_pred_dim]
            valid_steps = mask[:, 1:]  # [B, T-1]
            label_steps = y[:, 1:]  # [B, T-1]
            
            # Process hidden state for sequence-based CNN processing
            # For each time step, extract sequence up to that point (or last max_len steps)
            # For LSTM: hidden_state is (h_n, c_n) where h_n: [num_layers * num_directions, B, hidden_size]
            # For Transformer: hidden_state is [B, T, transformer_dim]
            if isinstance(hidden_state, tuple):
                # LSTM: take last layer's hidden state and expand to sequence
                h_n, c_n = hidden_state
                hidden_rep = h_n[-1]  # [B, hidden_size]
                # For LSTM, we don't have sequence, so use the same hidden state for all time steps
                # Expand to [B, T-1, hidden_size] - each time step uses the same hidden state
                hidden_sequences = hidden_rep.unsqueeze(1).expand(-1, pred_steps.shape[1], -1)  # [B, T-1, hidden_size]
                # For LSTM, we'll use the full sequence (all same values) for each step
                full_hidden_seq = hidden_rep.unsqueeze(1).expand(-1, max_len, -1)  # [B, max_len, hidden_size]
            else:
                # Transformer: hidden_state is [B, T, transformer_dim]
                # Use the full sequence (will be truncated/padded in judge model)
                hidden_sequences = hidden_state[:, :-1, :]  # [B, T-1, transformer_dim]
                full_hidden_seq = hidden_state  # [B, T, transformer_dim]
            
            # Only keep valid steps
            valid_mask = valid_steps == 1
            if valid_mask.any():
                # For each valid time step, use the full sequence up to that point
                # Pad or truncate to max_len (will be handled in judge model during forward pass)
                batch_size = hidden_sequences.shape[0]
                seq_len = hidden_sequences.shape[1]
                
                target_valid_list = []
                label_valid_list = []
                hidden_valid_list = []
                pred_valid_list = []
                
                for b in range(batch_size):
                    for t in range(seq_len):
                        if valid_mask[b, t]:
                            # For step t, use sequence from beginning to t+1 (or last max_len steps)
                            if isinstance(hidden_state, tuple):
                                # LSTM: use full sequence (all same values)
                                seq_to_use = full_hidden_seq[b].cpu().numpy()  # [max_len, hidden_dim]
                            else:
                                # Transformer: use sequence up to t+1, then pad/truncate
                                seq_end = min(t + 1, full_hidden_seq.shape[1])
                                seq_to_use = full_hidden_seq[b, :seq_end, :].cpu().numpy()  # [seq_end, hidden_dim]
                                # Pad or truncate to max_len
                                if seq_to_use.shape[0] < max_len:
                                    # Pad with zeros at the beginning
                                    pad_len = max_len - seq_to_use.shape[0]
                                    padding = np.zeros((pad_len, seq_to_use.shape[1]), dtype=seq_to_use.dtype)
                                    seq_to_use = np.vstack([padding, seq_to_use])  # [max_len, hidden_dim]
                                elif seq_to_use.shape[0] > max_len:
                                    # Truncate to last max_len steps
                                    seq_to_use = seq_to_use[-max_len:]  # [max_len, hidden_dim]
                            
                            target_valid_list.append(target_steps[b, t].cpu().numpy())  # [pred_dim]
                            label_valid_list.append(label_steps[b, t].cpu().numpy())  # scalar
                            hidden_valid_list.append(seq_to_use)  # [max_len, hidden_dim]
                            
                            if use_pred:
                                pred_valid_list.append(pred_steps[b, t].cpu().numpy())  # [pred_dim]
                
                if len(target_valid_list) > 0:
                    target_valid = np.array(target_valid_list)  # [N, pred_dim]
                    label_valid = np.array(label_valid_list)  # [N]
                    hidden_valid = np.array(hidden_valid_list)  # [N, max_len, hidden_dim]
                    
                    if use_pred:
                        pred_valid = np.array(pred_valid_list)  # [N, pred_dim]
                        all_predictions.append(pred_valid)
                    
                    all_targets.append(target_valid)
                    all_hidden_states.append(hidden_valid)
                    all_labels.append(label_valid)
    
    # Concatenate all data
    if all_targets:
        targets = np.concatenate(all_targets, axis=0)
        hidden_states = np.concatenate(all_hidden_states, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        if use_pred and all_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
        else:
            predictions = None
        
        print(f"Extracted {len(targets)} samples for judge training")
        print(f"Fraud rate: {np.mean(labels):.4f}")
        print(f"Hidden state shape: {hidden_states.shape}")
        if predictions is not None:
            print(f"Predictions shape: {predictions.shape}")
        
        return predictions, targets, hidden_states, labels
    else:
        print("No valid data extracted!")
        return None, None, None, None


def filter_fraud_batches(dataloader, min_fraud_rate=0.02):
    """
    Filter dataloader to only keep batches with fraud samples
    
    Args:
        dataloader: Original dataloader
        min_fraud_rate: Minimum fraud rate to keep batch
    
    Returns:
        Filtered batches as list of tuples
    """
    fraud_batches = []
    
    print("Filtering batches with fraud samples...")
    
    for batch_idx, (X, y, mask) in enumerate(tqdm(dataloader, desc="Filtering batches")):
        # Calculate fraud rate in this batch
        valid_mask = mask == 1
        if valid_mask.any():
            valid_y = y[valid_mask]
            fraud_rate = (valid_y == 1).float().mean().item()
            
            if fraud_rate >= min_fraud_rate:
                fraud_batches.append((X, y, mask))
    
    print(f"Kept {len(fraud_batches)} batches with fraud rate >= {min_fraud_rate}")
    return fraud_batches


def create_judge_dataset(predictions, targets, hidden_states, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train/val/test splits for judge model"""
    np.random.seed(seed)
    n_samples = len(targets)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'train': (predictions[train_idx] if predictions is not None else None, 
                  targets[train_idx], 
                  hidden_states[train_idx],
                  labels[train_idx]),
        'val': (predictions[val_idx] if predictions is not None else None, 
                targets[val_idx], 
                hidden_states[val_idx],
                labels[val_idx]),
        'test': (predictions[test_idx] if predictions is not None else None, 
                 targets[test_idx], 
                 hidden_states[test_idx],
                 labels[test_idx])
    }


def train_judge_model(judge_model, train_data, val_data, device, args, test_data=None):
    """Train the judge model"""
    # Build custom loss function if specified
    criterion = build_loss_from_args(args)
    trainer = FraudJudgeTrainer(judge_model, device, criterion)
    trainer.setup_optimizer(lr=args.judge_lr, weight_decay=args.judge_weight_decay)
    
    # Setup learning rate scheduler if specified
    if hasattr(args, 'judge_scheduler') and args.judge_scheduler is not None:
        scheduler_kwargs = {}
        if args.judge_scheduler == 'step':
            scheduler_kwargs = {
                'step_size': getattr(args, 'judge_scheduler_step_size', 30),
                'gamma': getattr(args, 'judge_scheduler_gamma', 0.1)
            }
        elif args.judge_scheduler == 'plateau':
            scheduler_kwargs = {
                'mode': 'max',  # Monitor validation accuracy
                'factor': getattr(args, 'judge_scheduler_factor', 0.5),
                'patience': getattr(args, 'judge_scheduler_patience', 10),
                'verbose': True
            }
        elif args.judge_scheduler == 'cosine':
            scheduler_kwargs = {
                'T_max': getattr(args, 'judge_scheduler_T_max', args.judge_epochs),
                'eta_min': getattr(args, 'judge_scheduler_eta_min', 0.0)
            }
        elif args.judge_scheduler == 'exponential':
            scheduler_kwargs = {
                'gamma': getattr(args, 'judge_scheduler_gamma', 0.95)
            }
        
        trainer.setup_scheduler(args.judge_scheduler, **scheduler_kwargs)
        print(f"Using {args.judge_scheduler} learning rate scheduler with initial LR={trainer.get_lr():.6f}")
    else:
        print(f"No learning rate scheduler (initial LR={trainer.get_lr():.6f})")
    
    train_pred, train_target, train_hidden, train_labels = train_data
    val_pred, val_target, val_hidden, val_labels = val_data
    
    # Convert to tensors
    train_target = torch.tensor(train_target, dtype=torch.float32).to(device)
    train_hidden = torch.tensor(train_hidden, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    val_target = torch.tensor(val_target, dtype=torch.float32).to(device)
    val_hidden = torch.tensor(val_hidden, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    if train_pred is not None:
        train_pred = torch.tensor(train_pred, dtype=torch.float32).to(device)
    if val_pred is not None:
        val_pred = torch.tensor(val_pred, dtype=torch.float32).to(device)
    
    # Prepare test data if available
    test_target = None
    test_hidden = None
    test_labels = None
    test_pred = None
    if test_data is not None:
        test_pred, test_target, test_hidden, test_labels = test_data
        test_target = torch.tensor(test_target, dtype=torch.float32).to(device)
        test_hidden = torch.tensor(test_hidden, dtype=torch.float32).to(device)
        test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
        if test_pred is not None:
            test_pred = torch.tensor(test_pred, dtype=torch.float32).to(device)
    
    # Training loop
    best_val_pr_auc = 0.0
    train_losses = []
    train_pr_aucs = []
    val_losses = []
    val_pr_aucs = []
    test_losses = []
    
    print(f"Training judge model for {args.judge_epochs} epochs...")
    
    for epoch in range(args.judge_epochs):
        # Training
        judge_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        batch_size = args.judge_batch_size
        n_train = len(train_target)
        
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            
            batch_target = train_target[i:end_idx]
            batch_hidden = train_hidden[i:end_idx]
            batch_labels = train_labels[i:end_idx]
            batch_pred = train_pred[i:end_idx] if train_pred is not None else None
            
            # Create mask for consistency with inference logic
            # Training data is already fixed length [B, max_len, hidden_dim], so use all-ones mask
            batch_mask = torch.ones(batch_hidden.shape[0], batch_hidden.shape[1], dtype=torch.float32).to(device)
            loss = trainer.train_step(batch_pred, batch_target, batch_labels, batch_hidden, batch_mask)
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # Evaluate on training set for PR-AUC
        judge_model.eval()
        train_mask = torch.ones(train_hidden.shape[0], train_hidden.shape[1], dtype=torch.float32).to(device)
        train_results = trainer.evaluate(train_pred, train_target, train_labels, train_hidden, train_mask)
        train_probs = train_results['probabilities']
        train_labels_np = train_labels.cpu().numpy()
        if train_probs.shape[1] > 1:
            train_positive_probs = train_probs[:, 1]
        else:
            train_positive_probs = train_probs[:, 0]
        if len(np.unique(train_labels_np)) > 1:
            train_pr_auc = average_precision_score(train_labels_np, train_positive_probs)
        else:
            train_pr_auc = 0.0
        train_pr_aucs.append(train_pr_auc)
        
        # Validation
        # Create mask for consistency with inference logic
        # Validation data is already fixed length [N, max_len, hidden_dim], so use all-ones mask
        val_mask = torch.ones(val_hidden.shape[0], val_hidden.shape[1], dtype=torch.float32).to(device)
        val_results = trainer.evaluate(val_pred, val_target, val_labels, val_hidden, val_mask)
        val_losses.append(val_results['loss'])
        val_probs = val_results['probabilities']
        val_labels_np = val_labels.cpu().numpy()
        if val_probs.shape[1] > 1:
            positive_probs = val_probs[:, 1]
        else:
            positive_probs = val_probs[:, 0]
        if len(np.unique(val_labels_np)) > 1:
            val_pr_auc = average_precision_score(val_labels_np, positive_probs)
        else:
            val_pr_auc = 0.0
        val_pr_aucs.append(val_pr_auc)
        
        # Evaluate on test set for loss
        if test_data is not None:
            test_mask = torch.ones(test_hidden.shape[0], test_hidden.shape[1], dtype=torch.float32).to(device)
            test_results = trainer.evaluate(test_pred, test_target, test_labels, test_hidden, test_mask)
            test_losses.append(test_results['loss'])
        
        # Step learning rate scheduler
        if hasattr(args, 'judge_scheduler') and args.judge_scheduler == 'plateau':
            trainer.step_scheduler(metric=val_pr_auc)
        elif trainer.scheduler is not None:
            trainer.step_scheduler()
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': judge_model.state_dict(),
                'val_pr_auc': val_pr_auc,
                'args': args
            }, f'checkpoints-ftenc/best_judge_model.pth')
        
        if epoch % 10 == 0:
            current_lr = trainer.get_lr()
            test_loss_str = f", Test Loss={test_losses[-1]:.4f}" if test_data is not None else ""
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Train PR-AUC={train_pr_auc:.4f}, Val PR-AUC={val_pr_auc:.4f}{test_loss_str}, LR={current_lr:.6f}")
    
    print(f"Best validation PR-AUC: {best_val_pr_auc:.4f}")
    
    return {
        'train_losses': train_losses,
        'train_pr_aucs': train_pr_aucs,
        'val_losses': val_losses,
        'val_pr_aucs': val_pr_aucs,
        'test_losses': test_losses if test_data is not None else [],
        'best_val_pr_auc': best_val_pr_auc
    }





def evaluate_judge_model(judge_model, test_data, device, hist_path=None):
    """Evaluate the judge model"""
    test_pred, test_target, test_hidden, test_labels = test_data
    
    # Convert to tensors
    test_target = torch.tensor(test_target, dtype=torch.float32).to(device)
    test_hidden = torch.tensor(test_hidden, dtype=torch.float32).to(device)  # [N, max_len, hidden_dim]
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
    if test_pred is not None:
        test_pred = torch.tensor(test_pred, dtype=torch.float32).to(device)
    
    # Debug: print shapes
    print(f"Debug - test_hidden shape: {test_hidden.shape}")
    print(f"Debug - test_target shape: {test_target.shape}")
    if test_pred is not None:
        print(f"Debug - test_pred shape: {test_pred.shape}")
    else:
        print(f"Debug - test_pred is None")
    
    # Evaluate
    judge_model.eval()
    with torch.no_grad():
        # test_hidden should be [N, max_len, hidden_dim] for CNN processing
        # Create mask for consistency with inference logic
        # Test data is already fixed length [N, max_len, hidden_dim], so use all-ones mask
        batch_size = test_hidden.shape[0]
        mask_for_judge = torch.ones(batch_size, test_hidden.shape[1], dtype=torch.float32).to(device)
        logits = judge_model(test_pred, test_target, test_hidden, mask_for_judge)
        probs = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(logits, dim=1)
        if probs.shape[1] > 1:
            positive_probs = probs[:, 1]
        else:
            positive_probs = probs[:, 0]
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels.cpu().numpy(), pred_labels.cpu().numpy())
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary'
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(test_labels.cpu().numpy(), positive_probs.cpu().numpy())
        except:
            auc = 0.5
        
        # Calculate PR-AUC (Average Precision)
        try:
            pr_auc = average_precision_score(test_labels.cpu().numpy(), positive_probs.cpu().numpy())
        except ValueError:
            pr_auc = 0.0
        
        print(f"\nJudge Model Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        
        # Plot probability histograms if requested
        if hist_path is not None:
            normal_probs = positive_probs[test_labels == 0].detach().cpu().numpy()
            fraud_probs = positive_probs[test_labels == 1].detach().cpu().numpy()
            
            base_dir = os.path.dirname(hist_path)
            base_name = os.path.splitext(os.path.basename(hist_path))[0]
            os.makedirs(base_dir, exist_ok=True)
            
            normal_path = os.path.join(base_dir, f"{base_name}_normal.png")
            fraud_path = os.path.join(base_dir, f"{base_name}_fraud.png")
            
            plt.figure(figsize=(6, 4))
            plt.hist(normal_probs, bins=20, alpha=0.8, color='tab:blue')
            plt.xlabel('Positive class probability')
            plt.ylabel('Count')
            plt.title('Normal probability distribution')
            plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(normal_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(6, 4))
            plt.hist(fraud_probs, bins=20, alpha=0.8, color='tab:orange')
            plt.xlabel('Positive class probability')
            plt.ylabel('Count')
            plt.title('Fraud probability distribution')
            plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(fraud_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Probability histograms saved to:\n  Normal: {normal_path}\n  Fraud:  {fraud_path}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'pr_auc': pr_auc
        }


def plot_training_curves(train_losses, train_pr_aucs, val_losses, val_pr_aucs, test_losses, save_dir):
    """Plot training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Train Loss and Val Loss
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2, linestyle='-')
    ax1.plot(val_losses, label='Val Loss', color='red', linewidth=2, linestyle='-')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Judge Model Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: PR-AUC curves
    ax2.plot(train_pr_aucs, label='Train PR-AUC', color='green', linewidth=2, linestyle='-')
    ax2.plot(val_pr_aucs, label='Val PR-AUC', color='orange', linewidth=2, linestyle='-')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Judge Model PR-AUC')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'judge_training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Train fraud judge model")
    
    # Required parameters
    parser.add_argument("--sequence_model_path", type=str, required=True,
                       help="Path to trained sequence model (e.g., checkpoints/best_model.pth)")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/final/matched",
                       help="Data directory")
    parser.add_argument("--max_len", type=int, default=50,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for sequence model")
    parser.add_argument("--min_fraud_rate", type=float, default=0.01,
                       help="Minimum fraud rate to keep batch")
    
    # Judge model parameters
    parser.add_argument("--judge_hidden_dims", type=int, nargs="+", default=[64, 32, 16],
                       help="Judge model hidden layer dimensions")
    parser.add_argument("--judge_dropout", type=float, default=0.2,
                       help="Judge model dropout rate")
    parser.add_argument("--judge_use_attention", action="store_true",
                       help="Use attention mechanism in judge model")
    
    # Loss function parameters for recall optimization
    parser.add_argument("--loss_type", type=str, default="focal",
                       choices=["cross_entropy", "focal", "weighted", "recall_focused", "adaptive", "hinge"],
                       help="Loss function type for recall optimization")
    parser.add_argument("--focal_alpha", type=float, default=0.9,
                       help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=1.5,
                       help="Focal loss gamma parameter")
    parser.add_argument("--class_weights", type=str, default="1.0,2.0",
                       help="Class weights for weighted loss (normal,fraud)")
    parser.add_argument("--recall_weight", type=float, default=0.5,
                       help="Weight for recall penalty in recall_focused loss")
    parser.add_argument("--target_recall", type=float, default=0.8,
                       help="Target recall rate for adaptive loss")
    parser.add_argument("--adaptation_rate", type=float, default=0.1,
                       help="Adaptation rate for adaptive loss")
    parser.add_argument("--hinge_margin", type=float, default=1.0,
                       help="Margin parameter for hinge loss")
    
    # Training parameters
    parser.add_argument("--judge_lr", type=float, default=1e-4,
                       help="Judge model learning rate")
    parser.add_argument("--judge_weight_decay", type=float, default=1e-4,
                       help="Judge model weight decay")
    parser.add_argument("--judge_batch_size", type=int, default=128,
                       help="Judge model batch size")
    parser.add_argument("--judge_epochs", type=int, default=80,
                       help="Judge model training epochs")
    
    # Learning rate scheduler parameters
    parser.add_argument("--judge_scheduler", type=str, default='cosine',
                       choices=['step', 'plateau', 'cosine', 'exponential', None],
                       help="Learning rate scheduler type (None to disable)")
    parser.add_argument("--judge_scheduler_step_size", type=int, default=30,
                       help="Step size for StepLR scheduler")
    parser.add_argument("--judge_scheduler_gamma", type=float, default=0.1,
                       help="Gamma (decay factor) for StepLR/ExponentialLR scheduler")
    parser.add_argument("--judge_scheduler_patience", type=int, default=10,
                       help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--judge_scheduler_factor", type=float, default=0.5,
                       help="Factor for ReduceLROnPlateau scheduler")
    parser.add_argument("--judge_scheduler_T_max", type=int, default=100,
                       help="T_max for CosineAnnealingLR scheduler")
    parser.add_argument("--judge_scheduler_eta_min", type=float, default=0.0,
                       help="Eta_min for CosineAnnealingLR scheduler")
    
    # Data split
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Training data ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="Validation data ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Test data ratio")
    
    # Judge model loading (resume training or test-only)
    parser.add_argument("--judge_model_path", type=str, default=None,
                       help="Path to trained judge model checkpoint (for resume training or test-only)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--save_dir", type=str, default="judge_results",
                       help="Results save directory")
    parser.add_argument("--no-stat", action="store_true",
                       help="Train judge model without statistical features (only predictions, targets, and errors)")
    parser.add_argument("--stat-only", action="store_true",
                       help="Train judge model using only statistical features (MSE, MAE, max_error, min_error, std_error, mean_error)")
    parser.add_argument("--use_pred", action="store_true",
                       help="Use predictions in addition to hidden representation (default: only use hidden representation)")
    
    # Sliding window parameters
    parser.add_argument("--use_sliding_window", action="store_true",
                       help="Use sliding window to increase data volume")
    parser.add_argument("--window_overlap", type=float, default=0.5,
                       help="Sliding window overlap ratio (0.0-0.9)")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load trained sequence model
    sequence_model, model_args = load_seq_model(args.sequence_model_path, device)
    
    # Check if only testing (train_ratio == 0 and val_ratio == 0)
    test_only = (args.train_ratio == 0.0 and args.val_ratio == 0.0)
    
    if test_only:
        if args.judge_model_path is None:
            parser.error("--judge_model_path is required when train_ratio=0 and val_ratio=0 (test only mode)")
        print("Test only mode: will load judge model and evaluate on test data")
    
    # Create data loader
    print("Loading data...")
    # Use test mode if only testing
    data_mode = "test" if test_only else "train"
    dataloader, feature_names = create_dataloader(
        matched_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=False,
        mode=data_mode,
        split=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        drop_all_zero_batches=False,
        use_sliding_window=getattr(args, 'use_sliding_window', False),
        window_overlap=getattr(args, 'window_overlap', 0.5)
    )
    
    print(f"Number of batches: {len(dataloader)}")
    if feature_names is None:
        print("No features inferred (no files loaded). Please check --data_dir and split ratios.")
        return
    print(f"Number of features: {len(feature_names)}")
    
    # Resolve target indices
    target_indices, resolved_target_names, target_names = resolve_target_indices(feature_names, model_args)
    print(f"Target features: {resolved_target_names}")
    print(f"Target indices: {target_indices}")
    
    # Filter batches with fraud samples
    fraud_batches = filter_fraud_batches(dataloader, args.min_fraud_rate)
    
    if len(fraud_batches) == 0:
        print("No batches with fraud samples found!")
        return
    
    # Create temporary dataloader from fraud batches
    class FraudBatchDataset:
        def __init__(self, batches):
            self.batches = batches
        def __len__(self):
            return len(self.batches)
        def __getitem__(self, idx):
            return self.batches[idx]
    
    fraud_dataset = FraudBatchDataset(fraud_batches)
    fraud_dataloader = torch.utils.data.DataLoader(fraud_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    # Extract training data using sequence model
    predictions, targets, hidden_states, labels = extract_judge_training_data(
        sequence_model, fraud_dataloader, device, feature_names, target_indices, target_names, 
        use_pred=args.use_pred, max_len=args.max_len
    )
    
    if targets is None:
        print("No valid training data extracted!")
        return
    
    # Create train/val/test splits
    dataset_splits = create_judge_dataset(
        predictions, targets, hidden_states, labels,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Dataset splits:")
    for split_name, (pred, target, hidden, label) in dataset_splits.items():
        if len(target) > 0:
            fraud_rate = np.mean(label)
            print(f"  {split_name}: {len(target)} samples, fraud rate: {fraud_rate:.4f}")
    
    # Determine hidden_dim from sequence model
    # For LSTM: hidden_dim = lstm_hidden * (2 if bidirectional else 1)
    # For Transformer: hidden_dim = transformer_dim
    model_type = getattr(model_args, 'model_type', 'lstm').lower()
    if model_type == 'lstm':
        lstm_hidden = getattr(model_args, 'lstm_hidden', 128)
        bidirectional = getattr(model_args, 'bidirectional', False)
        hidden_dim = lstm_hidden * (2 if bidirectional else 1)
    else:
        # Transformer models
        hidden_dim = getattr(model_args, 'transformer_dim', 128)
    
    print(f"Sequence model hidden dimension: {hidden_dim}")
    
    # Build or load judge model
    if args.judge_model_path is not None:
        # Load existing judge model
        # Pass pred_dim based on actual target_indices to ensure consistency
        # But load_judge_model can also infer from checkpoint, so pred_dim is optional
        judge_model, judge_checkpoint_args = load_judge_model(
            args.judge_model_path, device, pred_dim=len(target_indices)
        )
        # Update args with checkpoint args for model configuration
        for key, value in vars(judge_checkpoint_args).items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
        
        # If loaded model requires predictions but we didn't extract them, re-extract with predictions
        if getattr(judge_checkpoint_args, 'use_pred', False) and not args.use_pred:
            print(f"Warning: Loaded model requires predictions (use_pred=True), but data was extracted with use_pred=False.")
            print(f"Re-extracting data with use_pred=True...")
            args.use_pred = True
            predictions, targets, hidden_states, labels = extract_judge_training_data(
                sequence_model, fraud_dataloader, device, feature_names, target_indices, target_names, 
                use_pred=True, max_len=args.max_len
            )
            if targets is None:
                print("No valid training data extracted after re-extraction!")
                return
            
            # Recreate dataset splits with new data
            dataset_splits = create_judge_dataset(
                predictions, targets, hidden_states, labels,
                args.train_ratio, args.val_ratio, args.test_ratio, args.seed
            )
            
            print(f"Re-extracted dataset splits:")
            for split_name, (pred, target, hidden, label) in dataset_splits.items():
                if len(target) > 0:
                    fraud_rate = np.mean(label)
                    print(f"  {split_name}: {len(target)} samples, fraud rate: {fraud_rate:.4f}")
        # Set mode based on whether we're testing or training
        if test_only:
            judge_model.eval()
        else:
            judge_model.train()  # Continue training
    else:
        # Build new judge model
        pred_dim = len(target_indices)
        
        # Determine feature configuration
        use_statistical_features = not args.no_stat
        use_basic_features = args.use_pred and not args.stat_only
        
        print(f"Judge model configuration:")
        print(f"  Use predictions: {args.use_pred}")
        print(f"  Use hidden representation: True (default)")
        print(f"  Use statistical features: {use_statistical_features}")
        print(f"  Use basic features: {use_basic_features}")
        print(f"  Hidden dimension: {hidden_dim}")
        
        judge_model = build_judge_model(
            pred_dim=pred_dim,
            hidden_dims=args.judge_hidden_dims,
            dropout=args.judge_dropout,
            use_attention=args.judge_use_attention,
            use_statistical_features=use_statistical_features,
            use_basic_features=use_basic_features,
            hidden_dim=hidden_dim,
            use_pred=args.use_pred,
            max_len=getattr(args, 'hidden_len', 3),  # Use the same max_len as training data extraction
            cnn_out_channels=getattr(args, 'judge_cnn_out_channels', 16),
            cnn_kernel_sizes=getattr(args, 'judge_cnn_kernel_sizes', [1, 3])
        ).to(device)
        
        # Count and display judge model parameters
        total_params, trainable_params = count_parameters(judge_model)
        print(f"Judge model parameters:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    # Train or test based on split ratios
    if test_only:
        # Only test mode
        if 'test' not in dataset_splits or len(dataset_splits['test'][1]) == 0:
            print("No test data available!")
            return
        
        print("\nTesting judge model...")
        hist_path = os.path.join(args.save_dir, 'judge_test_prob_hist.png')
        test_results = evaluate_judge_model(judge_model, dataset_splits['test'], device, hist_path=hist_path)
        
        # Save test results
        results = {
            'test_results': test_results,
            'test_config': {
                'num_samples': len(dataset_splits['test'][1]),
                'fraud_rate': float(np.mean(dataset_splits['test'][3])),
                'target_features': resolved_target_names
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(args.save_dir, 'judge_test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Test results saved to: {results_path}")
        print("\nJudge model testing completed!")
        return
    
    # Training mode
    if 'train' not in dataset_splits or len(dataset_splits['train'][1]) == 0:
        print("No training data available!")
        return
    
    if 'val' not in dataset_splits or len(dataset_splits['val'][1]) == 0:
        print("Warning: No validation data available! Using training data for validation.")
        dataset_splits['val'] = dataset_splits['train']
    
    # Prepare test data if available
    test_data_for_training = None
    if 'test' in dataset_splits and len(dataset_splits['test'][1]) > 0:
        test_data_for_training = dataset_splits['test']
    
    # Train judge model
    training_results = train_judge_model(
        judge_model, dataset_splits['train'], dataset_splits['val'], device, args, test_data_for_training
    )
    
    # Plot training curves
    plot_training_curves(
        training_results['train_losses'],
        training_results['train_pr_aucs'],
        training_results['val_losses'],
        training_results['val_pr_aucs'],
        training_results['test_losses'],
        args.save_dir
    )
    
    # Evaluate on test set if available
    test_results = None
    if 'test' in dataset_splits and len(dataset_splits['test'][1]) > 0:
        hist_path = os.path.join(args.save_dir, 'judge_test_prob_hist.png')
        test_results = evaluate_judge_model(judge_model, dataset_splits['test'], device, hist_path=hist_path)
    
    # Get model parameters for saving
    sequence_total_params, sequence_trainable_params = count_parameters(sequence_model)
    judge_total_params, judge_trainable_params = count_parameters(judge_model)
    
    # Save results
    results = {
        'training_results': training_results,
        'test_results': test_results,
        'model_args': vars(args),
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'sequence_model': {
                'total_parameters': sequence_total_params,
                'trainable_parameters': sequence_trainable_params,
                'non_trainable_parameters': sequence_total_params - sequence_trainable_params,
                'model_size_mb': sequence_total_params * 4 / 1024 / 1024
            },
            'judge_model': {
                'total_parameters': judge_total_params,
                'trainable_parameters': judge_trainable_params,
                'non_trainable_parameters': judge_total_params - judge_trainable_params,
                'model_size_mb': judge_total_params * 4 / 1024 / 1024
            }
        }
    }
    
    results_path = os.path.join(args.save_dir, 'judge_training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {results_path}")
    print("\nJudge model training completed!")


if __name__ == "__main__":
    main()
