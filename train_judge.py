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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.models.backbone_model import build_arg_parser, build_model
from src.models.datasets import create_dataloader
from src.models.judge import FraudJudgeDNN, FraudJudgeTrainer, build_judge_model
from src.models.loss import build_loss_from_args



def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def load_trained_model(model_path, device):
    """Load trained sequence model"""
    print(f"Loading trained model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model parameters
    args = checkpoint['args']
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss (MSE): {checkpoint.get('val_loss', 0.0):.6f}")
    
    # Rebuild model
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count and display sequence model parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Sequence model parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    return model, args


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


def extract_judge_training_data(sequence_model, dataloader, device, feature_names, target_indices, target_names):
    """
    Extract training data for judge model using sequence model predictions
    
    Returns:
        predictions: [N, pred_dim] - sequence model predictions
        targets: [N, pred_dim] - ground truth targets  
        labels: [N] - fraud labels (0/1)
    """
    sequence_model.eval()
    
    all_predictions = []
    all_targets = []
    all_labels = []
    
    print("Extracting training data for judge model...")
    
    with torch.no_grad():
        for batch_idx, (X, y, mask) in enumerate(tqdm(dataloader, desc="Extracting data")):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Get sequence model predictions
            preds = sequence_model(X, mask, feature_names)
            
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
            
            # Only keep valid steps
            valid_mask = valid_steps == 1
            if valid_mask.any():
                # Flatten valid predictions and targets
                pred_valid = pred_steps[valid_mask].cpu().numpy()  # [N, pred_dim]
                target_valid = target_steps[valid_mask].cpu().numpy()  # [N, pred_dim]
                label_valid = label_steps[valid_mask].cpu().numpy()  # [N]
                
                all_predictions.append(pred_valid)
                all_targets.append(target_valid)
                all_labels.append(label_valid)
    
    # Concatenate all data
    if all_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        print(f"Extracted {len(predictions)} samples for judge training")
        print(f"Fraud rate: {np.mean(labels):.4f}")
        
        return predictions, targets, labels
    else:
        print("No valid data extracted!")
        return None, None, None


def filter_fraud_batches(dataloader, min_fraud_rate=0.1):
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


def create_judge_dataset(predictions, targets, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Create train/val/test splits for judge model"""
    np.random.seed(seed)
    n_samples = len(predictions)
    indices = np.random.permutation(n_samples)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    return {
        'train': (predictions[train_idx], targets[train_idx], labels[train_idx]),
        'val': (predictions[val_idx], targets[val_idx], labels[val_idx]),
        'test': (predictions[test_idx], targets[test_idx], labels[test_idx])
    }


def train_judge_model(judge_model, train_data, val_data, device, args):
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
    
    train_pred, train_target, train_labels = train_data
    val_pred, val_target, val_labels = val_data
    
    # Convert to tensors
    train_pred = torch.tensor(train_pred, dtype=torch.float32).to(device)
    train_target = torch.tensor(train_target, dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    val_pred = torch.tensor(val_pred, dtype=torch.float32).to(device)
    val_target = torch.tensor(val_target, dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"Training judge model for {args.judge_epochs} epochs...")
    
    for epoch in range(args.judge_epochs):
        # Training
        judge_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        batch_size = args.judge_batch_size
        n_train = len(train_pred)
        
        for i in range(0, n_train, batch_size):
            end_idx = min(i + batch_size, n_train)
            
            batch_pred = train_pred[i:end_idx]
            batch_target = train_target[i:end_idx]
            batch_labels = train_labels[i:end_idx]
            
            loss = trainer.train_step(batch_pred, batch_target, batch_labels)
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # Validation
        judge_model.eval()
        val_results = trainer.evaluate(val_pred, val_target, val_labels)
        val_acc = val_results['accuracy']
        val_accuracies.append(val_acc)
        
        # Step learning rate scheduler
        if hasattr(args, 'judge_scheduler') and args.judge_scheduler == 'plateau':
            trainer.step_scheduler(metric=val_acc)
        elif trainer.scheduler is not None:
            trainer.step_scheduler()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': judge_model.state_dict(),
                'val_accuracy': val_acc,
                'args': args
            }, f'checkpoints/best_judge_model.pth')
        
        if epoch % 10 == 0:
            current_lr = trainer.get_lr()
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}, LR={current_lr:.6f}")
    
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }


def load_judge_model(judge_model_path, device, pred_dim=None):
    """Load trained judge model from checkpoint
    
    Args:
        judge_model_path: Path to checkpoint file
        device: Device to load model on
        pred_dim: Optional pred_dim to use. If None, will infer from checkpoint state_dict
    """
    print(f"Loading judge model from: {judge_model_path}")
    
    checkpoint = torch.load(judge_model_path, map_location=device, weights_only=False)
    
    # Get model arguments from checkpoint
    checkpoint_args = checkpoint.get('args', None)
    if checkpoint_args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot load model configuration.")
    
    # Convert to argparse.Namespace if it's a dict
    if isinstance(checkpoint_args, dict):
        from argparse import Namespace
        args = Namespace(**checkpoint_args)
    else:
        args = checkpoint_args
    
    print(f"Model trained for {checkpoint.get('epoch', 0) + 1} epochs")
    print(f"Best validation accuracy: {checkpoint.get('val_accuracy', 0.0):.4f}")
    
    # Determine feature configuration from args
    use_statistical_features = not getattr(args, 'no_stat', False)
    use_basic_features = not getattr(args, 'stat_only', False)
    
    if getattr(args, 'stat_only', False):
        use_statistical_features = True
        use_basic_features = False
    
    # Always try to infer pred_dim from checkpoint state_dict first (most reliable)
    # This ensures consistency with the saved model
    state_dict = checkpoint['model_state_dict']
    inferred_pred_dim = None
    
    if 'dnn.0.weight' in state_dict:
        input_dim = state_dict['dnn.0.weight'].shape[1]
        # Infer pred_dim from input_dim
        if use_basic_features and use_statistical_features:
            # input_dim = pred_dim * 3 + 6
            inferred_pred_dim = (input_dim - 6) // 3
        elif use_basic_features and not use_statistical_features:
            # input_dim = pred_dim * 3
            inferred_pred_dim = input_dim // 3
        elif not use_basic_features and use_statistical_features:
            # input_dim = 6, pred_dim doesn't matter
            inferred_pred_dim = 1  # Dummy value, won't be used
        else:
            raise ValueError("Cannot infer pred_dim from model configuration")
        print(f"Inferred pred_dim={inferred_pred_dim} from checkpoint (input_dim={input_dim})")
    
    # Use inferred value if available, otherwise use provided pred_dim or fallback
    if inferred_pred_dim is not None:
        if pred_dim is not None and pred_dim != inferred_pred_dim:
            print(f"Warning: Provided pred_dim={pred_dim} differs from checkpoint pred_dim={inferred_pred_dim}. Using checkpoint value.")
        pred_dim = inferred_pred_dim
    elif pred_dim is None:
        # Fallback: try to get from checkpoint args or use default
        # This matches the target_names used during training
        target_names = [
            'Product ID_enc', 'Amount', 'Action Type_enc', 'Source T',
            'is_int',
            'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
        ]
        pred_dim = len(target_names)
        print(f"Using default pred_dim={pred_dim} based on target_names")
    
    # Build judge model with same configuration as training
    judge_model = build_judge_model(
        pred_dim=pred_dim,
        hidden_dims=getattr(args, 'judge_hidden_dims', [64, 32, 16]),
        dropout=getattr(args, 'judge_dropout', 0.2),
        use_attention=getattr(args, 'judge_use_attention', False),
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features
    ).to(device)
    
    # Load model state
    judge_model.load_state_dict(checkpoint['model_state_dict'])
    # Don't set eval mode here - let the caller decide (train or eval)
    
    # Count parameters
    total_params, trainable_params = count_parameters(judge_model)
    print(f"Judge model parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    return judge_model, args


def evaluate_judge_model(judge_model, test_data, device):
    """Evaluate the judge model"""
    test_pred, test_target, test_labels = test_data
    
    # Convert to tensors
    test_pred = torch.tensor(test_pred, dtype=torch.float32).to(device)
    test_target = torch.tensor(test_target, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)
    
    # Evaluate
    judge_model.eval()
    with torch.no_grad():
        logits = judge_model(test_pred, test_target)
        probs = torch.softmax(logits, dim=1)
        pred_labels = torch.argmax(logits, dim=1)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels.cpu().numpy(), pred_labels.cpu().numpy())
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels.cpu().numpy(), pred_labels.cpu().numpy(), average='binary'
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(test_labels.cpu().numpy(), probs[:, 1].cpu().numpy())
        except:
            auc = 0.5
        
        print(f"\nJudge Model Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }


def plot_training_curves(train_losses, val_accuracies, save_dir):
    """Plot training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Judge Model Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Judge Model Validation Accuracy')
    ax2.legend()
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
    parser.add_argument("--min_fraud_rate", type=float, default=0.1,
                       help="Minimum fraud rate to keep batch")
    
    # Judge model parameters
    parser.add_argument("--judge_hidden_dims", type=int, nargs="+", default=[64, 32, 16],
                       help="Judge model hidden layer dimensions")
    parser.add_argument("--judge_dropout", type=float, default=0.2,
                       help="Judge model dropout rate")
    parser.add_argument("--judge_use_attention", action="store_true",
                       help="Use attention mechanism in judge model")
    
    # Loss function parameters for recall optimization
    parser.add_argument("--loss_type", type=str, default="weighted",
                       choices=["cross_entropy", "focal", "weighted", "recall_focused", "adaptive", "hinge"],
                       help="Loss function type for recall optimization")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Focal loss alpha parameter")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
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
    parser.add_argument("--judge_lr", type=float, default=1e-3,
                       help="Judge model learning rate")
    parser.add_argument("--judge_weight_decay", type=float, default=1e-4,
                       help="Judge model weight decay")
    parser.add_argument("--judge_batch_size", type=int, default=64,
                       help="Judge model batch size")
    parser.add_argument("--judge_epochs", type=int, default=150,
                       help="Judge model training epochs")
    
    # Learning rate scheduler parameters
    parser.add_argument("--judge_scheduler", type=str, default=None,
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
    sequence_model, model_args = load_trained_model(args.sequence_model_path, device)
    
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
        drop_all_zero_batches=False
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
    predictions, targets, labels = extract_judge_training_data(
        sequence_model, fraud_dataloader, device, feature_names, target_indices, target_names
    )
    
    if predictions is None:
        print("No valid training data extracted!")
        return
    
    # Create train/val/test splits
    dataset_splits = create_judge_dataset(
        predictions, targets, labels,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Dataset splits:")
    for split_name, (pred, target, label) in dataset_splits.items():
        if len(pred) > 0:
            fraud_rate = np.mean(label)
            print(f"  {split_name}: {len(pred)} samples, fraud rate: {fraud_rate:.4f}")
    
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
        # Set mode based on whether we're testing or training
        if test_only:
            judge_model.eval()
        else:
            judge_model.train()  # Continue training
    else:
        # Build new judge model
        pred_dim = len(target_indices)
        
        # Determine feature configuration
        if args.stat_only:
            use_statistical_features = True
            use_basic_features = False
            input_dim = 6  # Only statistical features
            print(f"Judge model configuration:")
            print(f"  Mode: STAT-ONLY (only statistical features)")
            print(f"  Input dimension: {input_dim}")
        elif args.no_stat:
            use_statistical_features = False
            use_basic_features = True
            input_dim = pred_dim * 3  # Only basic features
            print(f"Judge model configuration:")
            print(f"  Mode: NO-STAT (only basic features)")
            print(f"  Input dimension: {input_dim}")
        else:
            use_statistical_features = True
            use_basic_features = True
            input_dim = pred_dim * 3 + 6  # Both basic and statistical features
            print(f"Judge model configuration:")
            print(f"  Mode: FULL (basic + statistical features)")
            print(f"  Input dimension: {input_dim}")
        
        judge_model = build_judge_model(
            pred_dim=pred_dim,
            hidden_dims=args.judge_hidden_dims,
            dropout=args.judge_dropout,
            use_attention=args.judge_use_attention,
            use_statistical_features=use_statistical_features,
            use_basic_features=use_basic_features
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
        if 'test' not in dataset_splits or len(dataset_splits['test'][0]) == 0:
            print("No test data available!")
            return
        
        print("\nTesting judge model...")
        test_results = evaluate_judge_model(judge_model, dataset_splits['test'], device)
        
        # Save test results
        results = {
            'test_results': test_results,
            'test_config': {
                'num_samples': len(dataset_splits['test'][0]),
                'fraud_rate': float(np.mean(dataset_splits['test'][2])),
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
    if 'train' not in dataset_splits or len(dataset_splits['train'][0]) == 0:
        print("No training data available!")
        return
    
    if 'val' not in dataset_splits or len(dataset_splits['val'][0]) == 0:
        print("Warning: No validation data available! Using training data for validation.")
        dataset_splits['val'] = dataset_splits['train']
    
    # Train judge model
    training_results = train_judge_model(
        judge_model, dataset_splits['train'], dataset_splits['val'], device, args
    )
    
    # Plot training curves
    plot_training_curves(
        training_results['train_losses'],
        training_results['val_accuracies'],
        args.save_dir
    )
    
    # Evaluate on test set if available
    test_results = None
    if 'test' in dataset_splits and len(dataset_splits['test'][0]) > 0:
        test_results = evaluate_judge_model(judge_model, dataset_splits['test'], device)
    
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
