'''
User guide: 

run

python train.py --save_model  --epochs 110 --max_len 50 --save_model --save_dir {dir}

after putting precessed csv files in data/precessed/

then trained model will be stored in /checkpoints/best_model.pth
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import time
import traceback
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import json

from src.models.backbone_model import TimeCatLSTM, build_arg_parser, build_model
from src.models.loss import build_loss_from_args
from src.models.datasets import create_dataloader


def setup_logging(log_file="error_log.txt"):
    """Setup logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def log_tensor_info(logger, tensor, name, batch_idx=None):
    """Log detailed tensor information"""
    try:
        logger.info(f"=== {name} ===")
        if batch_idx is not None:
            logger.info(f"Batch: {batch_idx}")
        logger.info(f"Shape: {tensor.shape}")
        logger.info(f"Dtype: {tensor.dtype}")
        logger.info(f"Device: {tensor.device}")
        logger.info(f"Min: {tensor.min().item()}")
        logger.info(f"Max: {tensor.max().item()}")
        logger.info(f"Unique values: {torch.unique(tensor).detach().cpu().numpy()}")
        logger.info(f"Has NaN: {torch.isnan(tensor).any().item()}")
        logger.info(f"Has Inf: {torch.isinf(tensor).any().item()}")
        logger.info(f"Sample values: {tensor.flatten()[:10].detach().cpu().numpy()}")
    except Exception as e:
        logger.error(f"Error logging tensor info for {name}: {e}")


def _masked_regression_metrics(preds, target, valid_mask):
    """Compute masked MSE/MAE given preds [B,T,D], target [B,T,D], valid_mask [B,T]."""
    m = valid_mask.unsqueeze(-1).to(preds.dtype)
    diff = (preds - target) * m
    # denom should be the total number of valid elements (B*T*D), not just B*T
    # m.sum() gives B*T (number of valid time steps), need to multiply by feature dim D
    denom = (m.sum() * preds.shape[-1]).clamp_min(1.0)
    mse = (diff ** 2).sum() / denom
    mae = diff.abs().sum() / denom
    return mse, mae


def train_epoch(model, train_loader, optimizer, criterion, device, feature_names, target_indices, logger, scaler=None, use_amp=False):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    
    for batch_idx, (X, y, mask) in enumerate(tqdm(train_loader, desc="Training")):
        try:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Log input tensor info (only for first batch)
            if batch_idx == 0:
                logger.info(f"=== Batch {batch_idx} Input Info ===")
                log_tensor_info(logger, X, "Input X", batch_idx)
                log_tensor_info(logger, y, "Labels y", batch_idx)
                log_tensor_info(logger, mask, "Mask", batch_idx)
                logger.info(f"Feature names: {feature_names}")
            
            optimizer.zero_grad()
            
            # Forward pass to get sequence predictions [B,T,pred_dim]
            if use_amp and scaler is not None:
                with autocast():
                    preds = model(X, mask, feature_names)
            else:
                preds = model(X, mask, feature_names)
            
            # Log model output info (only for first batch)
            if batch_idx == 0:
                log_tensor_info(logger, preds, "Model Output Preds", batch_idx)
            
            # Next-step target (shift by one): use mask[:,1:] to select valid steps
            if mask.any():
                pred_steps = preds[:, :-1, :]
                # Select target columns by indices
                target_steps = X[:, 1:, target_indices]
                valid_steps = mask[:, 1:]

                mse, mae = _masked_regression_metrics(pred_steps, target_steps, valid_steps)
                loss = criterion(pred_steps, target_steps, valid_steps)

                # Backward pass
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                    
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Log tensor state at error
            try:
                logger.error("=== Error State Tensor Info ===")
                log_tensor_info(logger, X, "X at error", batch_idx)
                log_tensor_info(logger, y, "y at error", batch_idx)
                log_tensor_info(logger, mask, "mask at error", batch_idx)
            except:
                logger.error("Could not log tensor info at error state")
            
            raise e  # Re-raise exception
    
    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_mae = total_mae / len(train_loader)
    return avg_loss, avg_mse, avg_mae


def evaluate(model, val_loader, criterion, device, feature_names, target_indices):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    
    with torch.no_grad():
        for X, y, mask in tqdm(val_loader, desc="Evaluating"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Forward pass
            preds = model(X, mask, feature_names)

            if mask.any():
                pred_steps = preds[:, :-1, :]
                target_steps = X[:, 1:, target_indices]
                valid_steps = mask[:, 1:]

                mse, mae = _masked_regression_metrics(pred_steps, target_steps, valid_steps)
                loss = criterion(pred_steps, target_steps, valid_steps)  # Use same criterion as training
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()

    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_mae = total_mae / len(val_loader)
    return avg_loss, avg_mse, avg_mae


def test_model(model, test_loader, criterion, device, feature_names, target_indices):
    """Test model performance"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    
    print("Starting test set evaluation...")
    
    with torch.no_grad():
        for X, y, mask in tqdm(test_loader, desc="Testing"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # Forward pass
            preds = model(X, mask, feature_names)

            if mask.any():
                pred_steps = preds[:, :-1, :]
                target_steps = X[:, 1:, target_indices]
                valid_steps = mask[:, 1:]

                mse, mae = _masked_regression_metrics(pred_steps, target_steps, valid_steps)
                loss = criterion(pred_steps, target_steps, valid_steps)  # Use same criterion as training
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()

    avg_loss = total_loss / len(test_loader)
    avg_mse = total_mse / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    
    print("=" * 60)
    print("Test set evaluation results (regression):")
    print("=" * 60)
    print(f"test loss: {avg_loss:.6f}")
    print(f"test MSE: {avg_mse:.6f}")
    print(f"test MAE: {avg_mae:.6f}")
    print("=" * 60)
    
    return avg_loss, avg_mse, avg_mae


def train_model(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging("training_error_log.txt")
    logger.info("=== Training Started ===")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    # Setup random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # Create data loaders
    print("Loading data...")
    logger.info("Loading data...")
    
    try:
        train_loader, feature_names = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=True,
            mode="train",
            drop_all_zero_batches=False,
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
            use_sliding_window=getattr(args, 'use_sliding_window', False),
            window_overlap=getattr(args, 'window_overlap', 0.5)
        )
        
        val_loader, _ = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=False,
            mode="val",
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
            drop_all_zero_batches=False  # Don't filter all-zero batches for validation set
        )
        
        # Create test set data loader
        test_loader, _ = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=False,
            mode="test",
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed,
            drop_all_zero_batches=False  # Don't filter all-zero batches for test set
        )
        
        print(f"Feature names: {feature_names}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
        # Statistics for positive samples in validation and test sets (time-step level and sequence level)
        def _count_pos(loader, name):
            seq_pos, seq_total = 0, 0
            step_pos, step_total = 0, 0
            with torch.no_grad():
                for Xb, yb, maskb in loader:
                    # Time-step level (by mask valid steps)
                    step_pos += ((yb == 1) & (maskb == 1)).sum().item()
                    step_total += (maskb == 1).sum().item()
                    # Sequence level (last valid label)
                    seq_lengths = maskb.sum(dim=1).long() - 1
                    seq_lengths = torch.clamp(seq_lengths, min=0)
                    last_valid = yb.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
                    seq_pos += (last_valid == 1).sum().item()
                    seq_total += last_valid.numel()
            print(f"[{name}] step-level positives: {step_pos}/{step_total}")
            print(f"[{name}] seq-level positives:  {seq_pos}/{seq_total}")

        _count_pos(val_loader, "val")
        _count_pos(test_loader, "test")
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e
    
    # Setup model parameters
    args.feature_names = feature_names
    args.cont_dim = len([f for f in feature_names if f not in [
        'Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized', 'cluster_id'
    ]])
    
    # Resolve target names (subset to predict)
    default_target_names = [
        'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
        'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
    ]
    # Use args.target_names if provided; otherwise default list
    target_names = getattr(args, 'target_names', None)
    if not target_names:
        target_names = default_target_names
    # Keep only those present in feature_names, preserving feature_names order
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_indices = [name_to_idx[n] for n in feature_names if n in set(target_names) and n in name_to_idx]
    # Also store resolved target_names in the same order as indices
    resolved_target_names = [feature_names[i] for i in target_indices]
    args.target_names = resolved_target_names
    # Ensure pred_dim matches
    args.pred_dim = len(target_indices) if len(target_indices) > 0 else len(feature_names)

    logger.info(f"Resolved target names: {args.target_names}")
    logger.info(f"Target indices: {target_indices}")

    # Create model
    print("Building model...")
    model = build_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = build_loss_from_args(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Mixed precision training setup
    use_amp = getattr(args, 'use_amp', False) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP)")
        logger.info("Using mixed precision training (AMP)")
    else:
        print("Using full precision training")
        logger.info("Using full precision training")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_auc = 0
    patience_counter = 0
    
    print("Starting training...")

    # Per-epoch metrics history
    history = {
        'train_loss': [],
        'train_mse': [],
        'train_mae': [],
        'val_loss': [],
        'val_mse': [],
        'val_mae': [],
        'epoch_time': []
    }
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_mse, train_mae = train_epoch(model, train_loader, optimizer, criterion, device, feature_names, target_indices, logger, scaler=scaler, use_amp=use_amp)
        
        # Validation
        val_loss, val_mse, val_mae = evaluate(model, val_loader, criterion, device, feature_names, target_indices)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train MSE: {train_mse:.6f}, Train MAE: {train_mae:.6f}")
        print(f"Val   MSE: {val_mse:.6f}, Val   MAE: {val_mae:.6f}")
        print("-" * 50)

        # Record history
        history['train_loss'].append(float(train_loss))
        history['train_mse'].append(float(train_mse))
        history['train_mae'].append(float(train_mae))
        history['val_loss'].append(float(val_loss))
        history['val_mse'].append(float(val_mse))
        history['val_mae'].append(float(val_mae))
        history['epoch_time'].append(float(epoch_time))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'args': args
                }, os.path.join(args.save_dir, f'best_model_enc.pth'))
                print(f"Saved best model with val MSE: {val_mse:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best val loss (MSE): {best_val_loss:.6f}")

    # Save training curves and history
    os.makedirs(args.save_dir, exist_ok=True)
    history_csv = os.path.join(args.save_dir, 'training_history.csv')
    history_json = os.path.join(args.save_dir, 'training_history.json')
    curves_png = os.path.join(args.save_dir, 'training_curves.png')

    # Write CSV
    with open(history_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'train_mse', 'train_mae', 'val_loss', 'val_mse', 'val_mae', 'epoch_time']
        writer.writerow(header)
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_mse'][i],
                history['train_mae'][i],
                history['val_loss'][i],
                history['val_mse'][i],
                history['val_mae'][i],
                history['epoch_time'][i],
            ])

    # Write JSON
    with open(history_json, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    # Plot
    try:
        epochs = list(range(1, len(history['train_loss']) + 1))
        plt.figure(figsize=(12, 8))

        # Subplot 1: Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Subplot 2: MSE/MAE
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['train_mse'], label='Train MSE')
        plt.plot(epochs, history['val_mse'], label='Val MSE')
        plt.plot(epochs, history['train_mae'], label='Train MAE')
        plt.plot(epochs, history['val_mae'], label='Val MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('MSE/MAE per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Subplot 3: Loss zoom
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['train_loss'], label='Train Loss (MSE)')
        plt.plot(epochs, history['val_loss'], label='Val Loss (MSE)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss (zoom)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Subplot 4: Val MSE
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['val_mse'], label='Val MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('Val MSE per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        plt.savefig(curves_png, dpi=150)
        plt.close()
        print(f"Saved training curves to: {curves_png}")
        print(f"Saved training history to: {history_csv}, {history_json}")
    except Exception as e:
        print(f"Failed to plot/save training curves: {e}")
    
    # After training completion, use best model for test set evaluation
    if args.save_model:
        # Load best model
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("Loading test datasets...")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"loaded model at epoch {checkpoint['epoch']+1}, val loss (MSE): {checkpoint.get('val_loss', 0.0):.6f}")
            
            # Perform test set evaluation
            test_loss, test_mse, test_mae = test_model(model, test_loader, criterion, device, feature_names, target_indices)
            
            # Log test results
            logger.info("=== Results ===")
            logger.info(f"loss (MSE): {test_loss:.6f}")
            logger.info(f"MSE: {test_mse:.6f}")
            logger.info(f"MAE: {test_mae:.6f}")
            
            # Save test results to file
            test_results = {
                'test_loss_mse': test_loss,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'best_val_loss': best_val_loss,
                'epoch': checkpoint['epoch']
            }
            results_file = os.path.join(args.save_dir, 'test_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"Saved result to: {results_file}")
        else:
            print("model not found")
    else:
        # If no model saved, directly use current model for testing
        print("eval on...")
        test_loss, test_mse, test_mae = test_model(model, test_loader, criterion, device, feature_names, target_indices)
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train sequence model (LSTM or Transformer)")
    
    # Data related parameters
    parser.add_argument("--data_dir", type=str, default="data/test/no_fraud", help="Data directory")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    # Training related parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Learning rate scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training (AMP)")
    
    # Loss options
    parser.add_argument("--loss_type", type=str, default="mse", choices=["bce", "huber", "pseudohuber", "quantile"], help="Choose loss function")
    parser.add_argument("--delta", type=float, default=10, help="Huber/Pseudo-Huber delta (δ)")
    parser.add_argument("--auto_delta_p", type=float, default=0.9, help="Auto delta: p-quantile of residual |e|, e.g., 0.9")
    parser.add_argument("--quantiles", type=str, default="0.1,0.5,0.9", help="Quantile list, comma-separated")
    parser.add_argument("--crossing_lambda", type=float, default=0.0, help="Anti-crossing regularization λ for quantile regression")
    parser.add_argument("--apply_sigmoid_in_regression", action="store_true", default=True, help="Apply sigmoid to logits before regression-style losses")

    # Model saving
    parser.add_argument("--save_model", action="store_true", help="Whether to save model")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Model save directory")
    
    # Sliding window parameters
    parser.add_argument("--use_sliding_window", action="store_true", help="Use sliding window to increase data volume")
    parser.add_argument("--window_overlap", type=float, default=0.8, help="Sliding window overlap ratio (0.0-0.9)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Get model parameters
    model_parser = build_arg_parser()
    model_args, remaining = model_parser.parse_known_args()
    
    # Merge parameters
    args = parser.parse_args()
    for key, value in vars(model_args).items():
        setattr(args, key, value)
    
    print("Training arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # Start training
    model = train_model(args)


if __name__ == "__main__":
    main()
