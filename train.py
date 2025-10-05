import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

from model import TimeCatLSTM, build_arg_parser, build_model
from datasets import create_dataloader


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


def train_epoch(model, train_loader, optimizer, criterion, device, feature_names, logger):
    """Train one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
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
            
            # Forward pass
            logits = model(X, mask, feature_names)
            
            # Log model output info (only for first batch)
            if batch_idx == 0:
                log_tensor_info(logger, logits, "Model Output Logits", batch_idx)
            
            # Calculate loss (model output is one prediction per sequence)
            # logits shape: [batch_size] - one prediction per sequence
            # y shape: [batch_size, seq_len] - labels for each time step
            # mask shape: [batch_size, seq_len] - validity for each time step
            
            # For sequence-level prediction, we need to aggregate labels
            # Use mask to weight average labels, or use last valid label
            if mask.any():
                # Use whether there are any fraud events in the sequence (more suitable for fraud detection)
                # For fraud detection, we care about whether fraud occurred in the sequence, not the last step's label
                valid_y = y * mask  # Only consider valid time steps
                seq_labels = (valid_y == 1).any(dim=1).float()  # Whether there is fraud in the sequence
                
                # Ensure logits and labels dimensions match
                if logits.dim() == 1 and seq_labels.dim() == 1:
                    loss = criterion(logits, seq_labels)
                else:
                    # If dimensions don't match, adjust logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, seq_labels.unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Collect predictions and labels for evaluation
                with torch.no_grad():
                    preds = torch.sigmoid(logits) > 0.5
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(seq_labels.cpu().numpy())
                    
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
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, feature_names):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Statistics for y==1 predictions
    y1_correct = 0  # y==1 and correctly predicted count
    y1_incorrect = 0  # y==1 but incorrectly predicted count
    y1_total = 0  # total y==1 count
    
    with torch.no_grad():
        for X, y, mask in tqdm(val_loader, desc="Evaluating"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # 前向传播
            logits = model(X, mask, feature_names)
            
            # 计算损失（使用与训练相同的逻辑）
            if mask.any():
                # 使用序列中是否有任何欺诈事件（更符合欺诈检测场景）
                valid_y = y * mask  # 只考虑有效时间步
                seq_labels = (valid_y == 1).any(dim=1).float()  # 序列中是否有欺诈
                
                # 确保logits和labels的维度匹配
                if logits.dim() == 1 and seq_labels.dim() == 1:
                    loss = criterion(logits, seq_labels)
                else:
                    # 如果维度不匹配，调整logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, seq_labels.unsqueeze(0))
                
                total_loss += loss.item()
                
                # 收集预测和标签
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(seq_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 统计y==1的预测情况
                labels_np = seq_labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                # 找到y==1的样本
                y1_mask = labels_np == 1
                y1_total += y1_mask.sum()
                
                # 统计y==1样本中的正确和错误预测
                y1_correct += ((preds_np == 1) & y1_mask).sum()
                y1_incorrect += ((preds_np == 0) & y1_mask).sum()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    # 计算其他指标
    if all_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
    else:
        precision = recall = f1 = auc = 0.0
    
    # Print y==1 prediction statistics
    print(f"Validation set y==1 prediction statistics:")
    print(f"  y==1 total count: {y1_total}")
    print(f"  y==1 correctly predicted: {y1_correct}")
    print(f"  y==1 incorrectly predicted: {y1_incorrect}")
    if y1_total > 0:
        print(f"  y==1 prediction accuracy: {y1_correct/y1_total:.4f}")
    else:
        print(f"  y==1 prediction accuracy: N/A (no y==1 samples)")
    
    return avg_loss, accuracy, precision, recall, f1, auc


def test_model(model, test_loader, criterion, device, feature_names):
    """Test model performance"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # Statistics for y==1 predictions
    y1_correct = 0  # y==1 and correctly predicted count
    y1_incorrect = 0  # y==1 but incorrectly predicted count
    y1_total = 0  # total y==1 count
    
    print("Starting test set evaluation...")
    
    with torch.no_grad():
        for X, y, mask in tqdm(test_loader, desc="Testing"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # 前向传播
            logits = model(X, mask, feature_names)
            
            # 计算损失（使用与训练相同的逻辑）
            if mask.any():
                # 使用序列中是否有任何欺诈事件（更符合欺诈检测场景）
                valid_y = y * mask  # 只考虑有效时间步
                seq_labels = (valid_y == 1).any(dim=1).float()  # 序列中是否有欺诈
                
                # 确保logits和labels的维度匹配
                if logits.dim() == 1 and seq_labels.dim() == 1:
                    loss = criterion(logits, seq_labels)
                else:
                    # 如果维度不匹配，调整logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, seq_labels.unsqueeze(0))
                
                total_loss += loss.item()
                
                # 收集预测和标签
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(seq_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # 统计y==1的预测情况
                labels_np = seq_labels.cpu().numpy()
                preds_np = preds.cpu().numpy()
                
                # 找到y==1的样本
                y1_mask = labels_np == 1
                y1_total += y1_mask.sum()
                
                # 统计y==1样本中的正确和错误预测
                y1_correct += ((preds_np == 1) & y1_mask).sum()
                y1_incorrect += ((preds_np == 0) & y1_mask).sum()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    # 计算其他指标
    if all_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
    else:
        precision = recall = f1 = auc = 0.0
    
    # Print test results
    print("=" * 60)
    print("Test set evaluation results:")
    print("=" * 60)
    print(f"test loss: {avg_loss:.4f}")
    print(f"test accuracy: {accuracy:.4f}")
    print(f"test precision: {precision:.4f}")
    print(f"test recall: {recall:.4f}")
    print(f"test F1: {f1:.4f}")
    print(f"test AUC: {auc:.4f}")
    
    # Print y==1 prediction statistics
    print(f"\nTest set y==1 prediction statistics:")
    print(f"  y==1 total count: {y1_total}")
    print(f"  y==1 correctly predicted: {y1_correct}")
    print(f"  y==1 incorrectly predicted: {y1_incorrect}")
    if y1_total > 0:
        print(f"  y==1 prediction accuracy: {y1_correct/y1_total:.4f}")
    else:
        print(f"  y==1 prediction accuracy: N/A (no y==1 samples)")
    print("=" * 60)
    
    return avg_loss, accuracy, precision, recall, f1, auc


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
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized'
    ]])
    
    # Create model
    print("Building model...")
    model = build_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
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
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'epoch_time': []
    }
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, feature_names, logger)
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device, feature_names
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print results
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)

        # Record history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_precision'].append(float(val_precision))
        history['val_recall'].append(float(val_recall))
        history['val_f1'].append(float(val_f1))
        history['val_auc'].append(float(val_auc))
        history['epoch_time'].append(float(epoch_time))
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'args': args
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")

    # Save training curves and history
    os.makedirs(args.save_dir, exist_ok=True)
    history_csv = os.path.join(args.save_dir, 'training_history.csv')
    history_json = os.path.join(args.save_dir, 'training_history.json')
    curves_png = os.path.join(args.save_dir, 'training_curves.png')

    # Write CSV
    with open(history_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1', 'val_auc', 'epoch_time']
        writer.writerow(header)
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                history['train_loss'][i],
                history['train_acc'][i],
                history['val_loss'][i],
                history['val_acc'][i],
                history['val_precision'][i],
                history['val_recall'][i],
                history['val_f1'][i],
                history['val_auc'][i],
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

        # Subplot 2: Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Subplot 3: Precision/Recall/F1
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['val_precision'], label='Val Precision')
        plt.plot(epochs, history['val_recall'], label='Val Recall')
        plt.plot(epochs, history['val_f1'], label='Val F1')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('PRF per Epoch')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.4)

        # Subplot 4: AUC
        plt.subplot(2, 2, 4)
        plt.plot(epochs, history['val_auc'], label='Val AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC per Epoch')
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
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"loaded model at epoch {checkpoint['epoch']+1}, auc on test set: {checkpoint['val_auc']:.4f}")
            
            # Perform test set evaluation
            test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test_model(
                model, test_loader, criterion, device, feature_names
            )
            
            # Log test results
            logger.info("=== Results ===")
            logger.info(f"loss: {test_loss:.4f}")
            logger.info(f"accuracy: {test_acc:.4f}")
            logger.info(f"precision: {test_precision:.4f}")
            logger.info(f"recall: {test_recall:.4f}")
            logger.info(f"F1: {test_f1:.4f}")
            logger.info(f"AUC: {test_auc:.4f}")
            
            # Save test results to file
            test_results = {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'best_val_auc': best_auc,
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
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test_model(
            model, test_loader, criterion, device, feature_names
        )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TimeCatLSTM model")
    
    # Data related parameters
    parser.add_argument("--data_dir", type=str, default="matched", help="Data directory")
    parser.add_argument("--max_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    
    # Training related parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Learning rate scheduler patience")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience")
    
    # Model saving
    parser.add_argument("--save_model", action="store_true", help="Whether to save model")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Model save directory")
    
    # Sliding window parameters
    parser.add_argument("--use_sliding_window", action="store_true", help="Use sliding window to increase data volume")
    parser.add_argument("--window_overlap", type=float, default=0.5, help="Sliding window overlap ratio (0.0-0.9)")
    
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
