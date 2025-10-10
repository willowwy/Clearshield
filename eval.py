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

from model import build_arg_parser, build_model
from datasets import create_dataloader


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
    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, args


def evaluate_model(model, test_loader, device, feature_names, target_indices, args):
    """Evaluate model performance"""
    model.eval()
    
    # Store all predictions and true values
    all_predictions = []
    all_targets = []
    all_masks = []
    all_labels = []  # aligned fraud labels for target/pred steps (t+1)
    
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
            preds = model(X, mask, feature_names)
            
            if mask.any():
                # Calculate regression metrics
                pred_steps = preds[:, :-1, :]
                target_steps = X[:, 1:, target_indices]
                valid_steps = mask[:, 1:]
                label_steps = y[:, 1:]  # align labels with next-step targets
                
                # Calculate MSE and MAE
                m = valid_steps.unsqueeze(-1).to(preds.dtype)
                diff = (pred_steps - target_steps) * m
                denom = m.sum().clamp_min(1.0)
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
        'labels': all_labels
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


def save_evaluation_results(results, analysis, target_names, save_dir):
    """Save evaluation results to file"""
    os.makedirs(save_dir, exist_ok=True)
    
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
        }
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Model file path (e.g., checkpoints/best_model.pth)")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="matched", 
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
    
    # Save evaluation results
    save_evaluation_results(results, analysis, resolved_target_names, args.save_dir)
    
    print("\nModel evaluation completed!")


if __name__ == "__main__":
    main()
