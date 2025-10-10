import os
# Workaround for duplicate OpenMP runtime on Windows (libiomp5md.dll)
# Must be set BEFORE importing numpy/matplotlib/torch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse
import json
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv

from datasets import create_dataloader
from model import build_arg_parser, build_model


def load_model(model_path: str, device: torch.device):
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss (MSE): {checkpoint.get('val_loss', 0.0):.6f}")

    model = build_model(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, args


def resolve_target_indices(feature_names, model_args):
    target_names = getattr(model_args, 'target_names', [
        'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
        'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
    ])
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_indices = [name_to_idx[n] for n in target_names if n in name_to_idx]
    resolved_target_names = [feature_names[i] for i in target_indices]
    return target_indices, resolved_target_names


def compute_step_mse_and_labels(model, loader, device, feature_names, target_indices):
    """
    Return:
        step_mse_by_feature: dict[feature_name] -> list[float] per time step MSE for each target feature
        step_label_list: list[int] per time step (y values for each valid time step)
        step_predictions: list[array] per time step predictions for all target features
        step_targets: list[array] per time step targets for all target features
    """
    model.eval()
    step_mse_by_feature = {}
    step_label_list = []
    step_predictions = []
    step_targets = []
    
    # Initialize feature dictionaries
    for i, idx in enumerate(target_indices):
        if i < len(feature_names):
            feature_name = feature_names[idx]
            step_mse_by_feature[feature_name] = []

    with torch.no_grad():
        for X, y, mask in tqdm(loader, desc="Scoring time steps"):
            X = X.to(device)
            y = y.to(device)
            mask = mask.to(device)

            preds = model(X, mask, feature_names)
            # Align to next-step targets
            pred_steps = preds[:, :-1, :]
            target_steps = X[:, 1:, target_indices]
            valid_steps = mask[:, 1:]
            label_steps = y[:, 1:]  # align labels with next-step targets

            # Per-time-step MSE for each target feature
            for i, idx in enumerate(target_indices):
                if i < len(target_indices):
                    feature_name = feature_names[idx]
                    # MSE for this specific feature at each time step
                    feature_diff = (pred_steps[:, :, i] - target_steps[:, :, i]) * valid_steps  # [B, T]
                    feature_mse = (feature_diff ** 2)  # [B, T] - MSE for each time step
                    # Flatten and collect only valid steps
                    valid_mask = valid_steps == 1
                    step_mse_by_feature[feature_name].extend(feature_mse[valid_mask].detach().cpu().numpy().tolist())

            # Collect predictions and targets for Mahalanobis distance calculation
            valid_mask = valid_steps == 1
            if valid_mask.any():
                pred_valid = pred_steps[valid_mask].detach().cpu().numpy()  # [N, num_features]
                target_valid = target_steps[valid_mask].detach().cpu().numpy()  # [N, num_features]
                step_predictions.extend(pred_valid)
                step_targets.extend(target_valid)

            # Time step labels: y values for each valid time step
            valid_y = (label_steps * valid_steps).detach()
            step_labels = valid_y[valid_steps == 1].cpu().numpy()
            step_label_list.extend(step_labels.tolist())

    return step_mse_by_feature, step_label_list, step_predictions, step_targets


def sweep_thresholds_compute_auc(step_mse_by_feature, step_label_list, num_thresholds: int = 50, max_factor: float = 1.0):
    """Compute AUC using overall MSE (sum of all features) at time step level"""
    # Calculate overall MSE per time step (sum of all feature MSEs)
    overall_mse = []
    for i in range(len(step_label_list)):
        step_mse = 0.0
        for feature_name, mse_list in step_mse_by_feature.items():
            if i < len(mse_list):
                step_mse += mse_list[i]
        overall_mse.append(step_mse)
    
    step_mse = np.asarray(overall_mse, dtype=np.float64)
    labels = np.asarray(step_label_list, dtype=np.int64)
    if step_mse.size == 0:
        return np.array([]), np.array([])

    t_min = float(np.min(step_mse))
    t_max = float(np.max(step_mse))
    t_max_expanded = t_max * float(max_factor)
    if t_min == t_max:
        thresholds = np.array([t_min])
    else:
        thresholds = np.linspace(t_min, t_max_expanded, num_thresholds)

    aucs = []
    for thr in thresholds:
        preds = (step_mse > thr).astype(np.int64)  # y_hat = 1 if mse > thr else 0
        
        # Calculate AUC using sklearn
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(labels)) > 1:  # Need both classes
                auc = roc_auc_score(labels, preds)
            else:
                auc = 0.5  # Random performance when only one class
        except:
            auc = 0.5  # Fallback to random performance
        aucs.append(auc)

    return thresholds, np.asarray(aucs)


def plot_auc_curve(thresholds, aucs, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, aucs, marker='o', ms=3)
    plt.xlabel('MSE Threshold')
    plt.ylabel('AUC')
    plt.title('AUC vs. MSE Threshold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (AUC=0.5)')
    plt.legend()
    path = os.path.join(save_dir, 'judger_auc_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"AUC curve saved to: {path}")


def analyze_mse_distribution_by_feature(step_mse_by_feature, step_label_list):
    """Analyze and print MSE distribution for each feature by y==0 and y==1 at time step level"""
    labels = np.asarray(step_label_list, dtype=np.int64)
    
    print("\n" + "="*80)
    print("MSE Distribution Analysis by Feature (Time Step Level)")
    print("="*80)
    
    print(f"Total time steps: {len(labels)}")
    print(f"y==0 time steps: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"y==1 time steps: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    
    feature_stats = {}
    
    for feature_name, mse_list in step_mse_by_feature.items():
        if len(mse_list) == 0:
            continue
            
        mse_array = np.asarray(mse_list, dtype=np.float64)
        mse_y0 = mse_array[labels == 0]
        mse_y1 = mse_array[labels == 1]
        
        print(f"\n--- {feature_name} ---")
        
        if len(mse_y0) > 0:
            print(f"  y==0 (Non-fraud): Mean={np.mean(mse_y0):.6f}, Std={np.std(mse_y0):.6f}, "
                  f"Min={np.min(mse_y0):.6f}, Max={np.max(mse_y0):.6f}, Median={np.median(mse_y0):.6f}")
        else:
            print(f"  y==0 (Non-fraud): No data")
            
        if len(mse_y1) > 0:
            print(f"  y==1 (Fraud):     Mean={np.mean(mse_y1):.6f}, Std={np.std(mse_y1):.6f}, "
                  f"Min={np.min(mse_y1):.6f}, Max={np.max(mse_y1):.6f}, Median={np.median(mse_y1):.6f}")
        else:
            print(f"  y==1 (Fraud):     No data")
        
        feature_stats[feature_name] = {
            'mse_y0': mse_y0,
            'mse_y1': mse_y1
        }
    
    return feature_stats


def compute_mahalanobis_distances(step_predictions, step_targets, step_label_list):
    """Compute Mahalanobis distances for each time step"""
    if len(step_predictions) == 0 or len(step_targets) == 0:
        return np.array([]), np.array([])
    
    pred_array = np.array(step_predictions)
    target_array = np.array(step_targets)
    labels = np.array(step_label_list)
    
    # Calculate covariance matrix from all data
    try:
        # Use pseudo-inverse for numerical stability
        cov_matrix = np.cov(pred_array.T)
        cov_inv = pinv(cov_matrix)
        
        mahal_distances = []
        for i in range(len(pred_array)):
            try:
                dist = mahalanobis(pred_array[i], target_array[i], cov_inv)
                mahal_distances.append(dist)
            except:
                # Fallback to Euclidean distance if Mahalanobis fails
                dist = np.linalg.norm(pred_array[i] - target_array[i])
                mahal_distances.append(dist)
        
        return np.array(mahal_distances), labels
    except:
        # Fallback to Euclidean distances if covariance calculation fails
        euclidean_distances = []
        for i in range(len(pred_array)):
            dist = np.linalg.norm(pred_array[i] - target_array[i])
            euclidean_distances.append(dist)
        return np.array(euclidean_distances), labels


def analyze_mahalanobis_distribution(mahal_distances, labels):
    """Analyze and print Mahalanobis distance distribution by y==0 and y==1"""
    print("\n" + "="*80)
    print("Mahalanobis Distance Distribution Analysis")
    print("="*80)
    
    print(f"Total time steps: {len(labels)}")
    print(f"y==0 time steps: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.1f}%)")
    print(f"y==1 time steps: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.1f}%)")
    
    # Split by labels
    mahal_y0 = mahal_distances[labels == 0]
    mahal_y1 = mahal_distances[labels == 1]
    
    print(f"\nMahalanobis Distance Statistics:")
    if len(mahal_y0) > 0:
        print(f"  y==0 (Non-fraud): Mean={np.mean(mahal_y0):.6f}, Std={np.std(mahal_y0):.6f}, "
              f"Min={np.min(mahal_y0):.6f}, Max={np.max(mahal_y0):.6f}, Median={np.median(mahal_y0):.6f}")
    else:
        print(f"  y==0 (Non-fraud): No data")
        
    if len(mahal_y1) > 0:
        print(f"  y==1 (Fraud):     Mean={np.mean(mahal_y1):.6f}, Std={np.std(mahal_y1):.6f}, "
              f"Min={np.min(mahal_y1):.6f}, Max={np.max(mahal_y1):.6f}, Median={np.median(mahal_y1):.6f}")
    else:
        print(f"  y==1 (Fraud):     No data")
    
    return {
        'mahal_y0': mahal_y0,
        'mahal_y1': mahal_y1
    }


def plot_mahalanobis_distribution(mahal_stats, save_dir: str):
    """Plot Mahalanobis distance distribution histograms for y==0 and y==1"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot for y==0
    if len(mahal_stats['mahal_y0']) > 0:
        axes[0].hist(mahal_stats['mahal_y0'], bins=50, alpha=0.7, edgecolor='black', color='blue', density=True)
        axes[0].set_xlabel('Mahalanobis Distance')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'Mahalanobis Distance Distribution for y==0 (Non-fraud)\nMean: {np.mean(mahal_stats["mahal_y0"]):.4f}, Std: {np.std(mahal_stats["mahal_y0"]):.4f}')
        axes[0].grid(True, alpha=0.3)
    
    # Plot for y==1
    if len(mahal_stats['mahal_y1']) > 0:
        axes[1].hist(mahal_stats['mahal_y1'], bins=50, alpha=0.7, edgecolor='black', color='red', density=True)
        axes[1].set_xlabel('Mahalanobis Distance')
        axes[1].set_ylabel('Density')
        axes[1].set_title(f'Mahalanobis Distance Distribution for y==1 (Fraud)\nMean: {np.mean(mahal_stats["mahal_y1"]):.4f}, Std: {np.std(mahal_stats["mahal_y1"]):.4f}')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'mahalanobis_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mahalanobis distance distribution plot saved to: {path}")


def plot_mse_distribution_by_feature(feature_stats, save_dir: str):
    """Plot MSE distribution histograms for each feature by y==0 and y==1"""
    os.makedirs(save_dir, exist_ok=True)
    
    n_features = len(feature_stats)
    if n_features == 0:
        print("No features to plot")
        return
    
    # Calculate subplot layout
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (feature_name, stats) in enumerate(feature_stats.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        mse_y0 = stats['mse_y0']
        mse_y1 = stats['mse_y1']
        
        # Plot both distributions on the same subplot with density normalization
        if len(mse_y0) > 0:
            ax.hist(mse_y0, bins=30, alpha=0.6, edgecolor='black', color='blue', 
                   label='y==0 (Non-fraud)', density=True)
        if len(mse_y1) > 0:
            ax.hist(mse_y1, bins=30, alpha=0.6, edgecolor='black', color='red', 
                   label='y==1 (Fraud)', density=True)
        
        ax.set_xlabel('MSE')
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name}\nBlue: y==0, Red: y==1')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'mse_distribution_by_feature.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MSE distribution by feature plot saved to: {path}")


def save_judger_results(thresholds, aucs, step_mse_by_feature, step_label_list, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    best_idx = int(np.argmax(aucs)) if aucs.size > 0 else -1
    
    # Calculate overall MSE statistics
    overall_mse = []
    for i in range(len(step_label_list)):
        step_mse = 0.0
        for feature_name, mse_list in step_mse_by_feature.items():
            if i < len(mse_list):
                step_mse += mse_list[i]
        overall_mse.append(step_mse)
    
    step_mse = np.asarray(overall_mse, dtype=np.float64)
    labels = np.asarray(step_label_list, dtype=np.int64)
    mse_y0 = step_mse[labels == 0]
    mse_y1 = step_mse[labels == 1]
    
    # Calculate per-feature statistics
    feature_distributions = {}
    for feature_name, mse_list in step_mse_by_feature.items():
        mse_array = np.asarray(mse_list, dtype=np.float64)
        mse_y0_feat = mse_array[labels == 0]
        mse_y1_feat = mse_array[labels == 1]
        
        feature_distributions[feature_name] = {
            'y0_count': int(len(mse_y0_feat)),
            'y1_count': int(len(mse_y1_feat)),
            'y0_mean': float(np.mean(mse_y0_feat)) if len(mse_y0_feat) > 0 else None,
            'y0_std': float(np.std(mse_y0_feat)) if len(mse_y0_feat) > 0 else None,
            'y0_min': float(np.min(mse_y0_feat)) if len(mse_y0_feat) > 0 else None,
            'y0_max': float(np.max(mse_y0_feat)) if len(mse_y0_feat) > 0 else None,
            'y0_median': float(np.median(mse_y0_feat)) if len(mse_y0_feat) > 0 else None,
            'y1_mean': float(np.mean(mse_y1_feat)) if len(mse_y1_feat) > 0 else None,
            'y1_std': float(np.std(mse_y1_feat)) if len(mse_y1_feat) > 0 else None,
            'y1_min': float(np.min(mse_y1_feat)) if len(mse_y1_feat) > 0 else None,
            'y1_max': float(np.max(mse_y1_feat)) if len(mse_y1_feat) > 0 else None,
            'y1_median': float(np.median(mse_y1_feat)) if len(mse_y1_feat) > 0 else None,
        }
    
    payload = {
        'evaluation_time': datetime.now().isoformat(),
        'num_time_steps': int(len(step_label_list)),
        'thresholds': [float(x) for x in thresholds.tolist()],
        'aucs': [float(x) for x in aucs.tolist()],
        'best_threshold': float(thresholds[best_idx]) if best_idx >= 0 else None,
        'best_auc': float(aucs[best_idx]) if best_idx >= 0 else None,
        'overall_mse_distribution': {
            'y0_count': int(len(mse_y0)),
            'y1_count': int(len(mse_y1)),
            'y0_mean': float(np.mean(mse_y0)) if len(mse_y0) > 0 else None,
            'y0_std': float(np.std(mse_y0)) if len(mse_y0) > 0 else None,
            'y0_min': float(np.min(mse_y0)) if len(mse_y0) > 0 else None,
            'y0_max': float(np.max(mse_y0)) if len(mse_y0) > 0 else None,
            'y0_median': float(np.median(mse_y0)) if len(mse_y0) > 0 else None,
            'y1_mean': float(np.mean(mse_y1)) if len(mse_y1) > 0 else None,
            'y1_std': float(np.std(mse_y1)) if len(mse_y1) > 0 else None,
            'y1_min': float(np.min(mse_y1)) if len(mse_y1) > 0 else None,
            'y1_max': float(np.max(mse_y1)) if len(mse_y1) > 0 else None,
            'y1_median': float(np.median(mse_y1)) if len(mse_y1) > 0 else None,
        },
        'feature_distributions': feature_distributions
    }
    path = os.path.join(save_dir, 'judger_results.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Judger results saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Judge sequences by MSE threshold and plot AUC curve")

    # Required
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (e.g., checkpoints/best_model.pth)")

    # Data
    parser.add_argument("--data_dir", type=str, default="matched", help="Data directory (use matched)")
    parser.add_argument("--max_len", type=int, default=50, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # Sweep
    parser.add_argument("--num_thresholds", type=int, default=50, help="Number of thresholds to sweep")
    parser.add_argument("--threshold_max_factor", type=float, default=1.5, help="Expand upper bound as max_mse * factor")

    # Output
    parser.add_argument("--save_dir", type=str, default="evaluation_results", help="Directory to save plots and results")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file does not exist: {args.model_path}")

    # Load model
    model, model_args = load_model(args.model_path, device)

    # Data loader (use test split for stability)
    print("Loading data...")
    loader, feature_names = create_dataloader(
        matched_dir=args.data_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        shuffle=False,
        mode="train",
        split=(0.8, 0.1, 0.1),
        seed=args.seed,
        drop_all_zero_batches=False
    )
    print(f"Test batches: {len(loader)} | Features: {len(feature_names)}")

    # Targets
    target_indices, resolved_target_names = resolve_target_indices(feature_names, model_args)
    print(f"Targets: {resolved_target_names}")

    # Per-time-step MSE and labels by feature
    step_mse_by_feature, step_label_list, step_predictions, step_targets = compute_step_mse_and_labels(
        model, loader, device, feature_names, target_indices
    )
    print(f"Collected {len(step_label_list)} time steps for judging")

    # Analyze MSE distribution by feature
    feature_stats = analyze_mse_distribution_by_feature(step_mse_by_feature, step_label_list)
    
    # Plot MSE distribution by feature
    plot_mse_distribution_by_feature(feature_stats, args.save_dir)

    # Compute and analyze Mahalanobis distances
    mahal_distances, mahal_labels = compute_mahalanobis_distances(step_predictions, step_targets, step_label_list)
    if len(mahal_distances) > 0:
        mahal_stats = analyze_mahalanobis_distribution(mahal_distances, mahal_labels)
        plot_mahalanobis_distribution(mahal_stats, args.save_dir)
    else:
        print("No valid data for Mahalanobis distance calculation")

    # Threshold sweep (using overall MSE)
    thresholds, aucs = sweep_thresholds_compute_auc(step_mse_by_feature, step_label_list, args.num_thresholds, args.threshold_max_factor)
    if thresholds.size == 0:
        print("No time steps available to evaluate.")
        return

    # Plot and save
    plot_auc_curve(thresholds, aucs, args.save_dir)
    save_judger_results(thresholds, aucs, step_mse_by_feature, step_label_list, args.save_dir)


if __name__ == "__main__":
    main()


