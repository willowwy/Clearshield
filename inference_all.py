'''
Inference script: Perform inference on a single CSV file, calculate and save sequence model MSE and judge model predictions

Usage:
python inference_all.py --csv_file path/to/file.csv --sequence_model_path checkpoints/best_model_enc.pth --judge_model_path checkpoints/best_judge_model.pth --max_len 50 --n_predictions 10 --output_dir results/
If n_predictions = -1, perform inference on all sequences in the file
'''
import os
import argparse
import torch
import numpy as np
import json
import pandas as pd
from pathlib import Path

# Import functions from inference.py
from src.models.inference import (
    infer_sequence,
    infer_judge,
    resolve_target_indices
)
from src.models.load_model import load_seq_model, load_judge_model
from src.models.datasets import _process_single_dataframe, build_sequences_from_dataframe


def calculate_mse(predictions, targets):
    """
    Calculate MSE (Mean Squared Error), following the calculation method in MSELoss from loss.py
    
    Args:
        predictions: [pred_dim] Prediction values (numpy array or torch tensor)
        targets: [pred_dim] Ground truth values (numpy array or torch tensor)
    
    Returns:
        mse: Scalar MSE value
    """
    # Convert to torch tensor to maintain consistency with MSELoss calculation
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    elif not isinstance(predictions, torch.Tensor):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    elif not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32)
    
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
    
    # Follow MSELoss calculation method
    # diff = (preds - target)
    diff = predictions - targets
    
    # For a single sample, all elements are valid
    # valid_count = pred_dim (feature dimension)
    # Reference MSELoss: valid_count = m.sum() * preds.size(-1)
    # For a single sample without mask, valid_count = 1 * pred_dim = pred_dim
    valid_count = predictions.size(-1) if len(predictions.shape) > 0 else 1
    valid_count = max(float(valid_count), 1.0)  # Avoid division by zero, equivalent to clamp_min(1.0)
    
    # MSE = sum((diff)^2) / valid_count
    # Reference MSELoss: return (diff ** 2).sum() / valid_count
    mse = (diff ** 2).sum() / valid_count
    
    return float(mse.item() if isinstance(mse, torch.Tensor) else mse)


def prepare_sequence_data(df, max_len, feature_names):
    """
    Prepare sequence data: Process the entire DataFrame and build sequences
    
    Args:
        df: Complete DataFrame
        max_len: Maximum sequence length
        feature_names: List of feature names
    
    Returns:
        X_seq: [T, feature_dim] Complete sequence
        y_seq: [T] Fraud label sequence, or None if Fraud column doesn't exist
        feature_names: List of feature names
        has_fraud_column: Boolean indicating if Fraud column exists
    """
    # Check if Fraud column exists in original DataFrame
    has_fraud_column = 'Fraud' in df.columns
    
    # Process data
    df_processed = _process_single_dataframe(df)
    
    # Build sequences
    sequences, feature_names_from_data = build_sequences_from_dataframe(df_processed)
    
    if len(sequences) == 0:
        raise ValueError("Cannot build sequences from data")
    
    # Take the first sequence (usually only one account)
    X_seq, y_seq = sequences[0]  # X_seq: [T, feature_dim], y_seq: [T] fraud labels
    
    # If Fraud column doesn't exist, set y_seq to None
    if not has_fraud_column:
        y_seq = None
    
    # Verify feature name matching
    if feature_names is None:
        feature_names = feature_names_from_data
    elif feature_names != feature_names_from_data:
        print(f"Warning: Feature names do not match")
        print(f"  Model expects: {feature_names}")
        print(f"  Data contains: {feature_names_from_data}")
        feature_names = feature_names_from_data
    
    return X_seq, y_seq, feature_names, has_fraud_column


def prepare_single_step_data(X_seq, t, max_len):
    """
    Prepare input data for time step t (using ground truth sequence)
    
    Args:
        X_seq: [T_total, feature_dim] Complete sequence
        t: Target time step (predict t+1, use [0:t+1] as input)
        max_len: Maximum sequence length
    
    Returns:
        X: [1, max_len, feature_dim] Input tensor
        mask: [1, max_len] Mask tensor
    """
    # To predict t+1, use [0:t+1] as input (t+1 time steps in total)
    # The model will output t+1 predictions, the last one corresponds to t+1
    if t + 1 >= len(X_seq):
        raise ValueError(f"Time step {t} exceeds sequence length {len(X_seq)}")
    
    # Input sequence: use [0:t+1] (t+1 time steps in total)
    # The model will output t+1 predictions, the last one (predictions[0, -1, :]) corresponds to t+1 prediction
    input_seq = X_seq[:t+1]  # [t+1, feature_dim]
    
    # Process sequence: pad or truncate to max_len
    seq_len = len(input_seq)
    if seq_len < max_len:
        # Left padding
        pad_len = max_len - seq_len
        X_padded = np.vstack([
            np.zeros((pad_len, input_seq.shape[1]), dtype=np.float32),
            input_seq.astype(np.float32)
        ])
        mask = np.hstack([
            np.zeros(pad_len, dtype=np.float32),
            np.ones(seq_len, dtype=np.float32)
        ])
    elif seq_len > max_len:
        # Truncate to max_len (take last max_len rows)
        X_padded = input_seq[-max_len:].astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    else:
        X_padded = input_seq.astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    
    # Convert to tensor and add batch dimension
    X = torch.tensor(X_padded, dtype=torch.float32).unsqueeze(0)  # [1, max_len, feature_dim]
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, max_len]
    
    return X, mask


def infer_single_csv(csv_file, sequence_model_path, judge_model_path, max_len=50, 
                     n_predictions=1, device=None, output_dir=None, verbose=False):
    """
    Perform inference on a single CSV file (supports multiple time steps)
    
    Args:
        csv_file: CSV file path
        sequence_model_path: Sequence model path
        judge_model_path: Judge model path
        max_len: Maximum sequence length
        n_predictions: Number of time steps to infer, -1 means infer all time steps
        device: Device (cuda/cpu), None for auto-selection
        output_dir: Output directory, None means do not save files
        verbose: Whether to print detailed inference information
    
    Returns:
        result_dict: Dictionary containing all results
    """
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file does not exist: {csv_file}")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Read CSV file
    print(f"\nReading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if len(df) < 2:
        raise ValueError(f"CSV file requires at least 2 rows, currently has {len(df)} rows")
    
    # Load sequence model
    print("\n" + "="*60)
    print("Loading Sequence Model")
    print("="*60)
    sequence_model, sequence_args = load_seq_model(sequence_model_path, device)
    
    # Get feature names (from model parameters)
    feature_names = getattr(sequence_args, 'feature_names', None)
    if feature_names is None or len(feature_names) == 0:
        print("Warning: Feature names not found in model parameters, will infer from data")
        feature_names = None
    
    # Prepare complete sequence data
    print("\n" + "="*60)
    print("Preparing Sequence Data")
    print("="*60)
    X_seq, y_seq, feature_names, has_fraud_column = prepare_sequence_data(df, max_len, feature_names)
    print(f"Complete sequence length: {len(X_seq)}")
    print(f"Feature dimension: {X_seq.shape[1]}")
    if has_fraud_column and y_seq is not None:
        print(f"Fraud label count: {len(y_seq)} (Fraud: {np.sum(y_seq)}, Normal: {len(y_seq) - np.sum(y_seq)})")
    else:
        print("No Fraud column found in CSV file")
    
    # Resolve target feature indices
    target_indices, resolved_target_names, target_names = resolve_target_indices(feature_names, None)
    print(f"Target features: {resolved_target_names}")
    
    # Determine time steps to infer
    total_steps = len(X_seq)
    if n_predictions == -1:
        # Infer all possible time steps (from max_len to total_steps-1)
        start_step = max_len - 1  # Start from max_len-1 (need at least max_len time steps as input)
        inference_steps = list(range(start_step, total_steps - 1))  # Exclude the last one as it has no target
    else:
        # Only infer the last n_predictions time steps
        start_step = max(total_steps - n_predictions - 1, max_len - 1)
        inference_steps = list(range(start_step, total_steps - 1))
    
    if len(inference_steps) == 0:
        raise ValueError(f"No time steps available for inference. Sequence length: {total_steps}, max_len: {max_len}, n_predictions: {n_predictions}")
    
    print(f"Will infer {len(inference_steps)} time steps: {inference_steps[0]} to {inference_steps[-1]}")
    
    # Get model target_names (for selecting prediction dimensions)
    model_target_names = getattr(sequence_args, 'target_names', None)
    if model_target_names is None or len(model_target_names) == 0:
        model_target_names = [
            'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
            'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
        ]
    
    # Find indices of target features in model predictions
    model_name_to_idx = {n: i for i, n in enumerate(model_target_names)}
    selected_pred_indices = [model_name_to_idx[n] for n in target_names if n in model_name_to_idx]
    
    # Load judge model
    print("\n" + "="*60)
    print("Loading Judge Model")
    print("="*60)
    judge_model, judge_args = load_judge_model(judge_model_path, device, pred_dim=len(target_indices))
    
    # Perform inference for each time step
    if verbose:
        print("\n" + "="*60)
        print("Starting Inference")
        print("="*60)
    else:
        print(f"\nStarting inference on {len(inference_steps)} time steps...")
    
    all_results = []
    all_mse = []
    
    for step_idx, t in enumerate(inference_steps):
        if verbose:
            print(f"\nInference time step {step_idx+1}/{len(inference_steps)}: t={t} (predict t+1={t+1})")
        
        # Prepare input data (using ground truth sequence)
        X, mask = prepare_single_step_data(X_seq, t, max_len)
        
        # Sequence model inference
        predictions = infer_sequence(sequence_model, X, mask, feature_names, device)
        # predictions: [1, T, pred_dim]
        # Get the last step prediction (corresponds to t+1)
        step_pred = predictions[0, -1, :].cpu().numpy()  # [pred_dim]
        
        # Select target feature dimensions
        if len(selected_pred_indices) > 0:
            step_pred_selected = step_pred[selected_pred_indices]  # [selected_pred_dim]
        else:
            if verbose:
                print("Warning: Cannot find target feature indices in model predictions, using all predictions")
            step_pred_selected = step_pred
        
        # Get ground truth (target features for t+1)
        target_features = X_seq[t+1, target_indices]  # [target_dim]
        
        # Get ground truth fraud label (t+1) if Fraud column exists
        ground_truth_fraud = None
        if has_fraud_column and y_seq is not None:
            ground_truth_fraud = int(y_seq[t+1]) if t+1 < len(y_seq) else 0
        
        # Calculate MSE
        mse = calculate_mse(step_pred_selected, target_features)
        all_mse.append(mse)
        
        # Judge model inference
        fraud_prob, is_fraud = infer_judge(judge_model, step_pred_selected, target_features, device)
        
        # Save results
        step_result = {
            'time_step': int(t),
            'predicted_step': int(t + 1),
            'sequence_mse': float(mse),
            'judge_model': {
                'fraud_probability': float(fraud_prob),
                'is_fraud': bool(is_fraud),
                'judgment': 'Fraud' if is_fraud else 'Normal'
            },
            'prediction': step_pred_selected.tolist(),
            'ground_truth': target_features.tolist()
        }
        # Only include ground_truth_fraud if Fraud column exists
        if ground_truth_fraud is not None:
            step_result['ground_truth_fraud'] = int(ground_truth_fraud)
        
        all_results.append(step_result)
        
        if verbose:
            if ground_truth_fraud is not None:
                print(f"  MSE: {mse:.6f}, Fraud probability: {fraud_prob:.4f}, Judgment: {'Fraud' if is_fraud else 'Normal'}, Ground truth: {'Fraud' if ground_truth_fraud == 1 else 'Normal'}")
            else:
                print(f"  MSE: {mse:.6f}, Fraud probability: {fraud_prob:.4f}, Judgment: {'Fraud' if is_fraud else 'Normal'}")
    
    # Calculate average MSE
    avg_mse = np.mean(all_mse) if len(all_mse) > 0 else 0.0
    
    # Build result dictionary
    result_dict = {
        'csv_file': csv_file,
        'total_steps': int(total_steps),
        'inference_steps': len(inference_steps),
        'average_sequence_mse': float(avg_mse),
        'target_features': target_names,
        'results': all_results
    }
    
    # Output summary results
    print("\n" + "="*60)
    print("Inference Results Summary")
    print("="*60)
    print(f"CSV file: {csv_file}")
    print(f"Total sequence length: {total_steps}")
    print(f"Number of inference steps: {len(inference_steps)}")
    print(f"Average Sequence Model MSE: {avg_mse:.6f}")
    print("="*60)
    
    # Save results to files
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract member ID from CSV filename (if possible)
        csv_filename = os.path.basename(csv_file)
        csv_stem = Path(csv_filename).stem  # Filename without extension
        
        # Save JSON results
        output_json = os.path.join(output_dir, f"{csv_stem}_results.json")
        # Remove existing file if it exists to ensure overwrite
        if os.path.exists(output_json):
            os.remove(output_json)
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_json}")
        
        # Save text summary
        output_txt = os.path.join(output_dir, f"{csv_stem}_summary.txt")
        # Remove existing file if it exists to ensure overwrite
        if os.path.exists(output_txt):
            os.remove(output_txt)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("Inference Results Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"CSV file: {csv_file}\n")
            f.write(f"Total sequence length: {total_steps}\n")
            f.write(f"Number of inference steps: {len(inference_steps)}\n")
            f.write(f"Average Sequence Model MSE: {avg_mse:.6f}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("Detailed Results for Each Time Step\n")
            f.write("="*60 + "\n")
            for step_result in all_results:
                f.write(f"\nTime step {step_result['time_step']} (predict {step_result['predicted_step']}):\n")
                f.write(f"  MSE: {step_result['sequence_mse']:.6f}\n")
                if 'ground_truth_fraud' in step_result:
                    f.write(f"  Ground truth label: {'Fraud' if step_result['ground_truth_fraud'] == 1 else 'Normal'}\n")
                f.write(f"  Fraud probability: {step_result['judge_model']['fraud_probability']:.4f}\n")
                f.write(f"  Judgment: {step_result['judge_model']['judgment']}\n")
        print(f"Summary saved to: {output_txt}")
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="Perform inference on a single CSV file, calculate MSE and judge predictions")
    
    # Required arguments
    parser.add_argument("--csv_file", type=str, required=True,
                       help="CSV file path")
    parser.add_argument("--sequence_model_path", type=str, required=True,
                       help="Sequence model path (e.g., checkpoints/best_model_enc.pth)")
    parser.add_argument("--judge_model_path", type=str, required=True,
                       help="Judge model path (e.g., checkpoints/best_judge_model.pth)")
    
    # Optional arguments
    parser.add_argument("--max_len", type=int, default=50,
                       help="Maximum sequence length")
    parser.add_argument("--n_predictions", type=int, default=1,
                       help="Number of time steps to infer, -1 means infer all time steps")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), default auto-select")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory, if specified save results to files")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Print detailed inference information (MSE, fraud probability for each time step, etc.)")
    
    args = parser.parse_args()
    
    # Perform inference
    result = infer_single_csv(
        csv_file=args.csv_file,
        sequence_model_path=args.sequence_model_path,
        judge_model_path=args.judge_model_path,
        max_len=args.max_len,
        n_predictions=args.n_predictions,
        device=args.device,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    return result


if __name__ == "__main__":
    result = main()

