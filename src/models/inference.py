'''
Inference script: Fraud detection inference for a single CSV file

Usage:
python inference.py --folder path/to/folder --member_id 12345 --sequence_model_path checkpoints/best_model_enc.pth --judge_model_path checkpoints/best_judge_model.pth --max_len 50
'''
import os
import argparse
import torch
import pandas as pd
import numpy as np
from src.models.backbone_model import build_seq_model
from src.models.judge import FraudJudgeDNN
from src.models.datasets import _process_single_dataframe, build_sequences_from_dataframe
from src.models.load_model import load_seq_model, load_judge_model



def resolve_target_indices(feature_names, model_args):
    """Resolve target feature indices"""
    # Keep consistent with train_judge.py
    target_names = ['Amount', 'Action Type_enc', 'Source Type_enc']
    
    print(f"Using target features: {target_names}")
    
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    target_indices = [name_to_idx[n] for n in target_names if n in name_to_idx]
    resolved_target_names = [feature_names[i] for i in target_indices]
    return target_indices, resolved_target_names, target_names


def prepare_inference_data(csv_file, max_len, feature_names):
    """
    Prepare inference data: Read CSV file, take last max_len rows (excluding last row), process and construct input
    
    Args:
        csv_file: CSV file path
        max_len: Maximum sequence length
        feature_names: List of feature names
    
    Returns:
        X: [1, max_len, feature_dim] Input tensor
        mask: [1, max_len] Mask tensor
        last_row_features: [feature_dim] Last row features (for judge model)
        last_row_target: [target_dim] Last row target features (for judge model)
        target_indices: List of target feature indices
    """
    print(f"Reading CSV file: {csv_file}")
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    if len(df) < 2:
        raise ValueError(f"CSV file requires at least 2 rows, currently has {len(df)} rows")
    
    # Take last max_len+1 rows: first max_len rows as input, last row as target
    # If data has less than max_len+1 rows, use all available rows
    n_rows = len(df)
    if n_rows > max_len + 1:
        # Take last max_len+1 rows: [-max_len-1:-1] as input, [-1] as target
        input_df = df.iloc[-max_len-1:-1].copy()
        target_row = df.iloc[-1:].copy()
    else:
        # Insufficient data, use all rows (except last row)
        input_df = df.iloc[:-1].copy()
        target_row = df.iloc[-1:].copy()
    
    print(f"Input data: {len(input_df)} rows")
    print(f"Target data: {len(target_row)} rows")
    
    # Process input data
    input_df = _process_single_dataframe(input_df)
    target_row = _process_single_dataframe(target_row)
    
    # Build sequences
    input_sequences, feature_names_from_data = build_sequences_from_dataframe(input_df)
    target_sequences, _ = build_sequences_from_dataframe(target_row)
    
    if len(input_sequences) == 0:
        raise ValueError("Cannot build sequences from input data")
    if len(target_sequences) == 0:
        raise ValueError("Cannot build sequences from target data")
    
    # Take first sequence (usually only one account)
    X_seq, _ = input_sequences[0]  # [T, feature_dim]
    target_X, _ = target_sequences[0]  # [1, feature_dim]
    
    # Verify feature name matching
    if feature_names is None:
        feature_names = feature_names_from_data
    elif feature_names != feature_names_from_data:
        print(f"Warning: Feature names do not match")
        print(f"  Model expects: {feature_names}")
        print(f"  Data contains: {feature_names_from_data}")
        # Use feature names from data
        feature_names = feature_names_from_data
    
    # Resolve target feature indices (model_args parameter is not used in resolve_target_indices, pass None)
    target_indices, resolved_target_names, target_names = resolve_target_indices(feature_names, None)
    
    # Get last row target features
    last_row_target = target_X[0, target_indices]  # [target_dim]
    
    # Get all features of last row (for subsequent processing)
    last_row_features = target_X[0, :]  # [feature_dim]
    
    # Process input sequence: padding to max_len
    seq_len = len(X_seq)
    if seq_len < max_len:
        # Left padding
        pad_len = max_len - seq_len
        X_padded = np.vstack([
            np.zeros((pad_len, X_seq.shape[1]), dtype=np.float32),
            X_seq.astype(np.float32)
        ])
        mask = np.hstack([
            np.zeros(pad_len, dtype=np.float32),
            np.ones(seq_len, dtype=np.float32)
        ])
    elif seq_len > max_len:
        # Truncate to max_len (take last max_len rows)
        X_padded = X_seq[-max_len:].astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    else:
        X_padded = X_seq.astype(np.float32)
        mask = np.ones(max_len, dtype=np.float32)
    
    # Convert to tensor and add batch dimension
    X = torch.tensor(X_padded, dtype=torch.float32).unsqueeze(0)  # [1, max_len, feature_dim]
    mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # [1, max_len]
    
    print(f"Input shape: {X.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Target features: {resolved_target_names}")
    
    return X, mask, last_row_features, last_row_target, target_indices, target_names, feature_names


def infer_sequence(sequence_model, X, mask, feature_names, device):
    """
    Perform inference using sequence model
    
    Args:
        sequence_model: Sequence model
        X: [1, T, feature_dim] Input tensor
        mask: [1, T] Mask tensor
        feature_names: List of feature names
        device: Device
    
    Returns:
        predictions: [1, T, pred_dim] Prediction results
        hidden_state: Hidden state from sequence model (tuple for LSTM or tensor for Transformer)
    """
    sequence_model.eval()
    X = X.to(device)
    mask = mask.to(device)
    
    with torch.no_grad():
        predictions, hidden_state = sequence_model(X, mask, feature_names, use_pack=False)
    
    return predictions, hidden_state


def infer_judge(judge_model, predictions, targets, device, hidden_representation=None):
    """
    Use judge model to determine if it is fraud
    
    Args:
        judge_model: Judge model
        predictions: [pred_dim] Sequence model prediction for last step (optional, used only when use_pred=True)
        targets: [pred_dim] Ground truth for last step
        device: Device
        hidden_representation: Hidden state from sequence model (tuple for LSTM or tensor for Transformer)
    
    Returns:
        fraud_prob: Fraud probability
        is_fraud: Whether it is fraud (bool)
    """
    judge_model.eval()
    
    # Convert to tensor and add batch dimension
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(0).to(device)  # [1, pred_dim]
    
    # Process predictions if provided
    pred_tensor = None
    if predictions is not None:
        pred_tensor = torch.tensor(predictions, dtype=torch.float32).unsqueeze(0).to(device)  # [1, pred_dim]
    
    # Process hidden representation
    hidden_tensor = None
    if hidden_representation is not None:
        if isinstance(hidden_representation, tuple):
            # LSTM: (h_n, c_n) where h_n: [num_layers * num_directions, B, hidden_size]
            # Take the last layer's hidden state
            h_n, c_n = hidden_representation
            hidden_tensor = h_n[-1]  # [B, hidden_size] -> [1, hidden_size] for single sample
        elif len(hidden_representation.shape) == 3:
            # Transformer: [B, T, hidden_dim], take the last time step
            hidden_tensor = hidden_representation[:, -1, :]  # [B, hidden_dim] -> [1, hidden_dim] for single sample
        else:
            # Already [B, hidden_dim]
            hidden_tensor = hidden_representation
    
    with torch.no_grad():
        logits = judge_model(pred_tensor, targets, hidden_tensor)  # [1, 2]
        probs = torch.softmax(logits, dim=1)  # [1, 2]
        fraud_prob = probs[0, 1].item()  # Fraud probability
        is_fraud = torch.argmax(logits, dim=1)[0].item() == 1
    
    return fraud_prob, is_fraud


def main():
    parser = argparse.ArgumentParser(description="Fraud detection inference for a single CSV file")
    
    # Required arguments
    parser.add_argument("--folder", type=str, required=True,
                       help="Folder path containing CSV files")
    parser.add_argument("--member_id", type=str, required=True,
                       help="Member ID (CSV file will be: member_<member_id>.csv)")
    parser.add_argument("--sequence_model_path", type=str, required=True,
                       help="Sequence model path (e.g., checkpoints/best_model_enc.pth)")
    parser.add_argument("--judge_model_path", type=str, required=True,
                       help="Judge model path (e.g., checkpoints/best_judge_model.pth)")
    
    # Optional arguments
    parser.add_argument("--max_len", type=int, default=50,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu), default auto-select")
    
    args = parser.parse_args()
    
    # Construct CSV file path from folder and member_id
    csv_file = os.path.join(args.folder, f"member_{args.member_id}.csv")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load sequence model
    sequence_model, sequence_args = load_seq_model(args.sequence_model_path, device)
    
    # Get feature names (from model parameters)
    feature_names = getattr(sequence_args, 'feature_names', None)
    if feature_names is None or len(feature_names) == 0:
        print("Warning: Feature names not found in model parameters, will infer from data")
        feature_names = None
    
    # Prepare inference data
    print("\n" + "="*60)
    print("Preparing inference data")
    print("="*60)
    X, mask, last_row_features, last_row_target, target_indices, target_names, inferred_feature_names = \
        prepare_inference_data(csv_file, args.max_len, feature_names)
    
    # Update feature names
    feature_names = inferred_feature_names
    
    # Perform inference using sequence model
    print("\n" + "="*60)
    print("Sequence model inference")
    print("="*60)
    predictions, hidden_state = infer_sequence(sequence_model, X, mask, feature_names, device)
    print(f"Prediction shape: {predictions.shape}")
    
    # Get prediction for last step (corresponding to last row)
    # predictions: [1, T, pred_dim]
    # We need the prediction for the last step, i.e., predictions[0, -1, :]
    last_step_pred = predictions[0, -1, :].cpu().numpy()  # [pred_dim]
    
    # However, the model predicts the next step, so we need to find the corresponding prediction dimensions
    # According to train_judge.py logic, need to select target feature dimensions from full predictions
    # Get target_names from model parameters, use default order if not available
    model_target_names = getattr(sequence_args, 'target_names', None)
    if model_target_names is None or len(model_target_names) == 0:
        # Default order (consistent with train_judge.py)
        model_target_names = [
            'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
            'is_int', 'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
        ]
    
    # Find indices of target features in model predictions
    model_name_to_idx = {n: i for i, n in enumerate(model_target_names)}
    selected_pred_indices = [model_name_to_idx[n] for n in target_names if n in model_name_to_idx]
    
    # Select corresponding prediction dimensions
    if len(selected_pred_indices) > 0:
        last_step_pred_selected = last_step_pred[selected_pred_indices]  # [selected_pred_dim]
    else:
        # If no match found, use all predictions (may have dimension mismatch)
        print("Warning: Cannot find target feature indices in model predictions, using all predictions")
        last_step_pred_selected = last_step_pred
    
    print(f"Last step prediction (selected target features): {last_step_pred_selected}")
    print(f"Last step ground truth: {last_row_target}")
    
    # Load judge model
    print("\n" + "="*60)
    print("Loading Judge model")
    print("="*60)
    judge_model, judge_args = load_judge_model(args.judge_model_path, device, pred_dim=len(target_indices))
    
    # Use judge model for judgment
    print("\n" + "="*60)
    print("Judge model inference")
    print("="*60)
    # Process hidden state for judge model
    # For LSTM: take last layer's hidden state
    # For Transformer: take last time step
    if isinstance(hidden_state, tuple):
        h_n, c_n = hidden_state
        hidden_rep = h_n[-1]  # [1, hidden_size]
    else:
        hidden_rep = hidden_state[:, -1, :]  # [1, hidden_dim] - last time step
    
    fraud_prob, is_fraud = infer_judge(judge_model, last_step_pred_selected, last_row_target, device, hidden_rep)
    
    # Output results
    print("\n" + "="*60)
    print("Inference results")
    print("="*60)
    print(f"Member ID: {args.member_id}")
    print(f"CSV file: {csv_file}")
    print(f"Fraud probability: {fraud_prob:.4f}")
    print(f"Judgment: {'Fraud' if is_fraud else 'Normal'}")
    print("="*60)
    
    # Return result dictionary (can be used for subsequent processing)
    return {
        'member_id': args.member_id,
        'csv_file': csv_file,
        'fraud_probability': fraud_prob,
        'is_fraud': is_fraud,
        'predictions': last_step_pred_selected.tolist(),
        'targets': last_row_target.tolist()
    }


if __name__ == "__main__":
    result = main()

