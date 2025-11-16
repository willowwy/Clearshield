import torch
from src.models.backbone_model import build_seq_model
from src.models.judge import build_judge_model

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
    
def load_seq_model(model_path, device):
    """Load trained sequence model"""
    print(f"Loading trained model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model parameters
    args = checkpoint['args']
    print(f"Model trained for {checkpoint['epoch']+1} epochs")
    print(f"Best validation loss (MSE): {checkpoint.get('val_loss', 0.0):.6f}")
    
    # Rebuild model
    model = build_seq_model(args).to(device)
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