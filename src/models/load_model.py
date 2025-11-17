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
    
    # Check if this is an old model (without hidden_dim and use_pred parameters)
    # Old models use: predictions + targets + errors + statistics
    # New models use: hidden_representation + (optionally predictions) + targets + statistics
    has_use_pred = hasattr(args, 'use_pred')
    has_hidden_dim = hasattr(args, 'hidden_dim') or 'hidden_dim' in checkpoint_args if isinstance(checkpoint_args, dict) else False
    
    # Always try to infer configuration from checkpoint state_dict first (most reliable)
    state_dict = checkpoint['model_state_dict']
    input_dim = None
    inferred_pred_dim = None
    inferred_hidden_dim = None
    inferred_use_pred = None
    
    if 'dnn.0.weight' in state_dict:
        input_dim = state_dict['dnn.0.weight'].shape[1]
        print(f"Checkpoint model input dimension: {input_dim}")
        
        # Try to determine if this is an old or new model
        # Old model: input_dim = pred_dim * 3 + 6 (if use_basic_features and use_statistical_features)
        # New model (default, use_pred=False): input_dim = hidden_dim + 6 (if use_statistical_features)
        # New model (use_pred=True): input_dim = hidden_dim + pred_dim * 3 + 6
        
        if not has_use_pred and not has_hidden_dim:
            # This is an old model - use old interface
            print("Detected old model format (without hidden_dim and use_pred)")
            inferred_use_pred = True  # Old models always use predictions
            inferred_hidden_dim = None
            
            # Infer pred_dim from input_dim using old formula
            # Try different possible configurations
            if use_basic_features and use_statistical_features:
                # Try standard formula: input_dim = pred_dim * 3 + 6
                remainder = (input_dim - 6) % 3
                if remainder == 0:
                    inferred_pred_dim = (input_dim - 6) // 3
                else:
                    # Try alternative formulas
                    # Maybe it's pred_dim * 3 + 8 (8 statistical features?)
                    remainder2 = (input_dim - 8) % 3
                    if remainder2 == 0:
                        inferred_pred_dim = (input_dim - 8) // 3
                        print(f"Warning: Using pred_dim*3+8 formula (input_dim={input_dim}, pred_dim={inferred_pred_dim})")
                    else:
                        # Try pred_dim * 3 + 7
                        remainder3 = (input_dim - 7) % 3
                        if remainder3 == 0:
                            inferred_pred_dim = (input_dim - 7) // 3
                            print(f"Warning: Using pred_dim*3+7 formula (input_dim={input_dim}, pred_dim={inferred_pred_dim})")
                        else:
                            # Use closest integer
                            inferred_pred_dim = round((input_dim - 6) / 3)
                            print(f"Warning: input_dim={input_dim} doesn't match standard formulas, using closest pred_dim={inferred_pred_dim}")
            elif use_basic_features and not use_statistical_features:
                # input_dim = pred_dim * 3
                inferred_pred_dim = input_dim // 3
            elif not use_basic_features and use_statistical_features:
                # input_dim = 6, pred_dim doesn't matter
                inferred_pred_dim = 1  # Dummy value, won't be used
            else:
                raise ValueError("Cannot infer pred_dim from old model configuration")
            print(f"Inferred pred_dim={inferred_pred_dim} from old model checkpoint (input_dim={input_dim})")
        else:
            # This is a new model - use new interface
            print("Detected new model format (with hidden_dim and/or use_pred)")
            inferred_use_pred = getattr(args, 'use_pred', False)
            inferred_hidden_dim = getattr(args, 'hidden_dim', None)
            
            # If hidden_dim is not in args, try to infer from input_dim
            if inferred_hidden_dim is None:
                # Try to infer from input_dim
                if inferred_use_pred and use_basic_features and use_statistical_features:
                    # input_dim = hidden_dim + pred_dim * 3 + 6
                    # We need pred_dim to infer hidden_dim, so use provided pred_dim
                    if pred_dim is not None:
                        inferred_hidden_dim = input_dim - pred_dim * 3 - 6
                    else:
                        # Can't infer without pred_dim, use a default
                        print("Warning: Cannot infer hidden_dim without pred_dim, will try to infer from sequence model")
                        inferred_hidden_dim = None
                elif use_statistical_features:
                    # input_dim = hidden_dim + 6 (default case)
                    inferred_hidden_dim = input_dim - 6
                else:
                    # input_dim = hidden_dim (only hidden representation)
                    inferred_hidden_dim = input_dim
                print(f"Inferred hidden_dim={inferred_hidden_dim} from new model checkpoint (input_dim={input_dim})")
            
            # Infer pred_dim if not provided
            if pred_dim is None:
                if inferred_use_pred and use_basic_features and use_statistical_features and inferred_hidden_dim is not None:
                    # input_dim = hidden_dim + pred_dim * 3 + 6
                    inferred_pred_dim = (input_dim - inferred_hidden_dim - 6) // 3
                elif inferred_use_pred and use_basic_features and not use_statistical_features and inferred_hidden_dim is not None:
                    # input_dim = hidden_dim + pred_dim * 3
                    inferred_pred_dim = (input_dim - inferred_hidden_dim) // 3
                else:
                    # Use default or from args
                    inferred_pred_dim = getattr(args, 'pred_dim', None)
    
    # Use inferred values
    if inferred_pred_dim is not None:
        if pred_dim is not None and pred_dim != inferred_pred_dim:
            print(f"Warning: Provided pred_dim={pred_dim} differs from checkpoint pred_dim={inferred_pred_dim}. Using checkpoint value.")
        pred_dim = inferred_pred_dim
    elif pred_dim is None:
        # Fallback: try to get from checkpoint args or use default
        pred_dim = getattr(args, 'pred_dim', None)
        if pred_dim is None:
            target_names = [
                'Product ID_enc', 'Amount', 'Action Type_enc', 'Source Type_enc',
                'is_int',
                'Post Date_doy', 'Post Time_hour', 'Post Time_minute'
            ]
            pred_dim = len(target_names)
            print(f"Using default pred_dim={pred_dim} based on target_names")
    
    # Determine final configuration
    final_use_pred = inferred_use_pred if inferred_use_pred is not None else getattr(args, 'use_pred', False)
    final_hidden_dim = inferred_hidden_dim if inferred_hidden_dim is not None else getattr(args, 'hidden_dim', None)
    
    print(f"Building judge model with:")
    print(f"  pred_dim={pred_dim}")
    print(f"  use_pred={final_use_pred}")
    print(f"  hidden_dim={final_hidden_dim}")
    print(f"  use_statistical_features={use_statistical_features}")
    print(f"  use_basic_features={use_basic_features}")
    
    # Build judge model with same configuration as training
    judge_model = build_judge_model(
        pred_dim=pred_dim,
        hidden_dims=getattr(args, 'judge_hidden_dims', [64, 32, 16]),
        dropout=getattr(args, 'judge_dropout', 0.2),
        use_attention=getattr(args, 'judge_use_attention', False),
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features,
        hidden_dim=final_hidden_dim,
        use_pred=final_use_pred
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