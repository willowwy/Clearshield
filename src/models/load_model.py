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
            # Note: New architecture uses hidden_dim * 2 (with delta)
            if inferred_hidden_dim is None:
                # Try to infer from input_dim
                # New architecture: input_dim = hidden_dim * 2 + (pred_dim * 3 if use_pred) + (6 if use_statistical_features)
                if inferred_use_pred and use_basic_features and use_statistical_features:
                    # input_dim = hidden_dim * 2 + pred_dim * 3 + 6
                    # We need pred_dim to infer hidden_dim, so use provided pred_dim
                    if pred_dim is not None:
                        # Try new architecture first (with delta)
                        remainder = (input_dim - pred_dim * 3 - 6) % 2
                        if remainder == 0:
                            inferred_hidden_dim = (input_dim - pred_dim * 3 - 6) // 2
                            print(f"Inferred hidden_dim={inferred_hidden_dim} from new architecture (with delta): input_dim={input_dim} = {inferred_hidden_dim}*2 + {pred_dim}*3 + 6")
                        else:
                            # Fallback to old architecture (without delta)
                            inferred_hidden_dim = input_dim - pred_dim * 3 - 6
                            print(f"Inferred hidden_dim={inferred_hidden_dim} from old architecture (without delta): input_dim={input_dim} = {inferred_hidden_dim} + {pred_dim}*3 + 6")
                    else:
                        # Can't infer without pred_dim, use a default
                        print("Warning: Cannot infer hidden_dim without pred_dim, will try to infer from sequence model")
                        inferred_hidden_dim = None
                elif use_statistical_features:
                    # input_dim = hidden_dim * 2 + 6 (new architecture with delta)
                    # or input_dim = hidden_dim + 6 (old architecture without delta)
                    remainder = (input_dim - 6) % 2
                    if remainder == 0:
                        inferred_hidden_dim = (input_dim - 6) // 2
                        print(f"Inferred hidden_dim={inferred_hidden_dim} from new architecture (with delta): input_dim={input_dim} = {inferred_hidden_dim}*2 + 6")
                    else:
                        inferred_hidden_dim = input_dim - 6
                        print(f"Inferred hidden_dim={inferred_hidden_dim} from old architecture (without delta): input_dim={input_dim} = {inferred_hidden_dim} + 6")
                else:
                    # input_dim = hidden_dim * 2 (new architecture) or hidden_dim (old architecture)
                    remainder = input_dim % 2
                    if remainder == 0:
                        inferred_hidden_dim = input_dim // 2
                        print(f"Inferred hidden_dim={inferred_hidden_dim} from new architecture (with delta): input_dim={input_dim} = {inferred_hidden_dim}*2")
                    else:
                        inferred_hidden_dim = input_dim
                        print(f"Inferred hidden_dim={inferred_hidden_dim} from old architecture (without delta): input_dim={input_dim}")
            
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
    
    # Get CNN-related parameters from checkpoint args
    # Check both max_len and hidden_len (hidden_len is the new parameter name)
    max_len = getattr(args, 'hidden_len', getattr(args, 'max_len', 50))
    cnn_out_channels = getattr(args, 'judge_cnn_out_channels', None)
    cnn_kernel_sizes = getattr(args, 'judge_cnn_kernel_sizes', None)
    
    # Try to infer CNN parameters from state_dict if not in args
    if cnn_out_channels is None or cnn_kernel_sizes is None:
        state_dict = checkpoint.get('model_state_dict', {})
        # Check if model has CNN layers (new architecture)
        if 'cnn_layers.0.0.weight' in state_dict:
            # Infer cnn_out_channels from first CNN layer
            cnn_weight_shape = state_dict['cnn_layers.0.0.weight'].shape
            inferred_cnn_out_channels = cnn_weight_shape[0]  # [out_channels, in_channels, kernel_size]
            inferred_hidden_dim_from_cnn = cnn_weight_shape[1]  # This is the actual hidden_dim
            
            # Infer cnn_kernel_sizes by checking all CNN layers
            inferred_kernel_sizes = []
            layer_idx = 0
            while f'cnn_layers.{layer_idx}.0.weight' in state_dict:
                kernel_size = state_dict[f'cnn_layers.{layer_idx}.0.weight'].shape[2]
                inferred_kernel_sizes.append(int(kernel_size))
                layer_idx += 1
            
            if cnn_out_channels is None:
                cnn_out_channels = inferred_cnn_out_channels
                print(f"Inferred cnn_out_channels={cnn_out_channels} from checkpoint")
            
            if cnn_kernel_sizes is None:
                cnn_kernel_sizes = inferred_kernel_sizes
                print(f"Inferred cnn_kernel_sizes={cnn_kernel_sizes} from checkpoint")
            
            # Update hidden_dim if inferred from CNN (more accurate)
            if inferred_hidden_dim_from_cnn is not None and final_hidden_dim != inferred_hidden_dim_from_cnn:
                print(f"Warning: hidden_dim mismatch. Inferred from input_dim: {final_hidden_dim}, from CNN: {inferred_hidden_dim_from_cnn}")
                print(f"Using hidden_dim={inferred_hidden_dim_from_cnn} from CNN weights (more accurate)")
                final_hidden_dim = inferred_hidden_dim_from_cnn
        else:
            # Old model without CNN, use defaults
            if cnn_out_channels is None:
                cnn_out_channels = 64
            if cnn_kernel_sizes is None:
                cnn_kernel_sizes = [3, 5, 7]
    
    print(f"Building judge model with:")
    print(f"  pred_dim={pred_dim}")
    print(f"  use_pred={final_use_pred}")
    print(f"  hidden_dim={final_hidden_dim}")
    print(f"  use_statistical_features={use_statistical_features}")
    print(f"  use_basic_features={use_basic_features}")
    print(f"  max_len={max_len}")
    print(f"  cnn_out_channels={cnn_out_channels}")
    print(f"  cnn_kernel_sizes={cnn_kernel_sizes}")
    
    # Build judge model with same configuration as training
    judge_model = build_judge_model(
        pred_dim=pred_dim,
        hidden_dims=getattr(args, 'judge_hidden_dims', [16, 8]),
        dropout=getattr(args, 'judge_dropout', 0.2),
        use_attention=getattr(args, 'judge_use_attention', False),
        use_statistical_features=use_statistical_features,
        use_basic_features=use_basic_features,
        hidden_dim=final_hidden_dim,
        use_pred=final_use_pred,
        max_len=max_len,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_sizes=cnn_kernel_sizes
    ).to(device)
    
    # Check if we need to adapt old model weights to new architecture (with delta)
    old_state_dict = checkpoint['model_state_dict']
    new_state_dict = judge_model.state_dict()
    
    # Check if input dimensions match
    old_input_weight = old_state_dict.get('dnn.0.weight', None)
    new_input_weight = new_state_dict.get('dnn.0.weight', None)
    
    if old_input_weight is not None and new_input_weight is not None:
        old_dim = old_input_weight.shape[1]
        new_dim = new_input_weight.shape[1]
        
        if old_dim != new_dim:
            # Need to adapt weights from old architecture to new architecture
            print(f"Detected dimension mismatch: old model input_dim={old_dim}, new model input_dim={new_dim}")
            print("Adapting weights from old model (without delta) to new model (with delta)...")
            
            # Calculate dimensions for old model
            # Old model: hidden_dim + (pred_dim * 3 if use_pred and use_basic_features) + (6 if use_statistical_features)
            old_hidden_dim = final_hidden_dim if final_hidden_dim is not None else 0
            old_pred_part = pred_dim * 3 if (final_use_pred and use_basic_features) else 0
            old_stat_part = 6 if use_statistical_features else 0
            old_total_expected = old_hidden_dim + old_pred_part + old_stat_part
            
            # New model: hidden_dim * 2 + (pred_dim * 3 if use_pred and use_basic_features) + (6 if use_statistical_features)
            new_hidden_dim = final_hidden_dim * 2 if final_hidden_dim is not None else 0
            new_pred_part = pred_dim * 3 if (final_use_pred and use_basic_features) else 0
            new_stat_part = 6 if use_statistical_features else 0
            new_total_expected = new_hidden_dim + new_pred_part + new_stat_part
            
            if old_dim == old_total_expected and new_dim == new_total_expected:
                # This is the case: old model without delta, new model with delta
                # Adapt weights: [old_hidden, old_pred, old_stat] -> [old_hidden, zeros, old_pred, old_stat]
                adapted_state_dict = {}
                
                for key, old_weight in old_state_dict.items():
                    if key == 'dnn.0.weight':
                        # First layer weight: [out_dim, in_dim]
                        # Move old_weight to device if needed
                        old_weight = old_weight.to(device)
                        out_dim = old_weight.shape[0]
                        old_in_dim = old_weight.shape[1]
                        
                        # Create new weight tensor
                        new_weight = torch.zeros(out_dim, new_dim, device=device, dtype=old_weight.dtype)
                        
                        # Map old weights to new positions
                        new_offset = 0
                        
                        # 1. Copy old hidden part to new hidden[t] part
                        if old_hidden_dim > 0:
                            new_weight[:, new_offset:new_offset + old_hidden_dim] = old_weight[:, 0:old_hidden_dim]
                            new_offset += old_hidden_dim
                            # 2. Add zeros for delta part
                            new_offset += old_hidden_dim  # Skip delta (zeros)
                        
                        # 3. Copy old pred part
                        if old_pred_part > 0:
                            old_pred_start = old_hidden_dim
                            new_weight[:, new_offset:new_offset + old_pred_part] = old_weight[:, old_pred_start:old_pred_start + old_pred_part]
                            new_offset += old_pred_part
                        
                        # 4. Copy old stat part
                        if old_stat_part > 0:
                            old_stat_start = old_hidden_dim + old_pred_part
                            new_weight[:, new_offset:new_offset + old_stat_part] = old_weight[:, old_stat_start:old_stat_start + old_stat_part]
                        
                        adapted_state_dict[key] = new_weight
                    else:
                        # Other layers remain the same, but move to device
                        adapted_state_dict[key] = old_weight.to(device) if isinstance(old_weight, torch.Tensor) else old_weight
                
                # Load adapted state dict
                judge_model.load_state_dict(adapted_state_dict)
                print("Successfully adapted old model weights to new architecture with delta")
            else:
                # Dimension mismatch but not due to delta addition - try direct load with strict=False
                print(f"Warning: Dimension mismatch detected but not due to delta addition.")
                print(f"Old expected: {old_total_expected}, Old actual: {old_dim}")
                print(f"New expected: {new_total_expected}, New actual: {new_dim}")
                print("Attempting to load with strict=False (may fail)...")
                judge_model.load_state_dict(old_state_dict, strict=False)
        else:
            # Dimensions match, load normally
            judge_model.load_state_dict(old_state_dict)
    else:
        # Load model state normally
        judge_model.load_state_dict(old_state_dict)
    # Don't set eval mode here - let the caller decide (train or eval)
    
    # Count parameters
    total_params, trainable_params = count_parameters(judge_model)
    print(f"Judge model parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    
    return judge_model, args