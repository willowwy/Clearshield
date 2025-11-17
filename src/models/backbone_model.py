import json
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeCatLSTM(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # Embedding dropout
        self.embedding_dropout = getattr(args, 'embedding_dropout', 0.0)
        self.emb_dropout = nn.Dropout(self.embedding_dropout) if self.embedding_dropout > 0 else nn.Identity()

        # Time embeddings
        self.post_day_emb = nn.Embedding(args.day_vocab, args.day_emb_dim)
        self.post_hour_emb = nn.Embedding(args.hour_vocab, args.hour_emb_dim)
        self.post_min_emb = nn.Embedding(args.minute_vocab, args.minute_emb_dim)
        self.open_day_emb = nn.Embedding(args.aod_day_vocab, args.aod_day_emb_dim)

        time_total_dim = (
            args.day_emb_dim
            + args.hour_emb_dim
            + args.minute_emb_dim
            + args.aod_day_emb_dim
        )

        # Categorical embeddings - direct configuration
        self.cat_embs = nn.ModuleDict()
        cat_total_dim = 0
        
        # Configure categorical embeddings based on feature names
        if hasattr(args, 'feature_names') and args.feature_names:
            for name in args.feature_names:
                if name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                    if name == "Member Age":
                        vocab, emb = 10, 10  # //10 then do embedding
                    elif name == "Amount":
                        vocab, emb = 12, 10  # 12 buckets
                    elif name == "is_int":
                        vocab, emb = 2, 4   # Boolean value
                    elif name == "account_age_quantized":
                        vocab, emb = 5, 5   # 5 age stages
                    elif name == "cluster_id":
                        vocab, emb = 60, 32  # cluster_id embedding
                    elif "Account Type" in name:
                        vocab, emb = 15, 16
                    elif "Product ID" in name:
                        vocab, emb = 160, 32  # Match Transformer configuration
                    elif "Action Type" in name:
                        vocab, emb = 5, 5
                    elif "Source Type" in name:
                        vocab, emb = 20, 16  # Match Transformer configuration
                    else:
                        vocab, emb = 50, 4  # Default configuration
                    
                    self.cat_embs[name] = nn.Embedding(vocab, emb)
                    cat_total_dim += emb

        # Total input dimension after embeddings (for consistency with Transformer)
        self.input_dim = time_total_dim + cat_total_dim + args.cont_dim
        lstm_input_dim = self.input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            batch_first=True,
            dropout=args.dropout if args.lstm_layers > 1 else 0.0,
            bidirectional=args.bidirectional,
        )
        last_dim = args.lstm_hidden * (2 if args.bidirectional else 1)
        # Prediction head for next-step vector forecasting
        # Determine prediction dimension: prefer explicit pred_dim, otherwise use number of target_names
        inferred_pred_dim = getattr(args, 'pred_dim', 0)
        target_names = getattr(args, 'target_names', []) or []
        if not inferred_pred_dim:
            if target_names:
                inferred_pred_dim = len(target_names)
            else:
                inferred_pred_dim = len(getattr(args, 'feature_names', []) or [])
        self.pred_dim = inferred_pred_dim
        # Store target_names for compatibility with train_judge.py
        self.target_names = target_names if target_names else (getattr(args, 'feature_names', []) or [])
        self.head = nn.Linear(last_dim, self.pred_dim)

    def forward(
        self,
        x: torch.Tensor,             # [B, T, total_feature_dim] - all features already prepared
        mask: torch.Tensor,          # [B, T]
        feature_names: list,         # feature names list to split feature types
        use_pack: bool = True,
    ) -> torch.Tensor:
        # Separate different types of features from feature names
        time_features = {}
        categorical_features = {}
        continuous_features = []
        
        # Record feature separation information
        debug_info = []
        debug_info.append("=== Feature Separation Debug ===")
        debug_info.append(f"Total features: {len(feature_names)}")
        debug_info.append(f"Feature names: {feature_names}")
        debug_info.append(f"Input X shape: {x.shape}")
        debug_info.append(f"Input X min/max: {x.min()}/{x.max()}")
        debug_info.append("=" * 50)
        
        for i, name in enumerate(feature_names):
            feature_values = x[:, :, i]
            
            # Record information for each feature
            debug_info.append(f"Feature {i} ({name}): min={feature_values.min()}, max={feature_values.max()}, dtype={feature_values.dtype}")
            
            if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                time_features[name] = feature_values.long()
                debug_info.append(f"  -> Time feature")
                debug_info.append(f"    Original range: {feature_values.min()} to {feature_values.max()}")
                debug_info.append(f"    Converted to long: {time_features[name].min()} to {time_features[name].max()}")
            elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                # Ensure categorical features are integers and non-negative
                # For negative values, we map them to 0
                clamped_values = torch.clamp(feature_values, min=0).long()
                categorical_features[name] = clamped_values
                debug_info.append(f"  -> Categorical feature (clamped to non-negative)")
                debug_info.append(f"    Original range: {feature_values.min()} to {feature_values.max()}")
                debug_info.append(f"    Clamped range: {clamped_values.min()} to {clamped_values.max()}")
            else:
                continuous_features.append(x[:, :, i:i+1])
                debug_info.append(f"  -> Continuous feature")
        
        debug_info.append(f"Time features: {list(time_features.keys())}")
        debug_info.append(f"Categorical features: {list(categorical_features.keys())}")
        debug_info.append(f"Continuous features count: {len(continuous_features)}")
        debug_info.append("=" * 50)
        
        # Write all debug information at once
        with open("time_features_debug.txt", "a") as f:
            for line in debug_info:
                f.write(line + "\n")
        
        # Time embeddings - add detailed safety checks and debug information
        time_debug_info = []
        time_debug_info.append("=== Time Features Debug (Before Clamp) ===")
        time_debug_info.append(f"Post Date_doy raw: {time_features['Post Date_doy'].cpu().numpy()}")
        time_debug_info.append(f"Post Time_hour raw: {time_features['Post Time_hour'].cpu().numpy()}")
        time_debug_info.append(f"Post Time_minute raw: {time_features['Post Time_minute'].cpu().numpy()}")
        time_debug_info.append(f"Account Open Date_doy raw: {time_features['Account Open Date_doy'].cpu().numpy()}")
        time_debug_info.append(f"Post Date_doy raw min/max: {time_features['Post Date_doy'].min()}/{time_features['Post Date_doy'].max()}")
        time_debug_info.append(f"Post Time_hour raw min/max: {time_features['Post Time_hour'].min()}/{time_features['Post Time_hour'].max()}")
        time_debug_info.append(f"Post Time_minute raw min/max: {time_features['Post Time_minute'].min()}/{time_features['Post Time_minute'].max()}")
        time_debug_info.append(f"Account Open Date_doy raw min/max: {time_features['Account Open Date_doy'].min()}/{time_features['Account Open Date_doy'].max()}")
        time_debug_info.append("=" * 50)
        
        # Ensure all time features are non-negative to avoid embedding layer index out of bounds
        post_date_doy = torch.clamp(time_features["Post Date_doy"], min=0, max=365)
        post_time_hour = torch.clamp(time_features["Post Time_hour"], min=0, max=23)
        post_time_minute = torch.clamp(time_features["Post Time_minute"], min=0, max=59)
        account_open_doy = torch.clamp(time_features["Account Open Date_doy"], min=0, max=365)
        
        # Record values after clamp and embedding vocabulary sizes
        time_debug_info.append("=== Time Features Debug (After Clamp) ===")
        time_debug_info.append(f"Post Date_doy: {post_date_doy.cpu().numpy()}")
        time_debug_info.append(f"Post Time_hour: {post_time_hour.cpu().numpy()}")
        time_debug_info.append(f"Post Time_minute: {post_time_minute.cpu().numpy()}")
        time_debug_info.append(f"Account Open Date_doy: {account_open_doy.cpu().numpy()}")
        time_debug_info.append(f"Post Date_doy min/max: {post_date_doy.min()}/{post_date_doy.max()}")
        time_debug_info.append(f"Post Time_hour min/max: {post_time_hour.min()}/{post_time_hour.max()}")
        time_debug_info.append(f"Post Time_minute min/max: {post_time_minute.min()}/{post_time_minute.max()}")
        time_debug_info.append(f"Account Open Date_doy min/max: {account_open_doy.min()}/{account_open_doy.max()}")
        time_debug_info.append(f"Embedding vocab sizes: day={self.post_day_emb.num_embeddings}, hour={self.post_hour_emb.num_embeddings}, min={self.post_min_emb.num_embeddings}, aod={self.open_day_emb.num_embeddings}")
        time_debug_info.append("=" * 50)
        
        # Write time debug information at once
        with open("time_features_debug.txt", "a") as f:
            for line in time_debug_info:
                f.write(line + "\n")
        
        try:
            t_emb = torch.cat([
                self.post_day_emb(post_date_doy),
                self.post_hour_emb(post_time_hour),
                self.post_min_emb(post_time_minute),
                self.open_day_emb(account_open_doy),
            ], dim=-1).float()  # [B,T,Dt] Ensure float32 type
            # Apply embedding dropout
            t_emb = self.emb_dropout(t_emb)
        except Exception as e:
            # Record error information to file
            with open("embedding_error.txt", "a") as f:
                f.write(f"=== Embedding Error ===\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Post Date_doy shape: {post_date_doy.shape}, dtype: {post_date_doy.dtype}\n")
                f.write(f"Post Time_hour shape: {post_time_hour.shape}, dtype: {post_time_hour.dtype}\n")
                f.write(f"Post Time_minute shape: {post_time_minute.shape}, dtype: {post_time_minute.dtype}\n")
                f.write(f"Account Open Date_doy shape: {account_open_doy.shape}, dtype: {account_open_doy.dtype}\n")
                f.write(f"Embedding vocab sizes: day={self.post_day_emb.num_embeddings}, hour={self.post_hour_emb.num_embeddings}, min={self.post_min_emb.num_embeddings}, aod={self.open_day_emb.num_embeddings}\n")
                f.write("=" * 50 + "\n")
            raise e

        # Categorical embeddings - add debug information
        c_emb = None
        if len(self.cat_embs) > 0 and categorical_features:
            # Record categorical feature information
            with open("time_features_debug.txt", "a") as f:
                f.write(f"=== Categorical Features Debug ===\n")
                for name, values in categorical_features.items():
                    f.write(f"{name}: {values.cpu().numpy()}\n")
                    f.write(f"{name} min/max: {values.min()}/{values.max()}\n")
                    if name in self.cat_embs:
                        f.write(f"{name} embedding vocab size: {self.cat_embs[name].num_embeddings}\n")
                f.write("=" * 50 + "\n")
            
            # Final clamp processing for categorical features to prevent out-of-bounds
            # Note: categorical_features have already been clamped to non-negative values in the feature separation stage
            final_categorical_features = {}
            for name, values in categorical_features.items():
                if name in self.cat_embs:
                    vocab_size = self.cat_embs[name].num_embeddings
                    # Ensure values are within embedding vocabulary range
                    final_values = torch.clamp(values, min=0, max=vocab_size-1)
                    final_categorical_features[name] = final_values
                    
                    # Record final clamp information
                    with open("time_features_debug.txt", "a") as f:
                        f.write(f"{name} final values: {final_values.cpu().numpy()}\n")
                        f.write(f"{name} final min/max: {final_values.min()}/{final_values.max()}\n")
                        f.write(f"{name} vocab size: {vocab_size}\n")
                else:
                    final_categorical_features[name] = values
            
            # Execute embedding
            cat_list = [self.cat_embs[name](final_categorical_features[name]) for name in self.cat_embs]
            c_emb = torch.cat(cat_list, dim=-1).float()  # Ensure float32 type
            # Apply embedding dropout
            c_emb = self.emb_dropout(c_emb)
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x_emb = torch.cat([t_emb, cont_x, c_emb], dim=-1)
        else:
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x_emb = torch.cat([t_emb, cont_x], dim=-1)
        
        # Verify dimension match (for consistency with Transformer and debugging)
        if x_emb.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"Dimension mismatch: x_emb has shape {x_emb.shape} (last dim={x_emb.shape[-1]}), "
                f"but LSTM expects {self.input_dim}. "
                f"Time dim: {t_emb.shape[-1]}, Cont dim: {cont_x.shape[-1]}, "
                f"Cat dim: {c_emb.shape[-1] if c_emb is not None else 0}, "
                f"Cat embeddings: {list(self.cat_embs.keys())}, "
                f"Categorical features found: {list(categorical_features.keys())}, "
                f"Feature names: {feature_names}"
            )
        
        x = x_emb

        if use_pack:
            lengths = mask.sum(dim=1).long().clamp_min(1)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            # For packed sequences, hidden state is the last valid hidden state
            hidden_state = (h_n, c_n)
        else:
            out, (h_n, c_n) = self.lstm(x)
            hidden_state = (h_n, c_n)

        # Time-distributed linear to produce per-step predictions [B, T, pred_dim]
        preds = self.head(out)
        return preds, hidden_state


class FraudEnc(nn.Module):
    """Encoder-only Transformer model for fraud detection sequence modeling."""
    
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        
        # Embedding dropout
        self.embedding_dropout = getattr(args, 'embedding_dropout', 0.0)
        self.emb_dropout = nn.Dropout(self.embedding_dropout) if self.embedding_dropout > 0 else nn.Identity()
        
        # Time embeddings
        self.post_day_emb = nn.Embedding(args.day_vocab, args.day_emb_dim)
        self.post_hour_emb = nn.Embedding(args.hour_vocab, args.hour_emb_dim)
        self.post_min_emb = nn.Embedding(args.minute_vocab, args.minute_emb_dim)
        self.open_day_emb = nn.Embedding(args.aod_day_vocab, args.aod_day_emb_dim)
        
        time_total_dim = (
            args.day_emb_dim
            + args.hour_emb_dim
            + args.minute_emb_dim
            + args.aod_day_emb_dim
        )
        
        # Categorical embeddings - direct configuration
        self.cat_embs = nn.ModuleDict()
        cat_total_dim = 0
        
        # Configure categorical embeddings based on feature names
        if hasattr(args, 'feature_names') and args.feature_names:
            for name in args.feature_names:
                if name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                    if name == "Member Age":
                        vocab, emb = 10, 10
                    elif name == "Amount":
                        vocab, emb = 12, 10
                    elif name == "is_int":
                        vocab, emb = 2, 4
                    elif name == "account_age_quantized":
                        vocab, emb = 5, 5
                    elif name == "cluster_id":
                        vocab, emb = 60, 32  # cluster_id embedding
                    elif "Account Type" in name:
                        vocab, emb = 15, 16
                    elif "Product ID" in name:
                        vocab, emb = 160, 32
                    elif "Action Type" in name:
                        vocab, emb = 5, 5
                    elif "Source Type" in name:
                        vocab, emb = 20, 16

                    else:
                        vocab, emb = 50, 4
                    
                    self.cat_embs[name] = nn.Embedding(vocab, emb)
                    cat_total_dim += emb
        
        # Total input dimension after embeddings
        self.input_dim = time_total_dim + cat_total_dim + args.cont_dim
        
        # Transformer embedding dimension (can be configured)
        transformer_dim = getattr(args, 'transformer_dim', args.lstm_hidden if hasattr(args, 'lstm_hidden') else 128)
        
        # Input projection to transformer dimension
        self.input_proj = nn.Linear(self.input_dim, transformer_dim)
        
        # Positional embedding (learned)
        max_seq_len = getattr(args, 'max_seq_len', 512)
        self.pos_emb = nn.Embedding(max_seq_len, transformer_dim)
        
        # Multi-layer Transformer Encoder
        dim_feedforward = getattr(args, 'dim_feedforward', 0)
        if dim_feedforward == 0:
            dim_feedforward = transformer_dim * 2
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=getattr(args, 'nhead', 3),
            dim_feedforward=dim_feedforward,
            dropout=getattr(args, 'dropout', 0.1),
            activation=getattr(args, 'activation', 'relu'),
            batch_first=True,
            norm_first=getattr(args, 'norm_first', False),
        )
        num_encoder_layers = getattr(args, 'num_encoder_layers', 1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Prediction head for next-step vector forecasting
        inferred_pred_dim = getattr(args, 'pred_dim', 0)
        target_names = getattr(args, 'target_names', []) or []
        if not inferred_pred_dim:
            if target_names:
                inferred_pred_dim = len(target_names)
            else:
                inferred_pred_dim = len(getattr(args, 'feature_names', []) or [])
        self.pred_dim = inferred_pred_dim
        # Store target_names for compatibility with train_judge.py
        self.target_names = target_names if target_names else (getattr(args, 'feature_names', []) or [])
        self.head = nn.Linear(transformer_dim, self.pred_dim)
        
        # Store transformer_dim for forward pass
        self.transformer_dim = transformer_dim
        
    def forward(
        self,
        x: torch.Tensor,             # [B, T, total_feature_dim]
        mask: torch.Tensor,          # [B, T]
        feature_names: list,         # feature names list to split feature types
        use_pack: bool = True,       # Not used for transformer, kept for compatibility
    ) -> torch.Tensor:
        # Separate different types of features from feature names
        time_features = {}
        categorical_features = {}
        continuous_features = []
        
        for i, name in enumerate(feature_names):
            feature_values = x[:, :, i]
            
            if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                time_features[name] = feature_values.long()
            elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                clamped_values = torch.clamp(feature_values, min=0).long()
                categorical_features[name] = clamped_values
            else:
                continuous_features.append(x[:, :, i:i+1])
        
        # Ensure all time features are non-negative and within valid ranges
        post_date_doy = torch.clamp(time_features["Post Date_doy"], min=0, max=365)
        post_time_hour = torch.clamp(time_features["Post Time_hour"], min=0, max=23)
        post_time_minute = torch.clamp(time_features["Post Time_minute"], min=0, max=59)
        account_open_doy = torch.clamp(time_features["Account Open Date_doy"], min=0, max=365)
        
        # Time embeddings
        t_emb = torch.cat([
            self.post_day_emb(post_date_doy),
            self.post_hour_emb(post_time_hour),
            self.post_min_emb(post_time_minute),
            self.open_day_emb(account_open_doy),
        ], dim=-1).float()  # [B, T, Dt]
        # Apply embedding dropout
        t_emb = self.emb_dropout(t_emb)
        
        # Categorical embeddings
        if len(self.cat_embs) > 0 and categorical_features:
            final_categorical_features = {}
            for name, values in categorical_features.items():
                if name in self.cat_embs:
                    vocab_size = self.cat_embs[name].num_embeddings
                    final_values = torch.clamp(values, min=0, max=vocab_size-1)
                    final_categorical_features[name] = final_values
                else:
                    final_categorical_features[name] = values
            
            cat_list = [self.cat_embs[name](final_categorical_features[name]) for name in self.cat_embs]
            c_emb = torch.cat(cat_list, dim=-1).float()
            # Apply embedding dropout
            c_emb = self.emb_dropout(c_emb)
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x_emb = torch.cat([t_emb, cont_x, c_emb], dim=-1)
        else:
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x_emb = torch.cat([t_emb, cont_x], dim=-1)
        
        # Verify dimension match
        if x_emb.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"Dimension mismatch: x_emb has shape {x_emb.shape} (last dim={x_emb.shape[-1]}), "
                f"but input_proj expects {self.input_dim}. "
                f"Time dim: {t_emb.shape[-1]}, Cont dim: {cont_x.shape[-1]}, "
                f"Cat dim: {c_emb.shape[-1] if len(self.cat_embs) > 0 and categorical_features else 0}, "
                f"Cat embeddings: {list(self.cat_embs.keys())}, "
                f"Categorical features found: {list(categorical_features.keys())}, "
                f"Feature names: {feature_names}"
            )
        
        # Project to transformer dimension
        x_proj = self.input_proj(x_emb)  # [B, T, transformer_dim]
        
        # Add positional embeddings
        B, T, _ = x_proj.shape
        positions = torch.arange(0, T, device=x_proj.device).unsqueeze(0).expand(B, -1)  # [B, T]
        pos_embeddings = self.pos_emb(positions)  # [B, T, transformer_dim]
        x_with_pos = x_proj + pos_embeddings
        
        # Create attention mask: 0 for valid positions, -inf for padding
        # Transformer expects mask where True means ignore (mask out)
        # Our mask: 1 = valid, 0 = padding, so we need to invert it
        attention_mask = (mask == 0).bool()  # [B, T], True where padding
        
        # Transformer encoder expects mask of shape [B*num_heads, T, T] or [T, T] or [B, T]
        # For batch_first=True, we use [B, T] mask which will be expanded internally
        # Pass None mask (all positions valid) and handle padding in the attention mask
        
        # Convert to src_key_padding_mask format: True = ignore
        src_key_padding_mask = attention_mask  # [B, T]
        
        # Pass through transformer encoder
        out = self.transformer(x_with_pos, src_key_padding_mask=src_key_padding_mask)  # [B, T, transformer_dim]
        
        # Prediction head
        preds = self.head(out)  # [B, T, pred_dim]
        # Hidden state is the transformer encoder output
        hidden_state = out  # [B, T, transformer_dim]
        return preds, hidden_state


class FraudFTEnc(nn.Module):
    """Feature-level Token Encoder: each feature is an independent token."""
    
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        
        # Embedding dropout
        self.embedding_dropout = getattr(args, 'embedding_dropout', 0.0)
        self.emb_dropout = nn.Dropout(self.embedding_dropout) if self.embedding_dropout > 0 else nn.Identity()
        
        # Time embeddings (same as FraudEnc)
        self.post_day_emb = nn.Embedding(args.day_vocab, args.day_emb_dim)
        self.post_hour_emb = nn.Embedding(args.hour_vocab, args.hour_emb_dim)
        self.post_min_emb = nn.Embedding(args.minute_vocab, args.minute_emb_dim)
        self.open_day_emb = nn.Embedding(args.aod_day_vocab, args.aod_day_emb_dim)
        
        # Categorical embeddings (same as FraudEnc)
        self.cat_embs = nn.ModuleDict()
        
        # Configure categorical embeddings based on feature names
        if hasattr(args, 'feature_names') and args.feature_names:
            for name in args.feature_names:
                if name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                    if name == "Member Age":
                        vocab, emb = 10, 10
                    elif name == "Amount":
                        vocab, emb = 12, 10
                    elif name == "is_int":
                        vocab, emb = 2, 4
                    elif name == "account_age_quantized":
                        vocab, emb = 5, 5
                    elif name == "cluster_id":
                        vocab, emb = 60, 32
                    elif "Account Type" in name:
                        vocab, emb = 15, 16
                    elif "Product ID" in name:
                        vocab, emb = 160, 32
                    elif "Action Type" in name:
                        vocab, emb = 5, 5
                    elif "Source Type" in name:
                        vocab, emb = 20, 16
                    else:
                        vocab, emb = 50, 4
                    
                    self.cat_embs[name] = nn.Embedding(vocab, emb)
        
        # Transformer embedding dimension
        transformer_dim = getattr(args, 'transformer_dim', args.lstm_hidden if hasattr(args, 'lstm_hidden') else 128)
        
        # Feature-level projection layers: each feature gets its own projection
        self.time_feature_projs = nn.ModuleDict()
        self.cat_feature_projs = nn.ModuleDict()
        self.cont_feature_projs = nn.ModuleList()
        
        # Count features and create projections
        num_features = len(getattr(args, 'feature_names', []) or [])
        
        if hasattr(args, 'feature_names') and args.feature_names:
            for name in args.feature_names:
                if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                    # Time feature: get embedding dim
                    if name == "Post Date_doy":
                        emb_dim = args.day_emb_dim
                    elif name == "Post Time_hour":
                        emb_dim = args.hour_emb_dim
                    elif name == "Post Time_minute":
                        emb_dim = args.minute_emb_dim
                    elif name == "Account Open Date_doy":
                        emb_dim = args.aod_day_emb_dim
                    self.time_feature_projs[name] = nn.Linear(emb_dim, transformer_dim)
                elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                    # Categorical feature: get embedding dim
                    if name in self.cat_embs:
                        emb_dim = self.cat_embs[name].embedding_dim
                    else:
                        emb_dim = 4  # default
                    self.cat_feature_projs[name] = nn.Linear(emb_dim, transformer_dim)
                else:
                    # Continuous feature: project from 1 dim
                    self.cont_feature_projs.append(nn.Linear(1, transformer_dim))
        
        # Positional embeddings: separate time position and feature position
        max_seq_len = getattr(args, 'max_seq_len', 512)
        self.time_pos_emb = nn.Embedding(max_seq_len, transformer_dim)  # Time step position
        self.feature_pos_emb = nn.Embedding(num_features, transformer_dim)  # Feature position
        
        # Feature type embedding: each feature gets its own embedding to identify which feature it is
        self.feature_emb = nn.Embedding(num_features, transformer_dim)
        
        # Multi-layer Transformer Encoder (same as FraudEnc)
        dim_feedforward = getattr(args, 'dim_feedforward', 0)
        if dim_feedforward == 0:
            dim_feedforward = transformer_dim * 2
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=getattr(args, 'nhead', 3),
            dim_feedforward=dim_feedforward,
            dropout=getattr(args, 'dropout', 0.1),
            activation=getattr(args, 'activation', 'relu'),
            batch_first=True,
            norm_first=getattr(args, 'norm_first', False),
        )
        num_encoder_layers = getattr(args, 'num_encoder_layers', 1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Prediction head (same as FraudEnc)
        inferred_pred_dim = getattr(args, 'pred_dim', 0)
        target_names = getattr(args, 'target_names', []) or []
        if not inferred_pred_dim:
            if target_names:
                inferred_pred_dim = len(target_names)
            else:
                inferred_pred_dim = len(getattr(args, 'feature_names', []) or [])
        self.pred_dim = inferred_pred_dim
        self.target_names = target_names if target_names else (getattr(args, 'feature_names', []) or [])
        self.head = nn.Linear(transformer_dim, self.pred_dim)
        
        # Store transformer_dim and num_features for forward pass
        self.transformer_dim = transformer_dim
        self.num_features = num_features
        
    def forward(
        self,
        x: torch.Tensor,             # [B, T, total_feature_dim]
        mask: torch.Tensor,          # [B, T]
        feature_names: list,         # feature names list to split feature types
        use_pack: bool = True,       # Not used for transformer, kept for compatibility
    ) -> torch.Tensor:
        # Separate different types of features from feature names
        time_features = {}
        categorical_features = {}
        continuous_feature_indices = {}  # Map feature name to (index, feature_values)
        feature_order = []  # Track feature order for token sequence
        
        for i, name in enumerate(feature_names):
            feature_values = x[:, :, i]
            feature_order.append(name)
            
            if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                time_features[name] = feature_values.long()
            elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                clamped_values = torch.clamp(feature_values, min=0).long()
                categorical_features[name] = clamped_values
            else:
                continuous_feature_indices[name] = i
        
        # Ensure all time features are non-negative and within valid ranges
        post_date_doy = torch.clamp(time_features["Post Date_doy"], min=0, max=365)
        post_time_hour = torch.clamp(time_features["Post Time_hour"], min=0, max=23)
        post_time_minute = torch.clamp(time_features["Post Time_minute"], min=0, max=59)
        account_open_doy = torch.clamp(time_features["Account Open Date_doy"], min=0, max=365)
        
        # Feature-level tokenization: each feature becomes an independent token
        feature_tokens = []
        cont_idx = 0
        
        for name in feature_order:
            if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                # Time feature: embed and project
                if name == "Post Date_doy":
                    emb = self.post_day_emb(post_date_doy)  # [B, T, day_emb_dim]
                elif name == "Post Time_hour":
                    emb = self.post_hour_emb(post_time_hour)  # [B, T, hour_emb_dim]
                elif name == "Post Time_minute":
                    emb = self.post_min_emb(post_time_minute)  # [B, T, minute_emb_dim]
                elif name == "Account Open Date_doy":
                    emb = self.open_day_emb(account_open_doy)  # [B, T, aod_day_emb_dim]
                
                emb = self.emb_dropout(emb.float())
                token = self.time_feature_projs[name](emb)  # [B, T, transformer_dim]
                feature_tokens.append(token)
                
            elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized", "cluster_id"]:
                # Categorical feature: embed and project
                if name in categorical_features and name in self.cat_embs:
                    final_values = torch.clamp(categorical_features[name], min=0, max=self.cat_embs[name].num_embeddings-1)
                    emb = self.cat_embs[name](final_values)  # [B, T, cat_emb_dim]
                    emb = self.emb_dropout(emb.float())
                    token = self.cat_feature_projs[name](emb)  # [B, T, transformer_dim]
                    feature_tokens.append(token)
                    
            else:
                # Continuous feature: project directly
                feat_idx = continuous_feature_indices[name]
                cont_feat = x[:, :, feat_idx:feat_idx+1]  # [B, T, 1]
                token = self.cont_feature_projs[cont_idx](cont_feat)  # [B, T, transformer_dim]
                feature_tokens.append(token)
                cont_idx += 1
        
        # Build token sequence: [B, T*F, transformer_dim]
        # Stack all feature tokens along time dimension, then reshape
        B, T, _ = feature_tokens[0].shape
        F = len(feature_tokens)
        
        # Concatenate all feature tokens: [B, T, F, transformer_dim]
        tokens_3d = torch.stack(feature_tokens, dim=2)  # [B, T, F, transformer_dim]
        # Reshape to [B, T*F, transformer_dim]
        tokens_flat = tokens_3d.reshape(B, T * F, self.transformer_dim)
        
        # Separate positional embeddings: time position and feature position
        B, T_F, _ = tokens_flat.shape
        
        # Compute time positions and feature positions for each token
        # Token at position i corresponds to time_idx = i // F, feature_idx = i % F
        time_positions = torch.arange(0, T, device=tokens_flat.device).unsqueeze(0).expand(B, -1)  # [B, T]
        time_positions_expanded = time_positions.unsqueeze(-1).expand(-1, -1, F).reshape(B, T * F)  # [B, T*F]
        time_pos_embeddings = self.time_pos_emb(time_positions_expanded)  # [B, T*F, transformer_dim]
        
        feature_positions = torch.arange(0, F, device=tokens_flat.device).unsqueeze(0).expand(T, -1)  # [T, F]
        feature_positions_expanded = feature_positions.reshape(T * F).unsqueeze(0).expand(B, -1)  # [B, T*F]
        feature_pos_embeddings = self.feature_pos_emb(feature_positions_expanded)  # [B, T*F, transformer_dim]
        
        # Feature type embeddings: identify which feature each token represents
        feature_indices = torch.arange(0, F, device=tokens_flat.device).unsqueeze(0).expand(T, -1)  # [T, F]
        feature_indices_expanded = feature_indices.reshape(T * F).unsqueeze(0).expand(B, -1)  # [B, T*F]
        feature_type_embeddings = self.feature_emb(feature_indices_expanded)  # [B, T*F, transformer_dim]
        
        # Combine all embeddings: token + time_pos + feature_pos + feature_type
        tokens_with_pos = tokens_flat + time_pos_embeddings + feature_pos_embeddings + feature_type_embeddings
        
        # Expand mask: [B, T] -> [B, T*F]
        # Each time step's mask is replicated F times
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, F).reshape(B, T * F)  # [B, T*F]
        attention_mask = (mask_expanded == 0).bool()  # [B, T*F], True where padding
        
        # Pass through transformer encoder
        out = self.transformer(tokens_with_pos, src_key_padding_mask=attention_mask)  # [B, T*F, transformer_dim]
        
        # Aggregate feature tokens back to time steps: [B, T*F, transformer_dim] -> [B, T, transformer_dim]
        out_3d = out.reshape(B, T, F, self.transformer_dim)  # [B, T, F, transformer_dim]
        out_agg = out_3d.mean(dim=2)  # [B, T, transformer_dim] - average pooling over features
        
        # Prediction head
        preds = self.head(out_agg)  # [B, T, pred_dim]
        # Hidden state is the aggregated transformer encoder output
        hidden_state = out_agg  # [B, T, transformer_dim]
        return preds, hidden_state


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Time+Category Embedding LSTM (binary)")
    
    # Model type selection
    p.add_argument("--model_type", type=str, default="fraudftenc", choices=["lstm", "transformer", "fraudenc", "fraudftenc"], 
                   help="Model type: lstm, transformer, fraudenc, or fraudftenc")
    
    # Continuous feature dimension (needs to match data)
    p.add_argument("--cont_dim", type=int, default=1, help="Continuous feature dimension")
    
    # Feature names (for automatic categorical embedding configuration)
    p.add_argument("--feature_names", type=str, nargs="*", default=[], help="Feature name list for automatic embedding configuration")

    # Prediction dimension for next-step vector forecasting (0 => infer from feature_names)
    p.add_argument("--pred_dim", type=int, default=0, help="Prediction dimension; 0 to infer from feature_names length")
    # Target feature names to predict (subset of feature_names); if empty, defaults to all
    p.add_argument("--target_names", type=str, nargs="*", default=[], help="Target feature names to predict; subset of feature_names")

    # Time embedding vocabulary and dimensions
    p.add_argument("--day_vocab", type=int, default=366, help="Day vocabulary size")
    p.add_argument("--hour_vocab", type=int, default=25, help="Hour vocabulary size")
    p.add_argument("--minute_vocab", type=int, default=61, help="Minute vocabulary size")
    p.add_argument("--aod_day_vocab", type=int, default=366, help="Account open date vocabulary size")

    p.add_argument("--day_emb_dim", type=int, default=8, help="Day embedding dimension")
    p.add_argument("--hour_emb_dim", type=int, default=4, help="Hour embedding dimension")
    p.add_argument("--minute_emb_dim", type=int, default=8, help="Minute embedding dimension")
    p.add_argument("--aod_day_emb_dim", type=int, default=8, help="Account open date embedding dimension")

    # LSTM
    p.add_argument("--lstm_hidden", type=int, default=128, help="LSTM hidden layer dimension")
    p.add_argument("--lstm_layers", type=int, default=2, help="LSTM layer count")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    p.add_argument("--bidirectional", action="store_true", help="Whether to use bidirectional LSTM")
    
    # Transformer (for FraudEnc)
    p.add_argument("--transformer_dim", type=int, default=128, help="Transformer embedding dimension")
    p.add_argument("--nhead", type=int, default=32, help="Number of attention heads in transformer")
    p.add_argument("--num_encoder_layers", type=int, default=2, help="Number of transformer encoder layers")
    p.add_argument("--dim_feedforward", type=int, default=0, help="Feedforward dimension (0 = 2 * transformer_dim)")
    p.add_argument("--activation", type=str, default="relu", help="Activation function (relu/gelu)")
    p.add_argument("--norm_first", action="store_true", default = True, help="Apply normalization before attention/ffn (Pre-LN)")
    p.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length for positional embeddings")

    # Training related (optional)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    p.add_argument("--embedding_dropout", type=float, default=0.1, help="Dropout rate for embeddings")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p


def build_seq_model(args: argparse.Namespace):
    """Build model based on model_type argument."""
    model_type = getattr(args, 'model_type', 'lstm').lower()
    if model_type == 'transformer' or model_type == 'fraudenc':
        return FraudEnc(args)
    elif model_type == 'fraudftenc':
        return FraudFTEnc(args)
    else:
        return TimeCatLSTM(args)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # Example feature names (adjust according to your actual features)
    example_features = [
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized',
        'Post Date_doy', 'Account Open Date_doy', 'Post Time_hour', 'Post Time_minute'
    ]
    
    # Set feature names and continuous feature dimension
    args.feature_names = example_features
    args.cont_dim = len([f for f in example_features if f not in [
        'Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized'
    ]])
    
    # Build model
    model = build_seq_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model built.")
    print("Total params:", n_params)
    print("Continuous features dim:", args.cont_dim)
    print("Feature names:", args.feature_names)

    # Inference sanity check: next-step vector prediction
    torch.manual_seed(0)
    B, T, F = 2, 6, len(args.feature_names)

    # Create synthetic input X respecting index ranges for time/categorical features
    X = torch.zeros(B, T, F, dtype=torch.float32)
    name_to_idx = {n: i for i, n in enumerate(args.feature_names)}

    # Time features within vocab ranges
    X[:, :, name_to_idx['Post Date_doy']] = torch.randint(0, args.day_vocab, (B, T)).float()
    X[:, :, name_to_idx['Post Time_hour']] = torch.randint(0, args.hour_vocab, (B, T)).float()
    X[:, :, name_to_idx['Post Time_minute']] = torch.randint(0, args.minute_vocab, (B, T)).float()
    X[:, :, name_to_idx['Account Open Date_doy']] = torch.randint(0, args.aod_day_vocab, (B, T)).float()

    # Categorical example ranges
    def clamp_fill(name: str, vocab: int):
        if name in name_to_idx:
            X[:, :, name_to_idx[name]] = torch.randint(0, vocab, (B, T)).float()

    clamp_fill('Account Type_enc', 15)
    clamp_fill('Product ID_enc', 160)
    clamp_fill('Action Type_enc', 5)
    clamp_fill('Source Type_enc', 20)
    clamp_fill('is_int', 2)
    clamp_fill('Member Age', 10)
    clamp_fill('Amount', 12)
    clamp_fill('account_age_quantized', 5)

    # Any remaining continuous features (if exist)
    for i, n in enumerate(args.feature_names):
        if n not in ['Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
                     'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount',
                     'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized']:
            X[:, :, i] = torch.randn(B, T)

    # Mask: first two steps are padding(0), rest are valid(1) -- only for pack/aggregation
    mask = torch.zeros(B, T, dtype=torch.float32)
    mask[:, 2:] = 1.0

    # Run inference
    model.eval()
    with torch.no_grad():
        preds, hidden_state = model(X, mask, args.feature_names, use_pack=False)  # [B,T,pred_dim]
        print("Preds shape:", tuple(preds.shape))
        # Show first 3 dimensions of predictions for first two time steps
        print("Preds[0, :2, :3]:\n", preds[0, :2, :3])
        # During training, should align with next step: preds[:, :-1] vs X[:, 1:]
        shift_l2 = (preds[:, :-1, :F] - X[:, 1:, :])**2
        # Demo only: aligned mean squared error (unweighted)
        print("Shifted MSE (demo):", shift_l2.mean().item())