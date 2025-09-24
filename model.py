import json
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TimeCatLSTM(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        # 时间嵌入
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

        # 类别嵌入 - 直接配置
        self.cat_embs = nn.ModuleDict()
        cat_total_dim = 0
        
        # 根据特征名称配置类别嵌入
        if hasattr(args, 'feature_names') and args.feature_names:
            for name in args.feature_names:
                if name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized"]:
                    if name == "Member Age":
                        vocab, emb = 10, 4  # //10 然后做embedding
                    elif name == "Amount":
                        vocab, emb = 12, 4  # 12个桶
                    elif name == "is_int":
                        vocab, emb = 2, 2   # 布尔值
                    elif name == "account_age_quantized":
                        vocab, emb = 5, 4   # 5个年龄阶段
                    elif "Account Type" in name:
                        vocab, emb = 15, 8
                    elif "Product ID" in name:
                        vocab, emb = 160, 8
                    elif "Action Type" in name:
                        vocab, emb = 5, 4
                    elif "Source Type" in name:
                        vocab, emb = 20, 4
                    else:
                        vocab, emb = 50, 4  # 默认配置
                    
                    self.cat_embs[name] = nn.Embedding(vocab, emb)
                    cat_total_dim += emb

        lstm_input_dim = time_total_dim + cat_total_dim + args.cont_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=args.lstm_hidden,
            num_layers=args.lstm_layers,
            batch_first=True,
            dropout=args.dropout if args.lstm_layers > 1 else 0.0,
            bidirectional=args.bidirectional,
        )
        last_dim = args.lstm_hidden * (2 if args.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(last_dim, last_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(last_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,             # [B, T, total_feature_dim] - 所有特征已经标准化
        mask: torch.Tensor,          # [B, T]
        feature_names: list,         # 特征名称列表，用于分离不同类型特征
        use_pack: bool = True,
    ) -> torch.Tensor:
        # 从特征名称中分离不同类型的特征
        time_features = {}
        categorical_features = {}
        continuous_features = []
        
        # 记录特征分离信息
        debug_info = []
        debug_info.append("=== Feature Separation Debug ===")
        debug_info.append(f"Total features: {len(feature_names)}")
        debug_info.append(f"Feature names: {feature_names}")
        debug_info.append(f"Input X shape: {x.shape}")
        debug_info.append(f"Input X min/max: {x.min()}/{x.max()}")
        debug_info.append("=" * 50)
        
        for i, name in enumerate(feature_names):
            feature_values = x[:, :, i]
            
            # 记录每个特征的信息
            debug_info.append(f"Feature {i} ({name}): min={feature_values.min()}, max={feature_values.max()}, dtype={feature_values.dtype}")
            
            if name in ["Post Date_doy", "Post Time_hour", "Post Time_minute", "Account Open Date_doy"]:
                time_features[name] = feature_values.long()
                debug_info.append(f"  -> Time feature")
                debug_info.append(f"    Original range: {feature_values.min()} to {feature_values.max()}")
                debug_info.append(f"    Converted to long: {time_features[name].min()} to {time_features[name].max()}")
            elif name.endswith("_enc") or name in ["is_int", "Member Age", "Amount", "account_age_quantized"]:
                # 确保类别特征是整数且非负
                # 对于负值，我们将其映射到0
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
        
        # 一次性写入所有调试信息
        with open("time_features_debug.txt", "a") as f:
            for line in debug_info:
                f.write(line + "\n")
        
        # 时间嵌入 - 添加详细的安全检查和调试信息
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
        
        # 确保所有时间特征都是非负的，避免嵌入层索引越界
        post_date_doy = torch.clamp(time_features["Post Date_doy"], min=0, max=365)
        post_time_hour = torch.clamp(time_features["Post Time_hour"], min=0, max=23)
        post_time_minute = torch.clamp(time_features["Post Time_minute"], min=0, max=59)
        account_open_doy = torch.clamp(time_features["Account Open Date_doy"], min=0, max=365)
        
        # 记录clamp后的值和embedding词汇表大小
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
        
        # 一次性写入时间调试信息
        with open("time_features_debug.txt", "a") as f:
            for line in time_debug_info:
                f.write(line + "\n")
        
        try:
            t_emb = torch.cat([
                self.post_day_emb(post_date_doy),
                self.post_hour_emb(post_time_hour),
                self.post_min_emb(post_time_minute),
                self.open_day_emb(account_open_doy),
            ], dim=-1).float()  # [B,T,Dt] 确保是float32类型
        except Exception as e:
            # 记录错误信息到文件
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

        # 类别嵌入 - 添加调试信息
        if len(self.cat_embs) > 0 and categorical_features:
            # 记录类别特征信息
            with open("time_features_debug.txt", "a") as f:
                f.write(f"=== Categorical Features Debug ===\n")
                for name, values in categorical_features.items():
                    f.write(f"{name}: {values.cpu().numpy()}\n")
                    f.write(f"{name} min/max: {values.min()}/{values.max()}\n")
                    if name in self.cat_embs:
                        f.write(f"{name} embedding vocab size: {self.cat_embs[name].num_embeddings}\n")
                f.write("=" * 50 + "\n")
            
            # 对类别特征进行最终的clamp处理，防止越界
            # 注意：categorical_features已经在特征分离阶段被clamp到非负值了
            final_categorical_features = {}
            for name, values in categorical_features.items():
                if name in self.cat_embs:
                    vocab_size = self.cat_embs[name].num_embeddings
                    # 确保值在embedding词汇表范围内
                    final_values = torch.clamp(values, min=0, max=vocab_size-1)
                    final_categorical_features[name] = final_values
                    
                    # 记录最终clamp信息
                    with open("time_features_debug.txt", "a") as f:
                        f.write(f"{name} final values: {final_values.cpu().numpy()}\n")
                        f.write(f"{name} final min/max: {final_values.min()}/{final_values.max()}\n")
                        f.write(f"{name} vocab size: {vocab_size}\n")
                else:
                    final_categorical_features[name] = values
            
            # 执行embedding
            cat_list = [self.cat_embs[name](final_categorical_features[name]) for name in self.cat_embs]
            c_emb = torch.cat(cat_list, dim=-1).float()  # 确保是float32类型
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x = torch.cat([t_emb, cont_x, c_emb], dim=-1)
        else:
            cont_x = torch.cat(continuous_features, dim=-1) if continuous_features else torch.zeros(x.size(0), x.size(1), 0, device=x.device, dtype=torch.float32)
            x = torch.cat([t_emb, cont_x], dim=-1)

        if use_pack:
            lengths = mask.sum(dim=1).long().clamp_min(1)
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
            last = out.gather(1, idx).squeeze(1)
        else:
            out, _ = self.lstm(x)
            masked = out * mask.unsqueeze(-1)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
            last = masked.sum(dim=1) / denom

        logits = self.head(last).squeeze(-1)
        return logits


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Time+Category Embedding LSTM (binary)")
    
    # 连续特征维度（需要和数据一致）
    p.add_argument("--cont_dim", type=int, default=1, help="连续特征维度")
    
    # 特征名称（用于自动配置类别嵌入）
    p.add_argument("--feature_names", type=str, nargs="*", default=[], help="特征名称列表，用于自动配置嵌入")

    # 时间 embedding 词表与维度
    p.add_argument("--day_vocab", type=int, default=366, help="日期词汇表大小")
    p.add_argument("--hour_vocab", type=int, default=25, help="小时词汇表大小")
    p.add_argument("--minute_vocab", type=int, default=61, help="分钟词汇表大小")
    p.add_argument("--aod_day_vocab", type=int, default=366, help="开户日期词汇表大小")

    p.add_argument("--day_emb_dim", type=int, default=8, help="日期嵌入维度")
    p.add_argument("--hour_emb_dim", type=int, default=4, help="小时嵌入维度")
    p.add_argument("--minute_emb_dim", type=int, default=8, help="分钟嵌入维度")
    p.add_argument("--aod_day_emb_dim", type=int, default=8, help="开户日期嵌入维度")

    # LSTM
    p.add_argument("--lstm_hidden", type=int, default=128, help="LSTM隐藏层维度")
    p.add_argument("--lstm_layers", type=int, default=1, help="LSTM层数")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    p.add_argument("--bidirectional", action="store_true", help="是否使用双向LSTM")

    # 训练相关（可选）
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    return p


def build_model(args: argparse.Namespace) -> TimeCatLSTM:
    return TimeCatLSTM(args)


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # 示例特征名称（根据你的实际特征调整）
    example_features = [
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized',
        'Post Date_doy', 'Account Open Date_doy', 'Post Time_hour', 'Post Time_minute'
    ]
    
    # 设置特征名称和连续特征维度
    args.feature_names = example_features
    args.cont_dim = len([f for f in example_features if f not in [
        'Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized'
    ]])
    
    model = build_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print("Model built.")
    print("Total params:", n_params)
    print("Continuous features dim:", args.cont_dim)
    print("Feature names:", args.feature_names)