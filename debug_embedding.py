#!/usr/bin/env python3
"""
调试embedding越界问题的脚本
"""

import torch
import argparse
from model import build_model, build_arg_parser
from datasets import create_dataloader

def main():
    # 创建参数解析器
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # 设置基本参数
    args.cont_dim = 1  # 临时设置，会在数据加载后更新
    
    print("=== 开始调试embedding越界问题 ===")
    
    try:
        # 创建数据加载器
        print("加载数据...")
        train_loader, feature_names = create_dataloader(
            matched_dir="matched",
            max_len=10,  # 使用较小的序列长度进行调试
            batch_size=2,  # 使用较小的batch size
            shuffle=False,
            mode="train",
            split=(0.8, 0.1, 0.1),
            seed=42
        )
        
        print(f"特征名称: {feature_names}")
        
        # 设置模型参数
        args.feature_names = feature_names
        args.cont_dim = len([f for f in feature_names if f not in [
            'Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
            'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
            'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized'
        ]])
        
        print(f"连续特征维度: {args.cont_dim}")
        
        # 创建模型
        print("创建模型...")
        model = build_model(args)
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 清空调试文件
        with open("time_features_debug.txt", "w") as f:
            f.write("=== Embedding调试开始 ===\n")
        
        # 测试第一个batch
        print("测试第一个batch...")
        model.eval()
        with torch.no_grad():
            for batch_idx, (X, y, mask) in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  X shape: {X.shape}")
                print(f"  y shape: {y.shape}")
                print(f"  mask shape: {mask.shape}")
                print(f"  X min/max: {X.min()}/{X.max()}")
                print(f"  X dtype: {X.dtype}")
                
                # 记录输入信息
                with open("time_features_debug.txt", "a") as f:
                    f.write(f"=== Batch {batch_idx} Input Info ===\n")
                    f.write(f"X shape: {X.shape}\n")
                    f.write(f"X min/max: {X.min()}/{X.max()}\n")
                    f.write(f"X dtype: {X.dtype}\n")
                    f.write(f"Feature names: {feature_names}\n")
                    f.write("=" * 50 + "\n")
                
                try:
                    # 前向传播
                    logits = model(X, mask, feature_names)
                    print(f"  前向传播成功! 输出shape: {logits.shape}")
                    print(f"  输出min/max: {logits.min()}/{logits.max()}")
                    
                except Exception as e:
                    print(f"  前向传播失败: {str(e)}")
                    print(f"  错误类型: {type(e).__name__}")
                    
                    # 记录错误信息
                    with open("time_features_debug.txt", "a") as f:
                        f.write(f"=== Error in Batch {batch_idx} ===\n")
                        f.write(f"Error: {str(e)}\n")
                        f.write(f"Error type: {type(e).__name__}\n")
                        f.write("=" * 50 + "\n")
                
                # 只测试第一个batch
                break
                
    except Exception as e:
        print(f"整体错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("=== 调试完成，请查看 time_features_debug.txt 文件 ===")

if __name__ == "__main__":
    main()
