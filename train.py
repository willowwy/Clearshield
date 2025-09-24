import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import time
import traceback
import logging
from datetime import datetime
from tqdm import tqdm

from model import TimeCatLSTM, build_arg_parser, build_model
from datasets import create_dataloader


def setup_logging(log_file="error_log.txt"):
    """设置日志记录"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)


def log_tensor_info(logger, tensor, name, batch_idx=None):
    """记录张量详细信息"""
    try:
        logger.info(f"=== {name} ===")
        if batch_idx is not None:
            logger.info(f"Batch: {batch_idx}")
        logger.info(f"Shape: {tensor.shape}")
        logger.info(f"Dtype: {tensor.dtype}")
        logger.info(f"Device: {tensor.device}")
        logger.info(f"Min: {tensor.min().item()}")
        logger.info(f"Max: {tensor.max().item()}")
        logger.info(f"Unique values: {torch.unique(tensor).detach().cpu().numpy()}")
        logger.info(f"Has NaN: {torch.isnan(tensor).any().item()}")
        logger.info(f"Has Inf: {torch.isinf(tensor).any().item()}")
        logger.info(f"Sample values: {tensor.flatten()[:10].detach().cpu().numpy()}")
    except Exception as e:
        logger.error(f"Error logging tensor info for {name}: {e}")


def train_epoch(model, train_loader, optimizer, criterion, device, feature_names, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (X, y, mask) in enumerate(tqdm(train_loader, desc="Training")):
        try:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # 记录输入张量信息（仅在第一个batch）
            if batch_idx == 0:
                logger.info(f"=== Batch {batch_idx} Input Info ===")
                log_tensor_info(logger, X, "Input X", batch_idx)
                log_tensor_info(logger, y, "Labels y", batch_idx)
                log_tensor_info(logger, mask, "Mask", batch_idx)
                logger.info(f"Feature names: {feature_names}")
            
            optimizer.zero_grad()
            
            # 前向传播
            logits = model(X, mask, feature_names)
            
            # 记录模型输出信息（仅在第一个batch）
            if batch_idx == 0:
                log_tensor_info(logger, logits, "Model Output Logits", batch_idx)
            
            # 计算损失（模型输出是每个序列一个预测）
            # logits shape: [batch_size] - 每个序列一个预测
            # y shape: [batch_size, seq_len] - 每个时间步的标签
            # mask shape: [batch_size, seq_len] - 每个时间步的有效性
            
            # 对于序列级别的预测，我们需要将标签聚合
            # 使用mask来加权平均标签，或者使用最后一个有效标签
            if mask.any():
                # 方法1：使用最后一个有效时间步的标签
                seq_lengths = mask.sum(dim=1).long() - 1  # 最后一个有效位置的索引
                last_valid_labels = y.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
                
                # 确保logits和labels的维度匹配
                if logits.dim() == 1 and last_valid_labels.dim() == 1:
                    loss = criterion(logits, last_valid_labels)
                else:
                    # 如果维度不匹配，调整logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, last_valid_labels.unsqueeze(0))
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # 收集预测和标签用于评估
                with torch.no_grad():
                    preds = torch.sigmoid(logits) > 0.5
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(last_valid_labels.cpu().numpy())
                    
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # 记录错误时的张量状态
            try:
                logger.error("=== Error State Tensor Info ===")
                log_tensor_info(logger, X, "X at error", batch_idx)
                log_tensor_info(logger, y, "y at error", batch_idx)
                log_tensor_info(logger, mask, "mask at error", batch_idx)
            except:
                logger.error("Could not log tensor info at error state")
            
            raise e  # 重新抛出异常
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device, feature_names):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X, y, mask in tqdm(val_loader, desc="Evaluating"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # 前向传播
            logits = model(X, mask, feature_names)
            
            # 计算损失（使用与训练相同的逻辑）
            if mask.any():
                # 使用最后一个有效时间步的标签
                seq_lengths = mask.sum(dim=1).long() - 1  # 最后一个有效位置的索引
                last_valid_labels = y.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
                
                # 确保logits和labels的维度匹配
                if logits.dim() == 1 and last_valid_labels.dim() == 1:
                    loss = criterion(logits, last_valid_labels)
                else:
                    # 如果维度不匹配，调整logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, last_valid_labels.unsqueeze(0))
                
                total_loss += loss.item()
                
                # 收集预测和标签
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(last_valid_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    # 计算其他指标
    if all_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
    else:
        precision = recall = f1 = auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


def test_model(model, test_loader, criterion, device, feature_names):
    """测试模型性能"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("开始测试集评估...")
    
    with torch.no_grad():
        for X, y, mask in tqdm(test_loader, desc="Testing"):
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            
            # 前向传播
            logits = model(X, mask, feature_names)
            
            # 计算损失（使用与训练相同的逻辑）
            if mask.any():
                # 使用最后一个有效时间步的标签
                seq_lengths = mask.sum(dim=1).long() - 1  # 最后一个有效位置的索引
                last_valid_labels = y.gather(1, seq_lengths.unsqueeze(1)).squeeze(1)
                
                # 确保logits和labels的维度匹配
                if logits.dim() == 1 and last_valid_labels.dim() == 1:
                    loss = criterion(logits, last_valid_labels)
                else:
                    # 如果维度不匹配，调整logits
                    if logits.dim() == 1:
                        logits = logits.unsqueeze(0)
                    loss = criterion(logits, last_valid_labels.unsqueeze(0))
                
                total_loss += loss.item()
                
                # 收集预测和标签
                probs = torch.sigmoid(logits)
                preds = probs > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(last_valid_labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds) if all_preds else 0
    
    # 计算其他指标
    if all_preds:
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
    else:
        precision = recall = f1 = auc = 0.0
    
    # 打印测试结果
    print("=" * 60)
    print("测试集评估结果:")
    print("=" * 60)
    print(f"测试损失: {avg_loss:.4f}")
    print(f"测试准确率: {accuracy:.4f}")
    print(f"测试精确率: {precision:.4f}")
    print(f"测试召回率: {recall:.4f}")
    print(f"测试F1分数: {f1:.4f}")
    print(f"测试AUC: {auc:.4f}")
    print("=" * 60)
    
    return avg_loss, accuracy, precision, recall, f1, auc


def train_model(args):
    """主训练函数"""
    # 设置日志
    logger = setup_logging("training_error_log.txt")
    logger.info("=== Training Started ===")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    logger.info(f"Random seed set to: {args.seed}")
    
    # 创建数据加载器
    print("Loading data...")
    logger.info("Loading data...")
    
    try:
        train_loader, feature_names = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=True,
            mode="train",
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed
        )
        
        val_loader, _ = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=False,
            mode="val",
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed
        )
        
        # 创建测试集数据加载器
        test_loader, _ = create_dataloader(
            matched_dir=args.data_dir,
            max_len=args.max_len,
            batch_size=args.batch_size,
            shuffle=False,
            mode="test",
            split=(args.train_ratio, args.val_ratio, args.test_ratio),
            seed=args.seed
        )
        
        print(f"Feature names: {feature_names}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        logger.info(f"Feature names: {feature_names}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"Error creating data loaders: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise e
    
    # 设置模型参数
    args.feature_names = feature_names
    args.cont_dim = len([f for f in feature_names if f not in [
        'Post Date_doy', 'Post Time_hour', 'Post Time_minute', 'Account Open Date_doy',
        'Account Type_enc', 'Member Age', 'Product ID_enc', 'Amount', 
        'Action Type_enc', 'Source Type_enc', 'is_int', 'account_age_quantized'
    ]])
    
    # 创建模型
    print("Building model...")
    model = build_model(args).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.patience
    )
    
    # 训练循环
    best_val_loss = float('inf')
    best_auc = 0
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, feature_names, logger)
        
        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device, feature_names
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 打印结果
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        print("-" * 50)
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            best_val_loss = val_loss
            patience_counter = 0
            
            if args.save_model:
                os.makedirs(args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'args': args
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"Saved best model with AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best AUC: {best_auc:.4f}")
    
    # 训练完成后，使用最佳模型进行测试集评估
    if args.save_model:
        # 加载最佳模型
        best_model_path = os.path.join(args.save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print("Loading test datasets...")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"loaded model at epoch {checkpoint['epoch']+1}, auc on test set: {checkpoint['val_auc']:.4f}")
            
            # 进行测试集评估
            test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test_model(
                model, test_loader, criterion, device, feature_names
            )
            
            # 记录测试结果到日志
            logger.info("=== Results ===")
            logger.info(f"loss: {test_loss:.4f}")
            logger.info(f"accuracy: {test_acc:.4f}")
            logger.info(f"precision: {test_precision:.4f}")
            logger.info(f"recall: {test_recall:.4f}")
            logger.info(f"F1: {test_f1:.4f}")
            logger.info(f"AUC: {test_auc:.4f}")
            
            # 保存测试结果到文件
            test_results = {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'best_val_auc': best_auc,
                'epoch': checkpoint['epoch']
            }
            
            import json
            results_file = os.path.join(args.save_dir, 'test_results.json')
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(test_results, f, indent=2, ensure_ascii=False)
            print(f"Saved result to: {results_file}")
        else:
            print("model not found")
    else:
        # 如果没有保存模型，直接使用当前模型进行测试
        print("eval on test set...")
        test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = test_model(
            model, test_loader, criterion, device, feature_names
        )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train TimeCatLSTM model")
    
    # 数据相关参数
    parser.add_argument("--data_dir", type=str, default="matched", help="数据目录")
    parser.add_argument("--max_len", type=int, default=50, help="序列最大长度")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    
    # 训练相关参数
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--patience", type=int, default=5, help="学习率调度耐心值")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早停耐心值")
    
    # 模型保存
    parser.add_argument("--save_model", action="store_true", help="是否保存模型")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    # 获取模型参数
    model_parser = build_arg_parser()
    model_args, remaining = model_parser.parse_known_args()
    
    # 合并参数
    args = parser.parse_args()
    for key, value in vars(model_args).items():
        setattr(args, key, value)
    
    print("Training arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # 开始训练
    model = train_model(args)


if __name__ == "__main__":
    main()
