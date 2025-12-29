import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model import ImprovedDualBranchCNN
from src.data import SimpleMNISTDataset
from src.utils import get_device
import config

class SimpleCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weight)

def monitor_confusion(model, test_loader, device):
    """监控混淆矩阵和关键数字误判率"""
    model.eval()
    confusion_counts = torch.zeros(10, 10, dtype=torch.int32).to(device)
    total_acc, total = 0, 0
    zero_err = two_err = six_err = eight_err = 0
    zero_total = two_total = six_total = eight_total = 0
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            
            total += labels.size(0)
            total_acc += (preds == labels).sum().item()
            
            # 统计关键数字误判
            for i, (label, pred) in enumerate(zip(labels, preds)):
                if label == 0:
                    zero_total += 1
                    if pred != label: zero_err += 1
                elif label == 2:
                    two_total += 1
                    if pred != label: two_err += 1
                elif label == 6:
                    six_total += 1
                    if pred != label: six_err += 1
                elif label == 8:
                    eight_total += 1
                    if pred != label: eight_err += 1
                
                # 混淆矩阵计数
                if label != pred:
                    confusion_counts[label, pred] += 1
    
    # 计算比率
    total_acc_rate = total_acc / total
    zero_err_rate = zero_err / zero_total if zero_total > 0 else 0.0
    two_err_rate = two_err / two_total if two_total > 0 else 0.0
    six_err_rate = six_err / six_total if six_total > 0 else 0.0
    eight_err_rate = eight_err / eight_total if eight_total > 0 else 0.0
    
    return total_acc_rate, zero_err_rate, two_err_rate, six_err_rate, eight_err_rate, confusion_counts

def train_model():
    # 初始化设备和模型
    device = get_device()
    model = ImprovedDualBranchCNN().to(device)
    
    # 数据加载
    train_dataset = SimpleMNISTDataset(root=config.DATA_DIR, train=True)
    test_dataset = SimpleMNISTDataset(root=config.DATA_DIR, train=False)
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True if device.type == "cuda" else False
    )
    
    # 损失函数和优化器
    class_weights = torch.tensor(config.CLASS_WEIGHTS).to(device)
    criterion = SimpleCrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    
    # 训练记录
    best_acc = 0.0
    best_metrics = {
        "zero_err": 1.0, "two_err": 1.0, "six_err": 1.0, "eight_err": 1.0
    }
    counter = 0
    train_losses = []
    val_accuracies = []
    
    # 创建保存目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.EPOCHS}')
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100*train_correct/train_total:.2f}%"})
        
        # 验证
        train_losses.append(train_loss / len(train_loader))
        val_acc, zero_err, two_err, six_err, eight_err, _ = monitor_confusion(model, test_loader, device)
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        
        # 打印日志
        print(f"\nEpoch {epoch+1} | 训练损失: {train_losses[-1]:.4f} | 验证准确率: {val_acc:.4f}")
        print(f"关键数字误判率: 0:{zero_err:.4f} | 2:{two_err:.4f} | 6:{six_err:.4f} | 8:{eight_err:.4f}")
        
        # 保存最佳模型
        save_condition = (val_acc > best_acc and
                          zero_err <= best_metrics["zero_err"] * 1.1 and
                          two_err <= best_metrics["two_err"] * 1.1 and
                          six_err <= best_metrics["six_err"] * 1.1 and
                          eight_err <= best_metrics["eight_err"] * 1.1)
        
        if save_condition:
            best_acc = val_acc
            best_metrics = {"zero_err": zero_err, "two_err": two_err, "six_err": six_err, "eight_err": eight_err}
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc": best_acc
            }, config.BEST_MODEL_PATH)
            print(f"保存最佳模型到 {config.BEST_MODEL_PATH}")
            counter = 0
        else:
            if epoch >= config.MIN_TRAIN_EPOCHS:
                counter += 1
                if counter >= config.PATIENCE:
                    print(f"早停触发！最佳准确率: {best_acc:.4f}")
                    break
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(train_losses, label="训练损失")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(122)
    plt.plot(val_accuracies, label="验证准确率", color="orange")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{config.RESULT_DIR}/training_curves.png")
    plt.close()
    
    # 加载最佳模型并保存最终版本
    checkpoint = torch.load(config.BEST_MODEL_PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    torch.save(model.state_dict(), config.FINAL_MODEL_PATH)
    print(f"最终模型保存到 {config.FINAL_MODEL_PATH}")
    
    return model, test_loader, device