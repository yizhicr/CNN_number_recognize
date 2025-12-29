import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.train import monitor_confusion
import config

def visualize_results(model, test_loader, device, num_samples=20):
    """可视化正确/错误预测样本"""
    model.eval()
    correct_samples, wrong_samples = [], []
    
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)
            
            for img, label, pred in zip(imgs, labels, preds):
                if label in [0, 2, 6, 8]:  # 重点关注易混淆数字
                    if pred == label and len(correct_samples) < num_samples//2:
                        correct_samples.append((img.cpu(), label.item(), pred.item()))
                    elif pred != label and len(wrong_samples) < num_samples//2:
                        wrong_samples.append((img.cpu(), label.item(), pred.item()))
                
                if len(correct_samples) >= num_samples//2 and len(wrong_samples) >= num_samples//2:
                    break
            if len(correct_samples) >= num_samples//2 and len(wrong_samples) >= num_samples//2:
                break
    
    # 绘图
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    for idx, (img, true, pred) in enumerate(correct_samples):
        axes[0, idx].imshow(img.squeeze(), cmap="gray")
        axes[0, idx].set_title(f"正确: {true} → 预测: {pred}", color="green")
        axes[0, idx].axis("off")
    axes[0, 0].set_ylabel("正确预测", fontweight="bold")
    
    for idx, (img, true, pred) in enumerate(wrong_samples):
        axes[1, idx].imshow(img.squeeze(), cmap="gray")
        axes[1, idx].set_title(f"正确: {true} → 预测: {pred}", color="red")
        axes[1, idx].axis("off")
    axes[1, 0].set_ylabel("错误预测", fontweight="bold")
    
    plt.suptitle("模型预测结果可视化")
    plt.tight_layout()
    plt.savefig(f"{config.RESULT_DIR}/predictions_visualization.png")
    plt.close()

def analyze_confusion_matrix(model, test_loader, device):
    """绘制混淆矩阵热力图"""
    _, _, _, _, _, confusion_counts = monitor_confusion(model, test_loader, device)
    confusion_np = confusion_counts.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_np, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵热力图")
    plt.tight_layout()
    plt.savefig(f"{config.RESULT_DIR}/confusion_matrix.png")
    plt.close()
    
    return confusion_np