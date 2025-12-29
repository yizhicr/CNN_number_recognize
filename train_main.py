from src.train import train_model
from src.evaluate import visualize_results, analyze_confusion_matrix
from src.utils import setup_matplotlib_fonts

if __name__ == "__main__":
    setup_matplotlib_fonts()
    print("开始训练模型...")
    model, test_loader, device = train_model()
    
    print("开始评估模型...")
    analyze_confusion_matrix(model, test_loader, device)
    visualize_results(model, test_loader, device)
    print("训练与评估完成！")