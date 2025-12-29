# 路径配置
DATA_DIR = "./data"
MODEL_DIR = "./models"
RESULT_DIR = "./results"
FINAL_MODEL_PATH = f"{MODEL_DIR}/mnist_cnn_final.pth"
BEST_MODEL_PATH = f"{MODEL_DIR}/mnist_cnn_best.pth"

# 训练超参数
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 8  # 早停耐心值
MIN_TRAIN_EPOCHS = 12  # 最小训练轮次
WEIGHT_DECAY = 1e-4  # 权重衰减

# 类别权重（解决易混淆数字问题）
CLASS_WEIGHTS = [1.5, 0.9, 1.2, 1.0, 1.0, 1.6, 1.5, 1.0, 1.5, 1.3]