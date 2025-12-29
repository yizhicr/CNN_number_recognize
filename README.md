# 手写数字识别系统

基于改进双分支CNN的手写数字识别系统，支持模型训练和交互式手写识别。

## 功能
- 模型训练：使用MNIST数据集训练双分支CNN模型
- 实时识别：通过GUI界面手写数字并实时预测

## 环境配置
```bash
pip install -r requirements.txt
```

## 使用方法

### 1.训练模型：
```bash
python train_main.py
```

### 2.启动识别界面：
```bash
python predict_main.py
```

在左侧画布手写数字，右侧将显示预测概率分布。
