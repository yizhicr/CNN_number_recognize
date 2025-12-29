import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import cv2
from PIL import Image

# 数据增强类（用于训练集）
class RandomBrightness:
    def __init__(self, min_factor=0.85, max_factor=1.15):
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    def __call__(self, x):
        factor = self.min_factor + (self.max_factor - self.min_factor) * torch.rand(1)
        return x * factor

# 自定义MNIST数据集
class SimpleMNISTDataset(Dataset):
    def __init__(self, root, train=True):
        self.mnist = datasets.MNIST(root=root, train=train, download=True)
        
        if train:
            self.transform = transforms.Compose([
                transforms.RandomAffine(degrees=8, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                RandomBrightness()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        return self.transform(img), label

# 图像预处理（用于GUI输入）
def adaptive_preprocess(img_pil):
    """将用户手绘图像转换为MNIST格式（28x28）"""
    img_resized = img_pil.resize((280, 280)).convert('L')
    img_np = np.array(img_resized)
    
    # 自适应二值化
    if np.std(img_np) < 10:
        _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(
            img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3
        )
    
    # 形态学处理与去噪
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    denoised = cv2.medianBlur(morph, 3)
    
    # 裁剪数字区域
    non_zero_points = np.column_stack(np.where(denoised > 0))
    if len(non_zero_points) == 0:
        return Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
    
    y_min, x_min = non_zero_points.min(axis=0)
    y_max, x_max = non_zero_points.max(axis=0)
    margin = 10
    y_min, y_max = max(0, y_min - margin), min(denoised.shape[0], y_max + margin)
    x_min, x_max = max(0, x_min - margin), min(denoised.shape[1], x_max + margin)
    cropped = denoised[y_min:y_max, x_min:x_max]
    
    # 缩放并居中到28x28画布
    h, w = cropped.shape
    scale = 18.0 / max(h, w) if max(h, w) != 0 else 1.0
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA) if (new_h > 0 and new_w > 0) else cropped
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    if resized.size > 0:
        moments = cv2.moments(resized)
        cx = int(moments['m10'] / moments['m00']) if moments['m00'] != 0 else new_w // 2
        cy = int(moments['m01'] / moments['m00']) if moments['m00'] != 0 else new_h // 2
        y_offset, x_offset = 14 - cy, 14 - cx
        
        for y in range(new_h):
            for x in range(new_w):
                if 0 <= y + y_offset < 28 and 0 <= x + x_offset < 28 and resized[y, x] > 127:
                    canvas[y + y_offset, x + x_offset] = 255
    
    return Image.fromarray(canvas)

def preprocess_for_model(img_pil):
    """将预处理后的图像转换为模型输入张量"""
    processed_img = adaptive_preprocess(img_pil)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(processed_img).unsqueeze(0)