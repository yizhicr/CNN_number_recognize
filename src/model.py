import torch
import torch.nn as nn

class ImprovedDualBranchCNN(nn.Module):
    def __init__(self):
        super(ImprovedDualBranchCNN, self).__init__()
        
        # 共享特征提取层
        self.shared_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 三个特征分支（全局、局部、细粒度）
        self.global_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4)
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.fine_grain_branch = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(9856, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_shared = self.shared_conv(x)
        attn_weights = self.attention(x_shared)
        x_attended = x_shared * attn_weights
        
        x_global = self.global_branch(x_attended)
        x_local = self.local_branch(x_attended)
        x_fine = self.fine_grain_branch(x_attended)
        
        x_global_flat = x_global.view(x_global.size(0), -1)
        x_local_flat = x_local.view(x_local.size(0), -1)
        x_fine_flat = x_fine.view(x_fine.size(0), -1)
        
        x_concat = torch.cat([x_global_flat, x_local_flat, x_fine_flat], dim=1)
        return self.fusion(x_concat)