import torch
import torch.nn as nn

from mmengine.model import BaseModule
import torch.nn.functional as F

class EnhancedCrossModalAttention(BaseModule):
    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        query = self.query_conv(x1).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(x2).view(B, -1, H*W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x2).view(B, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x1

class GatingMechanism(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)
        gate = self.fc(avg_pool).unsqueeze(-1).unsqueeze(-1)
        return x * gate
    
class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.avg_pool2d(x, kernel_size=2, stride=2)
        x2 = F.avg_pool2d(x1, kernel_size=2, stride=2)
        
        x2 = F.interpolate(self.conv3(x2), scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv2(x1) + x2
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x) + x1
        
        return x