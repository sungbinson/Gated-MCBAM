import torch
import torch.nn as nn

class BaseCrossModalChannelAttention(nn.Module):
    def __init__(self, in_planes1, in_planes2, ratio=16):
        super(BaseCrossModalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes1 + in_planes2, (in_planes1 + in_planes2) // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d((in_planes1 + in_planes2) // ratio, in_planes1, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        avg_out1 = self.avg_pool(x1)
        max_out1 = self.max_pool(x1)
        avg_out2 = self.avg_pool(x2)
        max_out2 = self.max_pool(x2)
        
        avg_out = torch.cat([avg_out1, avg_out2], dim=1)
        max_out = torch.cat([max_out1, max_out2], dim=1)
        
        out = self.fc2(self.relu1(self.fc1(avg_out))) + self.fc2(self.relu1(self.fc1(max_out)))
        return self.sigmoid(out)

class BaseCrossModalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(BaseCrossModalSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        
        x = torch.cat([avg_out1, max_out1, avg_out2, max_out2], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BaseCrossModalCBAM(nn.Module):
    def __init__(self, in_planes1, in_planes2, ratio=16, kernel_size=7):
        super(BaseCrossModalCBAM, self).__init__()
        self.ca = BaseCrossModalChannelAttention(in_planes1, in_planes2, ratio)
        self.sa = BaseCrossModalSpatialAttention(kernel_size)

    def forward(self, x1, x2):
        x1_ca = self.ca(x1, x2) * x1
        x2_ca = self.ca(x2, x1) * x2
        
        x1_refined = self.sa(x1_ca, x2_ca) * x1_ca
        x2_refined = self.sa(x2_ca, x1_ca) * x2_ca
        
        return x1_refined, x2_refined


