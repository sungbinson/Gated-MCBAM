import torch
import torch.nn as nn
# import torchvision.models as models
import torch.nn.functional as F


class CrossModalChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CrossModalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes * 2, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()
        
        
        self.weight_msi = nn.Parameter(torch.ones(1) * 2)
        self.weight_sar = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):  
        avg_out1, avg_out2 = self.avg_pool(x1), self.avg_pool(x2)
        max_out1, max_out2 = self.max_pool(x1), self.max_pool(x2)
        
        
        weights = F.softmax(torch.stack([self.weight_msi, self.weight_sar]), dim=0)
        
        avg_out = torch.cat([weights[0] * avg_out1, weights[1] * avg_out2], dim=1)
        max_out = torch.cat([weights[0] * max_out1, weights[1] * max_out2], dim=1)
        
        out = self.fc2(self.relu1(self.fc1(avg_out))) + self.fc2(self.relu1(self.fc1(max_out)))
        return self.sigmoid(out)

class CrossModalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(CrossModalSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        
        self.weight_msi = nn.Parameter(torch.ones(1))
        self.weight_sar = nn.Parameter(torch.ones(1))

    def forward(self, x1, x2):  
        avg_out1 = torch.mean(x1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2, dim=1, keepdim=True)
        

        weights = F.softmax(torch.stack([self.weight_msi, self.weight_sar]), dim=0)
        
        x = torch.cat([
            weights[0] * avg_out1, weights[0] * max_out1,
            weights[1] * avg_out2, weights[1] * max_out2
        ], dim=1)
        x = torch.cat([avg_out1, max_out1, avg_out2, max_out2], dim=1)
#         x = self.conv1(x)
        x = self.conv1(x)
        return self.sigmoid(x)

class CrossModalCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CrossModalCBAM, self).__init__()
        self.ca = CrossModalChannelAttention(in_planes, ratio)
        self.sa = CrossModalSpatialAttention(kernel_size)

    def forward(self, x1, x2):  # x1: MSI, x2: SAR
        x1_ca = self.ca(x1, x2) * x1
        x2_ca = self.ca(x2, x1) * x2
        
        x1_refined = self.sa(x1_ca, x2_ca) * x1_ca
        x2_refined = self.sa(x2_ca, x1_ca) * x2_ca
        
        return x1_refined, x2_refined
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelWiseWeightPredictor(nn.Module):
    def __init__(self, in_channels):
        super(PixelWiseWeightPredictor, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))

class PixelWiseCrossModalCBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(PixelWiseCrossModalCBAM, self).__init__()
        self.ca = PixelWiseCrossModalChannelAttention(in_planes, ratio)
        self.sa = PixelWiseCrossModalSpatialAttention(kernel_size)

    def forward(self, x1, x2, weight1, weight2):
        x1_ca = self.ca(x1, x2, weight1, weight2) * x1
        x2_ca = self.ca(x2, x1, weight2, weight1) * x2
        
        x1_refined = self.sa(x1_ca, x2_ca, weight1, weight2) * x1_ca
        x2_refined = self.sa(x2_ca, x1_ca, weight2, weight1) * x2_ca
        
        return x1_refined, x2_refined

class PixelWiseCrossModalChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(PixelWiseCrossModalChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes * 2, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, weight1, weight2):
        avg_out1 = self.avg_pool(x1 * weight1)
        max_out1 = self.max_pool(x1 * weight1)
        avg_out2 = self.avg_pool(x2 * weight2)
        max_out2 = self.max_pool(x2 * weight2)
        
        avg_out = torch.cat([avg_out1, avg_out2], dim=1)
        max_out = torch.cat([max_out1, max_out2], dim=1)
        
        out = self.fc2(self.relu1(self.fc1(avg_out))) + self.fc2(self.relu1(self.fc1(max_out)))
        return self.sigmoid(out)

class PixelWiseCrossModalSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(PixelWiseCrossModalSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(4, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, weight1, weight2):
        avg_out1 = torch.mean(x1 * weight1, dim=1, keepdim=True)
        max_out1, _ = torch.max(x1 * weight1, dim=1, keepdim=True)
        avg_out2 = torch.mean(x2 * weight2, dim=1, keepdim=True)
        max_out2, _ = torch.max(x2 * weight2, dim=1, keepdim=True)
        
        x = torch.cat([avg_out1, max_out1, avg_out2, max_out2], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)