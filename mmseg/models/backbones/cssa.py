import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
import numpy as np

class ECABlock(nn.Module):

    def __init__(self, kernel_size=3, channel_first=None):
        super().__init__()

        self.channel_first = channel_first

        self.GAP = torch.nn.AdaptiveAvgPool2d(1)
        self.f = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding = kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x):

        x = self.GAP(x)

        # need to squeeze 4d tensor to 3d & transpose so convolution happens correctly
        x = x.squeeze(-1).transpose(-1, -2)
        x = self.f(x)
        x = x.transpose(-1, -2).unsqueeze(-1) # return to correct shape, reverse ops

        x = self.sigmoid(x)

        return x
  
class ChannelSwitching(nn.Module):
    def __init__(self, switching_thresh):
        super().__init__()
        self.k = switching_thresh

    def forward(self, x, x_prime, w):

        self.mask = w < self.k
        # If self.mask is True, take from x_prime; otherwise, keep x's value
        x = torch.where(self.mask, x_prime, x)

        return x
    
class SpatialAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, rgb_feats, ir_feats):
        # get shape
        B, C, H, W = rgb_feats.shape

        # channel concatenation (x_cat -> B,2C,H,W)
        x_cat = torch.cat((rgb_feats, ir_feats), axis=1)

        # create w_avg attention map (w_avg -> B,1,H,W)
        cap = torch.mean(x_cat, dim=1)
        w_avg = self.sigmoid(cap)
        w_avg = w_avg.unsqueeze(1)

        # create w_max attention maps (w_max -> B,1,H,W)
        cmp = torch.max(x_cat, dim=1)[0]
        w_max = self.sigmoid(cmp)
        w_max = w_max.unsqueeze(1)

        # weighted feature map (x_cat_w -> B,2C,H,W)
        x_cat_w = x_cat * w_avg * w_max

        # split weighted feature map (x_ir_w, x_rgb_w -> B,C,H,W)
        x_rgb_w = x_cat_w[:,:C,:,:]
        x_ir_w = x_cat_w[:,C:,:,:]

        # fuse feature maps (x_fused -> B,H,W,C)
        x_fused = (x_ir_w + x_rgb_w)/2

        return x_fused
    

class CSSA(torch.nn.Module):

    def __init__(self, switching_thresh=0.5, kernel_size=3, channel_first=None):
        super().__init__()

        # self.eca = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
        self.eca_rgb = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
        self.eca_ir = ECABlock(kernel_size=kernel_size, channel_first=channel_first)
        self.cs = ChannelSwitching(switching_thresh=switching_thresh)
        self.sa = SpatialAttention()

    def forward(self, rgb_input, ir_input):
        # channel switching for RGB input
        rgb_w = self.eca_rgb(rgb_input)
        rgb_feats = self.cs(rgb_input, ir_input, rgb_w)

        # channel switching for IR input
        ir_w = self.eca_ir(ir_input)
        ir_feats = self.cs(ir_input, rgb_input, ir_w)

        # spatial attention
        fused_feats = self.sa(rgb_feats, ir_feats)

        return fused_feats
  
class FPN(torch.nn.Module):

    def __init__(self, channel_sizes, feature_dim=256):
        super().__init__()

        self.n = len(channel_sizes)

        # top layer
        self.top_layer = torch.nn.Conv2d(channel_sizes[-1], feature_dim, kernel_size=1, stride=1, padding=0)

        # smoothing layer
        self.smooth_layers = [torch.nn.Conv2d(feature_dim, feature_dim, kernel_size=3, stride=1, padding=1) for i in range(self.n)]
        self.smooth_layers = torch.nn.Sequential(*self.smooth_layers)

        # lateral layers
        self.lateral_layers = [torch.nn.Conv2d(channel_sizes[i], feature_dim, kernel_size=1, stride=1, padding=0) for i in range(self.n-1)]
        self.lateral_layers = torch.nn.Sequential(*self.lateral_layers)


    def upsample_add(self, top_down_path, lat_connection):
        _, _, h, w = lat_connection.size()
        upsampled_map = torch.nn.functional.interpolate(top_down_path, size=(h,w), mode='bilinear')
        return upsampled_map + lat_connection


    def forward(self, c):
        # top-down path
        p = [self.top_layer(c[-1])]
        for i in range(self.n-2,-1,-1):
            p.append(self.upsample_add(p[i-self.n+1], self.lateral_layers[i](c[i])))
        p = list(reversed(p))

        # smoothing
        for i in range(self.n):
            p[i] = self.smooth_layers[i](p[i])

        return p


class BackbonePipeline(torch.nn.Module):

    def __init__(self, config, custom=False):
        super().__init__()

        # initialize config values
        self.num_classes = config['num_classes']
        self.reduction_factor = config['reduction_factor']
        self.cssa_switching_thresh = config['cssa_switching_thresh']
        self.cssa_kernel_size = config['cssa_kernel_size']
        self.eca_channel_first = config['eca_channel_first']
        self.channel_sizes = config['channel_sizes']
        self.out_channels = config['fpn_feature_dim']

        if custom:
            # create channels for ResNet-50
            channels = []
            channels += [self.channel_sizes[0] for _ in range(3)]
            channels += [self.channel_sizes[1] for _ in range(4)]
            channels += [self.channel_sizes[2] for _ in range(6)]
            channels += [self.channel_sizes[3] for _ in range(3)]

            # get pretrained ResNet-50 weights
            self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', "resnet50", pretrained=True)

            # initialize ResNet-50 backbones
            self.rgb_backbone = TresNetBottleneck(3, channels, self.reduction_factor)
            self.ir_backbone = TresNetBottleneck(3, channels, self.reduction_factor)
            self.rgb_backbone.compare_model_components(self.resnet50)
            self.ir_backbone.compare_model_components(self.resnet50)
        else:
            # Load the pre-trained ResNet-50 model
            self.rgb_backbone = ResNet50FeatureExtractor()
            self.ir_backbone = ResNet50FeatureExtractor()

        # ECA kernel sizes
        ECA_kernels = [self.find_ECA_k(channel) for channel in self.channel_sizes]

        # initialize CSSA module
        self.cssa_0 = CSSA(self.cssa_switching_thresh, ECA_kernels[0])
        self.cssa_1 = CSSA(self.cssa_switching_thresh, ECA_kernels[1])
        self.cssa_2 = CSSA(self.cssa_switching_thresh, ECA_kernels[2])
        self.cssa_3 = CSSA(self.cssa_switching_thresh, ECA_kernels[3])


        # initialize FPN
        extra_pool = LastLevelMaxPool()
        self.fpn = FeaturePyramidNetwork(self.channel_sizes, self.out_channels, extra_blocks=extra_pool)


    def find_ECA_k(self, channel):

        gamma, beta = 2, 1

        k = int(abs((np.log2(channel) / 2) + (beta/gamma) ))

        if k % 2 == 0:
            k -= 1

        return k


    def forward(self, rbg_images, ir_images):
        
        feats_rgb = self.rgb_backbone(rbg_images)
        feats_ir = self.ir_backbone(ir_images)

        c = [
            self.cssa_0(feats_rgb[0], feats_ir[0]),
            self.cssa_1(feats_rgb[1], feats_ir[1]),
            self.cssa_2(feats_rgb[2], feats_ir[2]),
            self.cssa_3(feats_rgb[3], feats_ir[3])
        ]

        # pass through FPN
        c_feats = OrderedDict()
        c_feats['0'] = c[0]
        c_feats['1'] = c[1]
        c_feats['2'] = c[2]
        c_feats['3'] = c[3]

        p = self.fpn(c_feats)

        return p