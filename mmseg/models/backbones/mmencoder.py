import torch
import torch.nn as nn

from mmengine.model import BaseModule
from ..builder import BACKBONES
from .resnet import ResNet
from .swin import SwinTransformer
from .cross_cbam_v2 import CrossModalCBAM
from. cross_cbam import BaseCrossModalCBAM
import pdb
from .convnext import ConvNeXt

from .cross_attention import GatingMechanism,  EnhancedCrossModalAttention, MultiScaleFusion



@BACKBONES.register_module()
class MMEncoder_v3(BaseModule):
    def __init__(self,
                 # 'resnet' or 'swin'
                 # ResNet parameters
                 resnet_in_channel = 3,
                 resnet_depth=50,
                 resnet_deep_stem = False,
                 resnet_num_stages=4,
                 resnet_out_indices=(0, 1, 2, 3),
                 resnet_frozen_stages=-1,
                 resnet_norm_cfg=dict(type='SyncBN', requires_grad=False),
                 resnet_style='pytorch',
                 resnet_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                 # Swin Transformer parameters
                 swin_patch_init = True,
                 swin_in_channel = 4,
                 swin_pretrain_img_size=384,
                 swin_embed_dims=128,
                 swin_depths=[2, 2, 18, 2],
                 swin_num_heads=[4, 8, 16, 32],
                 swin_window_size=12,
                 swin_mlp_ratio=4,
                 swin_qkv_bias=True,
                 swin_qk_scale=None,
                 swin_drop_rate=0.,
                 swin_attn_drop_rate=0.,
                 swin_drop_path_rate=0.3,
                 swin_patch_norm=True,
                 swin_out_indices=(0, 1, 2, 3),
                 swin_with_cp=False,
                 swin_frozen_stages=-1,
                 swin_init_cfg=dict(type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        self.swin_in_channel = swin_in_channel
        
        self.sar_extractor = ResNet(
            in_channels= resnet_in_channel,
            depth=resnet_depth,
            deep_stem= resnet_deep_stem,
            num_stages=resnet_num_stages,
            out_indices=resnet_out_indices,
            frozen_stages=resnet_frozen_stages,
            norm_cfg=resnet_norm_cfg,
            style=resnet_style,
            init_cfg=resnet_init_cfg
        )
    
        self.msi_extractor = SwinTransformer(
            patch_embed_init= swin_patch_init,
            in_channels=swin_in_channel,
            pretrain_img_size=swin_pretrain_img_size,
            embed_dims=swin_embed_dims,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=swin_qkv_bias,
            qk_scale=swin_qk_scale,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate,
            patch_norm=swin_patch_norm,
            out_indices=swin_out_indices,
            with_cp=swin_with_cp,
            frozen_stages=swin_frozen_stages,
            init_cfg=swin_init_cfg
        )

        self.swin_adapt_layers = nn.ModuleList([
                nn.Conv2d(128, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.Conv2d(1024, 2048, kernel_size=1)
            ])
        
        self.cm_cbam_modules = nn.ModuleList([
            CrossModalCBAM(256, 256),
            CrossModalCBAM(512, 512),
            CrossModalCBAM(1024, 1024),
            CrossModalCBAM(2048, 2048)
        ])

        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.Conv2d(1024 * 2, 1024, kernel_size=1),
            nn.Conv2d(2048 * 2, 2048, kernel_size=1)
        ])
    def forward(self, x):
        if isinstance(x, list):
            x = x[0].unsqueeze(0)        
        msi = x[:, :self.swin_in_channel, :, :]
        sar = x[:, self.swin_in_channel:, :, :]
        
        msi_features = self.msi_extractor(msi) # 128, 32 ,32
        sar_features = self.sar_extractor(sar) # 256 , 32 ,32
        

        fused_features = []
        
        for msi_feat, sar_feat, adapt_layer, cm_cbam, fusion_layer in zip(
            msi_features, sar_features, self.swin_adapt_layers, self.cm_cbam_modules, self.fusion_layers):
            
            adapted_msi_feat = adapt_layer(msi_feat)
            
            
            msi_refined, sar_refined = cm_cbam(adapted_msi_feat, sar_feat)
            
            fused = fusion_layer(torch.cat([msi_refined, sar_refined], dim=1))
            fused_features.append(fused)
        
        return tuple(fused_features)
    

@BACKBONES.register_module()
class MMEncoder_v4(BaseModule):
    def __init__(self,
                # 'resnet' or 'swin'
                # ResNet parameters
                resnet_in_channel = 3,
                resnet_depth=50,
                resnet_deep_stem = False,
                resnet_num_stages=4,
                resnet_out_indices=(0, 1, 2, 3),
                resnet_frozen_stages=-1,
                resnet_norm_cfg=dict(type='SyncBN', requires_grad=False),
                resnet_style='pytorch',
                resnet_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                conv_channel_list = [128,256,512,1024],
                # Swin Transformer parameters
                convn_arch='base',
                convn_out_indices=[0, 1, 2, 3],
                convn_drop_path_rate=0.4,
                convn_layer_scale_init_value=1.0,
                convn_gap_before_final_norm=False,
                convn_init_cfg=dict(
                    type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        
        self.conv_channel_list = conv_channel_list

        self.sar_extractor = ResNet(
            in_channels= resnet_in_channel,
            depth=resnet_depth,
            deep_stem= resnet_deep_stem,
            num_stages=resnet_num_stages,
            out_indices=resnet_out_indices,
            frozen_stages=resnet_frozen_stages,
            norm_cfg=resnet_norm_cfg,
            style=resnet_style,
            init_cfg=resnet_init_cfg
        )
        self.msi_extractor = ConvNeXt(
            in_channels = 12,
            arch=convn_arch,
            out_indices=convn_out_indices,
            drop_path_rate=convn_drop_path_rate,
            layer_scale_init_value=convn_layer_scale_init_value,
            gap_before_final_norm=convn_gap_before_final_norm,
            init_cfg=convn_init_cfg
        )
       

        self.swin_adapt_layers = nn.ModuleList([
                nn.Conv2d(self.conv_channel_list[0], 256, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[1], 512, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[2], 1024, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[3], 2048, kernel_size=1)
            ])
        
        self.cm_cbam_modules = nn.ModuleList([
            CrossModalCBAM(256, 256),
            CrossModalCBAM(512, 512),
            CrossModalCBAM(1024, 1024),
            CrossModalCBAM(2048, 2048)
        ])

        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.Conv2d(1024 * 2, 1024, kernel_size=1),
            nn.Conv2d(2048 * 2, 2048, kernel_size=1)
        ])
    def forward(self, x):
        if isinstance(x, list):
            x = x[0].unsqueeze(0)        
        msi = x[:, :12, :, :]
        sar = x[:, 12:, :, :]
        
        msi_features = self.msi_extractor(msi) # 128, 32 ,32
        sar_features = self.sar_extractor(sar) # 256 , 32 ,32
      

        fused_features = []
        
        for msi_feat, sar_feat, adapt_layer, cm_cbam, fusion_layer in zip(
            msi_features, sar_features, self.swin_adapt_layers, self.cm_cbam_modules, self.fusion_layers):
            
            adapted_msi_feat = adapt_layer(msi_feat)
            
            
            msi_refined, sar_refined = cm_cbam(adapted_msi_feat, sar_feat)
            
            fused = fusion_layer(torch.cat([msi_refined, sar_refined], dim=1))
            fused_features.append(fused)
        
        return tuple(fused_features)
    

@BACKBONES.register_module()
class MMEncoder_v5(BaseModule):
    def __init__(self,
                 resnet_in_channel=3,
                 resnet_depth=50,
                 resnet_deep_stem=False,
                 resnet_num_stages=4,
                 resnet_out_indices=(0, 1, 2, 3),
                 resnet_frozen_stages=-1,
                 resnet_norm_cfg=dict(type='SyncBN', requires_grad=False),
                 resnet_style='pytorch',
                 resnet_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                 conv_channel_list=[128, 256, 512, 1024],
                 convn_arch='base',
                 convn_out_indices=[0, 1, 2, 3],
                 convn_drop_path_rate=0.4,
                 convn_layer_scale_init_value=1.0,
                 convn_gap_before_final_norm=False,
                 convn_init_cfg=dict(type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        
        self.conv_channel_list = conv_channel_list

        self.sar_extractor = ResNet(
            in_channels=resnet_in_channel,
            depth=resnet_depth,
            deep_stem=resnet_deep_stem,
            num_stages=resnet_num_stages,
            out_indices=resnet_out_indices,
            frozen_stages=resnet_frozen_stages,
            norm_cfg=resnet_norm_cfg,
            style=resnet_style,
            init_cfg=resnet_init_cfg
        )
        self.msi_extractor = ConvNeXt(
            in_channels=10,
            arch=convn_arch,
            out_indices=convn_out_indices,
            drop_path_rate=convn_drop_path_rate,
            layer_scale_init_value=convn_layer_scale_init_value,
            gap_before_final_norm=convn_gap_before_final_norm,
            init_cfg=convn_init_cfg
        )

        self.swin_adapt_layers = nn.ModuleList([
            nn.Conv2d(self.conv_channel_list[0], 256, kernel_size=1),
            nn.Conv2d(self.conv_channel_list[1], 512, kernel_size=1),
            nn.Conv2d(self.conv_channel_list[2], 1024, kernel_size=1),
            nn.Conv2d(self.conv_channel_list[3], 2048, kernel_size=1)
        ])

        # Enhanced fusion components
        self.cross_attn_layers = nn.ModuleList([
            EnhancedCrossModalAttention(256),
            EnhancedCrossModalAttention(512),
            EnhancedCrossModalAttention(1024),
            EnhancedCrossModalAttention(2048)
        ])

        self.multi_scale_fusion_layers = nn.ModuleList([
            MultiScaleFusion(256),
            MultiScaleFusion(512),
            MultiScaleFusion(1024),
            MultiScaleFusion(2048)
        ])
        
        
 

@BACKBONES.register_module()
class MMEncoder_v5_swinGCBAM(BaseModule):
    def __init__(self,
                 # 'resnet' or 'swin'
                 # ResNet parameters
                 resnet_in_channel = 3,
                 resnet_depth=50,
                 resnet_deep_stem = False,
                 resnet_num_stages=4,
                 resnet_out_indices=(0, 1, 2, 3),
                 resnet_frozen_stages=-1,
                 resnet_norm_cfg=dict(type='SyncBN', requires_grad=False),
                 resnet_style='pytorch',
                 resnet_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                 # Swin Transformer parameters
                 swin_patch_init = True,
                 swin_in_channel = 10,
                 swin_pretrain_img_size=384,
                 swin_embed_dims=128,
                 swin_depths=[2, 2, 18, 2],
                 swin_num_heads=[4, 8, 16, 32],
                 swin_window_size=12,
                 swin_mlp_ratio=4,
                 swin_qkv_bias=True,
                 swin_qk_scale=None,
                 swin_drop_rate=0.,
                 swin_attn_drop_rate=0.,
                 swin_drop_path_rate=0.3,
                 swin_patch_norm=True,
                 swin_out_indices=(0, 1, 2, 3),
                 swin_with_cp=False,
                 swin_frozen_stages=-1,
                 swin_init_cfg=dict(type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        self.swin_in_channel = swin_in_channel
        
        self.sar_extractor = ResNet(
            in_channels= resnet_in_channel,
            depth=resnet_depth,
            deep_stem= resnet_deep_stem,
            num_stages=resnet_num_stages,
            out_indices=resnet_out_indices,
            frozen_stages=resnet_frozen_stages,
            norm_cfg=resnet_norm_cfg,
            style=resnet_style,
            init_cfg=resnet_init_cfg
        )
    
        self.msi_extractor = SwinTransformer(
            patch_embed_init= swin_patch_init,
            in_channels=swin_in_channel,
            pretrain_img_size=swin_pretrain_img_size,
            embed_dims=swin_embed_dims,
            depths=swin_depths,
            num_heads=swin_num_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=swin_qkv_bias,
            qk_scale=swin_qk_scale,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate,
            patch_norm=swin_patch_norm,
            out_indices=swin_out_indices,
            with_cp=swin_with_cp,
            frozen_stages=swin_frozen_stages,
            init_cfg=swin_init_cfg
        )

        self.swin_adapt_layers = nn.ModuleList([
                nn.Conv2d(128, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.Conv2d(1024, 2048, kernel_size=1)
            ])
        
        self.cm_cbam_modules = nn.ModuleList([
            BaseCrossModalCBAM(256, 256),
            BaseCrossModalCBAM(512, 512),
            BaseCrossModalCBAM(1024, 1024),
            BaseCrossModalCBAM(2048, 2048)
        ])

        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.Conv2d(1024 * 2, 1024, kernel_size=1),
            nn.Conv2d(2048 * 2, 2048, kernel_size=1)
        ])

        self.gating_layers = nn.ModuleList([
            GatingMechanism(256),
            GatingMechanism(512),
            GatingMechanism(1024),
            GatingMechanism(2048)
        ])
    def forward(self, x):
        #if isinstance(x, list):
        #    x = x[0].unsqueeze(0)

        msi = x[:, :10, :, :]
        sar = x[:, 10:, :, :]
        
        msi_features = self.msi_extractor(msi) # 128, 32 ,32
        sar_features = self.sar_extractor(sar) # 256 , 32 ,32


        fused_features = []
        
        for msi_feat, sar_feat, adapt_layer, cm_cbam, fusion_layer, gating in zip(
            msi_features, sar_features, self.swin_adapt_layers, self.cm_cbam_modules, self.fusion_layers, self.gating_layers):
            
            adapted_msi_feat = adapt_layer(msi_feat)
            
            
            msi_refined, sar_refined = cm_cbam(adapted_msi_feat, sar_feat)
            
            msi_gated = gating(msi_refined)
            sar_gated = gating(sar_refined)
            

            fused = fusion_layer(torch.cat([msi_gated, sar_gated], dim=1))

            fused = fused + msi_gated + sar_gated
            
            fused_features.append(fused)
        
        return tuple(fused_features)


@BACKBONES.register_module()
class MMEncoder_v2(BaseModule):
    def __init__(self,
                # 'resnet' or 'swin'
                # ResNet parameters
                resnet_in_channel = 3,
                resnet_depth=50,
                resnet_deep_stem = False,
                resnet_num_stages=4,
                resnet_out_indices=(0, 1, 2, 3),
                resnet_frozen_stages=-1,
                resnet_norm_cfg=dict(type='SyncBN', requires_grad=False),
                resnet_style='pytorch',
                resnet_init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                conv_channel_list = [128,256,512,1024],
                # Swin Transformer parameters
                convn_arch='base',
                convn_out_indices=[0, 1, 2, 3],
                convn_drop_path_rate=0.4,
                convn_layer_scale_init_value=1.0,
                convn_gap_before_final_norm=False,
                convn_init_cfg=dict(
                    type='Pretrained', checkpoint=None),
                 **kwargs):
        super().__init__()
        
        self.conv_channel_list = conv_channel_list

        self.sar_extractor = ResNet(
            in_channels= resnet_in_channel,
            depth=resnet_depth,
            deep_stem= resnet_deep_stem,
            num_stages=resnet_num_stages,
            out_indices=resnet_out_indices,
            frozen_stages=resnet_frozen_stages,
            norm_cfg=resnet_norm_cfg,
            style=resnet_style,
            init_cfg=resnet_init_cfg
        )
        self.msi_extractor = ConvNeXt(
            in_channels = 10,
            arch=convn_arch,
            out_indices=convn_out_indices,
            drop_path_rate=convn_drop_path_rate,
            layer_scale_init_value=convn_layer_scale_init_value,
            gap_before_final_norm=convn_gap_before_final_norm,
            init_cfg=convn_init_cfg
        )
        

        self.swin_adapt_layers = nn.ModuleList([
                nn.Conv2d(self.conv_channel_list[0], 256, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[1], 512, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[2], 1024, kernel_size=1),
                nn.Conv2d(self.conv_channel_list[3], 2048, kernel_size=1)
            ])
        
        self.cm_cbam_modules = nn.ModuleList([
            BaseCrossModalCBAM(256, 256),
            BaseCrossModalCBAM(512, 512),
            BaseCrossModalCBAM(1024, 1024),
            BaseCrossModalCBAM(2048, 2048)
        ])

        self.fusion_layers = nn.ModuleList([
            nn.Conv2d(256 * 2, 256, kernel_size=1),
            nn.Conv2d(512 * 2, 512, kernel_size=1),
            nn.Conv2d(1024 * 2, 1024, kernel_size=1),
            nn.Conv2d(2048 * 2, 2048, kernel_size=1)
        ])

        self.gating_layers = nn.ModuleList([
            GatingMechanism(256),
            GatingMechanism(512),
            GatingMechanism(1024),
            GatingMechanism(2048)
        ])
    def forward(self, x):
        #if isinstance(x, list):
        #    x = x[0].unsqueeze(0)

        msi = x[:, :10, :, :]
        sar = x[:, 10:, :, :]
        
        msi_features = self.msi_extractor(msi) # 128, 32 ,32
        sar_features = self.sar_extractor(sar) # 256 , 32 ,32


        fused_features = []
        
        for msi_feat, sar_feat, adapt_layer, cm_cbam, fusion_layer, gating in zip(
            msi_features, sar_features, self.swin_adapt_layers, self.cm_cbam_modules, self.fusion_layers, self.gating_layers):
            
            adapted_msi_feat = adapt_layer(msi_feat)
            
            
            msi_refined, sar_refined = cm_cbam(adapted_msi_feat, sar_feat)
            
            msi_gated = gating(msi_refined)
            sar_gated = gating(sar_refined)
            
            fused = fusion_layer(torch.cat([msi_gated, sar_gated], dim=1))
            
            fused = fused + msi_gated + sar_gated
            
            fused_features.append(fused)
        
        return tuple(fused_features)