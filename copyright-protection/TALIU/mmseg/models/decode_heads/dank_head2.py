import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from mmseg.ops import resize
from mmseg.models.utils import *
import math
from timm.models.layers import DropPath, trunc_normal_
from segmentation_models_pytorch.base import modules as md
from IPython import embed


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.
    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 #  act_cfg=dict(type='GELU'),
                 act_layer=nn.GELU,
                 drop_path=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        # self.act = build_activation_layer(act_cfg)
        self.act = act_layer()

        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class ChannelReductionAttention(nn.Module):
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super().__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        # self.dim2 = dim2
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)  
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)  
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        B, N, C = x.shape  
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class global_meta_block(nn.Module):

    def __init__(self, dim1, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super().__init__()
        self.norm1 = norm_layer(dim1)
        self.norm3 = norm_layer(dim1)

        self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm3(x), h, w))
        return x



###############################
#
#
############################



class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class DecoderBlock(nn.Module):
    def __init__(self,cin,cadd,cout,):
        super().__init__()
        self.cin = (cin + cadd)
        self.cout = cout
        self.conv1 = md.Conv2dReLU(self.cin,self.cout,kernel_size=3,padding=1,use_batchnorm=True)
        self.conv2 = md.Conv2dReLU(self.cout,self.cout,kernel_size=3,padding=1,use_batchnorm=True)

    def forward(self, x1, x2=None):
        x1 = F.interpolate(x1, scale_factor=2.0, mode="nearest")
        if x2 is not None:
            x1 = torch.cat([x1, x2], dim=1)
        x1 = self.conv1(x1[:,:self.cin])
        x1 = self.conv2(x1)
        return x1

class ConvBNReLU(nn.Module):
    def __init__(self,in_c,out_c,ks,stride=1,norm=True,res=False):
        super(ConvBNReLU, self).__init__()
        if norm:
            self.conv = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False),nn.BatchNorm2d(out_c),nn.ReLU(True))
        else:
            self.conv = nn.Conv2d(in_c, out_c, kernel_size=ks, padding = ks//2, stride=stride,bias=False)
        self.res = res
    def forward(self,x):
        if self.res:
            return (x + self.conv(x))
        else:
            return self.conv(x)

class FUSE1(nn.Module):
    def __init__(self,in_channels_list=(96,192,384,768)):
        super(FUSE1, self).__init__()
        self.c31 = ConvBNReLU(in_channels_list[2],in_channels_list[2],1)
        self.c32 = ConvBNReLU(in_channels_list[3],in_channels_list[2],1)
        self.c33 = ConvBNReLU(in_channels_list[2],in_channels_list[2],3)

        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2,x3 = x
        h,w = x2.shape[-2:]
        x2 = self.c33(F.interpolate(self.c32(x3),size=(h,w))+self.c31(x2)) #D20
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w))+self.c21(x1)) #D10
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w))+self.c11(x))
        return x,x1,x2,x3

class FUSE2(nn.Module):
    def __init__(self,in_channels_list=(96,192,384)):
        super(FUSE2, self).__init__()
        
        self.c21 = ConvBNReLU(in_channels_list[1],in_channels_list[1],1)
        self.c22 = ConvBNReLU(in_channels_list[2],in_channels_list[1],1)
        self.c23 = ConvBNReLU(in_channels_list[1],in_channels_list[1],3)

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1,x2 = x
        
        h,w = x1.shape[-2:]
        x1 = self.c23(F.interpolate(self.c22(x2),size=(h,w),mode='bilinear',align_corners=True)+self.c21(x1))
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1,x2

class FUSE3(nn.Module):
    def __init__(self,in_channels_list=(96,192)):
        super(FUSE3, self).__init__()

        self.c11 = ConvBNReLU(in_channels_list[0],in_channels_list[0],1)
        self.c12 = ConvBNReLU(in_channels_list[1],in_channels_list[0],1)
        self.c13 = ConvBNReLU(in_channels_list[0],in_channels_list[0],3)

    def forward(self,x):
        x,x1 = x
        h,w = x.shape[-2:]
        x = self.c13(F.interpolate(self.c12(x1),size=(h,w),mode='bilinear',align_corners=True)+self.c11(x))
        return x,x1

class MID(nn.Module):
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        self.fuse1 = FUSE1(in_channels_list=encoder_channels)
        self.fuse2 = FUSE2(in_channels_list=encoder_channels[:-1])
        self.fuse3 = FUSE3(in_channels_list=encoder_channels[:-2])
        add_extra = encoder_channels[0]
        
        encoder_channels = encoder_channels[1:][::-1]
        self.in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        self.add_channels = list(encoder_channels[1:]) + [add_extra]
        self.out_channels = decoder_channels

        decoder_convs = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.add_channels[layer_idx]
                    skip_ch = self.add_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.add_channels[layer_idx - 1]
                decoder_convs[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch)
        decoder_convs[f"x_{0}_{len(self.in_channels)-1}"] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1])
        self.decoder_convs = nn.ModuleDict(decoder_convs)

    def forward(self, features):
        decoder_features = {}
        features = self.fuse1(features)[::-1] 

        decoder_features["x_0_0"] = self.decoder_convs["x_0_0"](features[0],features[1]) #384, 32, 32
        decoder_features["x_1_1"] = self.decoder_convs["x_1_1"](features[1],features[2]) #192, 64, 64
        decoder_features["x_2_2"] = self.decoder_convs["x_2_2"](features[2],features[3]) #768, 16, 16

        decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"] = self.fuse2((decoder_features["x_2_2"], decoder_features["x_1_1"], decoder_features["x_0_0"]))

        decoder_features["x_0_1"] = self.decoder_convs["x_0_1"](decoder_features["x_0_0"], torch.cat((decoder_features["x_1_1"], features[2]),1))
        decoder_features["x_1_2"] = self.decoder_convs["x_1_2"](decoder_features["x_1_1"], torch.cat((decoder_features["x_2_2"], features[3]),1))

        decoder_features["x_1_2"], decoder_features["x_0_1"] = self.fuse3((decoder_features["x_1_2"], decoder_features["x_0_1"]))
        decoder_features["x_0_2"] = self.decoder_convs["x_0_2"](decoder_features["x_0_1"], torch.cat((decoder_features["x_1_2"], decoder_features["x_2_2"], features[3]),1))

        
        return self.decoder_convs["x_0_3"](torch.cat((decoder_features["x_0_2"], decoder_features["x_1_2"], decoder_features["x_2_2"]),1))



@HEADS.register_module()
class DankSegHead2(BaseDecodeHead):
    def __init__(self, feature_strides, **kwargs):
        super(DankSegHead2, self).__init__(input_transform='multiple_select', **kwargs)
        _, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels 

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        pool_ratio = 2
        mlp_ratio = 2

        # self.attn4 = global_meta_block(dim1=c4_in_channels, num_heads=8, mlp_ratio=mlp_ratio,
        #                                drop_path=0.1, pool_ratio=pool_ratio)

        # self.attn3 = global_meta_block(dim1=c3_in_channels, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 2)

        # self.attn2 = global_meta_block(dim1=c2_in_channels, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
        #                                pool_ratio=pool_ratio * 4)


        self.c40toc41_k1     = ConvBNReLU(c4_in_channels, c4_in_channels, 1)
        self.c41toc31_k3     = ConvBNReLU(c4_in_channels, c4_in_channels, 3)
        self.c41toc32_k1     = ConvBNReLU(c4_in_channels, c3_in_channels, 1)

        self.c30toc31_k1     = ConvBNReLU(c3_in_channels, c4_in_channels, 1)
        self.c31toc32_k1     = ConvBNReLU(c4_in_channels, c3_in_channels, 1)
        self.c31toc21_k3     = ConvBNReLU(c4_in_channels, c4_in_channels, 3)
        self.c32toc22_k3     = ConvBNReLU(c3_in_channels, c3_in_channels, 3)
        self.c32toc22_k1     = ConvBNReLU(c3_in_channels, c2_in_channels, 1)

        self.c20toc21_k1     = ConvBNReLU(c2_in_channels, c4_in_channels, 1)
        self.c21toc22_k1     = ConvBNReLU(c4_in_channels, c3_in_channels, 1)        
        self.c22toc23_k1     = ConvBNReLU(c3_in_channels, c2_in_channels, 1)        
        self.c21toc23_k1     = ConvBNReLU(c4_in_channels, c2_in_channels, 1)        


        # self.linear_fuse = ConvModule(
        #     in_channels=(c2_in_channels + c3_in_channels + c4_in_channels),
        #     out_channels=embedding_dim,
        #     kernel_size=1,
        #     norm_cfg=dict(type='BN', requires_grad=True)
        # )
        # self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # decoder_channels=self.in_channels[::-1][1:]+[embedding_dim]
        # self.linear_fuse = MID(encoder_channels=self.in_channels , decoder_channels=decoder_channels)
        # self.linear_pred = SegmentationHead(in_channels=decoder_channels[-1] , out_channels=self.num_classes, upsampling=2.0)


        self.linear_fuse = ConvModule(
            in_channels=(c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # self.linear_pred = SegmentationHead(in_channels=3*c2_in_channels , out_channels=self.num_classes, kernel_size=1, upsampling=1.0)
        self.count =0
    def feed2gmb(self, block, gmb:global_meta_block, h,w):
        n = block.shape[0]
        block_ = block.flatten(2).transpose(1,2)
        block_ = gmb(block_, h, w)
        return block_.permute(0, 2, 1).reshape(n, -1, h, w)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  
        c1, c2, c3, c4 = x

        n, d4, h4, w4 = c4.shape #H/32
        _, d3, h3, w3 = c3.shape #H/16
        _, d2, h2, w2 = c2.shape #H/8
        _, d1, h1, w1 = c1.shape #H/4

        c41 = self.c40toc41_k1(c4)                                      #node 1
        
        c41_scale2_c31_conv3 = resize(self.c41toc31_k3(c41), size=(h3, w3), mode='bilinear', align_corners=False)
        c41_scale2_c32_conv1 = resize(self.c41toc32_k1(c41), size=(h3, w3), mode='bilinear', align_corners=False)
        
        c31 = self.c30toc31_k1(c3)   + c41_scale2_c31_conv3
        c32 = self.c31toc32_k1(c31)  + c41_scale2_c32_conv1              #node2
        
        
        c31_scale2_c21_conv3 = resize(self.c31toc21_k3(c31), size=(h2, w2), mode='bilinear', align_corners=False)
        c32_scale2_c22_conv3 = resize(self.c32toc22_k3(c32), size=(h2, w2), mode='bilinear', align_corners=False)
        c32_scale2_c22_conv1 = resize(self.c32toc22_k1(c32), size=(h2, w2), mode='bilinear', align_corners=False)
        
        c21 = self.c20toc21_k1( c2) + c31_scale2_c21_conv3
        c22 = self.c21toc22_k1(c21) + c32_scale2_c22_conv3 
        c23 = self.c22toc23_k1(c22) + c32_scale2_c22_conv1  + self.c21toc23_k1(c21) 



        _c   = torch.cat([c21, c22, c23], dim=1)
        _c = self.linear_fuse(_c)
        # _c = self.linear_fuse([c1, _c2, _c3, _c4]) #[B, embed_dim, uh, uw]
        _c = resize(_c, size=(h1, w1), mode='bilinear', align_corners=False)
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x









