import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import vmamba_for_fuse
from typing import Any, Optional, Tuple
from torch.autograd import Function

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 ffn_expansion_factor=1.,
                 qkv_bias=False, ):
        super(BaseFeatureExtraction, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias, )
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor, )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# =============================================================================

# =============================================================================
import numbers


##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class multihead(nn.Module):
    def __init__(self,
                 dim=128,
                 num_heads=8,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(multihead, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.reduce_channel = nn.Conv2d(int(dim), int(dim//2), kernel_size=1, bias=False)
    def forward(self, a,b):
        x = torch.concat((a,b),dim=1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.reduce_channel(x)
        return x

class multihead2(nn.Module):
    def __init__(self,
                 dim=128,
                 num_heads=8,
                 ffn_expansion_factor=1.,
                 qkv_bias=False,):
        super(multihead2, self).__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.attn = AttentionBase(dim, num_heads=num_heads, qkv_bias=qkv_bias,)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(in_features=dim,
                       ffn_expansion_factor=ffn_expansion_factor,)
        self.reduce_channel = nn.Conv2d(int(dim), int(dim//2), kernel_size=1, bias=False)
    def forward(self, a,b):
        x = torch.concat((a, b), dim=1)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = self.reduce_channel(x)
        return x

class VMamba_Encoder(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Encoder, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=64,  # 3
            num_classes=1000,
            depths=[4],
            dims=[64],  # 96, 192, 384, 768
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
        self.patch_embed = OverlapPatchEmbed(1, 64)
        self.baseFeature = BaseFeatureExtraction(dim=64, num_heads=8)
        self.detailFeature = DetailFeatureExtraction()
    def forward(self, x):
        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.mamba(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class VMamba_Fuse_BASE(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Fuse_BASE, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=128,  # 3
            num_classes=1000,
            depths=[1, 1],  # 2292
            dims=[64, 64],  # 96, 192, 384, 768
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
    def forward(self, x, y):
        z = torch.concat((x,y),dim=1)
        z = self.mamba(z)
        return z

class VMamba_Fuse_cnn(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Fuse_cnn, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=128,  # 3
            num_classes=1000,
            depths=[1, 1],  # 2292
            dims=[64, 64],  # 96, 192, 384, 768
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
    def forward(self, x, y):
        z = torch.concat((x,y),dim=1)
        z = self.mamba(z)
        return z

class VMamba_Decoder(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Decoder, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=64,  # 3
            num_classes=1000,
            depths=[4],  # 2292
            dims=[64],  # 是否需要慢慢下降
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
        self.reduce_channel = nn.Conv2d(int(64 * 2), int(64), kernel_size=1, bias=bias)

        self.output = nn.Sequential(
            nn.Conv2d(int(64), int(64) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(64) // 2, 1, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, x, y):
        out_enc_level0 = torch.cat((x, y), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.mamba(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0

class VMamba_Decoder1(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Decoder1, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=64,  # 3
            num_classes=1000,
            depths=[4],  # 2292
            dims=[64],  # 是否需要慢慢下降
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
        self.reduce_channel = nn.Conv2d(int(64 * 2), int(64), kernel_size=1, bias=bias)

        self.output = nn.Sequential(
            nn.Conv2d(int(64), int(64) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(64) // 2, 1, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()
    def forward(self, inp_img, out_enc_level0):

        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.mamba(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1), out_enc_level0

class VMamba_Fuse_base(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Fuse_base, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=64,  # 3
            num_classes=1000,
            depths=[1],  # 2292
            dims=[64],  # 96, 192, 384, 768
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
    def forward(self, x):
        x = self.mamba(x)
        return x

class classify_CNN(nn.Module):
    def __init__(self, inchannel=64, bias=False):
        super(classify_CNN, self).__init__()
        self.block1 = nn.Conv2d(inchannel, inchannel, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block2 = nn.Conv2d(inchannel, inchannel * 2, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block3 = nn.Conv2d(inchannel * 2, inchannel * 2, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block4 = nn.Conv2d(inchannel * 2, inchannel * 4, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block5 = nn.Conv2d(inchannel * 4, inchannel * 4, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(inchannel , momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn2 = nn.BatchNorm2d(inchannel *2, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn3 = nn.BatchNorm2d(inchannel *2, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn4 = nn.BatchNorm2d(inchannel *4, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn5 = nn.BatchNorm2d(inchannel *4, momentum=0.9, eps=1e-5)  # attentionFGAN没有
        self.ln1 = nn.Linear(inchannel * 4,inchannel * 16)
        self.ln2 = nn.Linear(inchannel * 16, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        x = F.leaky_relu(self.bn1(self.block1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.block2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.block3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.block4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.block5(x)), negative_slope=0.2)
        x = torch.mean(x, dim=(2, 3), keepdim=True).view(b, c*4)
        x = self.ln1(x)
        x = torch.sigmoid(self.ln2(x))
        return x

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class classify_BASE(nn.Module):
    def __init__(self, inchannel=64, bias=False):
        super(classify_BASE, self).__init__()
        self.block1 = nn.Conv2d(inchannel, inchannel, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block2 = nn.Conv2d(inchannel, inchannel * 2, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block3 = nn.Conv2d(inchannel * 2, inchannel * 2, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block4 = nn.Conv2d(inchannel * 2, inchannel * 4, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.block5 = nn.Conv2d(inchannel * 4, inchannel * 4, kernel_size=3,
                              stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(inchannel , momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn2 = nn.BatchNorm2d(inchannel *2, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn3 = nn.BatchNorm2d(inchannel *2, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn4 = nn.BatchNorm2d(inchannel *4, momentum=0.9, eps=1e-5)#attentionFGAN没有
        self.bn5 = nn.BatchNorm2d(inchannel *4, momentum=0.9, eps=1e-5)  # attentionFGAN没有
        self.ln1 = nn.Linear(inchannel * 4,inchannel * 16)
        self.ln2 = nn.Linear(inchannel * 16, 1)
        self.grl = GRL_Layer()

    def forward(self, x):
        b, c, h, w = x.size()
        x = F.leaky_relu(self.bn1(self.block1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.block2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.block3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.block4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.block5(x)), negative_slope=0.2)
        x = torch.mean(x, dim=(2, 3), keepdim=True).view(b, c*4)
        x = self.ln1(x)
        x = torch.sigmoid(self.ln2(x))
        return x

    def grl_forward(self, x):
        b, c, h, w = x.size()
        x = F.leaky_relu(self.bn1(self.block1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.block2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.block3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.block4(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn5(self.block5(x)), negative_slope=0.2)
        x = torch.mean(x, dim=(2, 3), keepdim=True).view(b, c * 4)
        x = self.ln1(x)
        x = torch.sigmoid(self.ln2(x))
        x = self.grl(x)
        return x

class VMamba_Encoder1(nn.Module):
    def __init__(self, inchannel=1, bias=False):
        super(VMamba_Encoder1, self).__init__()
        self.mamba = vmamba_for_fuse.VSSM(
            patch_size=1,
            in_chans=64,  # 3
            num_classes=1000,
            depths=[4],
            dims=[64],  # 96, 192, 384, 768
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version="v2",  # "v1", "v2", "v3"
            patchembed_version="v1",  # "v1", "v2"
            use_checkpoint=False,
        )
        self.patch_embed = OverlapPatchEmbed(1, 64)
        self.baseFeature1 = BaseFeatureExtraction(dim=64, num_heads=8)
        self.baseFeature2 = BaseFeatureExtraction(dim=64, num_heads=8)
        # self.detailFeature = DetailFeatureExtraction()
    def forward(self, x):
        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.mamba(inp_enc_level1)
        base_feature = self.baseFeature1(out_enc_level1)
        detail_feature = self.baseFeature2(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class downsample(nn.Module):
    def __init__(self, inchannel=128, embed_dim=64, bias=False):
        super(downsample, self).__init__()
        self.conv1 = nn.Conv2d(inchannel, embed_dim, kernel_size=1,
                              stride=1, padding=0, bias=bias)
        self.bn1 = nn.BatchNorm2d(embed_dim, momentum=0.9, eps=1e-5)


    def forward(self, a,b):
        x = torch.concat((a, b), dim=1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        return x