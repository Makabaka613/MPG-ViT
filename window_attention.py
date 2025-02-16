import torch
from torch import nn
import warnings
from timm.models.layers import DropPath, Mlp, trunc_normal_
from fvcore.nn.distributed import differentiable_all_reduce
import fvcore.nn.weight_init as weight_init
import torch.distributed as dist
from torch.nn import functional as F
import functools
from functools import partial
from einops import rearrange
from math import isqrt
import math, copy


"""
Thanks for the open-source of detectron2, part of codes are from their implementation:
https://github.com/facebookresearch/detectron2
"""

def get_world_size() -> int:                     #获取分布式训练的世界大小（即当前有多少个进程在参与训练）。
    if not dist.is_available():                  #检查分布式训练是否可用。如果不可用，直接返回 1，表示只有一个进程。
        return 1
    if not dist.is_initialized():                #检查分布式训练是否已经初始化。如果没有初始化，同样返回 1。
        return 1
    return dist.get_world_size()                 #如果以上检查通过，返回世界大小，即参与训练的进程数。

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

def get_rel_pos(q_size, k_size, rel_pos):         #根据查询 (query) 和键 (key) 的大小，从给定的相对位置编码中提取对应的相对位置嵌入。
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.  查询的大小（如特征图的高度或宽度）。
        k_size (int): size of key k.    键的大小（如特征图的高度或宽度）。
        rel_pos (Tensor): relative position embeddings (L, C).    对位置嵌入，形状为 (L, C)，其中 L 是相对位置的数量，C 是嵌入维度。

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)        #计算最大相对距离。相对位置编码通常涵盖所有可能的相对位移，所以最大距离是 2 * max(q_size, k_size) - 1。
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:    #如果输入的相对位置编码的长度与计算出的 max_rel_dist 不一致，需要进行插值处理（使用线性插值）。插值后，重新调整维度，确保与预期的形状匹配。
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.  计算查询和键的坐标。通过将较小的尺寸按比例缩放到与较大尺寸匹配，得到相对坐标。
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]         #根据相对坐标，从处理过的相对位置编码中提取嵌入。


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.   注意力图，形状为 (B, q_h * q_w, k_h * k_w)。
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).    查询张量，形状为 (B, q_h * q_w, C)。
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.   用于高度轴的相对位置嵌入，形状为 (Lh, C)。
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.    用于宽度轴的相对位置嵌入，形状为 (Lw, C)。
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).           查询张量的空间尺寸 (q_h, q_w)。
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).             键张量的空间尺寸 (k_h, k_w)。

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.      添加了相对位置嵌入的注意力图。
    """
    q_h, q_w = q_size     #首先，提取查询和键的空间尺寸。
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)          #使用 get_rel_pos 函数来获取高度和宽度方向的相对位置嵌入。
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)              #将查询张量 q 重新调整为 (B, q_h, q_w, C) 的形状。
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh) #通过 torch.einsum 计算查询张量与高度和宽度方向的相对位置嵌入的点积，得到相对位置嵌入的加权值 rel_h 和 rel_w。
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (                  #将注意力图调整为 (B, q_h, q_w, k_h, k_w) 的形状，以便逐元素加上高度和宽度方向的相对位置嵌入。最后，将加上相对位置嵌入后的注意力图恢复为原来的形状 (B, q_h * q_w, k_h * k_w)。
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn               #返回添加了相对位置嵌入的注意力图。


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,                      #dim: 输入通道数。 
        num_heads=8,              #num_heads: 多头注意力的头数。
        qkv_bias=True,            #qkv_bias: 是否为查询、键、值向量添加可学习的偏置。
        use_rel_pos=False,        #use_rel_pos: 是否使用相对位置嵌入。
        rel_pos_zero_init=True,   #rel_pos_zero_init: 如果为 True，则将相对位置嵌入参数初始化为 0。
        input_size=None,          #input_size: 输入的分辨率（用于确定相对位置嵌入的大小）。
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads     #计算每个头的通道数 head_dim，并通过 self.scale 对查询向量进行缩放。
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)    #使用一个线性层 self.qkv 来生成查询（Q）、键（K）和值（V）向量，维度为 dim * 3，这三个向量是并列生成的。
        self.proj = nn.Linear(dim, dim)                      #self.proj 是用于输出的线性层。

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:           #如果使用相对位置嵌入，则为高度和宽度方向分别创建两个参数矩阵 self.rel_pos_h 和 self.rel_pos_w，用于存储相对位置嵌入。
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:   #如果 rel_pos_zero_init 为 False，则使用截断正态分布对这两个参数进行初始化。
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape           
        # qkv with shape (3, B, nHead, H * W, C)         首先，通过 self.qkv 将输入 x 映射为查询、键和值的联合表示，并 reshape 为 (3, B, nHead, H * W, C) 的形状。
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)       然后，解开 qkv，得到形状为 (B * nHead, H * W, C) 的查询 q、键 k 和值 v。
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)     #通过矩阵乘法计算查询与键之间的点积，并将其缩放。

        if self.use_rel_pos:                  #如果启用了相对位置嵌入，则调用 add_decomposed_rel_pos 函数，将相对位置嵌入添加到注意力权重中。
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)     #对注意力权重进行 softmax 归一化。
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)   #将加权的值向量乘回权重，得到最终的注意力输出，然后 reshape 回原始形状 (B, H, W, C)。
        x = self.proj(x)               #最后，通过 self.proj 对输出进行线性变换，得到最终结果。

        return x

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    It contains non-trainable buffers called
    "weight" and "bias", "running_mean", "running_var",
    initialized to perform identity transformation.

    The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
    which are computed from the original four parameters of BN.
    The affine transform `x * weight + bias` will perform the equivalent
    computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
    When loading a backbone model from Caffe2, "running_mean" and "running_var"
    will be left unchanged as identity transformation.

    Other pre-trained backbone models may contain all 4 parameters.

    The forward is implemented by `F.batch_norm(..., training=False)`.
    """

    _version = 3

    def __init__(self, num_features, eps=1e-5):   #构造函数中，指定了输入的通道数 num_features 和一个小的 eps 值用于计算稳定性。
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))   #在这里，定义了四个不可训练的缓冲区，分别是 weight、bias、running_mean 和 running_var。这些缓冲区用于保持固定的批归一化参数。weight 初始化为 1，bias 初始化为 0，而 running_mean 初始化为 0，running_var 初始化为 1（减去一个小的 eps 值）。
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        if x.requires_grad:  #当输入需要梯度时，手动进行归一化计算。首先计算 scale 和 bias，然后调整它们的形状以匹配输入张量的形状。最后将输入张量按 scale 和 bias 进行仿射变换。
            # When gradients are needed, F.batch_norm will use extra memory
            # because its backward op computes gradients for weight/bias as well.
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype  # may be half
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:     #当不需要梯度时，使用 PyTorch 的 F.batch_norm 函数执行批归一化操作，并指定 training=False，表示在推理时不更新运行均值和方差。这样可以获得更好的优化效果。
            # When gradients are not needed, F.batch_norm is a single fused op
            # and provide more optimization opportunities.
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # No running_mean/var in early versions
            # This will silent the warnings
            if prefix + "running_mean" not in state_dict:
                state_dict[prefix + "running_mean"] = torch.zeros_like(self.running_mean)
            if prefix + "running_var" not in state_dict:
                state_dict[prefix + "running_var"] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def __repr__(self):
        return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module
        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


class CNNBlockBase(nn.Module):
    """
    A CNN block is assumed to have input channels, output channels and a stride.
    The input and output of `forward()` method must be NCHW tensors.
    The method can perform arbitrary computation but must match the given
    channels and stride specification.

    Attribute:
        in_channels (int):
        out_channels (int):
        stride (int):
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            in_channels (int):
            out_channels (int):
            stride (int):
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        """
        Make this block not trainable.
        This method sets all parameters to `requires_grad=False`,
        and convert all BatchNorm layers to FrozenBatchNorm

        Returns:
            the block itself
        """
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            with warnings.catch_warnings(record=True):
                if x.numel() == 0 and self.training:
                    # https://github.com/pytorch/pytorch/issues/12013
                    assert not isinstance(
                        self.norm, torch.nn.SyncBatchNorm
                    ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
BatchNorm2d = torch.nn.BatchNorm2d
  
class NaiveSyncBatchNorm(BatchNorm2d):
    """
    In PyTorch<=1.5, ``nn.SyncBatchNorm`` has incorrect gradient
    when the batch size on each worker is different.
    (e.g., when scale augmentation is used, or when it is applied to mask head).

    This is a slower but correct alternative to `nn.SyncBatchNorm`.

    Note:
        There isn't a single definition of Sync BatchNorm.

        When ``stats_mode==""``, this module computes overall statistics by using
        statistics of each worker with equal weight.  The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (N, H, W). This mode does not support inputs with zero batch size.

        When ``stats_mode=="N"``, this module computes overall statistics by weighting
        the statistics of each worker by their ``N``. The result is true statistics
        of all samples (as if they are all on one worker) only when all workers
        have the same (H, W). It is slower than ``stats_mode==""``.

        Even though the result of this module may not be the true statistics of all samples,
        it may still be reasonable because it might be preferrable to assign equal weights
        to all workers, regardless of their (H, W) dimension, instead of putting larger weight
        on larger images. From preliminary experiments, little difference is found between such
        a simplified implementation and an accurate computation of overall mean & variance.
    """

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        half_input = input.dtype == torch.float16
        if half_input:
            # fp16 does not have good enough numerics for the reduction here
            input = input.float()
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        if self._stats_mode == "":
            assert B > 0, 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = torch.cat([mean, meansqr], dim=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = torch.split(vec, C)
            momentum = self.momentum
        else:
            if B == 0:
                vec = torch.zeros([2 * C + 1], device=mean.device, dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
            else:
                vec = torch.cat(
                    [mean, meansqr, torch.ones([1], device=mean.device, dtype=mean.dtype)], dim=0
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clamp(max=1) * self.momentum  # no update if total_batch is 0
            mean, meansqr, _ = torch.split(vec / total_batch.clamp(min=1), C)  # avoid div-by-zero

        var = meansqr - mean * mean
        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        self.running_mean += momentum * (mean.detach() - self.running_mean)
        self.running_var += momentum * (var.detach() - self.running_var)
        ret = input * scale + bias
        if half_input:
            ret = ret.half()
        return ret

def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
            # expose stats_mode N as an option to caller, required for zero-len inputs
            "naiveSyncBN_N": lambda channels: NaiveSyncBatchNorm(channels, stats_mode="N"),
            "LN": lambda channels: LayerNorm(channels),
        }[norm]
    return norm(out_channels)

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ResBottleneckBlock(CNNBlockBase):  #这个 ResBottleneckBlock 实现了一个标准的瓶颈残差块，适用于构建深层卷积神经网络。它使用了 1x1、3x3 和 1x1 的卷积组合来高效地进行特征提取，同时通过残差连接保持了信息的流动。
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm="LN",
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__(in_channels, out_channels, 1)

        self.conv1 = Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = get_norm(norm, bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            3,
            padding=1,
            bias=False,
        )
        self.norm2 = get_norm(norm, bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = get_norm(norm, out_channels)

        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x  
        for layer in self.children():   #在前向传播中，输入 x 经过所有层的处理。
            out = layer(out)

        out = x + out   #最后，将输入 x 和经过处理的输出 out 相加，实现残差连接。这种跳跃连接使得梯度在深层网络中更容易传播。
        return out   #返回残差输出。

def window_partition(x, window_size):  #这两段代码在处理图像时提供了灵活的窗口划分和恢复功能。它们允许对图像进行局部操作，同时通过填充确保窗口的大小适合处理。
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

class Block(nn.Module):  #该 Block 类是一个灵活且强大的 Transformer 块，集成了多头注意力机制、MLP、窗口注意力以及可选的残差块，适用于需要局部和全局特征融合的视觉任务。
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x



class Block(nn.Module):   #这个 Block 类实现了一个灵活且功能丰富的 Transformer 块，结合了局部窗口注意力机制、全局特征聚合和可选的残差瓶颈块，特别适合复杂的视觉任务。
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        return x


class PatchEmbed(nn.Module):   #这个 PatchEmbed 类的主要作用是将输入图像转换为一系列固定大小的 patch，并将每个 patch 嵌入到一个高维向量空间中。
    """
    Image to Patch Embedding.
    """

    def __init__(
        self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0), in_chans=3, embed_dim=768
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):   #这个函数用于在输入图像的空间分辨率变化时，对位置嵌入进行相应调整，确保位置嵌入能够匹配新的输入尺寸，尤其在 Vision Transformer (ViT) 等模型中很常见。
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)

###############################################################新加###################################################################################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MSAAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MSAAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MSAAttention2(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MSAAttention2, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class CrossAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(CrossAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class CrossAttention2(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(CrossAttention2, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)



class Prompt_block(nn.Module):
    def __init__(self):
        super(Prompt_block, self).__init__()
        # 实例化类
        self.msa = MSAAttention(h=4, d_model=768, dropout=0.1)
        self.msa2 = MSAAttention2(h=4, d_model=768, dropout=0.1)
        self.mca = CrossAttention(h=4, d_model=768, dropout=0.1)
        self.mca2 = CrossAttention2(h=4, d_model=768, dropout=0.1)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)  # 选择 kernel_size 和 stride
        self.deconv = nn.ConvTranspose2d(in_channels=768, out_channels=768, kernel_size=8, stride=8)
        # self.deconv1 = nn.ConvTranspose2d(in_channels=768, out_channels=3, kernel_size=4, stride=4, padding=0)
        # self.conv4 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.linear = nn.Linear(1024, 768)


    def forward(self,x: torch.Tensor, text_tensor: torch.Tensor ,num: int):
        # print("初始 的x ,text",x.shape,text_tensor.shape)    #初始 的x ,text torch.Size([1, 64, 64, 768]) torch.Size([1, 1, 1024])
        x = x.float()  
        text_tensor = text_tensor.float()  

        # 将 x 的形状从 (1, 64, 64, 768) 转换为 (1, 768, 64, 64)
        x0 = x.permute(0, 3, 1, 2)  # 变为 (1, 768, 64, 64)

        # 使用平均池化将 x 转换为 (1, 768, 8, 8)
        x0 = F.avg_pool2d(x0, kernel_size=8, stride=8)

        # 将 x 重新调整为 (1, 768, 64)
        x0 = x0.view(x0.size(0), x0.size(1), -1)  # 变为 (1, 768, 64)

        # 最后将 x 变为 (1, 64, 768)
        x1 = x0.permute(0, 2, 1)  # 变为 (1, 64, 768)

        if num == 0:  # 第一次循环
            # 对 text_tensor 进行卷积和线性变换
            text_tensor = text_tensor.unsqueeze(2)  # 变为 (1, 1, 1, 1024)
            text_tensor = self.conv1(text_tensor)  # 变为 (1, 64, 1, 1024)
            text_tensor0 = self.linear(text_tensor.view(1, 64, 1024))  # 变为 (1, 64, 768)
        else:  # 后续循环，直接使用已经是 (1, 64, 64，768) 的 text_tensor
            text_tensor = text_tensor.permute(0, 3, 1, 2)  # 变为 (1, 768, 64, 64)
            text_tensor = F.avg_pool2d(text_tensor, kernel_size=8, stride=8)
            text_tensor = text_tensor.view(text_tensor.size(0), text_tensor.size(1), -1)  # 变为 (1, 768, 64)
            text_tensor0 = text_tensor.permute(0, 2, 1)  # 变为 (1, 64, 768)

        
        x2 = self.msa(query=x1, key=text_tensor0, value=text_tensor0)   
        # print("MSA 后的 x", x.shape)  # MSA 后的 x torch.Size([1, 64, 768])
        text_tensor1 = self.msa2(query=text_tensor0, key=text_tensor0, value=text_tensor0)
        # print("MSA 后的 text_tensor", text_tensor.shape)  # MSA 后的 text_tensor torch.Size([1, 64, 768])
        x3 = self.mca(query=x2, key=text_tensor1, value=text_tensor1)
        # print("MCA 后的 x", x.shape)  # MCA 后的 x torch.Size([1, 64, 768])
        text_tensor2 = self.mca2(query=text_tensor1, key=text_tensor1, value=text_tensor1)
        # print("MCA2 后的 text_tensor", text_tensor.shape)  # MCA2 后的 text_tensor torch.Size([1, 64, 768])

        residual_x = x1
        residual_t = text_tensor0
        # print('11111111111111111111111',residual_x.shape,residual_t.shape)    #torch.Size([1, 64, 768]) torch.Size([1, 64, 768])
        x4 = x3 + residual_x
        text_tensor3 = text_tensor2 + residual_t
        # print('22222222222222222222222',x3.shape,text_tensor3.shape)             #torch.Size([1, 64, 768]) torch.Size([1, 64, 768])

            # if i == depth - 1:  # 判断是否为最后一次循环
        # 将 x3 变回原来的维度
        x4 = x4.view(x4.size(0), 64, 768)  # (1, 64, 768)
        x4 = x4.permute(0, 2, 1)  # 变为 (1, 768, 64)
        x4 = x4.view(x4.size(0), 768, 8, 8)  # 变为 (1, 768, 8, 8)
        x4 = self.deconv(x4)  # 卷积转换为 (1, 768, 64, 64)
        x4 = x4.permute(0, 2, 3, 1)  # 变为 (1, 64, 64, 768)

        text_tensor3 = text_tensor3.view(text_tensor3.size(0), 64, 768)  # (1, 64, 768)
        text_tensor3 = text_tensor3.permute(0, 2, 1)  # 变为 (1, 768, 64)
        text_tensor3 = text_tensor3.view(text_tensor3.size(0), 768, 8, 8)  # 变为 (1, 768, 8, 8)
        text_tensor3 = self.deconv(text_tensor3)  # 卷积转换为 (1, 768, 64, 64)
        text_tensor3 = text_tensor3.permute(0, 2, 3, 1)  # 变为 (1, 64, 64, 768)

        # # 将 text_tensor3 变回原来的维度
        # text_tensor3 = text_tensor3.view(text_tensor3.size(0), 64, 768)  # (1, 64, 768)
        # text_tensor3 = text_tensor3.permute(0, 2, 1)  # 变为 (1, 768, 64)
        # text_tensor3 = text_tensor3.view(text_tensor3.size(0), 768, 8, 8)  # 变为 (1, 768, 8, 8)
        # text_tensor3 = self.deconv1(text_tensor3)  # 卷积转换为 (1, 3, 32, 32)
        # text_tensor3 = self.conv4(text_tensor3)  # 从 (1, 3, 32, 32) 转换为 (1, 1, 32, 32)
        # text_tensor3 = text_tensor3.view(text_tensor3.size(0), 1, -1)  # 输出形状为 (1, 1, 1024)
        
        return x4, text_tensor3

#########################################################################结束################################################################

class ViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,                #输入图像的尺寸。
        patch_size=16,                #每个 patch 的尺寸。
        in_chans=3,                   #输入图像的通道数。
        embed_dim=768,                #patch 嵌入的维度。
        depth=12,                     #ViT 模型的层数（深度）。
        num_heads=12,                 #注意力头的数量。
        mlp_ratio=4.0,                #MLP 层隐藏维度相对于嵌入维度的比率。
        qkv_bias=True,                #是否在查询、键和值中添加可学习的偏置。
        drop_path_rate=0.0,           #随机深度的概率。
        norm_layer=nn.LayerNorm,      #归一化层的类型。
        act_layer=nn.GELU,            #激活函数的类型。
        use_abs_pos=True,             #是否使用绝对位置嵌入。
        use_rel_pos=False,            #是否使用相对位置嵌入。
        rel_pos_zero_init=True,       #是否将相对位置参数初始化为零。
        window_size=0,                #窗口注意力块的窗口大小。
        window_block_indexes=(),      #使用窗口注意力的块的索引列表。
        residual_block_indexes=(),    #使用卷积传播的块的索引列表。
        use_act_checkpoint=False,     #是否使用激活检查点。
        pretrain_img_size=224,        #预训练模型使用的图像尺寸。
        pretrain_use_cls_token=True,  #预训练模型是否使用分类 token。
        out_feature="last_feat",      #从最后一个块提取的特征的名称。
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(   #PatchEmbed 层用于将输入图像划分为 patch 并进行嵌入。这里使用 patch_size 来定义每个 patch 的尺寸，in_chans 为输入图像的通道数，embed_dim 为每个 patch 的嵌入维度。
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
##########################################################新加###########################################################################################
       
        self.prompt_blocks = []  
        self.prompt_blocks = nn.ModuleList()
        for i in range(depth):
            prompt_block = Prompt_block()
            self.prompt_blocks.append(prompt_block)  
          

###########################################################结束########################################################################################

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule  使用线性空间从 0 到 drop_path_rate 生成一个深度为 depth 的随机深度率列表 dpr。这个列表用于在模型训练过程中应用随机深度（drop path）。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()   #self.blocks 用于存储 Transformer 块的列表。在实际实现中，你会将 Transformer 的每一层添加到这个 ModuleList 中。
        for i in range(depth):   #这个循环根据 depth 的值创建多个 Block 对象，并将它们添加到 self.blocks 列表中。每个 Block 的配置可能会根据当前的索引 i 进行调整，例如是否使用窗口注意力、是否使用残差块等。 
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            # if use_act_checkpoint:
                # block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}  #这些变量定义了输出特征的通道数、步幅、特征图大小等信息。这些信息在模型的后续处理和应用中非常重要。
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]


        self.output_size = (img_size // patch_size, img_size // patch_size)
        self.stride = patch_size
        self.channels = embed_dim
        
        if self.pos_embed is not None:  #如果使用绝对位置嵌入，则对其进行截断正态分布初始化。
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)  #使用 _init_weights 方法对模型的权重进行初始化。这个方法会为线性层和 LayerNorm 层初始化权重和偏置。

    def output_shape(self):   #output_shape 方法返回输出特征图的大小、步幅和通道数，用于描述模型的输出结构。
        return {
            '${.net.out_feature}':self.output_size,
            'stride': self.stride,
            'channels' : self.channels
            } 
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

########################################################新加##########################################################################################
    def sample_norm(self, input_tensor):
        # 计算均值和标准差
        mean = input_tensor.mean(dim=[0, 2, 3], keepdim=True)
        std = input_tensor.std(dim=[0, 2, 3], keepdim=True)

        # 归一化
        normalized_tensor = (input_tensor - mean) / std
        return normalized_tensor

    def forward_features(self, x, text_tensor):
        x = self.patch_embed(x)   #将输入图像通过 patch_embed 层转化为 patch 嵌入。
        x_ = self.add_position_embed(x)    #添加位置嵌入（如果有）。
        x = x + x_
        # print('111111111111111',x.shape,text_tensor.shape)    #torch.Size([1, 64, 64, 768]) torch.Size([1, 1, 1024])

        # with torch.cuda.amp.autocast(enable=False):
        #     x = x.to(torch.float32)
        #     text_tensor = text_tensor.to(torch.float32)
        x1, text_tensor = self.prompt_blocks[0](x, text_tensor, num=0)
        x = x + self.sample_norm(x1)

        for i, blk in enumerate(self.blocks):
            if i >= 1 and i < len(self.blocks)-1:
                x1, text_tensor = self.prompt_blocks[i](x, text_tensor, num=i)
                x = x + self.sample_norm(x1)
                # print('222222222222222222222222',x.shape,text_tensor.shape)    #torch.Size([1, 64, 64, 768]) torch.Size([1, 1, 1024])
            x= blk(x)
        
        return x
############################################################结束#########################################################################################
    def add_position_embed(self, x):   #add_position_embed 方法将绝对位置嵌入加到输入的特征图上。如果模型使用了位置嵌入，这一步是必须的，以将位置编码融入到输入特征中。
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
        return x
    def mae_forward(self, x):    #mae_forward 方法是为 MAE（Masked Autoencoder）模型设计的。它通过所有的 Block 进行前向传播，并在最后将特征图的维度重新排列。
        for blk in self.blocks:
            x = blk(x)
        return x.permute(0, 3, 1, 2)

############################################改动###########################################
    def forward(self, x, text_tensor):
        x = self.forward_features(x, text_tensor)
        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs
    
    # def forward(self, x):
    #     x = self.patch_embed(x)   #将输入图像通过 patch_embed 层转化为 patch 嵌入。
    #     x = self.add_position_embed(x)    #添加位置嵌入（如果有）。
    #     for blk in self.blocks:         #通过所有 Transformer 块。
    #         x = blk(x)

    #     outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}   #返回最终的特征图，其中 x.permute(0, 3, 1, 2) 用于调整维度顺序。
    #     return outputs
#########################################结束#######################################################

class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.  这个类定义了一个简单的池化操作，用于生成下采样后的特征图。它在特征金字塔中用于从 P5 层生成 P6 层的特征图。
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1  #将 num_levels 设置为 1，将 in_feature 设置为 "p5"，表示这个模块是用来处理 P5 特征层的。
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]  #对输入的特征图 x 应用 1x1 的最大池化操作，步幅为 2，填充为 0。这将特征图 x 下采样一倍。返回下采样后的特征图，以列表形式给出。

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".  检查 strides 列表中的每个步幅是否是前一个步幅的两倍。这确保了步幅在对数尺度上是连续的，这对维持特征金字塔的一致性至关重要。
    """
    for i, stride in enumerate(strides[1:], 1):   #函数从 strides 列表的第二个元素开始，验证每个步幅是否是前一个步幅的两倍。
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class SimpleFeaturePyramid(nn.Module):
    """
    An sequetial implementation of Simple-FPN in 'vitdet' paper.  定义了一个名为 SimpleFeaturePyramid 的类，它继承自 nn.Module用于实现特征金字塔网络。
    """
    def __init__(self,
        in_feature_shape,   #输入特征的形状，格式为 (N, C, H, W)，表示批量大小、通道数、高度和宽度。
        out_channels,       #每个特征图的输出通道数。
        scale_factors,      #每个特征图的缩放因子。
        input_stride = 16,  #输入特征的步幅，默认值为 16。
        top_block=None,     #一个可选的模块，用于在最小分辨率的特征图上进一步进行下采样。
        norm=None           #用于指定归一化层，通常用于规范化特征。
    ) -> None:
        """
        Args:
            in_feature_shape (4d tensor): (N, C, H, W) for shape of input feature come from backbone.
            out_channles (int): number of output channels for each feature map.
            scale_factors (list of int): scale factors for each feature map.
            top_block ( optional): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5). Defaults to None.
            norm (nn.Module, optional): norm layers. need to be implemented.
        """
        super().__init__()                    #调用父类的构造函数并提取输入特征的通道数、缩放因子等信息。
        
        _, dim, H, W = in_feature_shape
        self.dim = dim
        self.scale_factors = scale_factors
        
        self.stages = []                       #计算每个特征图的步幅，并验证步幅是否是 2 的幂。use_bias 变量根据是否使用归一化层决定是否在卷积层中使用偏置项。
        strides = [input_stride // s for s in scale_factors]
        _assert_strides_are_log2_contiguous(strides)
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):      #根据缩放因子，构建不同的上采样或下采样操作。例如：
            out_dim = dim
            if scale == 4.0:                   #当 scale=4.0 时，进行两次上采样，使特征图放大到原来的 4 倍。
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:                 #当 scale=2.0 时，进行一次上采样。
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:                #当 scale=0.5 时，进行一次下采样。 如果缩放因子不在支持的范围内，则抛出未实现的异常。
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(                #在上采样或下采样之后，添加了两个卷积层：第一个卷积层使用 1x1 的卷积核，用于调整通道数。第二个卷积层使用 3x3 的卷积核，并添加适当的填充以保持空间尺寸不变。
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)
            stage = int(math.log2(strides[idx]))          #根据步幅计算当前阶段的编号，并将构建好的层添加到模块中，同时保存到 self.stages 中，便于在前向传播时使用。
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

            self.top_block = top_block                     #保存每个阶段输出特征图的步幅，并生成对应的输出名称，例如 p2、p3 等。
            # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
            self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
            # top block output feature maps.
            if self.top_block is not None:                 #如果提供了 top_block，在最小分辨率的特征图上进一步进行下采样，并更新输出特征图的信息。
                for s in range(stage, stage + self.top_block.num_levels):
                    self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

            self._out_features = list(self._out_feature_strides.keys())           #确定所有输出特征的名称和通道数。
            self._out_feature_channels = {k: out_channels for k in self._out_features}

    def forward(self, x):
        bottom_up_features = x # input contains the output of backbone ViT   在前向传播中，提取输入中的特征。这里假设输入特征包含来自主干网络（例如 Vision Transformer）的输出，并且特征存储在 'last_feat' 键下。
        # print(bottom_up_features)
        features = bottom_up_features['last_feat']
        
        results = []                                  #依次通过每个阶段的特征金字塔网络进行处理，并将结果保存。

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:                #如果定义了 top_block，则在最后一个特征图上进一步处理，并将结果添加到输出中。
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)            #确保输出的特征图数量与预期相符，并返回一个字典，其中键为特征图名称（例如 p2、p3），值为对应的特征图。
        return {f: res for f, res in zip(self._out_features, results)}

"""
Code below is for testing
"""
if __name__ == '__main__':

    from functools import partial
    embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1

    net= ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    )
    
    model = SimpleFeaturePyramid(
        in_feature_shape= (1, 768, 256, 256),   #表示输入特征图的形状，其中：1：批大小 (batch size)，通常设置为 1。768：通道数，这是从 ViT 的嵌入维度（embed_dim）来的。256, 256：输入特征图的高和宽。
        out_channels=256,                       #输出特征图的通道数设置为 256。
        scale_factors=(4.0, 2.0, 1.0, 0.5),     #指定每个特征图缩放的比例。不同的缩放比例用于不同层级的特征图。
        top_block=LastLevelMaxPool(),           #这是一个额外的操作，用于处理最后一层输出，进行更进一步的下采样。
        norm="LN",                              #使用层归一化（Layer Norm）作为归一化方式。
    )
    
    model.cpu()                                 #这部分代码将模型放在 CPU 上进行测试，并输出模型结构。
    print("constructed model")
    print(model)
    x = torch.randn(1, 3, 1024, 1024)           #创建一个随机的输入张量，形状为 (1, 3, 1024, 1024)，表示一张具有 3 个通道的 1024x1024 图像。
    x = net(x)                                  #将输入图像通过 ViT 编码器 net 得到特征图，这里的 x 是编码后的特征。
    # last_feature = x["last_feat"]
    print(x)
    # y = model(last_feature)
    y = model(x)                                #将特征图传入 SimpleFeaturePyramid 模型进行特征金字塔的处理。
    for k in y.keys():
        print(k)
        print(y[k].shape)                       #y 是一个字典，其中包含不同层次的特征图，键名如 "p2", "p3", 等。这段代码打印每个特征图的名称及其形状，验证模型的输出。