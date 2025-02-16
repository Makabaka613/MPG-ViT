import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):     #这个类继承自 nn.Module，它实现了层归一化（Layer Normalization）的一个变体。该实现专注于输入形状为 (batch_size, channels, height, width) 的数据，并在通道维度上执行归一化操作。
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):        #normalized_shape：归一化的形状，即要归一化的通道数（channels），eps：防止除以零的小常数，默认值为 1e-6。
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))    #self.weight 和 self.bias：这两个参数用于缩放和偏移归一化后的输出，它们是可学习的参数
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)                 #self.normalized_shape：将输入的形状保存为一个元组，确保操作的一致性。

    def forward(self, x):
        u = x.mean(1, keepdim=True)                         #计算输入 x 在通道维度上的均值（mean），保留维度（keepdim=True）以便后续操作。
        s = (x - u).pow(2).mean(1, keepdim=True)            #计算方差（variance），即将均值去除后的结果平方，再在通道维度上求平均。
        x = (x - u) / torch.sqrt(s + self.eps)              #对输入进行归一化（减去均值并除以标准差），这里加上 eps 是为了避免除零错误。
        x = self.weight[:, None, None] * x + self.bias[:, None, None]     #应用可学习的缩放和偏移参数 weight 和 bias，并将它们扩展到 (channels, height, width) 形状以匹配输入。
        return x

"""Implementation borrowed from SegFormer.

https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py

Thanks for the open-source research environment.
"""

class MLP(nn.Module):            #这个 MLP 类继承自 nn.Module，表示一个简单的多层感知机模块。
    """MLP module."""
    def __init__(self, input_dim = 256, output_dim = 256) -> None:         #input_dim=256 和 output_dim=256 是输入和输出特征维度，默认值为 256。
        super().__init__()
        self.proj = nn.Linear(input_dim , output_dim)           #self.proj 是一个全连接层（线性层），它将输入特征维度从 input_dim 映射到 output_dim。
    def forward(self, x:torch.Tensor):
        x = x.flatten(2).transpose(1, 2)   #将输入张量 x 从第三个维度开始展平,假设 x 的形状为 [batch_size, channels, height, width]，则展平后的形状为 [batch_size, channels, height * width]。将第二个维度和第三个维度进行转置，得到的形状为 [batch_size, height * width, channels]。
        x = self.proj(x)                 
        return x.permute(0, 2, 1)   #将变换后的张量的第二个维度和第三个维度再次转置，返回的形状为 [batch_size, channels, height * width]。
    
class PredictHead(nn.Module):
    def __init__(self, 
                feature_channels : list,   #feature_channels 是一个包含五个特征图通道数的列表，对应五个不同尺度的特征图（比如来自不同的卷积层）。
                embed_dim = 256,
                predict_channels : int = 1,
                norm : str = "BN"
                ) -> None:
        """
        We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
        Some intuitive conclusions are as follows:
            - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
            - "BN" Batch norm : When include authentic images during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
            - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
        """
        
        super().__init__()
        c1_in_channel, c2_in_channel, c3_in_channel, c4_in_channel, c5_in_channel = feature_channels   #从 feature_channels 中解包五个特征图的通道数。
        assert len(feature_channels) == 5 , "feature_channels must be a list of 5 elements"            #使用断言确保输入的特征图数量为 5，否则会报错。
        # self.linear_c5 = MLP(input_dim = c5_in_channel, output_dim = embed_dim)
        # self.linear_c4 = MLP(input_dim = c4_in_channel, output_dim = embed_dim)
        # self.linear_c3 = MLP(input_dim = c3_in_channel, output_dim = embed_dim)
        # self.linear_c2 = MLP(input_dim = c2_in_channel, output_dim = embed_dim)
        # self.linear_c1 = MLP(input_dim = c1_in_channel, output_dim = embed_dim)

        self.linear_fuse = nn.Conv2d(
            in_channels= embed_dim * 5,       #这里定义了一个 1x1 卷积层，用于将五个特征图通道融合在一起，减少维度。输入通道数为 embed_dim * 5（每个特征图有 embed_dim 通道），输出为 embed_dim。
            out_channels= embed_dim,
            kernel_size= 1
        )
        
        assert norm in ["LN", "BN", "IN"], "Argument error when initialize the predict head : Norm argument should be one of the 'LN', 'BN' , 'IN', which represent Layer_norm, Batch_norm and Instance_norm"
        
        if norm == "LN":                                #根据选择的规范化方式，使用对应的层：LayerNorm、BatchNorm 或 InstanceNorm。不同的规范化方法适合不同的任务配置。
            self.norm = LayerNorm(embed_dim)
        elif norm == "BN" :
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.InstanceNorm2d(embed_dim, track_running_stats=True, affine=True)

        self.dropout = nn.Dropout()                     #定义一个 Dropout 层来防止过拟合。
        
        self.linear_predict = nn.Conv2d(embed_dim, predict_channels, kernel_size= 1)   #linear_predict 是一个 1x1 卷积层，将特征映射到最终的预测图，通道数为 predict_channels（比如输出是二值掩码时为 1）。
        
    def forward(self, x):
        c1, c2, c3, c4, c5 = x    # 1/4 1/8 1/16 1/32 1/64  x 是一个包含五个尺度特征图的列表，分别对应 1/4 到 1/64 尺度的特征图。
        
        n, _ , h, w = c1.shape # Target size of all the features   从第一个特征图中获取批次大小、通道数、高度和宽度，其他特征图会被插值到相同尺寸。
        
        # _c1 = self.linear_c1(c1).reshape(shape=(n, -1, c1.shape[2], c1.shape[3]))  对所有特征图进行双线性插值，使它们的尺寸统一为 (h, w)。
        
        _c1 =  F.interpolate(c1, size=(h, w), mode='bilinear', align_corners=False)   
        
        # _c2 = self.linear_c2(c2).reshape(shape=(n, -1, c2.shape[2], c2.shape[3]))
        
        _c2 = F.interpolate(c2, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c3 = self.linear_c3(c3).reshape(shape=(n, -1, c3.shape[2], c3.shape[3]))
        
        _c3 = F.interpolate(c3, size=(h, w), mode='bilinear', align_corners=False)
        
        # _c4 = self.linear_c4(c4).reshape(shape=(n, -1, c4.shape[2], c4.shape[3]))
        
        _c4 = F.interpolate(c4, size=(h, w), mode='bilinear', align_corners=False) 
        
        # _c5 = self.linear_c5(c5).reshape(shape=(n, -1, c5.shape[2], c5.shape[3]))
        
        _c5 = F.interpolate(c5, size=(h, w), mode='bilinear', align_corners=False)
        
        _c = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4, _c5], dim=1))     #将五个插值后的特征图在通道维度上拼接，形成一个大的特征图，然后通过 linear_fuse 进行融合。
        
        _c = self.norm(_c)                 #对融合后的特征图进行规范化和随机失活。
        
        x = self.dropout(_c)
        
        x = self.linear_predict(x)         #最后通过 linear_predict 输出最终的预测图。
        
        return x                           #返回预测结果。
