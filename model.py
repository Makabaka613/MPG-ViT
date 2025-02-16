from modules.window_attention_ViT import ViT as window_attention_vit, SimpleFeaturePyramid, LastLevelMaxPool
from modules.decoderhead import PredictHead

from einops import rearrange
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from math import isqrt

import sys
sys.path.append('./modules')   #这两行代码将 './modules' 目录添加到 Python 模块搜索路径中。这允许 Python 脚本在 './modules' 目录中查找和导入模块。


class iml_vit_model(nn.Module):
    
    def __init__(
        self, 
        # ViT backbone:
        input_size = 1024,  #输入图像的尺寸（宽度和高度），默认值为 1024。
        patch_size = 16,    #将输入图像划分为小块的尺寸，默认值为 16。
        embed_dim = 768,    #嵌入维度，即 ViT 中每个补丁的特征维度，默认值为 768。
        vit_pretrain_path = None, # wether to load pretrained weights  预训练权重的路径，如果 None 则不加载预训练权重。
        # Simple_feature_pyramid_network:
        fpn_channels = 256,   #特征金字塔网络中通道的数量，默认值为 256。
        fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),   #特征金字塔网络中的尺度因子，用于调整不同层的特征图大小。

        # MLP embedding:
        mlp_embeding_dim = 256,   #MLP 嵌入的维度，默认值为 256
        # Decoder head norm
        predict_head_norm = "BN",  #预测头部使用的归一化方法，可以是 'BN'（批量归一化）、'LN'（层归一化）或 'IN'（实例归一化）
        # Edge loss:
        edge_lambda = 20,   #边缘损失的权重超参数，默认值为 20。
    ):
        """init iml_vit_model
        # TODO : add more args
        Args:
            input_size (int): size of the input image, defalut to 1024
            patch_size (int): patch size of Vision Transformer
            embed_dim (int): embedding dim for the ViT
            vit_pretrain_path (str): the path to initialize the model before start training
            fpn_channels (int): the number of embedding channels for simple feature pyraimd
            fpn_scale_factors(list(float, ...)) : the rescale factor for each SFPN layers.
            mlp_embedding dim: dim of mlp, i.e. decoder head
            predict_head_norm: the norm layer of predict head, need to select amoung 'BN', 'IN' and "LN"
                                We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
                            Some intuitive conclusions are as follows:
                                - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
                                - "BN" Batch norm : When include authentic images during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
                                - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
            edge_lambda(float) : the hyper-parameter for edge loss (lambda in our paper)
        """
        super(iml_vit_model, self).__init__()   #调用父类的初始化方法。这里 iml_vit_model 继承了某个父类，这行代码确保父类的属性和方法被正确初始化。
        self.input_size = input_size    #将输入图像的尺寸 input_size 赋值给类的属性 self.input_size，用于后续模块中处理输入图像的大小。
        self.patch_size = patch_size   #将图像分割的补丁（patch）的大小 patch_size 赋值给类的属性 self.patch_size。这个参数通常用于确定图像在 ViT 中如何被分割。
      

        # window attention vit   初始化一个 Vision Transformer（ViT）模型，这个模型使用了基于窗口的注意力机制（Window Attention）。以下是 window_attention_vit 模型中的各个参数设置。
        self.encoder_net = window_attention_vit(  
            img_size = input_size,    #设置输入图像的大小为 input_size，与前面定义的 self.input_size 相同。
            patch_size=16,     #指定图像补丁的大小为 16，表示输入图像会被分割成 16x16 的块。
            embed_dim=embed_dim,     #设置嵌入维度为 embed_dim，即每个图像补丁在进入 Transformer 模型前会被转换成的特征维度。
            depth=12,      #Transformer 的深度为 12，表示模型包含 12 层编码器。
            num_heads=12,    #每层多头自注意力的头数为 12，表示每层有 12 个注意力机制用于捕捉不同的特征。
            drop_path_rate=0.1,    #路径丢弃率为 0.1，用于在训练过程中引入随机性，防止过拟合。
            window_size=14,      #窗口大小为 14，表示窗口注意力机制中使用 14x14 的窗口进行局部注意力计算。
            mlp_ratio=4,       #多层感知机（MLP）部分的宽度比为 4，决定了在 Transformer 中 MLP 层的隐含层宽度。
            qkv_bias=True,     #在查询、键、值（QKV）矩阵的计算中启用偏置项，有助于提高模型性能。
            norm_layer=partial(nn.LayerNorm, eps=1e-6),    #指定归一化层为 LayerNorm，并设置其 epsilon 值为 1e-6，用于在模型中规范化特征。
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
            ],    #定义哪些层使用窗口注意力。这里的索引对应模型的不同层，表示指定层将使用窗口注意力，而其他未列出的层可能使用全局注意力。
            residual_block_indexes=[],    #定义残差块（Residual Block）的索引。这里为空列表，意味着没有特别指定使用残差连接的层。
            use_rel_pos=True,       #启用相对位置编码，有助于模型更好地捕捉局部和全局位置关系。
            out_feature="last_feat",     #指定输出特征为最后一层的特征 last_feat，用于在下游任务中进一步处理。
            )
        
        self.vit_pretrain_path = vit_pretrain_path   #指定 ViT 模型的预训练权重路径 vit_pretrain_path，可以加载该路径下的预训练模型参数。  
        
        # simple feature pyramid network
        self.featurePyramid_net = SimpleFeaturePyramid(    #初始化一个简单的特征金字塔网络（Feature Pyramid Network, FPN），用于从不同尺度上提取特征。FPN 是一种多尺度特征提取方法，常用于检测和分割任务中。
            in_feature_shape= (1, embed_dim, 256, 256),   #指定输入特征的形状为 (1, embed_dim, 256, 256)。这表示输入特征有 1 个通道，特征维度为 embed_dim（之前定义的嵌入维度），空间尺寸为 256x256。
            out_channels= fpn_channels,          #指定 FPN 的输出通道数为 fpn_channels，这决定了 FPN 输出特征的深度。
            scale_factors=fpn_scale_factors,     #定义 FPN 中每个层次的缩放因子（scale factors）。这些因子决定了在特征金字塔中不同层之间如何进行上采样或下采样，以生成多尺度特征图。
            top_block=LastLevelMaxPool(),        #在 FPN 的顶部层添加一个最大池化操作（Max Pooling），用于进一步提取更高层次的特征。LastLevelMaxPool() 是一个模块，通常用于生成更粗糙但更具语义信息的特征图。
            norm="LN",                           #指定归一化方式为 LayerNorm ("LN")，用于规范化特征，以帮助模型更稳定地训练。
        )
    
        # MLP predict head
        self.predict_head = PredictHead(           #初始化一个多层感知机（MLP）预测头，用于最终的预测输出。预测头通常用于将特征映射到实际的任务输出，如分类、分割或检测结果。
            feature_channels=[fpn_channels for i in range(5)],   #定义输入给预测头的特征通道数。这里通过列表推导式创建了 5 层，每层的通道数都是 fpn_channels。这意味着预测头将接收来自 FPN 不同尺度的特征图
            embed_dim=mlp_embeding_dim,            #指定 MLP 预测头的嵌入维度为 mlp_embeding_dim，用于控制 MLP 中特征的表示空间。
            norm=predict_head_norm  # important! may influence the results  设置预测头中的归一化方式为 predict_head_norm
        )
    

        # Edge loss hyper-parameters    
        self.BCE_loss = nn.BCEWithLogitsLoss()  #定义二元交叉熵损失（BCE Loss），并与 Logits 一起使用。BCEWithLogitsLoss 是一种结合了 Sigmoid 激活和 BCE 损失的函数，通常用于二分类或像素级分割任务。
        self.edge_lambda = edge_lambda          #设置边缘损失的超参数 edge_lambda，这个参数控制边缘损失在整体损失函数中的权重。
        
        self.apply(self._init_weights)          #调用 apply 方法，遍历模型的所有子模块，并对每个子模块应用 _init_weights 函数，初始化模型的权重。
        self._mae_init_weights()                #调用 _mae_init_weights 方法，进一步初始化模型的权重。
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):             #对于 nn.Linear 层，使用 xavier_uniform_ 初始化权重，这是常用的用于保持前向传播方差稳定的初始化方法。
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):        #对于 nn.LayerNorm 层，将偏置设置为 0，权重设置为 1，以保证初始状态下不改变数据的分布。
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _mae_init_weights(self):
        # Load MAE pretrained weights for Window Attention ViT encoder
        if self.vit_pretrain_path != None:                             #如果 vit_pretrain_path 不为空，则从指定路径加载预训练的权重。
            self.encoder_net.load_state_dict(
                torch.load(self.vit_pretrain_path, map_location='cpu')['model'], # BEIT MAE  使用 torch.load 加载模型权重，这里指定 map_location='cpu' 以确保权重加载到 CPU 上，适用于没有 GPU 的环境。
                strict=False        #使用 strict=False 参数允许部分权重不匹配，这在模型结构稍有不同或部分层未包含在预训练权重中时很有用。
            )
            print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))


    
    def forward(self, x: torch.Tensor, masks, edge_masks, shape ,text_tensor):
        # print("一进来model",text_tensor.shape)     #一进来model torch.Size([1, 1, 1024])
        # print("一进来model",x.shape)
        # text_tensor.squeeze()
        # print("一进来model",text_tensor.shape)          #一进来model torch.Size([1, 1, 1024])
        # print("正确的维度是",x.shape, x.shape, text_tensor.shape)    #证确的维度是 torch.Size([1, 3, 1024, 1024]) torch.Size([1, 3, 1024, 1024]) torch.Size([1, 1, 1024])
        x = self.encoder_net(x,text_tensor)
        x = self.featurePyramid_net(x)  #将编码后的特征 x 传递给 self.featurePyramid_net（即特征金字塔网络），获取经过特征金字塔处理后的特征。
        feature_list = []   #将特征金字塔网络输出的字典 x 中的所有特征值（v）添加到 feature_list 中。这将特征字典转化为一个列表，以便于后续处理。
        for k, v in x.items():
            feature_list.append(v)
        x = self.predict_head(feature_list)  #将 feature_list 传递给 self.predict_head（即预测头），获取最终的预测结果。
        
        # up-sample to 1024x1024  使用 F.interpolate 将预测结果 x 上采样到 (self.input_size, self.input_size) 的尺寸（这里是 1024x1024）。mode='bilinear' 指定使用双线性插值方法进行上采样，align_corners=False 指定不对齐角点。
        mask_pred = F.interpolate(x, size = (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
        # compute the loss
        predict_loss = self.BCE_loss(mask_pred, masks)  #计算预测掩码 mask_pred 和目标掩码 masks 之间的二进制交叉熵损失（BCE Loss）
        edge_loss = F.binary_cross_entropy_with_logits(  #计算边缘损失。F.binary_cross_entropy_with_logits 用于计算带有权重的二进制交叉熵损失：
            input = mask_pred,   #预测的掩码。
            target= masks,       #目标掩码。
            weight = edge_masks  #边缘掩码的权重。
            ) * self.edge_lambda  #将计算得到的边缘损失乘以超参数 self.edge_lambda。
        predict_loss += edge_loss  #将边缘损失 edge_loss 加到预测损失 predict_loss 上，得到总损失。
        mask_pred = torch.sigmoid(mask_pred)  #对预测掩码 mask_pred 应用 sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内，作为最终的预测结果。
        
        return predict_loss, mask_pred, edge_loss  #返回三个值：predict_loss：计算的总损失。mask_pred：最终的预测掩码。edge_loss：计算的边缘损失。
        

#源代码：
# from modules.window_attention_ViT import ViT as window_attention_vit, SimpleFeaturePyramid, LastLevelMaxPool
# from modules.decoderhead import PredictHead

# import torch.nn as nn
# import torch
# import torch.nn.functional as F
# from functools import partial

# import sys
# sys.path.append('./modules')   #这两行代码将 './modules' 目录添加到 Python 模块搜索路径中。这允许 Python 脚本在 './modules' 目录中查找和导入模块。

# class iml_vit_model(nn.Module):
    
#     def __init__(
#         self, 
#         # ViT backbone:
#         input_size = 1024,  #输入图像的尺寸（宽度和高度），默认值为 1024。
#         patch_size = 16,    #将输入图像划分为小块的尺寸，默认值为 16。
#         embed_dim = 768,    #嵌入维度，即 ViT 中每个补丁的特征维度，默认值为 768。
#         vit_pretrain_path = None, # wether to load pretrained weights  预训练权重的路径，如果 None 则不加载预训练权重。
#         # Simple_feature_pyramid_network:
#         fpn_channels = 256,   #特征金字塔网络中通道的数量，默认值为 256。
#         fpn_scale_factors = (4.0, 2.0, 1.0, 0.5),   #特征金字塔网络中的尺度因子，用于调整不同层的特征图大小。

#         # MLP embedding:
#         mlp_embeding_dim = 256,   #MLP 嵌入的维度，默认值为 256
#         # Decoder head norm
#         predict_head_norm = "BN",  #预测头部使用的归一化方法，可以是 'BN'（批量归一化）、'LN'（层归一化）或 'IN'（实例归一化）
#         # Edge loss:
#         edge_lambda = 20,   #边缘损失的权重超参数，默认值为 20。
#     ):
#         """init iml_vit_model
#         # TODO : add more args
#         Args:
#             input_size (int): size of the input image, defalut to 1024
#             patch_size (int): patch size of Vision Transformer
#             embed_dim (int): embedding dim for the ViT
#             vit_pretrain_path (str): the path to initialize the model before start training
#             fpn_channels (int): the number of embedding channels for simple feature pyraimd
#             fpn_scale_factors(list(float, ...)) : the rescale factor for each SFPN layers.
#             mlp_embedding dim: dim of mlp, i.e. decoder head
#             predict_head_norm: the norm layer of predict head, need to select amoung 'BN', 'IN' and "LN"
#                                 We tested three different types of normalization in the decoder head, and they may yield different results due to dataset configurations and other factors.
#                             Some intuitive conclusions are as follows:
#                                 - "LN" -> Layer norm : The fastest convergence, but poor generalization performance.
#                                 - "BN" Batch norm : When include authentic images during training, set batchsize = 2 may have poor performance. But if you can train with larger batchsize (e.g. A40 with 48GB memory can train with batchsize = 4) It may performs better.
#                                 - "IN" Instance norm : A form that can definitely converge, equivalent to a batchnorm with batchsize=1. When abnormal behavior is observed with BatchNorm, one can consider trying Instance Normalization. It's important to note that in this case, the settings should include setting track_running_stats and affine to True, rather than the default settings in PyTorch.
#             edge_lambda(float) : the hyper-parameter for edge loss (lambda in our paper)
#         """
#         super(iml_vit_model, self).__init__()   #调用父类的初始化方法。这里 iml_vit_model 继承了某个父类，这行代码确保父类的属性和方法被正确初始化。
#         self.input_size = input_size    #将输入图像的尺寸 input_size 赋值给类的属性 self.input_size，用于后续模块中处理输入图像的大小。
#         self.patch_size = patch_size   #将图像分割的补丁（patch）的大小 patch_size 赋值给类的属性 self.patch_size。这个参数通常用于确定图像在 ViT 中如何被分割。
#         # window attention vit   初始化一个 Vision Transformer（ViT）模型，这个模型使用了基于窗口的注意力机制（Window Attention）。以下是 window_attention_vit 模型中的各个参数设置。
#         self.encoder_net = window_attention_vit(  
#             img_size = input_size,    #设置输入图像的大小为 input_size，与前面定义的 self.input_size 相同。
#             patch_size=16,     #指定图像补丁的大小为 16，表示输入图像会被分割成 16x16 的块。
#             embed_dim=embed_dim,     #设置嵌入维度为 embed_dim，即每个图像补丁在进入 Transformer 模型前会被转换成的特征维度。
#             depth=12,      #Transformer 的深度为 12，表示模型包含 12 层编码器。
#             num_heads=12,    #每层多头自注意力的头数为 12，表示每层有 12 个注意力机制用于捕捉不同的特征。
#             drop_path_rate=0.1,    #路径丢弃率为 0.1，用于在训练过程中引入随机性，防止过拟合。
#             window_size=14,      #窗口大小为 14，表示窗口注意力机制中使用 14x14 的窗口进行局部注意力计算。
#             mlp_ratio=4,       #多层感知机（MLP）部分的宽度比为 4，决定了在 Transformer 中 MLP 层的隐含层宽度。
#             qkv_bias=True,     #在查询、键、值（QKV）矩阵的计算中启用偏置项，有助于提高模型性能。
#             norm_layer=partial(nn.LayerNorm, eps=1e-6),    #指定归一化层为 LayerNorm，并设置其 epsilon 值为 1e-6，用于在模型中规范化特征。
#             window_block_indexes=[
#                 # 2, 5, 8 11 for global attention
#                 0,
#                 1,
#                 3,
#                 4,
#                 6,
#                 7,
#                 9,
#                 10,
#             ],    #定义哪些层使用窗口注意力。这里的索引对应模型的不同层，表示指定层将使用窗口注意力，而其他未列出的层可能使用全局注意力。
#             residual_block_indexes=[],    #定义残差块（Residual Block）的索引。这里为空列表，意味着没有特别指定使用残差连接的层。
#             use_rel_pos=True,       #启用相对位置编码，有助于模型更好地捕捉局部和全局位置关系。
#             out_feature="last_feat",     #指定输出特征为最后一层的特征 last_feat，用于在下游任务中进一步处理。
#             )
#         self.vit_pretrain_path = vit_pretrain_path   #指定 ViT 模型的预训练权重路径 vit_pretrain_path，可以加载该路径下的预训练模型参数。
        
#         # simple feature pyramid network
#         self.featurePyramid_net = SimpleFeaturePyramid(    #初始化一个简单的特征金字塔网络（Feature Pyramid Network, FPN），用于从不同尺度上提取特征。FPN 是一种多尺度特征提取方法，常用于检测和分割任务中。
#             in_feature_shape= (1, embed_dim, 256, 256),   #指定输入特征的形状为 (1, embed_dim, 256, 256)。这表示输入特征有 1 个通道，特征维度为 embed_dim（之前定义的嵌入维度），空间尺寸为 256x256。
#             out_channels= fpn_channels,          #指定 FPN 的输出通道数为 fpn_channels，这决定了 FPN 输出特征的深度。
#             scale_factors=fpn_scale_factors,     #定义 FPN 中每个层次的缩放因子（scale factors）。这些因子决定了在特征金字塔中不同层之间如何进行上采样或下采样，以生成多尺度特征图。
#             top_block=LastLevelMaxPool(),        #在 FPN 的顶部层添加一个最大池化操作（Max Pooling），用于进一步提取更高层次的特征。LastLevelMaxPool() 是一个模块，通常用于生成更粗糙但更具语义信息的特征图。
#             norm="LN",                           #指定归一化方式为 LayerNorm ("LN")，用于规范化特征，以帮助模型更稳定地训练。
#         )
#         # MLP predict head
#         self.predict_head = PredictHead(           #初始化一个多层感知机（MLP）预测头，用于最终的预测输出。预测头通常用于将特征映射到实际的任务输出，如分类、分割或检测结果。
#             feature_channels=[fpn_channels for i in range(5)],   #定义输入给预测头的特征通道数。这里通过列表推导式创建了 5 层，每层的通道数都是 fpn_channels。这意味着预测头将接收来自 FPN 不同尺度的特征图
#             embed_dim=mlp_embeding_dim,            #指定 MLP 预测头的嵌入维度为 mlp_embeding_dim，用于控制 MLP 中特征的表示空间。
#             norm=predict_head_norm  # important! may influence the results  设置预测头中的归一化方式为 predict_head_norm
#         )
#         # Edge loss hyper-parameters    
#         self.BCE_loss = nn.BCEWithLogitsLoss()  #定义二元交叉熵损失（BCE Loss），并与 Logits 一起使用。BCEWithLogitsLoss 是一种结合了 Sigmoid 激活和 BCE 损失的函数，通常用于二分类或像素级分割任务。
#         self.edge_lambda = edge_lambda          #设置边缘损失的超参数 edge_lambda，这个参数控制边缘损失在整体损失函数中的权重。
        
#         self.apply(self._init_weights)          #调用 apply 方法，遍历模型的所有子模块，并对每个子模块应用 _init_weights 函数，初始化模型的权重。
#         self._mae_init_weights()                #调用 _mae_init_weights 方法，进一步初始化模型的权重。
        
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):             #对于 nn.Linear 层，使用 xavier_uniform_ 初始化权重，这是常用的用于保持前向传播方差稳定的初始化方法。
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):        #对于 nn.LayerNorm 层，将偏置设置为 0，权重设置为 1，以保证初始状态下不改变数据的分布。
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
            
#     def _mae_init_weights(self):
#         # Load MAE pretrained weights for Window Attention ViT encoder
#         if self.vit_pretrain_path != None:                             #如果 vit_pretrain_path 不为空，则从指定路径加载预训练的权重。
#             self.encoder_net.load_state_dict(
#                 torch.load(self.vit_pretrain_path, map_location='cpu')['model'], # BEIT MAE  使用 torch.load 加载模型权重，这里指定 map_location='cpu' 以确保权重加载到 CPU 上，适用于没有 GPU 的环境。
#                 strict=False        #使用 strict=False 参数允许部分权重不匹配，这在模型结构稍有不同或部分层未包含在预训练权重中时很有用。
#             )
#             print('load pretrained weights from \'{}\'.'.format(self.vit_pretrain_path))
    
#     def forward(self, x:torch.Tensor, masks, edge_masks,text_tensor,shape= None):  #x：输入张量，通常是模型的输入图像。masks：目标掩码，用于计算损失。edge_masks：边缘掩码，用于边缘损失的加权。shape：可选参数，通常是目标图像的尺寸（默认值为 None）。
        
#         # print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%',text_tensor.shape)          #   torch.Size([2, 5123, 1, 512])     
#         x = self.encoder_net(x)  #将输入 x 传递给 self.encoder_net（即 Vision Transformer 编码器），获取编码后的特征。
#         x = self.featurePyramid_net(x)  #将编码后的特征 x 传递给 self.featurePyramid_net（即特征金字塔网络），获取经过特征金字塔处理后的特征。
#         feature_list = []   #将特征金字塔网络输出的字典 x 中的所有特征值（v）添加到 feature_list 中。这将特征字典转化为一个列表，以便于后续处理。
#         for k, v in x.items():
#             feature_list.append(v)
#         x = self.predict_head(feature_list)  #将 feature_list 传递给 self.predict_head（即预测头），获取最终的预测结果。
        
#         # up-sample to 1024x1024  使用 F.interpolate 将预测结果 x 上采样到 (self.input_size, self.input_size) 的尺寸（这里是 1024x1024）。mode='bilinear' 指定使用双线性插值方法进行上采样，align_corners=False 指定不对齐角点。
#         mask_pred = F.interpolate(x, size = (self.input_size, self.input_size), mode='bilinear', align_corners=False)
        
#         # compute the loss
#         predict_loss = self.BCE_loss(mask_pred, masks)  #计算预测掩码 mask_pred 和目标掩码 masks 之间的二进制交叉熵损失（BCE Loss）
#         edge_loss = F.binary_cross_entropy_with_logits(  #计算边缘损失。F.binary_cross_entropy_with_logits 用于计算带有权重的二进制交叉熵损失：
#             input = mask_pred,   #预测的掩码。
#             target= masks,       #目标掩码。
#             weight = edge_masks  #边缘掩码的权重。
#             ) * self.edge_lambda  #将计算得到的边缘损失乘以超参数 self.edge_lambda。
#         predict_loss += edge_loss  #将边缘损失 edge_loss 加到预测损失 predict_loss 上，得到总损失。
#         mask_pred = torch.sigmoid(mask_pred)  #对预测掩码 mask_pred 应用 sigmoid 激活函数，将输出值压缩到 [0, 1] 范围内，作为最终的预测结果。
        
#         return predict_loss, mask_pred, edge_loss  #返回三个值：predict_loss：计算的总损失。mask_pred：最终的预测掩码。edge_loss：计算的边缘损失。
    






   

        
    
        
    