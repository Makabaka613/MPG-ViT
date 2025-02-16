import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F

class EdgeGenerator(torch.nn.Module):
    """generate the 'edge bar' for a 0-1 mask Groundtruth of a image
    Algorithm is based on 'Morphological Dilation and Difference Reduction'
    
    Which implemented with fixed-weight Convolution layer with weight matrix looks like a cross,
    for example, if kernel size is 3, the weight matrix is:
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]]

    """
    def __init__(self, kernel_size = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
    
    def _dilate(self, image, kernel_size=3):    #使用十字形卷积核对图像进行膨胀。膨胀操作扩展了图像的前景区域。
        """Doings dilation on the image  对图像进行膨胀操作

        Args:
            image (_type_): 0-1 tensor in shape (B, C, H, W)     image (Tensor): 形状为 (B, C, H, W) 的 0-1 张量
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd"  #卷积核大小必须是奇数"
        assert image.shape[2] > kernel_size and image.shape[3] > kernel_size, "Image must be larger than kernel size"   #"图像尺寸必须大于卷积核大小"
        
        kernel = torch.zeros((1, 1, kernel_size, kernel_size))
        kernel[0, 0, kernel_size // 2: kernel_size//2+1, :] = 1
        kernel[0, 0, :,  kernel_size // 2: kernel_size//2+1] = 1
        kernel = kernel.float()
        # print(kernel)
        res = F.conv2d(image, kernel.view([1,1,kernel_size, kernel_size]),stride=1, padding = kernel_size // 2)
        return (res > 0) * 1.0


    def _find_edge(self, image, kernel_size=3, return_all=False):  #生成图像的边缘条。首先进行膨胀处理，然后计算前景和背景的差异。
        """Find 0-1 edges of the image

        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        image = torch.tensor(image).float()   #图像的膨胀结果。
        shape = image.shape 
        
        if len(shape) == 2:
            image = image.reshape([1, 1, shape[0], shape[1]])
        if len(shape) == 3:
            image = image.reshape([1, shape[0], shape[1], shape[2]])   
        assert image.shape[1] == 1, "Image must be single channel"
        
        img = self._dilate(image, kernel_size=kernel_size)
        
        erosion = self._dilate(1-image, kernel_size=kernel_size)   #胶卷的背景（1-图像）的膨胀结果。

        diff = -torch.abs(erosion - img) + 1
        diff = (diff > 0) * 1.0
        # res = dilate(diff)
        diff = diff.numpy()   #通过差异计算边缘区域，最后将其转换为 NumPy 数组。
        if return_all :
            return diff, img, erosion
        else:
            return diff   #如果设置为 True，返回边缘条、膨胀后的前景图像和背景图像；否则，仅返回边缘条。
     
    def forward(self, x, return_all=False):   #在前向传播中调用 _find_edge 方法，以获取图像的边缘条。
        """
        Args:
            image (_type_): 0-1 ndarray in shape (B, C, H, W)
        """
        return self._find_edge(x, self.kernel_size, return_all=return_all)
    
    
"""Codes below are for testing"""
if __name__ == '__main__':
    lists = ['NC2016_1504.jpg', '519_mask.jpg', '526_mask.jpg', '528_mask.jpg', '534_mask.jpg']   #从指定路径加载图像，并将其转换为 PyTorch 张量。
 
    for i in lists:                  #将图像数据二值化（大于 127 的值设置为 1，其余设置为 0），以适配边缘检测算法。
        img = plt.imread(f'./components/Edge_generator/{i}')
        img = torch.tensor(img)
        print(img)
        img = (img > 127).float()
        plt.subplot(1, 4, 1)             #显示处理后的二值图像。
        plt.imshow(img, cmap='gray')
        print(img)
        Edge = EdgeGenerator(kernel_size=11)    #初始化 EdgeGenerator 类，设置卷积核大小为 11。
        
        raw_img = img.view(1, 1, img.shape[0], img.shape[1])   #将图像维度调整为 (1, 1, H, W) 以适应 EdgeGenerator 的输入要求。


        # print(img)
        # plt.subplot(1,4, 2)
        # plt.imshow(diff.detach().numpy()[0, 0, :, :], cmap='gray')

        diff, img,erosion, = Edge(raw_img, return_all=True)   #调用 EdgeGenerator 的 forward 方法，获取边缘图像、膨胀后的前景图像和背景图像。
        
        plt.subplot(1,4, 2)                         #显示生成的边缘图像。
        plt.imshow(diff[0, 0, :, :], cmap='gray')
        
        # plt.subplot(1,4, 3)
        # plt.imshow(img.detach().numpy()[0, 0, :, :], cmap='gray')
        
        # plt.subplot(1, 4, 4)
        # plt.imshow(erosion.detach().numpy()[0, 0, :, :], cmap='gray')
        
        
        print(diff.shape)     #打印处理后的图像尺寸。
        print(img.shape)
        print(erosion.shape)
        plt.show()       #显示图像。 