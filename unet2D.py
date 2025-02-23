import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()     # 调用父类的构造方法
        # 进行下卷积
        self.down_conv = nn.Sequential(
            # 第一个卷积层。卷积核3*3
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 归一化，加快模型训练
            nn.BatchNorm2d(out_channels),
            # 激活函数。inplace=true是直接修改输入，减少内存使用
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 定义最大池化层，当图像不够一个2*2的时候向上取整，直接进行池化
        # ceil_mode参数取整的时候向上取整，该参数默认为False表示取整的时候向下取整
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    # 输出下卷积之后的特征图
    def forward(self, x):
        # 调用上面的函数，进行下卷积
        out = self.down_conv(x)
        out_pool = self.pool(out)
        return out, out_pool

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        # 定义反卷积操作，用于扩大特征图尺寸
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 定义向上卷积
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    # x_copy是下采样过程中的特征图。x是上采样传上来的特征图
    def forward(self, x_copy, x, interpolate=True):
        # 先调用反卷积，特征图长宽均*2
        out = self.up(x)
        # 进行上卷积的时候要让x（上采样传的特征图）和x_copy(下采样特征图)的大小一样，下面才能进行特征图的合并
        # interpolate表示是否使用双线性插值填充特征图
        if interpolate:
            # 使用双线性插值对特征图进行填充
            out = F.interpolate(out, size=(x_copy.size(2), x_copy.size(3)),
                                mode="bilinear", align_corners=True
                                )
        else:
        # 直接用0进行填充
            # 如果填充物体积大小不同
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            # 这句倒数第二个参数有点问题，应该是diffY//2，否则填充后的大小有误。
            out = F.pad(out, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # 连接，两个通道特征图进行合并。通道数变成两个的加和
        out = torch.cat([x_copy, out], dim=1)
        out_conv = self.up_conv(out)
        return out_conv

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # 创建日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)

    # 设置继承时必须重写forward方法，否则报错
    def forward(self):
        raise NotImplementedError

    # 这边向logger记录模型参数
    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parametersL {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f"\nNbr of trainable parameters: {nbr_params}"


class UNet2D(BaseModel):
    # num_classes是分割类别的数量，数据为黑白图inchannels=1。 freeze_bn禁用Batch Normalization，训练过程不会更新归一化的方差均值，防止过拟合。 **_表示可以传入其他的参数
    def __init__(self, num_classes, in_channels=1, freeze_bn=False, **_):
        super(UNet2D, self).__init__()
        # 每一个channel对应一张特征图
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initalize_weights()
        if freeze_bn:
            self.freeze_bn()

    def _initalize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()