"""
U-NET code based from git hub repository:
https://github.com/nyoki-mtl/pytorch-discriminative-loss/tree/master
Thanks for the author nyoki-mtl for sharing this code.

This implementation is based on following code:
https://github.com/milesial/Pytorch-UNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels = 4, out_channels_instance = 12, init_features = 16,
                 use_cuda = False, use_resnet = False, semantic_path = False):

        super(UNet, self).__init__()
        self.use_cuda = use_cuda
        self.use_resnet = use_resnet
        self.semantic_path = semantic_path

        features = init_features
        self.inc = InConv(in_channels, features, use_resnet)
        self.down1 = Down(features, features * 2, use_resnet)
        self.down2 = Down(features * 2, features * 4, use_resnet)
        self.down3 = Down(features * 4, features * 8, use_resnet)
        self.down4 = Down(features*8, features*16, use_resnet)
        # for instance segmentation
        self.up1 = Up(features*16, features*8, use_resnet)
        self.up2 = Up(features * 8, features * 4, use_resnet)
        self.up3 = Up(features * 4, features * 2, use_resnet)
        self.up4 = Up(features * 2, features, use_resnet)
        self.ins_out = OutConv(features, out_channels_instance)
        # for semantic segmentation
        if semantic_path:
            self.up21 = Up(features*16, features*8, use_resnet)
            self.up22 = Up(features*8, features*4, use_resnet)
            self.up23 = Up(features*4, features*2, use_resnet)
            self.up24 = Up(features*2, features, use_resnet)
            self.sem_out = OutConv(features, 2)
            self.soft = nn.Softmax(dim=1)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_instance = self.up1(x5, x3)
        x_instance = self.up2(x_instance, x3)
        x_instance = self.up3(x_instance, x2)
        x_instance = self.up4(x_instance, x1)
        ins = self.ins_out(x_instance)

        if not self.semantic_path:
            return ins

        x_semantic = self.up21(x5, x4)
        x_semantic = self.up22(x_semantic, x3)
        x_semantic = self.up23(x_semantic, x2)
        x_semantic = self.up24(x_semantic, x1)

        sem = self.sem_out(x_semantic)
        sem = self.soft(sem)
        return sem, ins

class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, use_resnet = False):
        super(DoubleConv, self).__init__()
        self.use_resnet = use_resnet
        if self.use_resnet:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.use_resnet:
            x = self.conv1(x)
            residual = x # residual block
            x = self.conv2(x)
            x = self.relu(x + residual)  # residual block
        else:
            x = self.conv(x)  # regular case
            x = self.relu(x) # regular case

        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_resnet):
        super(InConv, self).__init__()
        print("InConv: ", in_ch, out_ch, use_resnet)
        self.conv = DoubleConv(in_ch, out_ch, use_resnet)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, use_resnet):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_resnet)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, use_resnet):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_resnet)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
