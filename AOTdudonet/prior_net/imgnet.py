import torch
from torch import nn
# from torchinfo import summary

import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),#BatchNorm2d
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),#BatchNorm2d
            nn.LeakyReLU(0.1))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.InstanceNorm2d(out_ch),#BatchNorm2d
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super(UNet, self).__init__()

        channels = 64
        filters = [channels, channels * 2, channels * 4, channels * 8, channels * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.Softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        e0 = torch.cat([x, y], dim=1)
        e1 = self.Conv1(e0)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        # out = self.Softmax(out)
        out = out + x
        return out


# if __name__ == '__main__':
#     model = UNet()
#     summary(model, input_size=(4, 1, 256, 256))
