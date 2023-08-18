import torch
from torch import nn
# from torchinfo import summary
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm



class SE_net(nn.Module):
    def __init__(self):  # 1046
        super(SE_net, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(2, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True)
        )
        rates = list(map(int, list('1+2+4+8'.split('+'))))
        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(8)])

        self.decoder = nn.Sequential(
            UpConv(256, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(64, 1, 1)
        )

        # self.init_weights()

    def forward(self, x, mask):
        a = x
        import matplotlib.pyplot as pyplot
        # pyplot.imshow(x.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x  = F.pad(x, (0, 1,0, 0))
        x = self.outconv(x)
        x = torch.tanh(x)
        # pyplot.imshow(x.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        return x


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))


class AOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate),
                    nn.Conv2d(dim, dim//4, 3, padding=0, dilation=rate),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat
