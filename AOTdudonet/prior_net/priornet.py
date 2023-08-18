import odl.tomo
import torch.nn as nn
from odl.contrib import torch as odl_torch
from .imgnet import UNet
from .senet import SE_net
from .build_gemotry import initialization, build_gemotry

para_ini = initialization()
fp = build_gemotry(para_ini)
op_modfp = odl_torch.OperatorModule(fp)
op_modfbp = odl_torch.OperatorModule(odl.tomo.fbp_op(fp))
op_modpT = odl_torch.OperatorModule(fp.adjoint)

class PriorNet (nn.Module):
    def __init__(self, args):
        super(PriorNet, self).__init__(   )
        self.IENet = UNet()
        self.SENet = SE_net()

    def forward(self,SLI, Tr):
        # SE-Net
        # pyplot.imshow(XBHC.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        SLI = SLI/255.0
        # pyplot.imshow(SBHC.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        # pyplot.imshow(SLI.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        # pyplot.imshow(Tr.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        Sout = SLI + self.SENet(SLI,Tr)*Tr
        Sout = Sout * 255.0
        Xcorr = op_modfbp((Sout/255)*4.0)
        Xcorr = Xcorr * 255.0

        # pyplot.imshow(Sout.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        # pyplot.imshow(Xcorr.cpu().detach().numpy().squeeze(), 'Greys_r')
        # pyplot.axis('off')
        # pyplot.show()
        # # IE-Net
        # Xi = torch.cat((Xcorr,XLI), dim =1)
        # Xout = XLI + self.IENet(Xi)
        return Sout,Xcorr
