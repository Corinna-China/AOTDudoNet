import odl.tomo
import torch
import torch.nn as nn
import torch.nn.functional as  F
from odl.contrib import torch as odl_torch
from .imgnet import UNet
from .build_gemotry import initialization, build_gemotry
import matplotlib.pyplot as pyplot
para_ini = initialization()
fp = build_gemotry(para_ini)
op_modfp = odl_torch.OperatorModule(fp)
op_modfbp = odl_torch.OperatorModule(odl.tomo.fbp_op(fp))
op_modpT = odl_torch.OperatorModule(fp.adjoint)

class AOTdudo(nn.Module):
    def __init__(self, args):
        super(AOTdudo, self).__init__(   )
        self.IENet = UNet()

    def forward(self,Xprior,XBHC):

        Xout = self.IENet(XBHC,Xprior)
        return Xout
