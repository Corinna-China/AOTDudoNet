import numpy as np
import PIL.Image as Image

import BHC
import LI
import util
import getMetal
import torch
import os
import os.path
from AOTdudonet.network.AOTdudo import AOTdudo
from AOTdudonet.prior_net.priornet import PriorNet
import argparse

parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="AOTdudonet/model", help='path to model and log files')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
opt = parser.parse_args()


def print_network(name, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('name={:s}, Total number={:d}'.format(name, num_params))

def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data


def AOTdudoMAR(image,metal):
    #init
    Xmahu = np.array(Image.fromarray(image).resize((416, 416)))
    Xma = util.hu_to_coefficient(Xmahu)
    XLI,SLI = LI.LIMARwithoutmetal(Xma,metal)


    XBHC = BHC.fastBHC(Xma,XLI,SLI)
    # b = util.coefficient_to_hu(XBHC)
    # util.numpyToDicom("output1/BHC.dcm", b)
    Xma = normalize(Xma, image_get_minmax())
    SLI = normalize(SLI, proj_get_minmax())
    XBHC = normalize(XBHC, image_get_minmax())

    M = metal.astype(np.float32)
    # util.numpyToDicom("output1/M.dcm", M)
    Tr = util.fanBeam(M)
    Tr = np.array(Tr)
    Tr[Tr != 0] = 1

    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)),0)
    # util.showNumpyPlot(Tr.squeeze())
    # util.showNumpyPlot(SLI.squeeze())
    priornet = PriorNet(opt).cuda()
    pt =  os.path.join('AOTdudonet/prior_model', 'priornet.pt')
    checkpoint = torch.load(pt, map_location='cuda')
    priornet.load_state_dict(checkpoint)
    priornet.eval()
    print_network("priornet", priornet)
    ssLI = SLI.squeeze()
    ssTr = Tr.squeeze()
    ssLI = np.expand_dims(np.transpose(np.expand_dims(ssLI, 2), (2, 0, 1)), 0)
    ssTr = np.expand_dims(np.transpose(np.expand_dims(ssTr, 2), (2, 0, 1)), 0)
    import time
    t1 = time.time()
    with torch.no_grad():
        # import torchsummary
        # torchsummary.summary(priornet.cuda(), input_size=[(1, 640, 641), (1, 640, 641)],
        #                      batch_size=1)
        Sprior, Xprior = priornet(torch.Tensor(ssLI).cuda(), torch.Tensor(ssTr).cuda())
        # Xprior = Xprior.cpu().detach().numpy().squeeze()
        # util.numpyToDicom("output/ssMa.dcm", Sma.squeeze())
        # util.numpyToDicom("output/ssTr.dcm", ssTr.squeeze())
        # util.numpyToDicom("output/ssLI.dcm", ssLI.squeeze())
        # util.numpyToDicom("output/Saot.dcm", Sprior.cpu().detach().numpy().squeeze())
        xp =Xprior/ 255.0
        xp = util.coefficient_to_hu(xp).squeeze()
        # util.numpyToDicom("output1/Xpr.dcm", xp.cpu().detach().numpy().squeeze())

    print('Loading model ...\n')
    net = AOTdudo(opt).cuda()
    # print_network("AOTdudoMAR", net)
    pt = os.path.join(opt.model_dir, 'net_239.pt')
    checkpoint = torch.load(pt, map_location='cuda')
    net.load_state_dict(checkpoint)
    net.eval()
    # import torchsummary
    # torchsummary.summary(net.cuda(), input_size=[(1, 416, 416), (1, 416, 416)],
    #                      batch_size=1)
    with torch.no_grad():
        if opt.use_GPU:
            torch.cuda.synchronize()
        Xout = net(Xprior, torch.Tensor(XBHC).cuda())
    # util.showNumpyPlot(Xprior.cpu().detach().numpy().squeeze())
    # util.showNumpyPlot(Xcorr.cpu().detach().numpy().squeeze())
    # util.showNumpyPlot(Xout.cpu().detach().numpy().squeeze())
    # print(time.time()-t1)
    Xma = Xma / 255.0
    Xmahu = util.coefficient_to_hu(Xma).squeeze()

    Xprior =  Xprior / 255.0
    Xpriorhu = util.coefficient_to_hu(Xprior).cpu().detach().numpy().squeeze()
    Xpriorhu = util.replaceImage(Xpriorhu, Xmahu, metal)
    # util.numpyToDicom("output1/Xpriorhu.dcm", Xpriorhu)
    Xout = Xout / 255.0
    XoutHU = util.coefficient_to_hu(Xout).cpu().detach().numpy().squeeze()
    XoutHU = util.replaceImage(XoutHU, Xmahu, metal)
    return XoutHU



if __name__ == "__main__":
    test_img = util.dicomToNumpy("./test_img/ma-1.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    res = AOTdudoMAR(test_img,metal)
    util.showNumpyPlot(res)
    util.numpyToDicom("./results/aotdodu-1.dcm",res)

    test_img = util.dicomToNumpy("./test_img/ma-2.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    res = AOTdudoMAR(test_img,metal)
    util.showNumpyPlot(res)
    util.numpyToDicom("./results/aotdodu-2.dcm",res)

    test_img = util.dicomToNumpy("./test_img/ma-3.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    res = AOTdudoMAR(test_img,metal)
    util.showNumpyPlot(res)
    util.numpyToDicom("./results/aotdodu-3.dcm",res)

    test_img = util.dicomToNumpy("./test_img/ma-4.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    res = AOTdudoMAR(test_img,metal)
    util.showNumpyPlot(res)
    util.numpyToDicom("./results/aotdodu-4.dcm",res)