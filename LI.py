import numpy as np
import PIL.Image as Image
import util
import getMetal
import  torch
import time

def LIMAR(image,metal):
    # init
    img = util.hu_to_coefficient(image)
    res,_ = LIMARwithoutmetal(img,metal)
    res = util.coefficient_to_hu(res)
    res = util.replaceImage(res,image,metal)
    return res

# def LIMARwithoutmetal(image):
#     # init
#     image = np.array(Image.fromarray(image).resize((416, 416)))
#     sin = util.fanBeam(image)
#     # get metal trace
#     metal = getMetal.getMetal(image)
#     metal_trace = util.fanBeam(metal)
#     MPR = getMetal.segmentByThreshold(metal_trace, 0)
#     # interpolation
#     li_sin = inpaintByLinearInterpolation(np.array(sin).T,np.array(MPR).T)
#     res = util.ifanBeam(li_sin)
#     return res

def LIMARwithoutmetal(image,metal):
    # init
    image = np.array(Image.fromarray(image).resize((416, 416)))
    image = np.expand_dims(np.transpose(np.expand_dims(image, 2), (2, 0, 1)), 0)
    image = torch.tensor(image).cuda()
    sin = util.fanBeamcuda(image)
    # get metal trace
    metal = np.expand_dims(np.transpose(np.expand_dims(metal, 2), (2, 0, 1)), 0)
    metal = torch.tensor(metal).cuda()
    metal_trace = util.fanBeamcuda(metal)
    MPR = getMetal.segmentByThreshold(metal_trace.cpu().detach().numpy().squeeze(), 0)
    # interpolation
    sint = sin.cpu().detach().numpy().squeeze().T
    mprt = np.array(MPR).T
    li_sin = inpaintByLinearInterpolation(sint,mprt)
    li_sin = np.expand_dims(np.transpose(np.expand_dims(li_sin, 2), (2, 0, 1)), 0)
    li_sin = torch.tensor(li_sin).cuda()
    res = util.ifanBeamcuda(li_sin)
    return res.cpu().detach().numpy().squeeze(),li_sin.cpu().detach().numpy().squeeze()


def inpaintByLinearInterpolation(sin,MPR):
    img_res = np.array(sin)
    for j in range(sin.shape[1]):
        for i in range(1,sin.shape[0]-1):
            if (MPR[i][j] == 1):
                L = i
                R = i
                while (L > 0 and MPR[L][j] == 1):
                    L = L-1
                while (R < sin.shape[0] - 1 and MPR[R][j] == 1):
                    R = R+1
                w = R - L
                if (w > 0):
                    img_res[i][j] = (R-i)/w*sin[L][j]+(i-L)/w*sin[R][j]
    img_res = img_res.T
    return img_res


def inpaintByLinearInterpolationTensor(sin,MPR):
    img_res = sin.clone()
    for j in range(sin.shape[1]):
        for i in range(1,sin.shape[0]-1):
            if (MPR[i][j] == 1):
                L = i
                R = i
                while (L > 0 and MPR[L][j] == 1):
                    L = L-1
                while (R < sin.shape[0] - 1 and MPR[R][j] == 1):
                    R = R+1
                w = R - L
                if (w > 0):
                    img_res[i][j] = (R-i)/w*sin[L][j]+(i-L)/w*sin[R][j]
    img_res = torch.t(img_res)
    return img_res


if __name__ == "__main__":
    test_img = util.dicomToNumpy("./test_img/ma-1.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    li_img = LIMAR(test_img,metal)
    util.showNumpyPlot(li_img)
    util.numpyToDicom("./results/LI-1.dcm",li_img)

    test_img = util.dicomToNumpy("./test_img/ma-2.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    li_img = LIMAR(test_img,metal)
    util.showNumpyPlot(li_img)
    util.numpyToDicom("./results/LI-2.dcm",li_img)

    test_img = util.dicomToNumpy("./test_img/ma-3.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    li_img = LIMAR(test_img,metal)
    util.showNumpyPlot(li_img)
    util.numpyToDicom("./results/LI-3.dcm",li_img)    

    test_img = util.dicomToNumpy("./test_img/ma-4.dcm")
    test_img = np.array(Image.fromarray(test_img).resize((416, 416)))
    metal = getMetal.getMetal(test_img)
    li_img = LIMAR(test_img,metal)
    util.showNumpyPlot(li_img)
    util.numpyToDicom("./results/LI-4.dcm",li_img)    