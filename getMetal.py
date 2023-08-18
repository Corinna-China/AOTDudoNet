import os
import PIL.Image as Image
import numpy as np
import util
import SimpleITK as sitk

def getMetal(img,setting="threshold"):
    if (setting == "threshold"):
        return getMetalByThreshold(img,util.config.get("metal","threshold"))
    elif (setting == "tt"):
        return getMetalByThreshold(img,1000)
    elif (setting == "entropy"):
        return getMetalByEntropy(img)
    return getMetalByThreshold(img,util.config.get("metal","threshold"))

def getMetalByThreshold(img,threshold):
    return segmentByThreshold(img,threshold)

def getMetalByEntropy(img):
    simg = sitk.GetImageFromArray(img.astype('int16'))
    thresh_filter = sitk.MaximumEntropyThresholdImageFilter()
    thresh_img = thresh_filter.Execute(simg)
    #thresh_value = 1.4 * thresh_filter.GetThreshold()
    #thresh_value = 4.7*thresh_filter.GetThreshold()
    thresh_value = thresh_filter.GetThreshold()
    return segmentByThreshold(img, thresh_value)
    # print(thresh_value)
    # thresh_img = sitk.GetArrayFromImage(thresh_img)
    # thresh_img = np.where((thresh_img == 0) | (thresh_img == 1), thresh_img ^ 1, thresh_img)
    # return thresh_img

def getMetalByAuto(img):
    pass

def segmentByThreshold(img,threshold):
    metal = np.zeros([img.shape[0],img.shape[1]])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] > int(threshold)):
                metal[i][j] = 1
            else:
                metal[i][j] = 0
    return metal
#
# def cal(img,metal_img):
#     img = util.rawToNumpy(img)
#     img = util.coefficient_to_hu(img)
#     truemetal = util.rawToNumpy(metal_img)
#     for i  in range(truemetal.shape[0]):
#         for j in range(truemetal.shape[1]):
#             if (truemetal[i][j] > 0):
#                 truemetal[i][j] = 1
#             else:
#                 truemetal[i][j] = 0
#     metalEntropy = getMetal(img,setting= "entropy")
#     metalThreshold = getMetal(img,setting= "threshold")
#     util.showNumpyPlot(truemetal)
#     util.showNumpyPlot(metalEntropy)
#     util.showNumpyPlot(metalThreshold)
#     print(util.call_rmse(truemetal,metalEntropy))
#     print(util.call_rmse(truemetal, metalThreshold))
#     util.numpytopng("ma.png", img)
#     util.numpytopng("1.png",truemetal)
#     util.numpytopng("2.png", metalEntropy)
#     util.numpytopng("3.png", metalThreshold)


def showmetal(img):
    img = util.rawToNumpy(img)
    img = util.coefficient_to_hu(img)

    metalEntropy = getMetal(img,setting= "entropy")
    metalThreshold = getMetal(img,setting= "threshold")
    util.showNumpyPlot(metalEntropy)
    util.showNumpyPlot(metalThreshold)

if __name__ == "__main__":
    maimg = "CT_metal_DICOM/original/03.dcm"
    img = util.dicomToNumpy(maimg)
    img = np.array(Image.fromarray(img).resize((416, 416)))
    #img = np.array(util.fanBeam(img))
    util.showNumpyPlot(img)
    metalEntropy = getMetal(img,setting= "entropy")
    metalThreshold = getMetal(img,setting= "threshold")
    util.showNumpyPlot(metalEntropy)
    #util.showNumpyPlot(metalThreshold)
    # img = "C:\\Users\\kyon\\Documents\\mar\\SpineWeb\\test\\ma_img\\patient0164_4645998_042.raw"
    # showmetal(img)
