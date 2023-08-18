import os
import matplotlib.pyplot as pyplot
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from build_gemotry import initialization, build_gemotry
import odl.tomo
import configparser
from odl.contrib import torch as odl_torch
from sklearn.metrics import mean_squared_error
import SimpleITK as sitk
# initialize
para_ini = initialization()
fp = build_gemotry(para_ini)
op_fp = odl_torch.OperatorModule(fp)
op_modfbp = odl_torch.OperatorModule(odl.tomo.fbp_op(fp))

config = configparser.ConfigParser()
path = os.path.split(os.path.realpath(__file__))[0] + '\config.ini'
config.read(path, encoding="utf-8")

def spinewebToNumpy(filename):
    file_path = os.path.join(filename)
    data = np.fromfile(file_path, dtype='float32').reshape(512, 512)
    data = coefficient_to_hu(data)
    data -= 2024
    return data

def dicomToNumpy(filename):
    file_path = os.path.join(filename)
    dicom = pydicom.dcmread(file_path,force=True)
    data = np.array(dicom.pixel_array)
    return data

def numpyToDicom(filename,data):
    img = sitk.GetImageFromArray(data.astype('int16'))
    sitk.WriteImage(img,os.path.join(filename))
    numpytopng(filename[:-4]+'.jpg',data)

def numpytopng(filename,data):
    data = toWinodow(data,60,500)
    pyplot.imsave(filename, data,cmap="gray")

# 512*512 raw
def rawToNumpy(filename):
    file_path = os.path.join(filename)
    data = np.fromfile(file_path, dtype='float32').reshape(512, 512)
    return data

# 512*512 raw
def rawToNumpy640(filename):
    file_path = os.path.join(filename)
    data = np.fromfile(file_path, dtype='float32').reshape(640,640)
    return data

# 416*416 raw
def rawToNumpy416(filename):
    file_path = os.path.join(filename)
    data = np.fromfile(file_path, dtype='float32').reshape(416, 416)
    return data

def numpyToRaw(filename,data):
    # data = np.array(data,dtype='float32')
    data.tofile(filename)

from PIL import Image
def deeplesionpicToNumpy(filename):
    file_path = os.path.join(filename)
    img = Image.open(file_path, mode="r")
    img = np.array(img)
    img -= 32768
    return img

def showNumpyPlot(array):
    pyplot.imshow(array, 'Greys_r')
    pyplot.show()

# 416*416->640*641
def fanBeam(array):
    return fp(array)

# 640*641->416*416
def ifanBeam(array):
    return odl.tomo.fbp_op(fp)(array)

# 416*416->640*641
def fanBeamcuda(array):
    return op_fp(array)

# 640*641->416*416
def ifanBeamcuda(array):
    return op_modfbp(array)

def replaceImage(img_dst,img_src,mask):
    img_res = img_dst
    for i in range(img_res.shape[0]):
        for j in range(img_res.shape[1]):
            if (mask[i][j] == 1):
                img_res[i][j] = img_src[i][j]
    return img_res

def call_rmse(data1,data2):
    return np.sqrt(mean_squared_error(data1, data2))


# def call_rmse(data1, data2):
#     rmse = 0
#     data1 = torch.Tensor(data1)
#     data2 = torch.Tensor(data2)
#     for num in range(data1.shape[0]):
#         temp = torch.sqrt(torch.mean(torch.square(data1[num] - data2[num]))).item()
#         rmse += temp
#     return rmse / data1.shape[0]

def psnr(data1,data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    return 10 * np.log(255 * 255.0 / (((data1.astype(np.float) - data2) ** 2).mean())) / np.log(10)

def coefficient_to_hu(img):
    img = (img - 0.192) / 0.192 * 1000
    return img

def hu_to_coefficient(img):
    img = img * 0.192 / 1000 + 0.192
    return img


def normalize(data, data_min, data_max):
    data = (data - data_min) / (data_max - data_min)
    return data


def denormalize(data, data_min, data_max):
    data = data * (data_max - data_min) + data_min
    return data

# data2:gt
def calc_KL(data1,data2):
    # data1 = data1.flatten()
    # data2 = data2.flatten()
    # data1 = np.histogram(data1,bins=25,normed=0)[0]/float(len(data1))
    # data2 = np.histogram(data2, bins=25, normed=0)[0] / float(len(data2))
    data1 = F.log_softmax(torch.Tensor(data1),dim=-1)
    data2 = F.softmax(torch.Tensor(data2),dim=-1)
    kl_mean = F.kl_div(torch.Tensor(data1),torch.Tensor(data2),reduction='mean')
    return kl_mean

    # pyplot.figure(1)
    # pyplot.hist(data1,bins=25,normed=0)
    # pyplot.xlabel("x")
    # pyplot.show()

def toWinodow(hu,window_center,window_width):
    win_min = (2 * window_center - window_width) / 2.0 + 0.5
    win_max = (2 * window_center + window_width) / 2.0 + 0.5
    dFactor = 255.0 / (win_max - win_min)
    hu = np.array(hu)
    for pixel_val in np.nditer(hu, op_flags=['readwrite']):
        if pixel_val[...] < win_min:
            pixel_val[...] = 0
            continue;
        if pixel_val[...] > win_max:
            pixel_val[...] = 255
            continue;
        nPixelVal = (pixel_val[...] - win_min) * dFactor
        if nPixelVal < 0:
            pixel_val[...] = 0
        elif nPixelVal > 255:
            pixel_val[...] = 255
        else:
            pixel_val[...] = nPixelVal
        #hu = (hu - np.min(hu)) / (np.max(hu) - np.min(hu))
    return hu