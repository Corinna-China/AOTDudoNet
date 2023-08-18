import matplotlib.pyplot as plt
import cv2

import util
from util import fanBeamcuda,ifanBeamcuda,fanBeam,ifanBeam,hu_to_coefficient,replaceImage,showNumpyPlot,coefficient_to_hu
import numpy as np
import PIL.Image as Image
from LI import LIMARwithoutmetal
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression  # 导入线性回归模块
from sklearn.preprocessing import PolynomialFeatures
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

def BHC(img):
    img = coefficient_to_hu(img)
    metal = getMetal(img)
    smetal = fanBeam(metal)
    ssmetal = np.array(smetal)
    ssmetal[ssmetal!=0]=1
    prior, sprior = getPrior(img, metal)

    smetal_con = getMetalConnectedComponents(metal)

    img = hu_to_coefficient(img)
    simg = fanBeam(img)
    import time

    # start_time = time.time()
    sbhc = fit(np.array(simg),np.array(smetal),np.array(sprior),smetal_con)
    # end_time = time.time()
    # dur_time = end_time - start_time
    # print('Times: ', dur_time)

    # ssbhc = normalize(sbhc, proj_get_minmax())
    # util.numpyToDicom("output1/sBHC.dcm", ssbhc.squeeze())
    xbhc = ifanBeam(sbhc)
    # xbhc = coefficient_to_hu(xbhc)
    # showNumpyPlot(xbhc)
    # showNumpyPlot(xbhc)
    return np.array(xbhc)


def fit(simg,smetal,sprior,smetal_con):
    s = np.array(simg)
    metal_trace = np.array(smetal)
    metal_trace[metal_trace!=0]=1
    # showNumpyPlot(metal_trace)
    #flatten
    sprior = sprior.reshape(-1)
    smetal_con.append(simg)
    smetal_con.append(smetal)
    for i in range(len(smetal_con)):
        smetal_con[i] = smetal_con[i].reshape(-1)
    x = list(zip(*smetal_con))
    for i in range(len(x)):
        x[i] = list(x[i])
    y = sprior.tolist()
    for index in range(1, 10):
        data = pd.DataFrame({'IN': x, 'OUT': y})
        data_train = np.array(data['IN']).reshape(data['IN'].shape[0], 1)
        data_test = data['OUT']
        poly_reg = PolynomialFeatures(degree=index,include_bias=False)
        X_ploy = poly_reg.fit_transform(x)
        regr = LinearRegression()
        regr.fit(X_ploy, data_test)
        a = regr.predict(X_ploy)
        a = a.reshape(s.shape[0],s.shape[1])
        if index == 3:
            return a
    return

def getMetalConnectedComponents(metal):
    smetal = fanBeam(metal)
    smetal = np.array(smetal)
    metal[metal != 0] = 1

    num_labels,labels,stats,centroids = cv2.connectedComponentsWithStats(metal.astype(np.uint8))
    mask_trace_ans = np.zeros((640,641))
    for i in range(1,num_labels):
        mask = np.array(labels)
        mask[labels!=i] = 0
        # util.showNumpyPlot(mask)
        mask_trace = fanBeam(mask)
        mask_trace = np.array(mask_trace)
        mask_trace[mask_trace!=0] = 1
        mask_trace_ans[mask_trace!=0] += 1
        # util.showNumpyPlot(mask_trace_ans)
    # showNumpyPlot(mask_trace_ans)
    mask_list= []
    for i in range(2, num_labels):
        ssmetal = np.array(smetal)
        m = np.array(mask_trace_ans)
        m[mask_trace_ans!=i] = 0
        ssmetal[m == 0] = 0
        if np.count_nonzero(ssmetal)!=0:
            mask_list.append(ssmetal)
            # showNumpyPlot(ssmetal)
    return mask_list

def getPrior(img,metal):
    metal[metal!=0] = 1
    img = hu_to_coefficient(img)
    # showNumpyPlot(img)
    import time

    # start_time = time.time()
    LIimg, lisin = LIMARwithoutmetal(img, metal)
    # end_time = time.time()
    # dur_time = end_time - start_time
    # print('Times: ', dur_time)
    from sklearn.cluster import KMeans
    kMeans = KMeans(n_clusters=250,n_init=60)
    kMeans.fit(LIimg)
    img = kMeans.cluster_centers_[kMeans.labels_]
    print(img)
    img = coefficient_to_hu(img)
    # util.numpyToDicom("output1/kmeans.dcm",img)
    image = np.array(Image.fromarray(img).resize((416, 416)))
    image = np.expand_dims(np.transpose(np.expand_dims(image, 2), (2, 0, 1)), 0)
    import torch
    image = torch.tensor(image).cuda()
    image = hu_to_coefficient(image)
    simg = util.fanBeamcuda(image)
    simg = simg.cpu().detach().numpy().squeeze()
    # simg = normalize(simg, proj_get_minmax())
    # util.numpyToDicom("output1/skmeans.dcm", simg.squeeze())
    # showNumpyPlot(LIimg)
    return LIimg,lisin

def getMetal(image):
    img_res = np.array(image)
    img_res[img_res<2000] = -1000
    # showNumpyPlot(img_res)
    img_res = hu_to_coefficient(img_res)
    return img_res

def save_ma_file(ma_img):
    base_name = os.path.basename(ma_img)
    if os.path.isfile(bhc_file + base_name):
        print(ma_img)
        ma = util.rawToNumpy("C:\\Users\\kyon\\Documents\\mar\\DeepLesion\\test\\ma_img\\0001_02.raw")
        ma = util.coefficient_to_hu(ma)
        # ma = util.rawToNumpy416(ma_img)
        util.numpyToDicom("./ma.dcm",ma)
        import time


        start_time = time.time()
        bhc = BHC(ma)
        end_time = time.time()
        dur_time = end_time - start_time
        print('Times: ', dur_time)

        # util.numpyToRaw(bhc_file + base_name,bhc)

def fastBHC(img,XLI,SLI):
    img = coefficient_to_hu(img)
    metal = getMetal(img)
    smetal = fanBeam(metal)
    ssmetal = np.array(smetal)
    ssmetal[ssmetal != 0] = 1
    prior, sprior = XLI,SLI

    smetal_con = getMetalConnectedComponents(metal)

    img = hu_to_coefficient(img)
    simg = fanBeam(img)
    import time

    # start_time = time.time()
    sbhc = fit(np.array(simg), np.array(smetal), np.array(sprior), smetal_con)
    # end_time = time.time()
    # dur_time = end_time - start_time
    # print('Times: ', dur_time)

    # ssbhc = normalize(sbhc, proj_get_minmax())
    # util.numpyToDicom("output1/sBHC.dcm", ssbhc.squeeze())
    xbhc = ifanBeam(sbhc)
    # xbhc = coefficient_to_hu(xbhc)
    # showNumpyPlot(xbhc)
    # showNumpyPlot(xbhc)
    return np.array(xbhc)



if __name__ == "__main__":
    data_file = "D:\\DuDoNet\\MAR_Data\\test"
    ma_file = data_file+"\\ma_img\\"
    bhc_file  = data_file + "\\bhc_img\\"
    import glob
    import os
    ma_imgs = glob.glob(os.path.join(ma_file,"*.raw"))
    print(ma_imgs)
    from multiprocessing.pool import ThreadPool
    import random
    random.shuffle(ma_imgs)
    ma_imgs = set(ma_imgs)
    for ma_img in ma_imgs:
        save_ma_file(ma_img)