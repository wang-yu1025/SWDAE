# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:34:30 2022

@author: liujing
"""


from dtaidistance import dtw

import numpy as np
import scipy.io as sio #read mat


X1=sio.loadmat('/home/htu/workspace/wyy/2/data/S2TAE_GY.mat')['TAE1']

#X4=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B3_3']
#X5=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B1_5']
#X6=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B2_2']
#X7=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B2_3']
#X8=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B2_5']
#X9=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B2_4']
#X10=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B2_5']
#X11=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B3_1']
#X12=sio.loadmat(r'F:\1A代码以及数据备份\Leiyaguo_data\Leiyaguo_data\XJTU_RMS.mat')['B3_2']

X13=sio.loadmat('/home/htu/workspace/wyy/2/data/S2xbgy.mat')['ff']
#X14=sio.loadmat(r'F:\1A代码以及数据备份\Data_PHM\Data\B2_2.mat')['H_data_RMS']
#X15=sio.loadmat(r'F:\1A代码以及数据备份\Data_PHM\Data\B2_6.mat')['H_data_RMS']
#X3=sio.loadmat('./RMS_B2.mat')['B2_1']
#X4=sio.loadmat('./RMS_B2.mat')['B2_2']
#X5=sio.loadmat('./RMS_B2.mat')['B2_6']



#X_ALL = [X1,X13]




DTW=np.ones(X13.shape[1])



for i in range(X13.shape[1]):
    print(i)
    a = np.double(X1)
    b = np.double(X13[:,i])
    DTW[i]=dtw.distance(a, b)
    print('dtw distance:',DTW[i])


    #sio.savemat('dtw_C_PHM.mat',{'dtw':DTW})
    sio.savemat('/home/htu/workspace/wyy/2/out/dtw.mat',{'dtw':DTW})
