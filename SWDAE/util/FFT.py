# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:34:17 2017

@author: hjl
"""
import numpy as np
def fft_data(S,Fs):
    #can fft matrix  :      sample*Length
    SampleNum=S.shape[0]
    SampleLen=S.shape[1]
    outputY=np.zeros([SampleNum,SampleLen//2])
    for i in range(SampleNum):
        Y=S[i,:]
        Y=np.fft.fft(Y,SampleLen)
        Y_abs=np.abs(Y)
        Y_abs=Y_abs/(SampleLen/2)
        Y_abs[0]=Y_abs[0]/2
        Y_fft=Y_abs[:,0:SampleLen//2]
        outputY[i,:]=Y_fft
    #F=np.arange(1,SampleLen)*Fs/SampleLen
    #X=F[0:np.int16(SampleLen/2)]
    return outputY