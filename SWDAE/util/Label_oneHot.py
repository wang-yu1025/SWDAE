# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:35:33 2017

@author: hjl
"""
import numpy as np
def Label_oneHot(Ytr,classNum):
    Ytr=np.int64(Ytr)
    Ytr.shape=Ytr.shape[0]
    Nlen=Ytr.shape[0]
    Yz=np.zeros((Nlen,classNum))
    Yz[np.arange(Nlen),Ytr]=1
    return Yz