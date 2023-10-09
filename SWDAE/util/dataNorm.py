#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:10:16 2017

@author: hjl
"""

import random
import numpy as np
import scipy.io as sio

def datanorm(x,Xyuan):
    lb=0
    ub=1
    n=x.shape[1]
    sca=np.zeros((n,1))
    mina=np.zeros((n,1))
    maxa=np.zeros((n,1))
    for i in np.arange(n):
        mina[i]=np.min(Xyuan[:,i])
        maxa[i]=np.max(Xyuan[:,i])
        sca[i] = maxa[i] - mina[i]
    A=np.zeros((n,1))
    B=np.zeros((n,1))
    xscaled=np.zeros((x.shape[0],n))
    
    for i in np.arange(n):
        if sca[i]:
             A[i] = (ub - lb)/sca[i]
             B[i] = lb - A[i]*mina[i]
             xscaled[:,i] = A[i]*x[:,i] + B[i]
    return xscaled
def inversedatanorm(x,Xyuan):
    n=x.shape[1]
    sca=np.zeros((n,1))
    mina=np.zeros((n,1))
    maxa=np.zeros((n,1))
    for i in np.arange(n):
        mina[i]=np.min(Xyuan[:,i])
        maxa[i]=np.max(Xyuan[:,i])
        sca[i] = maxa[i] - mina[i]
        xreverse=np.zeros((x.shape[0],n))
        for i in np.arange(n):
            if sca[i]:
                xreverse[:,i] = 0.5*(x[:,i]+1)*(maxa[i]-mina[i])+mina[i]
    return xreverse