#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:26:03 2017

@author: hjl
"""


import numpy as np

class Next_batch:
    def __init__(self,Xtr,Ytr,batch_size):
        self.Xtr=Xtr
        self.Ytr=Ytr
        self.batch_size=batch_size
        self.num_example=Xtr.shape[0]
        self.Index_of_epoch=0
        self.epoch_competed=0
        
    def next_batch(self):
        start=self.Index_of_epoch
        if start==0 and self.epoch_competed==0:
            idx=np.arange(0,self.num_example)
            np.random.shuffle(idx)
            self.Xtr=self.Xtr[idx]
            self.Ytr=self.Ytr[idx]
            
        if start+self.batch_size>self.num_example:
            self.epoch_competed+=1
            rest_num_example=self.num_example-start
            Xtr_rest=self.Xtr[start:self.num_example]
            Ytr_rest=self.Ytr[start:self.num_example]
            idx0=np.arange(0,self.num_example)
            np.random.shuffle(idx0)
            self.Xtr = self.Xtr[idx0]
            self.Ytr = self.Ytr[idx0]
            
            start = 0
            self.Index_of_epoch = self.batch_size - rest_num_example
            end =  self.Index_of_epoch  
            Xtr_new_part =  self.Xtr[start:end]
            Ytr_new_part =  self.Ytr[start:end]
            return np.concatenate((Xtr_rest, Xtr_new_part), axis=0),np.concatenate((Ytr_rest, Ytr_new_part), axis=0)
        else:
            self.Index_of_epoch += self.batch_size
            end = self.Index_of_epoch
            return self.Xtr[start:end],self.Ytr[start:end]
        
    def next_batch_line(self):
        start=self.Index_of_epoch
        if start+self.batch_size>self.num_example:
            Xtr_rest=self.Xtr[start:self.num_example]
            Ytr_rest=self.Ytr[start:self.num_example]
            return Xtr_rest,Ytr_rest
        else:
            self.Index_of_epoch += self.batch_size
            end = self.Index_of_epoch
            return self.Xtr[start:end],self.Ytr[start:end]
#  test
#d = Next_batch(np.arange(0, 100),np.arange(0, 100),30)
#Xtr,Ytr=d.next_batch()


                
                

            
            
            
            
            
            
            
            
            
            
            
            
            
            