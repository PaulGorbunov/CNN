import numpy as np
import cv2

from sys import exit as ex
from importlib import reload as re

class Conv_lay:
    
    def __init__(self,info):
        self.bias = 1
        if type(info) == int:
            info = [info]
        self.out_chan = info[-1]
        self.stride = 1
        self.filt_s = 3
        if len(info) > 1:
            self.stride = info[-2]
        if len(info) > 2:
             self.filt_s = info[-3]
        if len(info) > 3:
            self.padding = info[-4]
        else:
            self.padding = self.filt_s//2
        self.cache = []
        self.delt = []
    
    def add_zeros(self,data):
        h,w,p = np.shape(data)+(0,)*(3-len(np.shape(data)))
        if p == 0 :
            data= data.reshape((h,w,1))
            p = 1
        try:
            self._filter
        except AttributeError:
            self._filter = []
            n_in = np.size(data)
            n_out = ((w + (self.padding-self.filt_s//2 )*2)*(h + (self.padding-self.filt_s//2)*2))/self.stride
            a = 1/np.sqrt(n_in/2)
            a*= 0.001
            for u in range(self.out_chan):
                _filter = np.random.uniform(low=min(n_in*a,n_out*a) ,high=max(n_in*a,n_out*a), size=(self.filt_s,1,self.filt_s))
                fi = _filter
                for u in range(p-1):
                    fi = np.append(fi,_filter,axis = 1)
                fi = np.transpose(fi,(0,2,1))
                self._filter.append(fi)
            
        data= np.append(np.zeros((h,self.padding,p)),data ,axis = 1)
        data= np.append(data ,np.zeros((h,self.padding,p)),axis = 1)
        data= np.append(np.zeros((self.padding,w+self.padding*2,p)),data ,axis = 0)
        data= np.append(data ,np.zeros((self.padding,w+self.padding*2,p)),axis = 0)
        return data,(h,w,p,delt)
        
    def filter_pass(self,data,res_sh):
        r_h,r_w,r_p = res_sh
        if (r_h<= 0) or (r_w <= 0) or (filt_s%2 == 0):
            return 1
        delt = self.filt_s//2
        res = []
        for u in range(r_p):
            lis = np.zeros((r_h*r_w))
            for y in range(delt,len(data)-delt,self.stride):
                for x in range(delt,len(data[0])-delt,self.stride):
                    cols = np.arange(y-delt,y+delt+1,1)
                    rows = slice(x-delt,x+(self.filt_s-delt))
                    lis[int(((y-delt)*r_w+(x-delt))/self.stride)]+= np.sum(data[cols,rows]*self._filter[u]) + self.bias
            res.append(lis.reshape(r_h,r_w))
        res = np.stack(res,axis=-1)
        return res
        
    def change_param(self,nu):
        
        self.cache = []
        self.delt = []
        
    def forv_pass(self,data):
        data = np.array(data)
        data,(h,w,p,delt) = self.add_zeros(data)
        new_w = int((w + (self.padding-delt)*2)/self.stride)
        new_h = int((h + (self.padding-delt)*2)/self.stride)
        res = self.filter_pass(data,(new_h,new_w,self.out_chan))
        self.cache.append(data)
        self.cache.append((delt,i_h,i_w,i_p))
        return res
        
    def back_pass(self,dy):
        data = self.cache[0]
        delt_w = len(data[0])//self.filt_s
        delt_h = 0
        in_c = self.cache[1][-1]
        new_w = self.filt_s
        new_h = self.filt_s
        lis = []
        
        return []
    
    
    
    
    
