#!/usr/bin python

import torch
import torch.nn.functional as F
import sys
sys.path.append('..')
from models import ReSeg
import pickle
import h5py
import numpy as np

class MLCollisions(torch.nn.Module):

    def __init__(self, file_model, device, channels=1, dropout=0.5, train_df=False, minmax=False, div_log_tran=False, div_log_tran_divisor=1e10, scaling='normalize'):
        super().__init__()
        self.channels = channels
        out = torch.load(file_model,map_location=device)
        model = ReSeg(channels=channels,dropout=dropout).to(device)
        state_dict = out['state_dict']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        #from collections import OrderedDict
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k[7:] # remove `module.`
        #    new_state_dict[name] = v
        #model.load_state_dict(new_state_dict)
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model
        self.mean = 0.0
        self.std = 0.0
        self.device = device
        self.train_df = train_df
        self.minmax = minmax
        self.div_log_tran = div_log_tran
        self.div_log_tran_divisor = div_log_tran_divisor
        self.scaling = scaling

    def stats(self,f):
        if self.minmax:
            if torch.is_tensor(f):
                self.mean = f.amin(dim=(2,3),keepdims=True)
                self.std = f.amax(dim=(2,3),keepdims=True) - self.mean
            else:
                self.mean = np.min(f,axis=(2,3),keepdims=True)
                self.std = np.max(f,axis=(2,3),keepdims=True) - self.mean
        else:
            if torch.is_tensor(f):
                self.mean = f.mean(dim=(2,3),keepdim=True)
                self.std = f.std(dim=(2,3),keepdim=True,unbiased=False)+1e-15 # make it consistent with the numpy implementation
            else:
                self.mean = np.mean(f,axis=(2,3),keepdims=True)
                self.std = np.std(f,axis=(2,3),keepdims=True)+1e-15

    def print_stats(self,f):
        torch.set_printoptions(8)
        print(self.mean)
        print(self.std)
        if self.minmax:
            if torch.is_tensor(f):
                mean = f.amin(dim=(2,3),keepdims=True)
                std = f.amax(dim=(2,3),keepdims=True) - mean
            else:
                mean = np.min(f,axis=(2,3),keepdims=True)
                std = np.max(f,axis=(2,3),keepdims=True) - mean
        else:
            if torch.is_tensor(f):
                mean = f.amean(dim=(2,3),keepdim=True)
                std = f.std(dim=(2,3),keepdim=True,unbiased=False)+1e-15 # make it consistent with the numpy implementation
            else:
                mean = np.mean(f,axis=(2,3),keepdims=True)
                std = np.std(f,axis=(2,3),keepdims=True)+1e-15
        print(mean)
        print(std)

    def preprocess(self, f):
        if self.div_log_tran:
            f = self.div_log_transform(f, self.div_log_tran_divisor)
        if self.scaling == 'normalize':
            self.stats(f)
            f = (f - self.mean)/self.std
        return f

    def postprocess(self, f):
        if self.scaling == 'normalize':
            f = f*self.std + self.mean
        if self.div_log_tran:
            f = self.div_log_untransform(f, self.div_log_tran_divisor)
        return f

    def forward(self, fnorm):
        fnorm = self.pad(fnorm)
        if self.train_df:
            fdfnorm = fnorm + self.model(fnorm)
        else:
            fdfnorm = self.model(fnorm)
        fdfnorm = self.crop(fdfnorm)
        return fdfnorm

    def div_log_transform(self, X, divisor):
        if torch.is_tensor(X):
            Xtran = torch.log(X/divisor)
        else:
            Xtran = np.log(X/divisor)
        return Xtran

    def div_log_untransform(self, Xtran, divisor):
        if torch.is_tensor(Xtran):
            X = torch.exp(Xtran)*divisor
        else:
            X = np.exp(Xtran)*divisor
        return X

    def crop(self, data):
        return data[...,:-1]

    def pad(self, data):
        return F.pad(data, (0,1,0,0), mode='replicate')
