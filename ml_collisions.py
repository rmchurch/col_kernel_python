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

    def __init__(self, file_model, zca, device, channels=1, dropout=0.5):
        super().__init__()
        self.channels = channels
        out = torch.load(file_model,map_location=device)
        model = ReSeg(channels=channels,dropout=dropout).to(device)
        state_dict = out['state_dict']
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
        self.zca = zca
        self.device = device

    def stats(self,f):
        self.mean = f.mean(dim=(2,3),keepdim=True)
        self.std = f.std(dim=(2,3),keepdim=True,unbiased=False)+1e-15 # make it consistent with the numpy implementation

    def print_stats(self,f):
        print(self.mean)
        print(self.std)
        mean = np.mean(f,axis=(2,3),keepdims=True)
        std = np.std(f,axis=(2,3),keepdims=True)+1e-15
        print(mean)
        print(std)

    def preprocess(self, f, temp):
        self.stats(f)
        fnorm = ((f - self.mean)/self.std).float()
        tempnorm = (temp/5400).float()
        return fnorm, tempnorm

    def postprocess(self, fnorm):
        return fnorm*self.std + self.mean

    def forward(self,fnorm,temp):
        # self.stats(f)
        #fnorm, tempnorm = self.preprocess(f.to(self.device),temp.to(self.device))
        fdfnorm = self.model(fnorm) #, tempnorm)
        #fdf = self.postprocess(fdfnorm)
        return fdfnorm

