#!/usr/bin python

import torch
import torch.nn.functional as F
import sys
sys.path.append('/scratch/gpfs/rmc2/ml_collisions/mic_parallel/')
from models import ReSeg
import pickle
import h5py

class MLCollisions(torch.nn.Module):

    def __init__(self,file_model, channels=1):
        super().__init__()
        self.channels = channels
        out = torch.load(file_model)
        model = ReSeg(channels=channels, temperature=True, usegpu=True).cuda()
        state_dict = out['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.eval()
        self.model = model
        self.mean = 0.0
        self.std = 0.0

    def stats(self,f):
        self.mean = f.mean(dim=(2,3),keepdim=True)
        self.std = f.std(dim=(2,3),keepdim=True)

    def preprocess(self,f,temp):
        fnorm = ((f - self.mean)/self.std).float()
        tempnorm = (temp/5400).float()
        return fnorm, tempnorm

    def postprocess(self,fnorm):
        return fnorm*self.std + self.mean

    def forward(self,f,temp):
        self.stats(f)
        fnorm, tempnorm = self.preprocess(f,temp)
        fdfnorm = self.model(fnorm, tempnorm)
        fdf = self.postprocess(fdfnorm)
        return fdf

