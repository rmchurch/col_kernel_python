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

    def forward(self,f,temp):
        return self.model((f - f.mean(dim=(2,3),keepdim=True)/f.std(dim=(2,3),keepdim=True)).float(), (temp/5400).float())*f.std(dim=(2,3),keepdim=True) + f.mean(dim=(2,3),keepdim=True)

