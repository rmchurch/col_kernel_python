#!/usr/bin python

import torch
import torch.nn.functional as F
import sys
sys.path.append('/scratch/gpfs/rmc2/ml_collisions/alps2/')
from models import ReSeg
import pickle
import h5py

class MLCollisions(torch.nn.Module):

    def __init__(self,file_model,file_log,file_stats):
        super().__init__()
        self.file_stats = file_stats
        self.read_stats()

        with open(file_log, 'rb') as pickled_log:
            log = pickle.load(pickled_log)
        num_targets = len(log['targets'])
        model = ReSeg(channels=num_targets)
        out = torch.load(file_model)
        model.load_state_dict(out['state_dict'])
        model.eval()
        self.model = model.cuda()

    def read_stats(self):
        fh = h5py.File(self.file_stats,'r')
        attrs = ['mean_f','std_f','mean_fdf','std_fdf']
        self.mean_f = torch.nn.Parameter(torch.unsqueeze(torch.from_numpy(fh['mean_f'][...]),0),requires_grad=False)
        self.std_f = torch.nn.Parameter(torch.unsqueeze(torch.from_numpy(fh['std_f'][...]),0),requires_grad=False)
        self.mean_fdf = torch.nn.Parameter(torch.unsqueeze(torch.from_numpy(fh['mean_fdf'][...]),0),requires_grad=False)
        self.std_fdf = torch.nn.Parameter(torch.unsqueeze(torch.from_numpy(fh['std_fdf'][...]),0),requires_grad=False)
        fh.close()

    def normalize_f(self,f):
        return ((f - self.mean_f)/self.std_f)

    def preprocess(self,f):
        #remove negative inds
        f[f<0] = torch.tensor(0.0).double()
        #if adiabatic electron, add dimension for electrons
        #if torch.tensor(f.size())[0]==torch.tensor(1): #(assume for now adiabatic electrons, otherwise trace issue)
        #f = F.pad(f,(0,0,0,0,0,0,1,0))
        #switch order for pytorch model to [Ngrid,Nsp,Nmu,Nvpara]
        f = f.permute(2,0,1,3)
        #pad mu direction (so 32,32)
        f = F.pad(f,(0,1,0,0),mode='replicate')
        #normalize and convert to float for input to model
        return f

    def postprocess(self,fdfnorm,fpre):
        #unnormalize (ions only)
        df = fdfnorm*self.std_fdf + self.mean_fdf - fpre
        #remove extra vpara dimension
        df = df[:,:,:,:-1]
        #switch order back to XGC order of [Nsp,Nmu,Ngrid,Nvpara] and convert to double
        return df.permute(1,2,0,3)

    def forward(self,f):
        #this assumed fpre comes in preprocessed and on the GPU
        fpre = self.preprocess(torch.from_numpy(f))
        fnorm = self.normalize_f(fpre)
        return self.postprocess(self.model(fnorm.float().cuda()).cpu().double(),fpre)
        #return self.postprocess(self.model(fnorm)).cpu()
        #return self.model((fpre - self.mean_f)/self.std_f)*self.std_fdf + self.mean_fdf - torch.unsqueeze(fpre[:,1,:,:],1)

