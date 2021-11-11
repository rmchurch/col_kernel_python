#!/usr/bin python

import torch
import torch.nn.functional as F
import sys
sys.path.append('/scratch/gpfs/rmc2/ml_collisions/alps2/')
from models import ReSeg
import pickle
import h5py

class MLCollisions(torch.nn.Module):

    def __init__(self,file_model,file_log,file_stats,normalize=True):
        super().__init__()
        self.file_stats = file_stats
        self.read_stats()

        with open(file_log, 'rb') as pickled_log:
            log = pickle.load(pickled_log)
        self.channels = len(log['targets'])
        model = ReSeg(channels=self.channels)
        out = torch.load(file_model)
        model.load_state_dict(out['state_dict'])
        model.eval()
        self.model = model.cuda()
        self.normalize = normalize

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
    
    def unnormalize_f(self,f):
        return f*self.std_f + self.mean_f
    
    def normalize_fdf(self,fdf):
        return ((fdf - self.mean_fdf)/self.std_fdf)
    
    def unnormalize_fdf(self,fdfnorm):
        if self.channels==1:
            return fdfnorm*self.std_fdf[:,1,:,:] + self.mean_fdf[:,1,:,:]
        else:
            return fdfnorm*self.std_fdf + self.mean_fdf
    
    def preprocess(self,f):
        #remove negative inds
        f[f<0] = torch.tensor(0.0).double()
        #if adiabatic electron, add dimension for electrons
        #if self.channels>1:
        #    f = F.pad(f,(0,0,0,0,0,0,1,0))
        #switch order for pytorch model to [Ngrid,Nsp,Nmu,Nvpara]
        f = f.permute(2,0,1,3)
        #pad mu direction (so 32,32)
        f = F.pad(f,(0,1,0,0),mode='replicate')
        return f.contiguous()

    def postprocess(self,df):
        #remove extra vpara dimension
        df = df[:,:,:,:-1]
        #switch order back to XGC order of [Nsp,Nmu,Ngrid,Nvpara] and convert to double
        return df.permute(1,2,0,3).contiguous()
    
    def forward_nn(self,fnorm):
        return self.model(fnorm.float().cuda()).cpu().double()

    def forward(self,f):
        fpre = self.preprocess(torch.from_numpy(f))
        if normalize:
            fpre = self.normalize_f(fpre)
        dfout = self.forward_nn(fpre)
        if normalize:
            dfout = self.unnormalize_fdf(dfout) - fpre
        return self.postprocess(dfout)
        #return self.postprocess(self.model(fnorm)).cpu()
        #return self.model((fpre - self.mean_f)/self.std_f)*self.std_fdf + self.mean_fdf - torch.unsqueeze(fpre[:,1,:,:],1)

