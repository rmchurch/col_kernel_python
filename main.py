#!/usr/bin python
import numpy as np
import f90nml
from modules import sml_param,f0_param,col_f_param
#from ml_collisions import MLCollisions
from ml_collisions import MLCollisions
import matplotlib.pyplot as plt; plt.ion()
import torch
import torch.nn.functional as F

#TODO put into arg parser
plot=True
use_jet=False
use_ion=True
normalize = True#this should be in the log eventually

##setup - read in namedlist
with open('input') as nml_file:
    nml_in = f90nml.read(nml_file)
nml = f90nml.Namelist()
nml['sml_param'] = sml_param().__dict__
nml['sml_param'].update(nml_in['sml_param'])
nml['f0_param'] = f0_param().__dict__
nml['f0_param'].update(nml_in['f0_param'])
nml['col_f_param'] = col_f_param().__dict__
nml['col_f_param'].update(nml_in['col_f_param'])

sml = nml['sml_param']
sml['sml_mass'] = sml['sml_prot_mass']*sml['sml_mass']
sml['sml_charge'] = sml['sml_e_charge']*sml['sml_charge']
## Compute electron collision time using the ion-electron Coulomb logarithm
l_ii  = 24. - np.log(np.sqrt(sml['sml_den']*1e-6)/sml['sml_t_ev']) #TODO: update with actual l_ii, this is l_ie with ion
sml_tau_i = np.sqrt(sml['sml_t_ev']**3)/(2.91e-6*sml['sml_den']*1e-6*l_ii)
sml['sml_dt'] = sml['sml_dt']*sml_tau_i
Nsp = sml['sml_nsp']-sml['sml_isp']+1


##f0_init
f0 = nml['f0_param']
f0_dsmu = f0['f0_smu_max']/float(f0['f0_nmu']) 
f0_dvp = f0['f0_vp_max']/float(f0['f0_nvp']) 
vperp = np.arange(f0['f0_nmu']+1)*f0_dsmu
vpara = np.arange(-f0['f0_nvp'],f0['f0_nvp']+1)*f0_dvp
f0_nnode = 1 #hardcode for now
#f0_isp is 0 if electrons, 1 if no electrons
f0_f = np.zeros((f0_nnode, Nsp, vperp.size, vpara.size))
for isp in range(f0_f.shape[1]):
    for inode in range(f0_nnode):
        energy_para = 0.5*(vpara - sml['sml_flow'])**2./f0['f0_velo_aniso']
        energy_perp = 0.5*vperp**2.
        energy = np.einsum('i,j->ij',energy_perp,np.ones(energy_para.shape)) + \
                 np.einsum('i,j->ij',np.ones(energy_perp.shape),energy_para)
        f0_f[inode, isp, :,:] = vperp[:,np.newaxis]*sml['sml_den']/(np.sqrt(f0['f0_velo_aniso'])*sml['sml_t_ev'])*np.exp(-energy)
if use_jet:
    #try instead data directly from JET simulation
    import h5py
    datapath = '/scratch/gpfs/marcoam/ml_collisions/data/xgc1/ti272_JET_heat_load/00094/'
    ind = 50000
    phiind = 0
    fh = h5py.File(datapath+'hdf_f.h5','r')
    f0_f[...] = 0.0
    f0_f[0,:,0,:] = fh['e_f'][phiind,:,ind,:]
    f0_f[1,:,0,:] = fh['i_f'][phiind,:,ind,:]
    fh.close()
    fh = h5py.File(datapath+'hdf_cons_fullvol.h5','r')
    f0_T_ev = fh['f0_T_ev'][:,ind]
    sml['sml_t_ev'] = f0_T_ev
    fh.close()
elif use_ion:
    fh = np.load('outputs.0.npz_fno_original')
    f0_f[0,0,...] = fh['data_true'][0,0,:,:-1]
    sml['sml_t_ev'] = fh['temp'][0,0]

#save initial
f0_f_init = f0_f.copy()



##collison operator setup
#file_model = '/scratch/gpfs/rmc2/ml_collisions/mic_parallel/model_best.285218.pth.tar'
#file_model = '/scratch/gpfs/rmc2/ml_collisions/col_kernel_python/model_iononly_hong_v2/iononly_model_best.pth.tar'
#file_model = '/scratch/gpfs/rmc2/ml_collisions/col_kernel_python/model_iononly_hong_entropyon/model_iononly_entropyon.pth.tar'
#model_best.254407.pth.tar'
file_model = '../iononly_entropyon_03_24_2022/logs/model_best_278.pth.tar'
collisions = MLCollisions(file_model, channels=1)


## moment function
#this leaves out a vperp, since its cancelled with the vol factor
conv_factor = np.array([1./np.sqrt(np.array(sml['sml_t_ev'])*(2*np.pi*sml['sml_e_charge']/np.array(sml['sml_mass']))**3)])
vperp1 = vperp.copy()
vperp1[0] = vperp1[1]/3.


#vperp1[-1] = vperp1[1]*(vperp1.size-1./3.)
#this leaves out a vperp, since its cancelled with the conv_factor factor
#TODO: The ends have different values, decide what to do
m = np.array(sml['sml_mass']); q = np.array(sml['sml_charge'])
vth = np.array([np.sqrt(np.array(sml['sml_t_ev'])*sml['sml_ev2j']/m)])
vol = 2*np.pi*f0_dsmu*f0_dvp*np.einsum('i,j->ij',vth**3.,vperp)
vol[:,0] = 0.5/3.*vol[:,1] #would the 1/3 be cancelled by conv_factor?
vol[:,-1] = 0.5*(vperp.size-1./3.)/(vperp.size-1.)*vol[:,-2]

#convert f0_f to the collision operator representation
if use_ion:
    fhat = f0_f.copy()
else:
    fhat = np.einsum('ijkl,j,k->ijkl',f0_f,conv_factor,1./vperp1)

def calcMoments(f):
    den = np.einsum('kijl,ij->ik',f,vol)
    upar = np.einsum('l,kijl,ij->ik',vpara,f,vol)/den
    prefac = np.array([0.5*np.array(sml['sml_t_ev'])]) #this is the 1/2*m*vth^2 factor, but normalizes out to this
    Tperp = np.einsum('i,j,kijl,ij->ik',prefac,vperp1**2.,f,vol)/den
    Tpara = 2.*np.einsum('i,l,kijl,ij->ik',prefac,vpara**2.,f,vol)/den
    entropy = np.einsum('kijl,ij->ik',-np.log(f+1e-16)*f,vol)
    return den,upar,Tperp,Tpara,entropy


## TIME-STEPPING
ions = True
electrons = True
if collisions.channels==1: electrons=False
#Ngrid, Nsp
Nneg = 0
den = np.zeros((f0_f.shape[1],f0_f.shape[0],sml['sml_nstep']))
upar = np.zeros(den.shape)
Tperp = np.zeros(den.shape); Tpara = np.zeros(den.shape)
entropy = np.zeros(den.shape)
dden = np.zeros(den.shape)
dupar = np.zeros(den.shape)
dTperp = np.zeros(den.shape); dTpara = np.zeros(den.shape)
if plot:
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable 
    fig,ax = plt.subplots(1,1)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")

for i in range(sml['sml_nstep']):
    print('Start step %d' % i)
    den[...,i],upar[...,i],Tperp[...,i],Tpara[...,i],entropy[...,i] = calcMoments(fhat)
    print('n: %.2e \t upar: %.2e \t Tperp: %.2e \t Tpara: %.2e \t Nneg: %d ' % (den[0,0,i],upar[0,0,i],Tperp[0,0,i],Tpara[0,0,i], Nneg))

    temp = (Tpara[...,i] + 2*Tperp[...,i])/3.
    print(temp)
    fin = F.pad(torch.from_numpy(fhat), (0,1,0,0),mode='replicate') 
    fdf = collisions.forward(fin.cuda(), torch.from_numpy(temp).cuda()).detach().cpu().numpy() #pre- and post-processing in forward() pass
    df = np.maximum(fdf[...,:-1], fhat) - fhat

    dden[...,i],dupar[...,i],dTperp[...,i],dTpara[...,i],_ = calcMoments(df)
    if plot:
        ax.clear()
        cf = ax.contourf(df[0,0,...],100)  
        cb = fig.colorbar(cf,cax=cax)
        plt.suptitle('Step %d' % i)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(2)
    #remove negative
    Nneg = np.sum(fhat<0)
    fhat[fhat<0] = 1e-16
    fhat += df

#save moment data
np.savez('col_output.npz',den=den,upar=upar,Tperp=Tperp,Tpara=Tpara,
                          dden=dden,dupar=dupar,dTperp=dTperp,dTpara=dTpara)

plt.figure()
plt.plot(Tperp[0,0,:],label='$T_{e,\perp}$')
plt.plot(Tpara[0,0,:],label='$T_{e,\parallel}$')
plt.plot(Tperp[1,0,:],label='$T_{i,\perp}$')
plt.plot(Tpara[1,0,:],label='$T_{i,\parallel}$')
plt.legend()
plt.xlabel('Timestep')
