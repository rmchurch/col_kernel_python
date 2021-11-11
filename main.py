#!/usr/bin python
import numpy as np
import f90nml
from modules import sml_param,f0_param,col_f_param
from ml_collisions import MLCollisions
import matplotlib.pyplot as plt; plt.ion()
import torch

#TODO put into arg parser
plot=False #True
use_jet=False
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
sml['sml_mass'] = [m*sml['sml_prot_mass'] for m in sml['sml_mass']]
sml['sml_charge'] = [q*sml['sml_e_charge'] for q in sml['sml_charge']]
## Compute electron collision time using the ion-electron Coulomb logarithm
l_ie  = 24. - np.log(np.sqrt(sml['sml_den'][0]*1e-6)/sml['sml_t_ev'][0])
sml_tau_e = np.sqrt(sml['sml_t_ev'][0]**3)/(2.91e-6*sml['sml_den'][0]*1e-6*l_ie)
sml['sml_dt'] = sml['sml_dt']*sml_tau_e


##f0_init
f0 = nml['f0_param']
Nsp = f0['f0_nsp']-f0['f0_isp']+1
if not type(f0['f0_velo_aniso'])==list: f0['f0_velo_aniso'] = Nsp*[f0['f0_velo_aniso']]
f0_dsmu = f0['f0_smu_max']/float(f0['f0_nmu']) 
f0_dvp = f0['f0_vp_max']/float(f0['f0_nvp']) 
vperp = np.arange(f0['f0_nmu']+1)*f0_dsmu
vpara = np.arange(-f0['f0_nvp'],f0['f0_nvp']+1)*f0_dvp
f0_nnode = 1 #hardcode for now
#f0_isp is 0 if electrons, 1 if no electrons
f0_f = np.zeros((Nsp,vperp.size,f0_nnode,vpara.size))
for isp in range(f0_f.shape[0]):
    for inode in range(f0_nnode):
        energy_para = 0.5*(vpara - sml['sml_flow'][isp])**2./f0['f0_velo_aniso'][isp]
        energy_perp = 0.5*vperp**2.
        energy = np.einsum('i,j->ij',energy_perp,np.ones(energy_para.shape)) + \
                 np.einsum('i,j->ij',np.ones(energy_perp.shape),energy_para)
        f0_f[isp,:,inode,:] = vperp[:,np.newaxis]*sml['sml_den'][isp]/(np.sqrt(f0['f0_velo_aniso'][isp])*sml['sml_t_ev'][isp])*np.exp(-energy)
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

#save initial
f0_f_init = f0_f.copy()



##collison operator setup
file_log = 'checkpoint_log.pickle'
file_model = 'checkpoint.pth.tar'
file_stats = 'hdf_stats.h5'
collisions = MLCollisions(file_model,file_log,file_stats,normalize=False)


## moment function
#this leaves out a vperp, since its cancelled with the vol factor
conv_factor = 1./np.sqrt(np.array(sml['sml_t_ev'])*(2*np.pi*sml['sml_e_charge']/np.array(sml['sml_mass']))**3)
vperp1 = vperp.copy()
vperp1[0] = vperp1[1]/3.
#vperp1[-1] = vperp1[1]*(vperp1.size-1./3.)
#this leaves out a vperp, since its cancelled with the conv_factor factor
#TODO: The ends have different values, decide what to do
m = np.array(sml['sml_mass']); q = np.array(sml['sml_charge'])
vth = np.sqrt(np.array(sml['sml_t_ev'])*sml['sml_ev2j']/m)
vol = 2*np.pi*f0_dsmu*f0_dvp*np.einsum('i,j->ij',vth**3.,np.ones(vperp.shape))
vol[:,0] = 0.5/3.*vol[:,1] #would the 1/3 be cancelled by conv_factor?
vol[:,-1] = 0.5*(vperp.size-1./3.)/(vperp.size-1.)*vol[:,-2]
def calcMoments(f):
    den = np.einsum('ijkl,i,ij->ik',f,conv_factor,vol)
    upar = np.einsum('l,ijkl,i,ij->ik',vpara,f,conv_factor,vol)/den
    prefac = 0.5*np.array(sml['sml_t_ev']) #this is the 1/2*m*vth^2 factor, but normalizes out to this
    Tperp = np.einsum('i,j,ijkl,i,ij->ik',prefac,vperp1**2.,f,conv_factor,vol)/den
    Tpara = 2.*np.einsum('i,l,ijkl,i,ij->ik',prefac,vpara**2.,f,conv_factor,vol)/den
    entropy = np.einsum('ijkl,i,ij->ik',-np.log(f+1e-16)*f,conv_factor,vol)
    return den,upar,Tperp,Tpara,entropy


## TIME-STEPPING
ions = True
electrons = True
if collisions.channels==1: electrons=False
den = np.zeros((f0_f.shape[0],f0_f.shape[2],sml['sml_nstep']))
upar = np.zeros(den.shape)
Tperp = np.zeros(den.shape); Tpara = np.zeros(den.shape)
entropy = np.zeros(den.shape)
dden = np.zeros(den.shape)
dupar = np.zeros(den.shape)
dTperp = np.zeros(den.shape); dTpara = np.zeros(den.shape)
if plot: fig,ax = plt.subplots(2,4)
for i in range(sml['sml_nstep']):
    print('Start step %d' % i)
    den[...,i],upar[...,i],Tperp[...,i],Tpara[...,i],entropy[...,i] = calcMoments(f0_f)
    print('n: %.2e \t upar: %.2e \t Tperp: %.2e \t Tpara: %.2e ' % (den[1,0,i],upar[1,0,i],Tperp[1,0,i],Tpara[1,0,i]))

    fpre = collisions.preprocess(torch.from_numpy(f0_f))
    if normalize:
        fnorm = collisions.normalize_f(fpre)
    else:
        fnorm = fpre
    fdfnorm = collisions.forward_nn(fnorm.contiguous())
    if normalize:
        dfout = collisions.unnormalize_fdf(fdfnorm) - fpre
    else:
        dfout = fdfnorm
    df = collisions.postprocess(dfout).contiguous().detach().numpy()

    
    dden[...,i],dupar[...,i],dTperp[...,i],dTpara[...,i],_ = calcMoments(df)
    if electrons:
        if plot:
            ax[0,0].contourf(f0_f[0,:,0,:],100)
            ax[0,1].contourf(fnorm[0,0,:,:].detach().numpy(),100)
            ax[0,2].contourf(fdfnorm[0,0,:,:].detach().numpy(),100)
            ax[0,3].contourf(df[0,:,0,:],100)
        f0_f[0,...] += df[0,...]
    if ions:
        if plot:
            ax[1,0].contourf(f0_f[1,:,0,:],100)
            ax[1,1].contourf(fnorm[0,1,:,:].detach().numpy(),100)
            ax[1,2].contourf(fdfnorm[0,1,:,:].detach().numpy(),100)
            ax[1,3].contourf(df[1,:,0,:],100)
        f0_f[1,...] += df[collisions.channels-1,...]
    if plot:
        plt.suptitle('Step %d' % i)
        plt.draw()
        plt.pause(0.1)
    #remove negative
    #f0_f[f0_f<0] = 1e-16
    #f0_f += df

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
