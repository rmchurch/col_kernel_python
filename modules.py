#!/usr/bin python

class params(dict):
    def __init__(self):
        pass #may be used in future for head dict

class sml_param(object):
    def __init__(self):
        #super().__init__()
        self.sml_prot_mass = 1.6720e-27 #            !< proton mass (MKS)
        self.sml_e_charge  = 1.6022e-19 #           !< elementary charge (MKS)
        self.sml_ev2j      = self.sml_e_charge          #!< eV --> J conversion
        self.sml_j2ev      = 1.0/self.sml_e_charge      #!< J --> eV conversion
        self.sml_pi        = 3.1415926535897932    #!< pi
        self.sml_2pi       = 6.2831853071795862  #!< 2 pi
        self.sml_nsp_max         = 3                     #!< Maximum number of species minus 1

        #! Input variables
        self.sml_dt = 0.25e0 #!< Collision time step in units of the electron collision time
        self.sml_tau_e = 0e0 #!< Electron collision time
        self.sml_nstep = 10        #!< Number of time steps
        self.sml_isp = 1           #!< Index of first species (0 is for electrons, 1 for main ion species, >1 for impurities)
        self.sml_nsp = 1           #!< Index of last species
        self.sml_den    = [1e19,9.95003e18,3.33e16,1.667e16 ]  #!< Density of each species make sure that the sum of
        #!! species 1-sml_nsp is equal to the density of species 0
        self.sml_t_ev   = [ 1e3,1e3,1e3,1e3 ]                   #!< Temperature of each species
        self.sml_flow   = [ 0.2,0.1,0.1,0.1 ]                   #!< Flow of each species
        self.sml_mass   = [ 5.448e-4,2e0,7e0,12e0 ]             #!< Mass of each species in units of the proton mass
        self.sml_charge = [ -1e0,1e0,3e0,6e0 ]                  #!< Charge of each species, in units of the elementary charge
        #self.__dict__ = self
        #! Time and time step
        #self.sml_time    # !< Time of the simulation in seconds
        #self.sml_istep          #!< Time step index

        #! MPI variables
        #integer :: sml_totalpe   = 1  !< Number of MPI ranks
        #integer :: sml_mype      = 0  !< MPI rank ID
        #integer :: sml_comm      = 0  !< Global communicator (MPI_COMM_WORLD)
        #integer :: sml_comm_null = 0  !< Self-communicator

        #!OpenMP
        #integer :: sml_nthreads  = 1  !< Number of OpenMP threads (not an input variable)

                                                  #!! ~ number of configuration space grid points
                                                  #!! f0_nnode is #(MPI ranks)**m, with positive integer m
class f0_param(dict):
    def __init__(self):
        self.f0_nnode = 1                         #!< Total number of (independent) collision operations per time step
        self.f0_node_mult = 1                     #!< Multiplier for number of grid points ("m" in comment on f0_nnode)
        self.f0_isp = 0                           #!< Index of first species (copied from sml_isp)
        self.f0_nsp = 1                           #!< Index of last species (copied from sml_nsp; total number of species: f0_nsp-f0_isp+1)
        self.f0_nmu = 44                          #!< Grid size in v_perp direction --> N_perp = f0_nmu+1
        self.f0_nvp = 44                          #!< Grid size in v_para direction --> N_para = 2*f0_nvp+1
        self.f0_smu_max = 4.               #!< Perp. velocity cutoff in units of the thermal velocity
        self.f0_vp_max = 4.                #!< Parallel velocity cutoff in units of the thermal velocity
        #self.f0_dsmu                        #!< Grid resolution in v_perp direction
        #self.f0_dvp                         #!< Grid resolution in v_para direction
        self.f0_velo_aniso = 0.3          #!< Velocity anisotropy of the initial distribution function #!! (T_para = f0_velo_aniso*f0_t_ev)

class col_f_param(dict):
    def __init__(self):
        pass #these are really for the Picard iteration solver
