Simple test of the ability of the ML collision operator to isotropize a bi-Maxwellian distribution function, 
with ion and electron temperatures also different. Should replicate Figure 3 from Hager, R., & Chang, C. S. (2016). "Gyrokinetic neoclassical study of the bootstrap current in the tokamak edge pedestal with fully non-linear Coulomb collisions." Physics of Plasmas, 23(4), 042503. https://doi.org/10.1063/1.4945615

<img src="fortran_collision_kernel.png" width="50%">

**NOTE: Currently not working, density rising too much**

Code relies on the ReSeg model and trained weights, and training on a GPU (can modify for CPU testing).
* ReSeg model: https://gitlab.com/adener/ml-collision-python.git
* Trained weights (ions+electrons): https://drive.google.com/drive/folders/1PB-BqBy-i8aQI1AaophbfTRkuodXukV2?usp=sharing

