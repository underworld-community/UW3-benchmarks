#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # Thust Wedge Tutorial
#
# [UW2 example](https://github.com/underworldcode/underworld2/blob/master/docs/UWGeodynamics/tutorials/Tutorial_10_Thrust_Wedges.ipynb)

# %%
import numpy as np
import os
import math
import underworld3


from underworld3.utilities import generateXdmf


# %%
expt_name = 'output/thrustWedge/'

# Make output directory if necessary.
from mpi4py import MPI
if MPI.COMM_WORLD.rank==0:
    ### delete previous model run
    if os.path.exists(expt_name):
        for i in os.listdir(expt_name):
            os.remove(expt_name + i)
            
    ### create folder if not run before
    if not os.path.exists(expt_name):
        os.makedirs(expt_name)


# %%
### For visualisation
render = True


# %%
#Basic direct solver parameters from Firedrake docs, for ref

#mumps_solver_parameters = {
#    "mat_type": "aij",
#    "snes_type": "ksponly",
#    "ksp_type": "preonly",
#    "pc_type": "lu",
#    "pc_factor_mat_solver_type": "mumps",
#}


# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes

options = PETSc.Options()

options["snes_converged_reason"] = None
options["snes_monitor_short"] = None



# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

velocity = 1 * u.centimeter / u.year
model_length = 100. * u.kilometer
mu           = 1e22 * u.pascal*u.second
# bodyforce = 2700. * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = model_length
Kt = KL / velocity
KM = mu * KL * Kt #bodyforce * KL**2 * Kt**2


### create unit registry
scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients

# %%
xres = 256
yres = 128

xmin, xmax = 0, nd(128. * u.kilometer)
ymin, ymax = nd(-7 * u.kilometer), nd(9 * u.kilometer)

# %% [markdown]
# ### Create mesh and mesh vars

# %%
mesh = uw.meshing.StructuredQuadBox(elementRes=(    xres,yres), 
                    minCoords =(xmin, ymin), 
                    maxCoords =(xmax, ymax) )


v = uw.discretisation.MeshVariable("Velocity", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("Pressure", mesh, 1, degree=1)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=1)
node_viscosity   = uw.discretisation.MeshVariable("Viscosity", mesh, 1, degree=1)
materialField    = uw.discretisation.MeshVariable("Material", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=True)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)


# %% [markdown]
# ### Create swarm and swarm vars
# - 'swarm.add_variable' is a traditional swarm, can't be used to map material properties. Can be used for sympy operations, similar to mesh vars.
# - 'uw.swarm.IndexSwarmVariable', creates a mask for each material and can be used to map material properties. Can't be used for sympy operations.
#

# %%
swarm  = uw.swarm.Swarm(mesh)

# %%
## # Add variable for material
materialVariable      = swarm.add_variable(name="materialVariable", num_components=1, dtype=PETSc.IntType, proxy_degree=1)

material              = uw.swarm.IndexSwarmVariable("M", swarm, indices=5, proxy_degree=1) 

cumulativeStrain      = swarm.add_variable(name="strain", num_components=1, dtype=PETSc.RealType, proxy_degree=1)

swarm.populate()

# # Add some randomness to the particle distribution
# import numpy as np
# np.random.seed(0)

# with swarm.access(swarm.particle_coordinates):
#     factor = 0.5*boxLength/n_els/ppcell
#     swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)
      


# %% [markdown]
# #### Project fields to mesh vars
# Useful for visualising stuff on the mesh (Viscosity, material, strain rate etc) and saving to a grouped xdmf file


# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
# nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
# nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

meshMat = uw.systems.Projection(mesh, materialField)
meshMat.uw_function = materialVariable.sym[0]
# meshMat.smoothing = 1.0e-3
meshMat.petsc_options.delValue("ksp_monitor")

def updateFields():
    ### update strain rate
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

    ### update viscosity
    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    nodal_visc_calc.solve(_force_setup=True)
    
    ### update material field from swarm
    meshMat.uw_function = materialVariable.sym[0] 
    meshMat.solve(_force_setup=True)



# %% [markdown]
# ## Setup the material distribution


# %% [markdown]
# ### Update the material variable of the swarm

# %%
airIndex       = 0
ridgidBase     = 1
fritionalBase  = 2
sediment0      = 3
sediment1      = 4

# %%
top_pile = 0.
bottom_pile = nd(-6.0 * u.kilometer)

NLayers = 12
layer_thickness = (top_pile - bottom_pile) / NLayers

# %%
sed0_list = np.arange(top_pile, bottom_pile-layer_thickness, -2*layer_thickness)

sed_list = np.arange(top_pile, bottom_pile-layer_thickness, -layer_thickness)

# %%
with swarm.access(materialVariable, material):
    materialVariable.data[:] = airIndex
    for i in sed_list:
        if np.isin(i, sed0_list):
            materialVariable.data[swarm.data[:,1] <= i] = sediment0
        else:
            materialVariable.data[swarm.data[:,1] <= i] = sediment1
    
    
    
    materialVariable.data[swarm.data[:,1] <= ymin+nd(1*u.kilometer)]  = fritionalBase
    materialVariable.data[swarm.data[:,1] <= ymin+nd(0.5*u.kilometer)] = ridgidBase
    
    
        
    
    
    
    

# %%
def plot_mat():

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)


    with swarm.access():
        point_cloud.point_data["M"] = materialVariable.data.copy()



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
 
if render == True:
    plot_mat()


# %% [markdown]
# ### Function to save output of model
# Saves both the mesh vars and swarm vars

# %%
def saveData(step, outputPath):
    
#     ### save mesh vars
#     fname = f"./{expt_name}{'step_'}{step:02d}.h5"
#     xfname = f"./{expt_name}{'step_'}{step:02d}.xdmf"
#     viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE, comm=PETSc.COMM_WORLD)

#     viewer(mesh.dm)

#     ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
#     viewer(stokes.u._gvec)         # add velocity
#     viewer(stokes.p._gvec)         # add pressure
#     viewer(materialField._gvec)    # add material projection
#     viewer(strain_rate_inv2._gvec) # add strain rate
#     viewer(node_viscosity._gvec)   # add viscosity
#     viewer.destroy() 
#     generateXdmf(fname, xfname)

    mesh.petsc_save_checkpoint(meshVars=[v, p, materialField, strain_rate_inv2, node_viscosity], index=step, outputPath=outputPath)
    
    swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputPath)
    


# %% [markdown]
# ### Rheology

# %%
from sympy import Piecewise, ceiling, Abs, Min, sqrt, eye, Matrix, Max

# %%
minVisc = nd(5e19 * u.pascal * u.second)
maxVisc = nd(1e23 * u.pascal * u.second)

# %%
sedCohesion = nd(20*u.megapascal)
sedYield     = sedCohesion / (2*stokes._Einv2 + 1.0e-18) #(sed_C + (sed_FC * lithoP)) / (2*stokes._Einv2)



# %%
sed_Visc = Min(sedYield, maxVisc) #1. / ((1/sedYield) + (1/maxVisc))

# %%
FBCohesion = nd(0.1*u.megapascal)

FBYield     = FBCohesion / (2*stokes._Einv2 + 1.0e-18) #(sed_C + (sed_FC * lithoP)) / (2*stokes._Einv2)


FB_Visc = sed_Visc = Min(FBYield, maxVisc) #1. / ((1/FBYield) + (1/maxVisc))

# %%
viscosityFn      = Piecewise(
                              (FB_Visc,  Abs(materialVariable.sym[0] -  ridgidBase) < 0.5),
                              (FB_Visc,  Abs(materialVariable.sym[0] - fritionalBase) < 0.5),
                              (sed_Visc, Abs(materialVariable.sym[0] - sediment0) < 0.5),
                              (sed_Visc, Abs(materialVariable.sym[0] - sediment1) < 0.5),
                              (sed_Visc,                                True )
                                )


# %%
# ### weakening due to plasticity
# isYielding = Piecewise( 
#                         (stokes._Einv2, viscosityFn < BGviscosityFn),
#                         (0.0, True)
#                         )

# %% [markdown]
# #### Density

# %%
nd_gravity = nd(9.81*u.meter/u.second**2)

nd_density = nd(2700 * u.kilogram / u.metre**3)


# density = mantleDensity * material.sym[0] + \
#           mantleDensity * material.sym[1] + \
#           slabDensity   * material.sym[2] + \
#           slabDensity   * material.sym[3] + \
#           slabDensity   * material.sym[4]



stokes.bodyforce =  Matrix([0, -1 * nd_gravity*nd_density]) # -density*mesh.N.j


# %% [markdown]
# ### Boundary conditions
#
# Free slip by only constraining one component of velocity 

# %%
#free slip
stokes.add_dirichlet_bc( (1.,0.), 'Left',   (0) ) # left/right: function, boundaries, components
stokes.add_dirichlet_bc( (0,0.), 'Right',  (0) )

# stokes.add_dirichlet_bc( (0.,0.), 'Top',    (1) )
stokes.add_dirichlet_bc( (0.,0.), 'Bottom', (1) )# top/bottom: function, boundaries, components 

# %% [markdown]
# ###### initial first guess of constant viscosity

# %%
stokes.petsc_options['pc_type'] = 'lu'

stokes.petsc_options["snes_max_it"] = 500

stokes.petsc_options["snes_atol"] = 1e-6
stokes.petsc_options["snes_rtol"] = 1e-6

# %%
### initial linear solve
stokes.constitutive_model.Parameters.viscosity  = 1.

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity

stokes.solve(zero_init_guess=True)


# %% [markdown]
# #### add in NL viscosity for solve loop

# %%
stokes.constitutive_model.Parameters.viscosity= viscosityFn



stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.viscosity


stokes.solve(zero_init_guess=True)

# %% [markdown]
# ### Main loop
# Stokes solve loop

# %%
step      = 0
max_steps = 50
time      = 0


#timing setup
#viewer.getTimestep()
#viewer.setTimestep(1)


while step<max_steps:
    
    print(f'\nstep: {step}, time: {time}')
          
    #viz for parallel case - write the hdf5s/xdmfs 
    if step%10==0:
        if uw.mpi.rank == 0:
            print(f'\nVisualisation: ')
            
        ### updates projection of fields to the mesh
        updateFields()
        
        ### saves the mesh and swarm
        saveData(step, expt_name)
        

            
    
    if uw.mpi.rank == 0:
        print(f'\nStokes solve: ')  
        
    stokes.solve(zero_init_guess=False)
    
    ### get the timestep
    dt = stokes.estimate_dt()
 
    ### advect the particles according to the timestep
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt, corrector=False)
        
    step += 1
    
    time += dt



