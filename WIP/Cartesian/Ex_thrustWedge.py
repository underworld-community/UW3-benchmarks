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

import sympy


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

velocity     = 1 * u.centimeter / u.year
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
xmin, xmax = 0, nd(128. * u.kilometer)
ymin, ymax = nd(-7 * u.kilometer), nd(9 * u.kilometer)

# yres = 24
# xres = yres * (round((xmax-xmin)/(ymax - ymin)))

xres = 128
yres = 64

# %% [markdown]
# ### Create mesh and mesh vars

# %%
mesh = uw.meshing.StructuredQuadBox(elementRes=(    xres,yres), 
                    minCoords =(xmin, ymin), 
                    maxCoords =(xmax, ymax) )

# mesh = uw.meshing.UnstructuredSimplexBox( 
#                     minCoords =(xmin, ymin), 
#                     maxCoords =(xmax, ymax),
#                     cellSize=0.02)


v       = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p       = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

lithoP  = uw.discretisation.MeshVariable("P_l", mesh, 1, degree=2)
vlithoP = uw.discretisation.MeshVariable("V_l", mesh, mesh.dim, degree=2)

### strain rate and velocity should have same degree for projection
strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=v.degree)
node_viscosity   = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1)
materialField    = uw.discretisation.MeshVariable("Mat", mesh, 1, degree=1)
density_proj     = uw.discretisation.MeshVariable("rho", mesh, 1, degree=2)



# %% [markdown]
# ### Stokes solver

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p,)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)

# %% [markdown]
# #### Solver for lithostatic pressure calc

# %%
lithoP_solver = uw.systems.SteadyStateDarcy(mesh, lithoP, vlithoP, solver_name='lithoP_solver')
lithoP_solver.constitutive_model = uw.systems.constitutive_models.DiffusionModel(lithoP)
lithoP_solver.constitutive_model.Parameters.diffusivity = 1e-6
lithoP_solver.petsc_options.delValue("ksp_monitor")


def determine_lithostatic_pressure(bodyforce, solver=lithoP_solver): 
    solver.f = 0.
    solver.s = bodyforce
    solver.add_dirichlet_bc(0.0, "Top")
    solver.solve(zero_init_guess=True)
    # with mesh.access(lithoP):
    #     lithoP.data[lithoP.data <= nd(1*u.atmosphere)] = nd(1*u.atmosphere)


# %% [markdown]
# ### Create swarm and swarm vars
# - 'swarm.add_variable' is a traditional swarm, can't be used to map material properties. Can be used for sympy operations, similar to mesh vars.
# - 'uw.swarm.IndexSwarmVariable', creates a mask for each material and can be used to map material properties. Can't be used for sympy operations.
#

# %%
swarm  = uw.swarm.Swarm(mesh)

# %%
## # Add variable for material
materialVariable      = swarm.add_variable(name="material", size=1, dtype=PETSc.IntType, proxy_degree=1)

material              = uw.swarm.IndexSwarmVariable("M", swarm, indices=5, proxy_degree=1) 

strain                = swarm.add_variable(name="strain", size=1, dtype=PETSc.RealType, proxy_degree=1)

swarm.populate(4)

# # Add some randomness to the particle distribution
# import numpy as np
# np.random.seed(0)

# with swarm.access(swarm.particle_coordinates):
#     factor = 0.5*boxLength/n_els/ppcell
#     swarm.particle_coordinates.data[:] += factor*np.random.rand(*swarm.particle_coordinates.data.shape)
      


# %% [markdown]
# ##### passive swarm to track the surface

# %%
surfaceSwarm = uw.swarm.Swarm(mesh)

# %%
x = np.linspace(mesh.data[:,0].min(), mesh.data[:,0].max(), 500 )
y = np.zeros_like(x)

# %%
surface_coords = np.ascontiguousarray(np.array([x,y]).T)

# %%
surfaceSwarm.add_particles_with_coordinates(surface_coords)

# %%
from scipy.interpolate import griddata, interp1d

# %% [markdown]
# #### Project fields to mesh vars
# Useful for visualising stuff on the mesh (Viscosity, material, strain rate etc) and saving to a grouped xdmf file


# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
# nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
# nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")

meshMat = uw.systems.Projection(mesh, materialField)
meshMat.uw_function = materialVariable.sym[0]
# meshMat.smoothing = 1.0e-3
meshMat.petsc_options.delValue("ksp_monitor")

nodal_rho_calc = uw.systems.Projection(mesh, density_proj)
density_fn = 1
nodal_rho_calc.uw_function = density_fn
nodal_rho_calc.smoothing = 0.
nodal_rho_calc.petsc_options.delValue("ksp_monitor")

def update_density():
    nodal_rho_calc.uw_function = density_fn
    nodal_rho_calc.solve(_force_setup=True)

def updateSR():
    ### update strain rate
    nodal_strain_rate_inv2.uw_function = stokes._Einv2
    nodal_strain_rate_inv2.solve()

def updateFields():
    updateSR()
    
    update_density()

    ### update viscosity
    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    nodal_visc_calc.solve(_force_setup=True)
    
    ### update material field from swarm
    meshMat.uw_function = materialVariable.sym[0] 
    meshMat.solve(_force_setup=True)



# %%
def update_strain(dt, strain_var, healingRate=0.):
    
    updateSR()
    
    with swarm.access(strain):
        ### rbf interpolate is quicker, does not produce negative results.
        ### how does this work in parallel?
        SR_swarm = strain_rate_inv2.rbf_interpolate(strain_var.swarm.data)[:,0] 
        
        ### function evaluate (projection) produces negative SR results
        # SR_swarm = uw.function.evalf(strain_rate_inv2.sym[0], strain_var.swarm.data)
        
        #### dt / SR 
        ### add the strain into the model
        strain_var.data[:,0] += (dt * SR_swarm)
        ### heal the strain at a given rate
        strain_var.data[:,0] -= (dt * healingRate)
        ### make sure the healing does not go below zero
        strain_var.data[strain_var.data < 0] = 0.

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

# %% [markdown]
# ###### Create a layered material

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
    
    
    
    materialVariable.data[swarm.data[:,1] <= ymin+nd(1*u.kilometer)]   = fritionalBase
    materialVariable.data[swarm.data[:,1] <= ymin+nd(0.5*u.kilometer)] = ridgidBase
    
    material.data[:,0] = materialVariable.data[:,0]
    
    
        
    
    
    
    

# %%
def plot_mat():

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    # pv.global_theme.window_size = [750, 750]
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
# #### Density

# %%
nd_gravity = nd(9.81*u.meter/u.second**2)

nd_air_density = nd(1. * u.kilogram / u.metre**3)

nd_rock_density = nd(2700 * u.kilogram / u.metre**3)

density_fn = material.createMask([nd_air_density, nd_rock_density, nd_rock_density, nd_rock_density, nd_rock_density])




stokes.bodyforce =  sympy.Matrix([0, -1 * nd_gravity*density_fn])

# %%
stokes.bodyforce


# %%
def determine_lithostatic_pressure(dt):
    update_density()
    if uw.mpi.rank == 0:
        print('calculating lithostatic pressure')

    ### get original coords
    with surfaceSwarm.access(surfaceSwarm.particle_coordinates):
        x_old = uw.utilities.gather_data(surfaceSwarm.particle_coordinates.data[:,0], bcast=True)
        y_old = uw.utilities.gather_data(surfaceSwarm.particle_coordinates.data[:,1], bcast=True)

    ### advect surface
    with surfaceSwarm.access():
        v_x = uw.utilities.gather_data(uw.function.evalf(v.sym[0], surfaceSwarm.data), bcast=True)
        v_y = uw.utilities.gather_data(uw.function.evalf(v.sym[1], surfaceSwarm.data), bcast=True)
        
    x_new = x_old + (v_x*dt)
    y_new = y_old + (v_y*dt)

    ### create an interpolated surface from the new coordinates
    f = interp1d(x_new, y_new, kind='cubic', fill_value='extrapolate')
    
    ### update the y coords only of the passive tracers
    with surfaceSwarm.access(surfaceSwarm.particle_coordinates):
        surfaceSwarm.particle_coordinates.data[:,1] = f(surfaceSwarm.particle_coordinates.data[:,0])

    ### calculate the density based on rho*g*h
    with mesh.access(lithoP, density_proj):
        rho = density_proj.data[:,0]
        g = nd_gravity
        h = f(lithoP.coords[:,0]) - lithoP.coords[:,1] ### dh between point and surface
        lithoP.data[:,0] = rho  * g * h ### rho * g * h
        lithoP.data[lithoP.data < 0] = nd(1*u.atmosphere)


# %%
determine_lithostatic_pressure(dt=0)

# %%
# determine_lithostatic_pressure(bodyforce=stokes.bodyforce, solver=lithoP_solver)

# %%
if uw.mpi.size == 1 and uw.is_notebook:
    import matplotlib.pyplot as plt
    with mesh.access(lithoP):
        scatter = plt.scatter(lithoP.coords[:,0], lithoP.coords[:,1], c=lithoP.data)
        plt.colorbar(scatter)

# %%
if uw.mpi.size == 1 and uw.is_notebook:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v.data.copy()

    pvmesh.point_data["P"] = uw.function.evalf(lithoP.sym[0], mesh.data, mesh.N)

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0] = uw.function.evalf(v.sym[0], mesh.data, mesh.N)
    v_vectors[:, 1] = uw.function.evalf(v.sym[1], mesh.data, mesh.N)
    pvmesh.point_data["V"] = v_vectors

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        max_steps=1000,
        max_time=0.2,
        initial_step_length=0.001,
        max_step_length=0.01,
    )

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="P",
        use_transparency=False,
        opacity=1.0,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=0.5, opacity=0.75)

    # pl.add_mesh(pvstream, line_width=10.0)

    pl.show(cpos="xy")

# %% [markdown]
# ### Boundary conditions
#
# Free slip by only constraining one component of velocity 

# %%
flthick = nd(0.5*u.kilometer)#(nd(-1*u.kilometer)-nd(-0.5*u.kilometer))

tapeL_top = ymin+nd(1*u.kilometer)
tapeL_bottom = ymin+nd(0.5*u.kilometer)
rigidBase_top =ymin+nd(0.5*u.kilometer)

# %%
vel_LHS_wall = sympy.Piecewise( (nd(-velocity), mesh.X[1] <= nd(rigidBase_top)),
                                (nd(-velocity)*(flthick-(mesh.X[1]-nd(tapeL_bottom)))/flthick, (mesh.X[1]  < nd(tapeL_top))),
                                (0., True)
                                )

vel_LHS_wall


# %%
# # tapeL=frictionalBasal


# conditions = [(mesh.X[1] <= nd(rigidBase_top), nd(-velocity)),
#               (mesh.X[1]  < nd(tapeL_top),
#                nd(-velocity)*(flthick-(Model.y-nd(tapeL_bottom)))/flthick),
#               (True, nd(0. * u.centimeter / u.year))]

# fn_condition = fn.branching.conditional(conditions)

# Model.set_velocityBCs(left=[vel_LHS_wall, 0.],
#                       right=[-velocity, None],
#                       top=[None, None],
#                       bottom=[-velocity, 0.])

# %%
#free slip
stokes.add_dirichlet_bc( (vel_LHS_wall, 0), 'Left',   (0, 1) ) # left/right: function, boundaries, components

stokes.add_dirichlet_bc( (-nd(velocity)), 'Right',  (0) )

# stokes.add_dirichlet_bc( (0.,0.), 'Top',    (1) )
stokes.add_dirichlet_bc( (-nd(velocity),0.), 'Bottom', (0, 1) )# top/bottom: function, boundaries, components 

# %% [markdown]
# ###### initial first guess of constant viscosity

# %%
if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.petsc_options["snes_max_it"] = 500


# %%
### initial linear solve
stokes.constitutive_model.Parameters.shear_viscosity_0  = 1.

stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.shear_viscosity_0

stokes.solve(zero_init_guess=True)


# %% [markdown]
# #### add in NL viscosity for solve loop

# %%
minVisc = nd(1e19*u.pascal*u.second)
maxVisc = nd(1e23*u.pascal*u.second)

# %%
airViscosity            = nd(1e19*u.pascal*u.second)
rigidBaseViscosity      = nd(1e23*u.pascal*u.second)
frictionalBaseVsicosity = nd(1e23*u.pascal*u.second)
materialViscosity       = nd(1e23*u.pascal*u.second)


sedCohesion = nd(20*u.megapascal)
FBCohesion = nd(0.1*u.megapascal)

sedCohesion_w = sedCohesion / 5 #nd(4*u.megapascal)
FBCohesion_w = FBCohesion / 5


# %%
def material_weakening_piecewise(strain_var, val1, val2, epsilon1, epsilon2):
    val = sympy.Piecewise(  (val1, strain_var.sym[0] < epsilon1),
                            (val2, strain_var.sym[0] > epsilon2),
                            (val1 + ((val1 - val2) / (epsilon1 - epsilon2)) * (strain_var.sym[0] - epsilon1), True) )
    
    return val 


# %%
C_sed = material_weakening_piecewise(strain, sedCohesion, sedCohesion_w, 0.01, 0.06)
C_FB  = material_weakening_piecewise(strain, FBCohesion, FBCohesion_w, 0.01, 0.06)

fc_sed = material_weakening_piecewise(strain, np.tan(np.radians(25.0)), np.tan(np.radians(20.0)), 0.01, 0.06)
fc_FB =  material_weakening_piecewise(strain, np.tan(np.radians(12.0)), np.tan(np.radians(6.0)), 0.01, 0.06)
                                                     
                                                     
tau_sed = (C_sed * sympy.cos(fc_sed))  + (lithoP.sym[0] * sympy.sin(fc_sed))                                                     
tau_FB  = (C_FB * sympy.cos(fc_FB))  + (lithoP.sym[0] * sympy.sin(fc_FB))                                                    
                                                     

                                                     

# %%
fb_visc_fn = minVisc + (1/ ((1/frictionalBaseVsicosity)  + (1/tau_FB)))

materal_visc_fn = minVisc + (1/ ((1/materialViscosity)  + (1/tau_sed)))

# %%
# airIndex       = 0
# ridgidBase     = 1
# fritionalBase  = 2
# sediment0      = 3
# sediment1      = 4


visc_fn = material.createMask([airViscosity, rigidBaseViscosity, 
                               fb_visc_fn, materal_visc_fn, 
                               materal_visc_fn])



stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn

# %%
stokes.saddle_preconditioner = 1 / stokes.constitutive_model.Parameters.shear_viscosity_0


stokes.solve(zero_init_guess=False)

# %%
stokes.tolerance = 1e-4

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
    
    print(f'\nstep: {step}, time: {dim(time, u.megayear)}')
          
    #viz for parallel case - write the hdf5s/xdmfs 
    if step%5==0:
        if uw.mpi.rank == 0:
            print(f'\nSave data: ')
            
        ### updates projection of fields to the mesh
        updateFields()
        
        ### saves the mesh and swarm
        mesh.petsc_save_checkpoint(meshVars=[v, p, materialField, strain_rate_inv2, node_viscosity, density_proj, lithoP], index=step, outputPath=expt_name)
    
        swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=expt_name) 
        surfaceSwarm.petsc_save_checkpoint(swarmName='surfaceSwarm', index=step, outputPath=expt_name) 
    
    if uw.mpi.rank == 0:
        print(f'\nStokes solve: ')  
        
    stokes.solve(zero_init_guess=False)
    
    ### get the timestep
    dt = stokes.estimate_dt()
        
    # determine_lithostatic_pressure(bodyforce=stokes.bodyforce, solver=lithoP_solver)
    determine_lithostatic_pressure(dt=dt)
    
    update_strain(dt=dt, strain_var=strain, healingRate=nd(1e-18/u.second))
 
    ### advect the particles according to the timestep
    swarm.advection(V_fn=stokes.u.sym, delta_t=dt, corrector=False, evalf=True)
        
    step += 1
    time += dt




# %%
### saves the mesh and swarm
mesh.petsc_save_checkpoint(meshVars=[v, p, materialField, strain_rate_inv2, node_viscosity, density_proj, lithoP], index=step, outputPath=expt_name)

swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=expt_name) 
surfaceSwarm.petsc_save_checkpoint(swarmName='surfaceSwarm', index=step, outputPath=expt_name) 

# %%
import matplotlib.pyplot as plt
with surfaceSwarm.access():
    plt.scatter(surfaceSwarm.data[:,0], surfaceSwarm.data[:,1])
    print(surfaceSwarm.data.shape)

# %%
