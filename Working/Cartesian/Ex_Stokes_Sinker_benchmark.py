# %% [markdown]
# # Linear stokes sinker
#
# The stokes sinker is a benchmark to test sinking velocities determined by the stokes solver.
#
# Two materials are used, one for the background and one for the sinking ball.
#
# Stokes velocity is calculated and compared with the velocity from the UW model to benchmark the stokes solver
#
# # Other model benchmarks:
# - [Aspect benchmarks](https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/cookbooks/stokes/doc/stokes.html)
# - [UW2 example](https://github.com/underworldcode/underworld2/blob/master/docs/examples/04_StokesSinker.ipynb)

# %%
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
import petsc4py
import pyvista as pv

import os

# %%
petsc4py.__version__

# %%
petsc4py.get_config()


# %%
### plot figs
if uw.mpi.size == 1:
    render = True
else:
    render = False


# %%
## number of steps
nsteps = 1

## swarm gauss point count (particle distribution)
swarmGPC = 2


### resolution of model
res = 128

### Recycle rate of particles
recycle_rate = 0


### stokes tolerance
stokes_tol = 1e-8

# %%
# Set constants for the viscosity of each material
viscBG     =  1
viscSphere =  10

# %%
# set density of the different materials
densityBG     =  1
densitySphere =  10

# %%
# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1

# %%
# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0., 0.7)

# %%
#### output folder name
outputPath = f"output/sinker_eta{viscSphere}_rho{densitySphere}/"

if uw.mpi.rank==0:      
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

# %% [markdown]
# ### Create mesh

# %%

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / resy, regular=False)


mesh = uw.meshing.StructuredQuadBox(minCoords=(-1.0, 0.0), maxCoords=(1.0, 1.0),  elementRes=(res,res))


# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    mesh.vtk(outputPath + "Stokes_sinker_mesh.vtk")
    pvmesh = pv.read(outputPath + "Stokes_sinker_mesh.vtk")

    # with mesh1.access():
        # usol = stokes.u.data.copy()


    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    # )

    # with swarm.access('M'):
    #     points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
    #     points[:, 0] = swarm.particle_coordinates.data[:, 0]
    #     points[:, 1] = swarm.particle_coordinates.data[:, 1]
        
    #     point_cloud = pv.PolyData(points)
        
    #     point_cloud.point_data["M"] = material.data.copy()
        
    # pl.add_points(point_cloud, render_points_as_spheres=False, point_size=5, opacity=0.5)
    
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# %% [markdown]
# ### Create Stokes object

# %%
v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
# p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1,  continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=2)
dev_stress_inv2 = uw.discretisation.MeshVariable("stress", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("viscosity", mesh, 1, degree=1)

# timeField      = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)
# materialField  = uw.discretisation.MeshVariable("material", mesh, 1, degree=1)


# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )

# %%
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# %% [markdown]
# #### Setup swarm

# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate_petsc(2)

with swarm.access(material):
    material.data[...] = materialLightIndex

    cx, cy, r, m = sphereCentre[0], sphereCentre[1], sphereRadius, materialHeavyIndex
    inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
    material.data[inside] = m





# %%

# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1]

tracer_coord = np.vstack([x_pos, y_pos]).T

tracer = uw.swarm.Swarm(mesh=mesh)
tracer.add_particles_with_coordinates(tracer_coord)

# %%
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True


    pvmesh = pv.read(outputPath + "Stokes_sinker_mesh.vtk")

    # with mesh1.access():
        # usol = stokes.u.data.copy()


    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    # )

    with swarm.access(material):
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]
        
        point_cloud = pv.PolyData(points)
        
        point_cloud.point_data["M"] = material.data.copy()
        
    pl.add_points(point_cloud, render_points_as_spheres=False, point_size=5, opacity=0.5)
    
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")

# %% [markdown]
# ##### Additional mesh vars to save
# - create projections to save variables to the mesh

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes.Unknowns.Einv2
nodal_strain_rate_inv2.smoothing = 0.
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
nodal_visc_calc.smoothing = 0.
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


nodal_tau_inv2 = uw.systems.Projection(mesh, dev_stress_inv2)
nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes.Unknowns.Einv2
nodal_tau_inv2.smoothing = 0.
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

# matProj = uw.systems.Projection(mesh, materialField)
# matProj.uw_function = materialVariable.sym[0]
# matProj.smoothing = 0.
# matProj.petsc_options.delValue("ksp_monitor")


### create function to update fields
def updateFields(time):
    
    # with mesh.access(timeField):
    #     timeField.data[:,0] = dim(time, u.megayear).m

    nodal_strain_rate_inv2.solve()

    
    # matProj.uw_function = materialVariable.sym[0] 
    # matProj.solve(_force_setup=True)


    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.shear_viscosity_0
    nodal_visc_calc.solve(_force_setup=True)

    nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.shear_viscosity_0 * stokes.Unknowns.Einv2
    nodal_tau_inv2.solve(_force_setup=True)


# %% [markdown]
# ##### Create fig function to visualise mat

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


    pvmesh = pv.read(outputPath + "Stokes_sinker_mesh.vtk")

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)
    

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
        



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)
    



    pl.show(cpos="xy")


# %% [markdown]
# #### Boundary conditions

# %%
# Freeslip left & right & top & bottom

sol_vel = sympy.Matrix([0., 0.])
# No slip left & right & free slip top & bottom
stokes.add_dirichlet_bc( sol_vel, "Left",  [0] )  # left/right: components, function, markers
stokes.add_dirichlet_bc( sol_vel, "Right",  [0] )
stokes.add_dirichlet_bc( sol_vel, "Top",  [1] )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( sol_vel, "Bottom",  [1] )





# %% [markdown]
# #### Visualise swarm and passive tracers

# %%
# if render == True & uw.mpi.size==1:
#     plot_mat()

# %%
visc_fn = material.createMask([viscBG, viscSphere])

# %%
stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn

# %% [markdown]
# #### Set up density and viscosity of materials

# %%
density_fn = material.createMask([densityBG, densitySphere])

# %%
stokes.bodyforce = sympy.Matrix([0, -1. * density_fn])

# %% [markdown]
# ### set some petsc options

# %%
# Set solve options here (or remove default values
# stokes.petsc_options["ksp_monitor"] = None

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = stokes_tol



# %% [markdown]
# #### Save mesh to h5/xdmf file
# Function to put in solver loop

# %%
def saveData(step, outputPath, time):

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    mesh.petsc_save_checkpoint(index=step, 
                               meshVars=[v, p, strain_rate_inv2,node_viscosity], 
                                outputPath=outputPath)
    
    
    #### save the swarm and selected variables
    swarm.petsc_save_checkpoint('swarm', step, outputPath)
    

    


# %%
def v_rms(mesh, v_solution): 
    # v_soln must be a variable of mesh
    v_rms = np.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

v_rms(mesh, v)

# %%
tSinker = np.zeros(nsteps)*np.nan
ySinker = np.zeros(nsteps)*np.nan

vrms    = np.zeros(nsteps)*np.nan


step = 0
time = 0.

# %% [markdown]
# ### Solver loop for multiple iterations

# %%
while step < nsteps:

    with tracer.access():
        ySinker[step] = tracer.data[:,1][0]
    
    tSinker[step] = time
    vrms[step]    = v_rms(mesh, v)

    if uw.mpi.rank == 0:
        print('\n\nstep = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}; height = {3:.3e}\n\n'
              .format(step,time,vrms[step],ySinker[step]))

    ### save loop
    if step % 5 == 0:
        if uw.mpi.rank==0:
            print(f'\n\nSave data: \n\n')
        ### update fields first
        updateFields(time = time)
        ### save mesh variables
        saveData(step=step, outputPath=outputPath, time = time)

    
    ### solve stokes 
    stokes.solve()
    ### estimate dt
    dt = stokes.estimate_dt()


    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False, evalf=True)
    
    tracer.advection(stokes.u.sym, dt, corrector=False, evalf=True)
    
        
    step+=1
    time+=dt

# %% [markdown]
# #### Check the results against the benchmark 

# %%
stokesSink_vel = (2/9)*(((densitySphere-densityBG)*(sphereRadius**2)*1)/viscBG)
stokesSink_vel

# %%
with tracer.access():
    tracer_coord = tracer.data
    
vel = uw.function.evalf(v.sym[1], tracer_coord)[0]
vel

# %%
if uw.mpi.rank==0:
    print('Initial position: t = {0:.3f}, y = {1:.3f}'.format(tSinker[0], ySinker[0]))
    print('Final position:   t = {0:.3f}, y = {1:.3f}'.format(tSinker[nsteps-1], ySinker[nsteps-1]))

    # uw.utils.matplotlib_inline()
    import matplotlib.pyplot as pyplot
    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(tSinker, ySinker, label='UW results') 
    ax.plot(tSinker, 0.6-(tSinker*stokesSink_vel), label='analytical solution', ls=':')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Sinker position')

# %%
if uw.mpi.size==1 and render == True:
    plot_mat()

# %%
if not np.isclose(-1*vel, stokesSink_vel, atol=1e-2):
    raise RuntimeError('Analytical and numerical solution not close')

# %%

# %%
