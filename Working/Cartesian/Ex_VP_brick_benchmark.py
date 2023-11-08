# %% [markdown]
# # The brick benchmark
#
# As outlined in [Kaus, 2010](http://jupiter.ethz.ch/~kausb/k10.pdf) and [Glerum et al., 2018](https://se.copernicus.org/articles/9/267/2018/se-9-267-2018.pdf)
#
# [UWGeodynamics (UW2) version](https://github.com/underworldcode/underworld2/blob/master/docs/UWGeodynamics/benchmarks/Kaus_BrickBenchmark-Compression.ipynb)

# %%
import underworld3 as uw
import numpy as np
import sympy

import os

# %%
### plot figs
if uw.mpi.size == 1:
    render = True
else:
    render = False


# %%
outputPath = f'./output/brickBenchmark/'


if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outputPath):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outputPath)

# %% [markdown]
# #### Set up scaling of model

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

velocity     = 2e-11 * u.meter / u.second
model_height = 10. * u.kilometer
bodyforce    = 2700 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
mu           = 1e20 * u.pascal * u.second

KL = model_height
Kt = KL / velocity
# KM = bodyforce * KL**2 * Kt**2
KM = mu * KL * Kt


scaling_coefficients  = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM

scaling_coefficients

# %%
nd_gravity = nd( 9.81 * u.meter / u.second**2 )

# %%
### Key model parameters
nd_C = nd(40 *u.megapascal)

# %% [markdown]
# Set up dimensions of model and sinking block

# %%
xmin, xmax = 0., ndim(40*u.kilometer)
ymin, ymax = 0., ndim(10*u.kilometer)

## set brick height and length
BrickHeight = nd(625.*u.meter)
BrickLength = nd(1250.*u.meter)

### set the res in x and y
resx = 128
resy = 32

### add material index
BrickIndex = 0
BGIndex    = 1


# %%
if uw.mpi.rank == 0:
    print(f'')
    print(f'resx: {resx}')
    print(f'resy: {resy}')
    print(f'proc: {uw.mpi.size}')
    print(f'')

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / resy, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(resx),int(resy)),
                                    minCoords=(xmin,ymin), 
                                    maxCoords=(xmax,ymax))


# %% [markdown]
# ### Create Stokes object

# %%
v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
# p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1,  continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=v.degree)
dev_stress_inv2 = uw.discretisation.MeshVariable("stress", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("viscosity", mesh, 1, degree=1)

rank = uw.discretisation.MeshVariable("rank", mesh, 1, degree=1)

# materialField  = uw.discretisation.MeshVariable("material", mesh, 1, degree=1)


# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# %% [markdown]
# #### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)

material  = uw.swarm.IndexSwarmVariable("material", swarm, indices=2, proxy_degree=1)



swarm.populate(fill_param=2)

# %%
for i in [material, ]:
        with swarm.access(i):
            i.data[:] = BGIndex
            i.data[(swarm.data[:,1] <= BrickHeight) & 
                  (swarm.data[:,0] >= (((xmax - xmin) / 2.) - (BrickLength / 2.)) ) & 
                  (swarm.data[:,0] <= (((xmax - xmin) / 2.) + (BrickLength / 2.)) )] = BrickIndex


# %%
with mesh.access(rank):
    rank.data[:,] = uw.mpi.rank

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

    # pv.start_xvfb()

    mesh.vtk(outputPath + "stokes_brick_VP_mesh.vtk")
    pvmesh = pv.read(outputPath + "stokes_brick_VP_mesh.vtk")

    # with mesh1.access():
        # usol = stokes.u.data.copy()

    # pvmesh.point_data["T"] = uw.function.evaluate(T_soln.sym[0], meshball.data)

    pl = pv.Plotter(window_size=(750, 750))

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_mesh(
    #     pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    # )

    with swarm.access('M'):
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
def updateFields():

    nodal_strain_rate_inv2.solve()

    

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


    pvmesh = pv.read(outputPath + "stokes_brick_VP_mesh.vtk")

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


# %%
def plot_field(field):

    import numpy as np
    import pyvista as pv
    import vtk

    updateFields()

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    pvmesh = pv.read(outputPath + "stokes_brick_VP_mesh.vtk")

    with mesh.access(node_viscosity, strain_rate_inv2):
        pvmesh.point_data["edot"] = strain_rate_inv2.rbf_interpolate(mesh.data, nnn=1)
        pvmesh.point_data["eta"]  = node_viscosity.rbf_interpolate(mesh.data, nnn=1)

    

        



    pl = pv.Plotter(notebook=True)

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh,
        cmap="RdYlGn",
        scalars=field,
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        # clim=[0.1, 1.5],
        opacity=1.0,
        log_scale=True
    )





    pl.show(cpos="xy")

# %% [markdown]
# #### Boundary conditions

# %%
# Velocity boundary conditions
vel = nd(2e-11 * u.meter / u.second)

stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((vel,0.0), "Left")
stokes.add_dirichlet_bc((-vel,0.0), "Right")
# stokes.add_dirichlet_bc((sympy.oo,sympy.oo), "Top")


# %% [markdown]
# #### Visualise swarm and passive tracers

# %%
if render == True & uw.mpi.size==1:
    plot_mat()

# %% [markdown]
# #### Set up density and viscosity of materials

# %%
### set density of materials
nd_gravity = nd(9.81*u.meter/u.second**2)
nd_density = nd(2.7e3*u.kilogram/u.meter**3)

# %%
stokes.bodyforce = sympy.Matrix([0, -1*nd_gravity*nd_density])

# %% [markdown]
# ### set some petsc options

# %%
# Set solve options here (or remove default values
# stokes.petsc_options["ksp_monitor"] = None

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-10

stokes.petsc_options["snes_max_it"] = 500

### see the SNES output
stokes.petsc_options["snes_converged_reason"] = None


# stokes.petsc_options["snes_atol"] = 1e-6
# stokes.petsc_options["snes_rtol"] = 1e-6

# %% [markdown]
# ### Initial linear solve
# - Save the data

# %%
### linear solve

viscosity_L_fn = material.createMask([nd(1e20*u.pascal*u.second),
                                      nd(1e25*u.pascal*u.second)])




stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_L_fn
# stokes.saddle_preconditioner = 1.0 / viscosity_L_fn
stokes.solve()

# %%
updateFields()
mesh.petsc_save_checkpoint(index=0, meshVars=[strain_rate_inv2, node_viscosity, p, v], outputPath=outputPath)

# %% [markdown]
# #### Introduce NL viscosity
# Test a range of friction angles (0 to 30)

# %%
nd_lithoP = nd_density * nd_gravity * (ymax-mesh.X[1])

step = 1

for phi in [0, 5, 10, 15, 20, 25, 30]:

    ### initial linear solve
    viscosity_L_fn = material.createMask([nd(1e20*u.pascal*u.second),
                                          nd(1e25*u.pascal*u.second)])
    
    
    
    
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_L_fn
    stokes.solve(zero_init_guess=True)

    

    ### add in the plasticity
    fc = np.arctan(np.radians(phi))


    tau_y_dd_vm = ( nd_C * np.cos(fc) ) + ( np.sin(fc) * nd_lithoP )




    viscosity_Y = (tau_y_dd_vm / (2 * stokes.Unknowns.Einv2 + 1.0e-18))
    
    
    # nl_visc_bg = sympy.Min(viscosity_Y, nd(1e25*u.pascal*u.second)) #, nd(1e21*u.pascal*u.second))
    
    nl_visc_bg = nd(1e19*u.pascal*u.second) + (1/((1/viscosity_Y)  + (1/nd(1e25*u.pascal*u.second)))) # sympy.Max( nd(1e19*u.pascal*u.second) , sympy.Min( nd(1e25*u.pascal*u.second) , viscosity_Y) ) # 1/ ( ( 1/ nd(1e25*u.pascal*u.second ) ) + ( 1/viscosity_Y ) ) ) )


    ### Set constants for the viscosity and density of the sinker.
    visc_brick        = nd(1e20*u.pascal*u.second)
    
    
    visc_fn = material.createMask([visc_brick,
                                    nl_visc_bg])


    stokes.constitutive_model.Parameters.shear_viscosity_0 = visc_fn
    

        
    stokes.solve(zero_init_guess=False)


    updateFields()
    mesh.petsc_save_checkpoint(index=step, meshVars=[strain_rate_inv2, node_viscosity, p, v], outputPath=outputPath)


    step += 1


# %% [markdown]
# #### Check the results against the benchmark 

# %%
x = np.linspace(xmin+1e-6, xmax-1e-6, 5001)
y0 = np.repeat(0.1, x.shape[0])
y1 = np.repeat(0.15, x.shape[0])

profile0 = np.vstack([x, y0]).T

profile1 = np.vstack([x,y1]).T


# %%
SR_profile0 = uw.function.evaluate(strain_rate_inv2.sym, profile0)

SR_profile1 = uw.function.evaluate(strain_rate_inv2.sym, profile1)


# %%
import scipy

peaks0, _ = scipy.signal.find_peaks(SR_profile0[:,0], height=1)
peaks1, _ = scipy.signal.find_peaks(SR_profile1[:,0], height=1)

# %%
dx0 = (x[peaks0][0] - x[peaks1][0])
dx1 = (x[peaks1][1] - x[peaks0][1])

### get average of shear zones on each side
dx = (dx0 + dx1) / 2

dy = (y1[0]  - y0[0])

shear_angle0 = np.rad2deg( np.arctan(dx0/dy) )

shear_angle1 = np.rad2deg( np.arctan(dx1/dy) ) 

print(f'shear angle 0: {shear_angle0} degrees')
print(f'shear angle 1: {shear_angle1} degrees')

# %%
import matplotlib.pyplot as plt
plt.plot(x, SR_profile0, c='blue')
plt.plot(x[peaks0], SR_profile0[peaks0], "x", c='blue')

plt.plot(x, SR_profile1, c='red')
plt.plot(x[peaks1], SR_profile1[peaks1], "x", c='red')

# %%

# %%

# %%

# %%
