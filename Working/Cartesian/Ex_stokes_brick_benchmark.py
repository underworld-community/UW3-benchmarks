# %% [markdown]
# # The brick benchmark
#
# Brick compression benchark, as outlined in [Kaus, 2010](http://jupiter.ethz.ch/~kausb/k10.pdf) and [Glerum et al., 2018](https://se.copernicus.org/articles/9/267/2018/se-9-267-2018.pdf)

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import os

# %%
options = PETSc.Options()


options["snes_converged_reason"] = None
options["snes_monitor_short"] = None

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# %%
### plot figs
if uw.mpi.size == 1:
    render = True
else:
    render = False
    
    
### linear or nonlinear version
linear = False ### False for NL version

# %%
### set pyvista options
if render == True & uw.mpi.size==1:
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.anti_aliasing = 'ssaa'
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

# %%
## swarm gauss point count (particle distribution)
swarmGPC = 2

# %%
outputPath = './output/swarmTest/'


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
ndim = uw.scaling.non_dimensionalise
nd   = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# %%
velocity     = 2e-11 * u.meter / u.second
model_height = 10. * u.kilometer
bodyforce    = 2700 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
mu           = 1e22 * u.pascal * u.second

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
### Key ND values
ND_gravity     = nd( 9.81 * u.meter / u.second**2 )

# %%
nd(3000  * u.kilogram / u.metre**3) * ND_gravity

# %%
nd(1e20 *u.pascal *u.second )

# %%
### add material index
BGIndex    = 0
BrickIndex = 1

# %% [markdown]
# Set up dimensions of model and brick

# %%
xmin, xmax = 0., ndim(40*u.kilometer)
ymin, ymax = 0., ndim(10*u.kilometer)

## set brick height and length
BrickHeight = nd(400.*u.meter)
BrickLength = nd(800.*u.meter)

# %%
resx = 120
resy =  30

# %%
vel = ndim(2e-11 * u.meter / u.second)

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(resx),int(resy)),
                                    minCoords=(xmin,ymin), 
                                    maxCoords=(xmax,ymax))


# %%
v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
# p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1,  continuous=True)

strain_rate_inv2 = uw.discretisation.MeshVariable("SR", mesh, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("stress", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("viscosity", mesh, 1, degree=1)

timeField      = uw.discretisation.MeshVariable("time", mesh, 1, degree=1)
materialField  = uw.discretisation.MeshVariable("material", mesh, 1, degree=1)

# %% [markdown]
# ### Create Stokes object

# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )
stokes.constitutive_model = uw.systems.constitutive_models.ViscoPlasticFlowModel(mesh.dim)

# %% [markdown]
# #### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)

# material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=False, proxy_degree=0)
material  = uw.swarm.IndexSwarmVariable("material", swarm, indices=2)

materialVariable      = swarm.add_variable(name="materialVariable", num_components=1, dtype=PETSc.IntType)


swarm.populate(fill_param=swarmGPC)

# %%
for i in [material, materialVariable]:
        with swarm.access(i):
            i.data[:] = BGIndex
            i.data[(swarm.data[:,1] <= BrickHeight) & 
                  (swarm.data[:,0] >= (((xmax - xmin) / 2.) - (BrickLength / 2.)) ) & 
                  (swarm.data[:,0] <= (((xmax - xmin) / 2.) + (BrickLength / 2.)) )] = BrickIndex



# %% [markdown]
# #### Additional files to save

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
nodal_strain_rate_inv2.smoothing = 0.
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.smoothing = 0.
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


nodal_tau_inv2 = uw.systems.Projection(mesh, dev_stress_inv2)
nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
nodal_tau_inv2.smoothing = 0.
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")

matProj = uw.systems.Projection(mesh, materialField)
matProj.uw_function = materialVariable.sym[0]
matProj.smoothing = 0.
matProj.petsc_options.delValue("ksp_monitor")


# %%
def updateFields(time):
    
    with mesh.access(timeField):
        timeField.data[:,0] = dim(time, u.megayear).m

    nodal_strain_rate_inv2.solve()

    
    matProj.uw_function = materialVariable.sym[0] 
    matProj.solve(_force_setup=True)


    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    nodal_visc_calc.solve(_force_setup=True)

    nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
    nodal_tau_inv2.solve(_force_setup=True)

# %% [markdown]
# #### Boundary conditions

# %%
stokes.add_dirichlet_bc(sympy.Matrix([   vel,0.]), "Left", [0,1])
stokes.add_dirichlet_bc(sympy.Matrix([-1*vel,0.]), "Right", [0,1])


stokes.add_dirichlet_bc(0.0, "Bottom", 1)

# %% [markdown]
# #### Set up density of materials

# %%
### set density of materials
densityBG      = nd(2700 * u.kilogram / u.metre**3 )
densityBrick   = nd(2700 * u.kilogram / u.metre**3 )

# %%
mat_density = np.array([densityBG, densityBrick])

density = mat_density[0] * material.sym[0] + \
          mat_density[1] * material.sym[1]

stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density])


# %% [markdown]
# ### Create figure function

# %%
def plot_mat():

    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)
    
    # ### create point cloud for passive tracers
    # with passiveSwarm.access():
    #     passiveCloud = pv.PolyData(np.vstack((passiveSwarm.data[:,0],passiveSwarm.data[:,1], np.zeros(len(passiveSwarm.data)))).T)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
        



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)
    
    # ### add points of passive tracers
    # pl.add_mesh(passiveCloud, color='black', show_edges=True,
    #                 use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
    
if render == True & uw.mpi.size==1:
    
    plot_mat()

# %% [markdown]
# ### Initial linear solve
# viscosity is limited between 10$^{20}$ and 10$^{24}$ Pa S

# %%
minVisc = nd(1e20 *u.pascal*u.second)

maxVisc = nd(1e24 *u.pascal*u.second)


stokes.constitutive_model.Parameters.min_viscosity = minVisc

stokes.constitutive_model.Parameters.max_viscosity =  maxVisc

# %%
### linear solve
stokes.constitutive_model.Parameters.bg_viscosity = nd(1e22 *u.pascal*u.second)


# %%
stokes.petsc_options["snes_atol"] = 1.0e-7
stokes.petsc_options["ksp_atol"]  = 1.0e-7

stokes.tolerance = 1.0e-7

# %%
stokes.solve(zero_init_guess=True)

# %% [markdown]
# #### Linear solve with different viscosities

# %%
# ### linear viscosity

# viscosityL = np.array([maxVisc, minVisc])

# viscosityL   = viscosityL[0] * material.sym[0] + \
#                viscosityL[1] * material.sym[1]  


# stokes.constitutive_model.Parameters.bg_viscosity = viscosityL

# stokes.solve(zero_init_guess=False)



# %% [markdown]
# #### Solve for NL BG material

# %% [markdown]
# ###### This is a NL visco-platic material

# %%
### Set the viscosity of the brick
viscBrick        = nd(1e20 *u.pascal*u.second)

C = nd(40e6 *u.pascal)

mat_viscosity_bg = np.array([maxVisc, viscBrick])

viscosityMat_bg = mat_viscosity_bg[0] * material.sym[0] + \
                  mat_viscosity_bg[1] * material.sym[1] 

mat_viscosity_ys = np.array([C, 0.])

viscosityMat_ys = mat_viscosity_ys[0] * material.sym[0] + \
                  mat_viscosity_ys[1] * material.sym[1] 


stokes.constitutive_model.Parameters.bg_viscosity = viscosityMat_bg

stokes.constitutive_model.Parameters.yield_stress = viscosityMat_ys

stokes.solve(zero_init_guess=False)
dt = stokes.estimate_dt()

# %% [markdown]
# #### Check the results against the benchmark 

# %%
updateFields(0)

if render == True & uw.mpi.size==1:

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]


    pvmesh.point_data["pres"] = uw.function.evaluate(
        p.sym[0], mesh.data
    )
    


    # pvmesh.point_data["edot"] = uw.function.evaluate(strain_rate_inv2.sym[0], mesh.data)
    # # pvmesh.point_data["tauy"] = uw.function.evaluate(tau_y, mesh.data, mesh.N)
    # pvmesh.point_data["eta"] = uw.function.evaluate(node_viscosity.sym[0], mesh.data)
    # pvmesh.point_data["str"] = uw.function.evaluate(dev_stress_inv2.sym[0], mesh.data)
    
    with mesh.access():
        pvmesh.point_data["edot"] = strain_rate_inv2.data
        pvmesh.point_data["eta"]  = node_viscosity.data
        pvmesh.point_data["str"]  = dev_stress_inv2.data

    with mesh.access():
        usol = v.data.copy()

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    point_cloud0 = pv.PolyData(points)

    with swarm.access():
        point_cloud = pv.PolyData( np.zeros((swarm.data.shape[0], 3))  )
        point_cloud.points[:,0:2] = swarm.data[:]
        point_cloud.point_data["M"] = material.data.copy()
        point_cloud.point_data["edot"] = uw.function.evaluate(strain_rate_inv2.sym[0], swarm.data)



# %%
if uw.mpi.size==1 and render == True:
    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.0001, opacity=0.75)

    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     scalars="edot",
    #     edge_color="Grey",
    #     show_edges=True,
    #     use_transparency=False,
    #     log_scale=True,
    #     # clim=[0.1,2.1],
    #     opacity=1.0,
    # )
    
    # pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="edot",
    #                     use_transparency=False, opacity=0.1, log_scale=True)

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        scalars="edot",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.3,
        log_scale=True
    )

    pl.show(cpos="xy")
