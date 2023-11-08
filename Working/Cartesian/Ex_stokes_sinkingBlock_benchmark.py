# %% [markdown]
# # Stokes sinker - sinking block
#
# Sinking block benchmark as outlined in:
#
# - [Gerya, T.V. and Yuen, D.A., 2003. Characteristics-based marker-in-cell method with conservative finite-differences schemes for modeling geological flows with strongly variable transport properties. Physics of the Earth and Planetary Interiors, 140(4), pp.293-318.](http://jupiter.ethz.ch/~tgerya/reprints/2003_PEPI_method.pdf)
#
# -  Value to change : **viscBlock**, **dRho**
#
# - utilises the UW scaling module to convert from dimensional to non-dimensional values
#
# - [ASPECT benchmark](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/sinking_block/doc/sinking_block.html)

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
options = PETSc.Options()


# %%
# Set the resolution, a structured quad box of 51x51 is used in the paper
res = 101

### plot figs
if uw.mpi.size == 1:
    render = True
else:
    render = False

# %%
## swarm gauss point count (particle distribution)
swarmGPC = 3

# %% [markdown]
# #### Value to change:

# %%
# viscosity of block in Pa s (the value is ND later), reference is 1e21
viscBlock = 1e21
dRho      = 100

# %% [markdown]
# #### Set up scaling of model

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# %%
### set reference values
model_height = 500.  * u.kilometer
velocity     = 1e-11 * u.meter / u.second
bodyforce    = 3200  * u.kilogram / u.metre**3 * 10 * u.meter / u.second**2
mu           = 1e21  * u.pascal * u.second

KL = model_height
Kt = KL / velocity
# KM = bodyforce * KL**2 * Kt**2
KM = mu * KL * Kt



scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients

# %%
### Key ND values
ND_gravity     = ndim(9.81*u.meter/u.second**2)

# %%
BGIndex = 0
BlockIndex = 1

# %%
## Set constants for the viscosity and density of the sinker.
viscBG        = ndim(1e21*u.pascal*u.second)
viscBlock     = ndim(viscBlock*u.pascal*u.second)



## set density of blocks
densityBG     = ndim(3.2e3 *u.kilogram/u.meter**3)
densityBlock  = ndim((3.2e3 + dRho) *u.kilogram/u.meter**3)

# %% [markdown]
# Set up dimensions of model and sinking block

# %%
xmin, xmax = 0., ndim(500*u.kilometer)
ymin, ymax = 0., ndim(500*u.kilometer)

# %%
# Set the box min and max coords
boxCentre_x, boxCentre_y = ndim(250.0*u.kilometer), ndim(400.0*u.kilometer)

box_xmin, box_xmax = boxCentre_x-ndim(50*u.kilometer), boxCentre_x+ndim(50*u.kilometer)
box_ymin, box_ymax = boxCentre_y-ndim(50*u.kilometer), boxCentre_y+ndim(50*u.kilometer)

# location of tracer at bottom of sinker
x_pos = box_xmax - ((box_xmax - box_xmin)/2.)
y_pos = box_ymin

### add a tracer
tracer = np.zeros(shape=(1,2))
tracer[:,0] = boxCentre_x
tracer[:,1] = boxCentre_y

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(res),int(res)),
                                    minCoords=(xmin,ymin), 
                                    maxCoords=(xmax,ymax),
                                    qdegree=3)


# %% [markdown]
# ### Create Stokes object

# %%
v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=3 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=2 )

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

# %%
sol_vel = sympy.Matrix([0.,0.])

### free slip
# No slip left & right & free slip top & bottom
stokes.add_dirichlet_bc( sol_vel, "Left",  [0] )  # left/right: components, function, markers
stokes.add_dirichlet_bc( sol_vel, "Right",  [0] )
stokes.add_dirichlet_bc( sol_vel, "Top",  [1] )  # top/bottom: components, function, markers 
stokes.add_dirichlet_bc( sol_vel, "Bottom",  [1] )




# %% [markdown]
# ### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)
material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=swarmGPC)

# %%
with swarm.access(material):
    material.data[:] = BGIndex
    material.data[(swarm.data[:,0] >= box_xmin) & (swarm.data[:,0] <= box_xmax) &
                  (swarm.data[:,1] >= box_ymin) & (swarm.data[:,1] <= box_ymax)] = BlockIndex


# %% [markdown]
# ### Add a passive tracer(s)

# %%
passiveSwarm = uw.swarm.Swarm(mesh)

passiveSwarm.add_particles_with_coordinates(tracer)

# %% [markdown]
# #### Set up density and viscosity of materials

# %%
density_fn = material.createMask([densityBG, densityBlock])

# %%
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density_fn])

# %%
viscosity_fn = material.createMask([viscBG, viscBlock])

# %%
### add in material-based viscosity
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn


# %% [markdown]
# ### Create figure function

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


    # mesh.vtk("tempMsh.vtk")
    # pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)
    
    ### create point cloud for passive tracers
    with passiveSwarm.access():
        passiveCloud = pv.PolyData(np.vstack((passiveSwarm.data[:,0],passiveSwarm.data[:,1], np.zeros(len(passiveSwarm.data)))).T)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
        



    pl = pv.Plotter(notebook=True)

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)
    
    ### add points of passive tracers
    pl.add_mesh(passiveCloud, color='black', show_edges=True,
                    use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
    
if render == True & uw.mpi.size==1:
    plot_mat()

# %% [markdown]
# ### Initial linear solve

# %%
if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'

stokes.tolerance = 1.0e-5

# %%
stokes.solve()

# %%

stokesSink_vel = (2/9)*((densityBlock-densityBG)*(ndim(50*u.kilometer)**2)*ndim(9.81*u.meter/u.second**2)/viscBG)

stokesSink_vel_dim = dim(-1*stokesSink_vel, u.meter/u.second).m



# %%
vel = dim(uw.function.evalf(v.sym[1], tracer)[0], u.meter/u.second).m

# %%
vel, stokesSink_vel_dim

# %%
if not np.isclose(vel, stokesSink_vel_dim):
    raise RuntimeError('Analytical and numerical solution not close')

# %%
