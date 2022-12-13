# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Linear stokes sinker
#
# The stokes sinker is a benchmark to test sinking velocities determined by the stokes solver.
#
# Two materials are used, one for the background and one for the sinking ball.
#
# Stokes velocity is calculated and compared with the velocity from the UW model to benchmark the stokes solver
#

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI


import pyvista as pv
import vtk

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# +
#### visualisation within script
render = True

#### save output
save_output = False
# -

### number of steps for the model
nstep = 2

tol = 1e-5

expt_name = f"output/stinker_eta1e6_rho10"

res = 80

# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0.0, 0.7)

# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1

# Set constants for the viscosity of each material
viscBG     =  1.0
viscSphere = 10.0

# set density of the different materials
densityBG     =  0.0
densitySphere = 10.0

# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1] - sphereRadius

nsteps = 10

swarmGPC = 2

# +
# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(-1.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / res, regular=False
# )

mesh = uw.meshing.StructuredQuadBox(minCoords=(-1.0, 0.0), maxCoords=(1.0, 1.0),  elementRes=(res,res))
# -

# ####  Create Stokes object and the required mesh variables (velocity and pressure)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p, verbose=True)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)


### No slip BC
sol_vel = sympy.Matrix([0, 0])
stokes.add_dirichlet_bc(
    sol_vel, ["Top", "Bottom"], [0, 1]
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    sol_vel, ["Left", "Right"], [0, 1]
)  # left/right: components, function, markers


# ####  Add a particle swarm which is used to track material properties 

swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=4, proxy_continuous=True)
swarm.populate(fill_param=4)

# Create an array which contains:
# - the x and y coordinate of the sphere (0,1), 
# - the radius (2) and 
# - the material index (3)

sphere = np.array(
    [[sphereCentre[0], sphereCentre[1], sphereRadius, 1]]
    )


# Update the material variable to include the background and sphere

with swarm.access(material):
    material.data[...] = materialLightIndex

    for i in range(sphere.shape[0]):
        cx, cy, r, m = sphere[i, :]
        inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
        material.data[inside] = m

### add tracer for sinker velocity
tracer = np.zeros(shape=(1, 2))
tracer[:, 0], tracer[:, 1] = x_pos, y_pos

# +
### assign the density of each material

mat_density = np.array([densityBG, densitySphere])

density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]



# +
### assign material viscosity

mat_viscosity = np.array([viscBG, viscSphere])

viscosityMat = mat_viscosity[0] * material.sym[0] + mat_viscosity[1] * material.sym[1]

viscosity = viscosityMat
# -
pv.global_theme.background = "white"
pv.global_theme.window_size = [750, 750]
pv.global_theme.antialiasing = True
pv.global_theme.jupyter_backend = "panel"
pv.global_theme.smooth_shading = True
pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

pl = pv.Plotter(notebook=True)


def plot_T_mesh(filename):

    if not render:
        return

    mesh.vtk("tmpMsh.vtk")
    pvmesh = pv.read("tmpMsh.vtk")

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        points[:, 2] = 0.0

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    ## Plotting into existing pl (memory leak in panel code)
    pl.clear()

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.5,
    )

    pl.screenshot(
        filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False
    )


stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * density])
stokes.saddle_preconditioner = 1.0 / viscosity


stokes.petsc_options["snes_converged_reason"] = None
stokes.petsc_options["snes_rtol"] = tol
stokes.petsc_options["ksp_rtol"]  = tol

step = 0
time = 0.0
nprint = 0.0

tSinker = np.zeros(nsteps)*np.nan
ySinker = np.zeros(nsteps)*np.nan

# #### Stokes solver loop

while step < nstep:
    ### Get the position of the sinking ball
    ymin = tracer[:, 1].min()
    ySinker[step] = ymin
    tSinker[step] = time

    ### solve stokes
    stokes.solve(zero_init_guess=True)
    ### estimate dt
    dt = stokes.estimate_dt()

    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False)


    ### advect tracer
    vel_on_tracer = uw.function.evaluate(stokes.u.fn, tracer)
    tracer += dt * vel_on_tracer

    ### print some stuff
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.2f}, tracer:  {ymin:6.2f}")
        
        if save_output == True:
            plot_T_mesh(filename="{}_step_{}".format(expt_name, step))

    step += 1
    time += dt


stokesSink_vel = (((2*sphereRadius)**2)*(densitySphere - densityBG)*1)/18*viscBG
print(f'stokes sinking velocity: {stokesSink_vel}')

# #### Compare velocity from the model with numerical solution to benchmark the Stokes solver

if uw.mpi.rank==0:
    
    ### remove nan values, if any
    ySinker = ySinker[~np.isnan(ySinker)]
    tSinker = tSinker[~np.isnan(tSinker)]
    
    t_benchmark = np.arange(0, tSinker.max(), 0.1)
    v_benchmark = 0.6 - (t_benchmark*stokesSink_vel)
    
    
    print('Initial position: t = {0:.3f}, y = {1:.3f}'.format(tSinker[0], ySinker[0]))
    print('Final position:   t = {0:.3f}, y = {1:.3f}'.format(tSinker[-1], ySinker[-1]))
    
    
    velocity = (ySinker[0] - ySinker[-1]) / (tSinker[-1] - tSinker[0])
    print(f'Velocity:         v = {velocity}')

    import matplotlib.pyplot as pyplot

    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(tSinker, ySinker, label='UW sinker velocity') 
    
    ax.plot(t_benchmark, v_benchmark, ls='--', label='benchmark velocity') 
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Sinker position')

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 250]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    mesh.vtk("tmpMsh.vtk")
    pvmesh = pv.read("tmpMsh.vtk")

    # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

    with mesh.access():
        vsol = v.data.copy()

    with swarm.access():
        points = np.zeros((swarm.data.shape[0], 3))
        points[:, 0] = swarm.data[:, 0]
        points[:, 1] = swarm.data[:, 1]
        points[:, 2] = 0.0

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = vsol[...]

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Black", "wireframe")

    pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="rho",
    #                 use_transparency=False, opacity=0.95)

    # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
    #               use_transparency=False, opacity=0.5)

    pl.add_mesh(
        point_cloud,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=False,
        scalars="M",
        use_transparency=False,
        opacity=0.95,
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=5.0, opacity=0.5)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


