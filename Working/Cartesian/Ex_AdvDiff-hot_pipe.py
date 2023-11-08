# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Advection-diffusion of a hot pipe
#
# - Using the adv_diff solver.
# - Advection of the hot pipe vertically as it also diffuses. The velocity is 1/mesh.get_min_radius() and has a diffusivity value of 1.
# - Benchmark comparison between 1D analytical solution and 2D UW numerical model.
#
# ### How to test advection or diffusion only
# - Set velocity to 0 to test diffusion only.
# - Set diffusivity (k) to 0 to test advection only.
#
#
# ### Change benchmark_version:
# - 0 - for the mesh only version
# - 1 - for the swarm version

import underworld3 as uw
import numpy as np
import sympy
import math
import os

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


# ### Set up variables of the model

# +
res = 64


nsteps = 1


kappa = 1.0 # diffusive constant

velocity = 1/res #/res

### min and max temps
tmin = 0.5 # temp min
tmax = 1.0 # temp max

### Thickness of hot pipe in centre of box
pipe_thickness = 0.4 

Tdegree = 3
Vdegree = 1

# +
outputPath = f'./output/adv_diff-hot_pipe_example-res={res}_kappa={kappa}/'

if uw.mpi.rank == 0:
    # checking if the directory
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
# -

# ### Set up the mesh

xmin, xmax = 0, 1
ymin, ymax = 0, 1

### Quads
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=max(Tdegree, Vdegree),
)


# +
### triangles
# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1/res, regular=False )
# -

# ### Create mesh variables
# To be used in the solver

# Create an mesh vars
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=Vdegree)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=Tdegree)

# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_fn=v,
    solver_name="adv_diff",
)

# ### Set up properties of the adv_diff solver
# - Constitutive model (Diffusivity)
# - Boundary conditions
# - Internal velocity
# - Initial temperature distribution 

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = kappa

### fix temp of top and bottom walls
adv_diff.add_dirichlet_bc(tmin, "Bottom")
adv_diff.add_dirichlet_bc(tmin, "Top")
# adv_diff.add_dirichlet_bc(0., "Left")
# adv_diff.add_dirichlet_bc(0., "Right")


with mesh.access(v):
    # initialise fields
    # v.data[:,0] = -1*v.coords[:,1]
    v.data[:,1] =  velocity

with mesh.access(T):
    T.data[...] = tmin

    pipePosition = ((ymax - ymin) - pipe_thickness) / 2.0

    T.data[
        (T.coords[:, 1] >= (T.coords[:, 1].min() + pipePosition))
        & (T.coords[:, 1] <= (T.coords[:, 1].max() - pipePosition))
    ] = tmax

    T_old = np.copy(T.data[:,0])

# ### Create points to sample the UW results

# +
### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), 0.1 * mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y)  ### LHS of box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y
# -

### get the initial temp profile
T_orig = uw.function.evaluate(T.sym[0], sample_points)


# ### 1D diffusion function
# To compare UW results with a numerical results

def diffusion_1D(sample_points, T0, diffusivity, vel, time_1D):
    x = sample_points
    T = T0
    k = diffusivity
    time = time_1D

    dx = sample_points[1] - sample_points[0]

    dt_dif = (dx**2 / k)
    dt_adv = (dx/velocity)

    dt = 0.5 * min(dt_dif, dt_adv)


    if time > 0:

        """ determine number of its """
        nts = math.ceil(time / dt)
    
        """ get dt of 1D model """
        final_dt = time / nts

    
        for i in range(nts):
            qT = -k * np.diff(T) / dx
            dTdt = -np.diff(qT) / dx
            T[1:-1] += dTdt * final_dt

    

    return T


# ### Function to visualise temp field

# %%
def plot_fig():
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
        mesh.vtk(outputPath + "advDiff_mesh.vtk")
        # mesh.vtk("ignore_periodic_mesh.vtk")
        pvmesh = pv.read(outputPath + "advDiff_mesh.vtk")

        # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

        with mesh.access():
            vsol = v.data.copy()
            pvmesh["T"] = uw.function.evalf(T.sym, mesh.data)

        arrow_loc = np.zeros((v.coords.shape[0], 3))
        arrow_loc[:, 0:2] = v.coords[...]

        arrow_length = np.zeros((v.coords.shape[0], 3))
        arrow_length[:, 0:2] = vsol[...]

        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        # pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.95,
        )

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
        #               use_transparency=False, opacity=0.5)

        # pl.add_mesh(
        #     point_cloud,
        #     cmap="coolwarm",
        #     edge_color="Black",
        #     show_edges=False,
        #     scalars="M",
        #     use_transparency=False,
        #     opacity=0.95,
        # )

        pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol


plot_fig()

# ### Solver loop

step = 0
model_time = 0.0

from time import time as track_time

while step < nsteps:
    start = track_time()
    ### print some stuff
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {model_time:6.5f}\n")

    
    
    dt = adv_diff.estimate_dt()

    print(f'dt: {dt}\n')
    
    ### diffuse through underworld
    adv_diff.solve(timestep=dt)

    step += 1
    model_time += dt

    end = track_time()

    solve_time = end - start
    print(f'solve time: {solve_time}\n')

# ### Check the results

plot_fig()

with mesh.access(T):
    T_new = np.copy(T.data[:,0])

# +
"""compare 1D and 2D models"""

T_UW = uw.function.evalf(T.sym[0], sample_points)


T_1D_model = diffusion_1D(
    sample_points=sample_points[:, 1], T0=T_orig.copy(), diffusivity=kappa, vel=velocity, time_1D=model_time
)

#### 1D numerical advection
new_y = sample_points[:,1] + (velocity*model_time)

### profile from UW
plt.plot(T_UW, sample_points[:, 1], ls="-", c="red", label="UW numerical solution")
### numerical solution
plt.plot(T_1D_model, new_y, ls="-.", c="k", label="1D numerical solution")
plt.title(f'time: {round(model_time, 5)}', fontsize=8)
plt.legend(fontsize=8)
# -
# ### Check that the values are close
# Some issues due to the interp on the UW profile

if not np.allclose(T_UW, T_1D_model, rtol=1e-1):
    raise RuntimeError('Analytical and numerical solution not close')



