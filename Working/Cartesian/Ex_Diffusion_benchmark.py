# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Linear diffusion of a hot pipe
#
# - Using the adv_diff solver.
# - No advection as the velocity field is not updated (and set to 0).
# - Benchmark comparison between 1D analytical solution and 2D UW numerical model.
#

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


# Set the resolution.
res = 32

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

pipe_thickness = 0.4  ###

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
)


# +
# Set some values of the system
k = 1.0 # diffusive constant

tmin = 0.5 # temp min
tmax = 1.0 # temp max
# -

# Create an adv
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_Field=v,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k

# %%
### fix temp of top and bottom walls
adv_diff.add_dirichlet_bc(0.5, "Bottom")
adv_diff.add_dirichlet_bc(0.5, "Top")


# %%
maxY = mesh.data[:, 1].max()
minY = mesh.data[:, 1].min()

with mesh.access(T):
    T.data[...] = tmin

    pipePosition = ((maxY - minY) - pipe_thickness) / 2.0

    T.data[
        (mesh.data[:, 1] >= (mesh.data[:, 1].min() + pipePosition))
        & (mesh.data[:, 1] <= (mesh.data[:, 1].max() - pipePosition))
    ] = tmax


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
        mesh.vtk("tmpMsh.vtk")
        # mesh.vtk("ignore_periodic_mesh.vtk")
        pvmesh = pv.read("tmpMsh.vtk")

        # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

        with mesh.access():
            vsol = v.data.copy()
            pvmesh["T"] = T.data.copy()

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

        pl.add_arrows(arrow_loc, arrow_length, mag=5.0, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol


plot_fig()

# ##### Vertical profile across the centre of the box

### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y)  ### LHS of box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y

t0 = uw.function.evaluate(adv_diff.u.fn, sample_points)

### estimate the timestep based on diffusion only
dt = mesh.get_min_radius() ** 2 / k  ### dt = length squared / diffusivity
print(f"dt: {dt}")


# %%
def diffusion_1D(sample_points, tempProfile, k, model_dt):
    x = sample_points
    T = tempProfile

    dx = sample_points[1] - sample_points[0]

    dt = 0.5 * (dx**2 / k)

    """ max time of model """
    total_time = model_dt

    """ get min of 1D and 2D model """
    time_1DModel = min(model_dt, dt)

    """ determine number of its """
    nts = math.ceil(total_time / time_1DModel)

    """ get dt of 1D model """
    final_dt = total_time / nts

    for i in range(nts):
        qT = -k * np.diff(T) / dx
        dTdt = -np.diff(qT) / dx
        T[1:-1] += dTdt * final_dt

    return T


### get the initial temp profile
tempData = uw.function.evaluate(adv_diff.u.fn, sample_points)

step = 0
time = 0.0

nsteps = 21

while step < nsteps:
    ### print some stuff
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.5f}")

    ### 1D profile from underworld
    t1 = uw.function.evaluate(adv_diff.u.fn, sample_points)

    if uw.mpi.size == 1 and step % 10 == 0:
        """compare 1D and 2D models"""
        plt.figure()
        ### profile from UW
        plt.plot(t1, sample_points[:, 1], ls="-", c="red", label="UW numerical solution")
        ### numerical solution
        plt.plot(tempData, sample_points[:, 1], ls=":", c="k", label="1D analytical solution")
        plt.legend()
        plt.show()

    ### 1D diffusion
    tempData = diffusion_1D(
        sample_points=sample_points[:, 1], tempProfile=tempData, k=k, model_dt=dt
    )
    
    dt0 = adv_diff.estimate_dt()
    
    dt1 = mesh.get_min_radius() ** 2 / k
    
    print(dt0, dt1)
    
    dt = adv_diff.estimate_dt()
    
    ### diffuse through underworld
    adv_diff.solve(timestep=dt)

    step += 1
    time += dt

plot_fig()
