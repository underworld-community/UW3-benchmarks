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

# # Advection of a hot pipe
#
# - Using the adv_diff solver.
# - Advection of the hot pipe vertically. The velocity is 1/mesh.get_min_radius() to advect the pipe upwards.
# - Benchmark comparison between 1D analytical solution and 2D UW numerical model.
#

# +
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI

import time
# -

import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt


sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


# +
# Set the resolution.
res = 64

Tdegree = 3
Vdegree = 2
# -

mesh_qdegree = max(Tdegree, Vdegree)
mesh_qdegree

xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

pipe_thickness = 0.4 

# +
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=mesh_qdegree)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
#                                          , cellSize=1/res, qdegree=2)
                                    
# -


velocity = (1/mesh.get_min_radius())

# +
# Set some values of the system
k = 0.0 # diffusive constant

tmin = 0.5 # temp min
tmax = 1.0 # temp max
# -

# Create an adv
v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=Vdegree)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=Tdegree)

# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh,
    u_Field=T,
    V_Field=v,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(T)
adv_diff.constitutive_model.Parameters.diffusivity = k

### fix temp of top and bottom walls
adv_diff.add_dirichlet_bc(tmin, "Bottom")
adv_diff.add_dirichlet_bc(tmin, "Top")


maxY = mesh.data[:, 1].max()
minY = mesh.data[:, 1].min()

with mesh.access(T):
    T.data[...] = tmin

    pipePosition = ((maxY - minY) - pipe_thickness) / 2.0

    T.data[
        (T.coords[:, 1] >= (T.coords[:, 1].min() + pipePosition))
        & (T.coords[:, 1] <= (T.coords[:, 1].max() - pipePosition))
    ] = tmax

with mesh.access(v):
    v.data[:,0] = 0.
    v.data[:,1] = velocity


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
            pvmesh["T"] =  uw.function.evaluate(T.fn, mesh.data)

        arrow_loc = np.zeros((v.coords.shape[0], 3))
        arrow_loc[:, 0:2] = v.coords[...]

        arrow_length = np.zeros((v.coords.shape[0], 3))
        arrow_length[:, 0:2] = vsol[...]

        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        with mesh.access():
            points = np.zeros((mesh._centroids.shape[0],3))
            points[:,0] = mesh._centroids[:,0]
            points[:,1] = mesh._centroids[:,1]
            points[:,2] = 0.0

        point_cloud = pv.PolyData(points)

        with mesh.access():
            point_cloud.point_data["T"] = uw.function.evaluate(T.sym[0], mesh._centroids)

        # pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

        # pl.add_mesh(
        #     pvmesh,
        #     cmap="coolwarm",
        #     edge_color="Black",
        #     show_edges=True,
        #     scalars="T",
        #     use_transparency=False,
        #     opacity=0.95,
        # )

        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
                        use_transparency=False, opacity=0.95)

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
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

        # pl.add_arrows(arrow_loc, arrow_length, mag=5.0, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol


plot_fig()

# +
adv_diff.petsc_options['snes_rtol'] = 1e-12
adv_diff.petsc_options['snes_atol'] = 1e-6

### see the SNES output
adv_diff.petsc_options["snes_converged_reason"] = None
# -

# ##### Vertical profile across the centre of the box

### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), 0.2 * mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y)  ### LHS of box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y

### get the initial temp profile
T_orig = uw.function.evaluate(T.sym[0], sample_points)


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

step = 0
model_time = 0.0

fig_step = 0
nfigs = 4
nsteps = nfigs*4

# +
fig, ax = plt.subplots(1, nfigs+1, figsize=(15,3), sharex=True, sharey=True)

while step < nsteps+1:
    ### print some stuff
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {model_time:6.5f}\n\n")
        
    ### 1D temp profile from underworld
    T_UW = uw.function.evaluate(T.sym[0], sample_points)

    ### 1D numerical diffusion
    T_1D_model = diffusion_1D(
        sample_points=sample_points[:, 1], T0=T_orig.copy(), diffusivity=k, vel=velocity, time_1D=model_time
    )

    #### 1D numerical advection
    new_y = sample_points[:,1] + (velocity*model_time)

    if uw.mpi.size == 1 and step % (nsteps/nfigs) == 0:
        """compare 1D and 2D models"""
        ### profile from UW
        ax[fig_step].plot(T_UW, sample_points[:, 1], ls="-", c="red", label="UW numerical solution")
        ### numerical solution
        ax[fig_step].plot(T_1D_model, new_y, ls="-.", c="k", label="1D numerical solution")
        ax[fig_step].set_title(f'time: {round(model_time, 5)}', fontsize=8)
        ax[fig_step].legend(fontsize=8)
        fig_step += 1


    dt = 0.5 * adv_diff.estimate_dt()
    
    start = time.time()
    
    ### diffuse through underworld
    adv_diff.solve(timestep=dt)
    finish = time.time()

    solver_time = finish - start
    print(f'\n\nTime to solve = {solver_time}\n\n')

    step += 1
    model_time += dt
    
plt.savefig(f'./adv_benchmark_res={res}_Tdegree={T.degree}_Vdegree={v.degree}.pdf', bbox_inches='tight', dpi=500)
# -

plot_fig()




