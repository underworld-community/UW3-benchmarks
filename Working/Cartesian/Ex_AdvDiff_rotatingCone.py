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

# # The rotating cone problem
#
# Over a unit square a blob of heat is rotated one revolution subject to a Péclet number of 1e6.  
#
# The evolution of the heat is governed by the advection diffusion equation.  
#
# In Underworld this equation can be numerically approximated using the Semi-Lagrangian Crank-Nicholson methd (**SLCN**). Various numerical parameters can be investigated in the model below.
#
# This classical test problem is taken from J. Donea and A. Huerta, _Finite Element Methods for Flow Problems (2003)_ with an initial field (temperature in this case) given:
#
#     
# $$
# T(x,y,0) = 
# \begin{array}{cc}
#   \biggl\{ & 
#     \begin{array}{cc}
#       \frac{1}{4}(1+\cos(\pi X))(1+\cos(\pi Y)) & X^2 + Y^2 \leq 1, \\
#       0 & X^2 + Y^2 \gt 1, \\
#     \end{array}
# \end{array}
# $$
#
# where $$
# (X, Y) = (\space (x-x_0) / \sigma ,\space (y-y_0) / \sigma)\space)
# $$ 
#
# with $(x_0, y_0)$ as the center of the blob and the boundary condition is $T=0$ on all walls.
#
# The domain is a unit square $[-0.5,0.5]x[-0.5,0.5]y$. The advection field is of pure rotation with velocity as $V(x,y) = (-y,x)$.
#
# [UW2 version](https://github.com/underworldcode/underworld2/blob/master/docs/test/2DCosineHillAdvection.ipynb)
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

xmin, xmax = -0.5, 0.5
ymin, ymax = -0.5, 0.5

# +
# mesh = uw.meshing.StructuredQuadBox(
#     elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax)
# )
# -


mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=0.03, regular=False )

# default model parameters
sigma = 0.2          # width of blob
x_0   = (1./6, 1./6) # position of blob
kappa = 1e-6         # thermal diffusion (entire domain)

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
adv_diff.constitutive_model.Parameters.diffusivity = kappa

# +
# %%
### fix temp of top and bottom walls
# adv_diff.add_dirichlet_bc(1.0, "Bottom")
# adv_diff.add_dirichlet_bc(1.0, "Top")
# adv_diff.add_dirichlet_bc(1.0, "Left")
# adv_diff.add_dirichlet_bc(1.0, "Right")
# -


# %%
maxY = mesh.data[:, 1].max()
minY = mesh.data[:, 1].min()

with mesh.access(v):
    # initialise fields
    v.data[:,0] = -1*v.coords[:,1]
    v.data[:,1] =  1*v.coords[:,0]

fn_X = (mesh.X[0]-x_0[0])/sigma
fn_Y = (mesh.X[1]-x_0[1])/sigma

fn_inside = ( sympy.Pow(fn_X,2.) + sympy.Pow(fn_Y,2.) ) <= 1.
fn_hill   = 0.25 * ( 1.+sympy.cos(np.pi*fn_X) ) * ( 1.+sympy.cos(np.pi*fn_Y) )


### opposite order to UW branching conditions
fn_ic = sympy.Piecewise( (fn_hill, fn_inside),
                         (0., True ) )

with mesh.access(T):
    T.data[:,0] = uw.function.evaluate(fn_ic, T.coords, mesh.N)
    T_old = np.copy(T.data[:,0])
    
    print(T.data.min(), T.data.max())


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

        pl.add_arrows(arrow_loc, arrow_length, mag=0.1, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")

        # return vsol


plot_fig()

# ##### Vertical profile across the centre of the box

step = 0
time = 0.0

nsteps = 101

while step < nsteps:
    ### print some stuff
    if uw.mpi.rank == 0:
        print(f"Step: {str(step).rjust(3)}, time: {time:6.5f}")

    
    dt0 = adv_diff.estimate_dt()
    
    dt1 = mesh.get_min_radius() ** 2 / kappa
    
    print(dt0, dt1)
    
    dt = adv_diff.estimate_dt()
    
    ### diffuse through underworld
    adv_diff.solve(timestep=dt)

    step += 1
    time += dt

plot_fig()

with mesh.access(T):
    T_new = np.copy(T.data[:,0])

# %%
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
        pvmesh["T_new"] = T_new
        pvmesh["T_old"] = T_old
        pvmesh["dT"] = T_old - T_new

    arrow_loc = np.zeros((v.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v.coords[...]

    arrow_length = np.zeros((v.coords.shape[0], 3))
    arrow_length[:, 0:2] = vsol[...]

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, "Black", "wireframe")


    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="dT",
        use_transparency=False,
        opacity=0.95,
    )


    # )

    pl.add_arrows(arrow_loc, arrow_length, mag=0.0001, opacity=0.5)


    pl.show(cpos="xy")

    # return vsol


