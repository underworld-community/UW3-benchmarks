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

# # Stokes Benchmark SolCx
#
# The SolCx benchmark tests the stokes solver with a large viscosity change at the centre of the domain. These situations are common in geodynamics, for example cold subducting slabs into the astenospheric mantle.
#


import petsc4py
from petsc4py import PETSc

# +
import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function
import numpy as np

import sympy
from sympy import Piecewise
# -

# #### Create the mesh

# +
res = 64
# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 50, qdegree=2
# )

mesh  = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), elementRes=(res, res), qdegree=3)
# -


# #### Add some mesh vars

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

# #### Create Stokes solver

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)


x, y = mesh.CoordinateSystem.X

# #### Parameters for the model

# +
eta_0 = 1.0
eta_1 = 1.0e6

x_c = 0.5
f_0 = 1.0

n_z = 1
# -


# ### Set up the bodyforce 
#
# - density:
#
# \begin{aligned}
# \rho=\sin(\pi y)\cos(\pi x)
# \end{aligned}
#
# - gravity
# \begin{aligned}
# g=(0,1)
# \end{aligned}
#

# +
rho = sympy.sin(n_z * sympy.pi * y)*sympy.cos(sympy.pi * x)

g = sympy.Matrix([0, 1])

stokes.bodyforce = rho*g
# -

# ### Set up the viscosity
#
#
#
# \begin{split}\begin{aligned}
#   \eta(\mathbf x) = \left\{
#     \begin{matrix}
#       1 & \text{for}\ x \le 0.5, \\
#       10^6 & \text{for}\ x  > 0.5.
#     \end{matrix}
#   \right.\end{aligned}\end{split}

# +

viscosity_fn = sympy.Piecewise(
    (
        eta_1,
        x > x_c,
    ),
    (eta_0, True),
)

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn
# -

# #### Add the boundary conditions
# All sides are free slip in this case

# free slip.
# note with petsc we always need to provide a vector of correct cardinality.
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Top", "Bottom"], 1
)  # top/bottom: components, function, markers
stokes.add_dirichlet_bc(
    (0.0, 0.0), ["Left", "Right"], 0
)  # left/right: components, function, markers


# #### Change some of the default petsc options
# We may need to adjust the tolerance if $\Delta \eta$ is large

stokes.petsc_options["snes_rtol"] = 1.0e-6
stokes.petsc_options["ksp_rtol"] = 1.0e-6
stokes.petsc_options["snes_max_it"] = 10

# Solve time
stokes.solve()

# ### Get the velocity from uw3 solution and uw2 analytical solution

# +
### placeholder for uw2 analytical solution
vel_soln_analytic = None

### get uw3 velocity
vel_soln_uw3 = uw.function.evaluate(v.fn, mesh._centroids )

# -

# ##### Attempt to get analytical solution from uw2

# +
try:
    import underworld as uw2

    solCx = uw2.function.analytic.SolCx(n_z=n_z, eta_A=eta_0, eta_B=eta_1, x_c=x_c)
    vel_soln_analytic = solCx.fn_velocity.evaluate(mesh._centroids)


    

    from numpy import linalg as LA
    if uw.mpi.rank == 0:
        # print("Diff norm a. = {}".format(LA.norm(v.data - vel_soln_analytic)))
        print("Diff norm = {}".format(LA.norm(vel_soln_uw3 - vel_soln_analytic)))

        # if not np.allclose(vel_soln_uw3, vel_soln_analytic, rtol=1):
        #     raise RuntimeError("Solve did not produce expected result.")

    

except ImportError:
    import warnings

    warnings.warn("Unable to test SolC results as UW2 not available.")
# -

# ### Visualise it !

# +
# check the mesh if in a notebook / serial

import mpi4py

if mpi4py.MPI.COMM_WORLD.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 1200]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")


    pvmesh.point_data["P"] = uw.function.evaluate(p.sym[0], mesh.data)
    
    pvmesh.point_data["rho"] = uw.function.evaluate(rho, mesh.data)

    def gen_arrows(v_data, mesh):

        arrow_loc = np.zeros((mesh._centroids.shape[0], 3))
        arrow_loc[:, 0:2] = mesh._centroids
    
        arrow_length = np.zeros((v_data.shape[0], 3))
        arrow_length[:, 0] = v_data[:,0] #uw.function.evaluate(stokes.u.sym[0], stokes.u.coords)
        arrow_length[:, 1] = v_data[:,1] #uw.function.evaluate(stokes.u.sym[1], stokes.u.coords)

        return arrow_loc, arrow_length
    


    pl = pv.Plotter(window_size=[1000, 1000])
    pl.add_axes()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        scalars="rho",
        use_transparency=False,
        opacity=1.0,
    )

    uw3_loc, uw3_len = gen_arrows(vel_soln_uw3, mesh)
    pl.add_arrows(uw3_loc, uw3_len, mag=15, color='k',opacity=0.8)
    
    if vel_soln_analytic is not None:
        uw2_loc, uw2_len = gen_arrows(vel_soln_analytic, mesh)
        pl.add_arrows(uw2_loc, uw2_len, mag=15, color='green', opacity=0.8)

    pl.show(cpos="xy", screenshot='SolCx_sol.png')

# -


