# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# # Darcy flow (1d) using xy coordinates to define permeability distribution
#
#
# Change benchmark version to test the piecewise (0) or swarm (1) approach for the hydraulic conductivity layout

# +
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()
# -

# #### Test different versions of the benchmark
# Mapping of the hydraulic conductivity:<br />
# *0* for Piecewise function<br />
# *1* for swarm-based
#

params = uw.Params(
    uw_res          = 20,
    uw_k1           = 1.0,
    uw_k2           = 1.0e-4,
    uw_max_pressure = 0.5,
    uw_interface_y  = -0.25,
)

res          = params.uw_res
k1           = params.uw_k1
k2           = params.uw_k2
max_pressure = params.uw_max_pressure
interfaceY   = params.uw_interface_y

benchmark_version = 0  # 0 = piecewise, 1 = swarm-based

show_plot = False

# #### Set up the mesh and mesh vars

# +
minX, maxX = -1.0, 0.0
minY, maxY = -1.0, 0.0

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=1.0/res, qdegree=2, regular=False)

# mesh = uw.meshing.StructuredQuadBox(elementRes=(20,20),
#                                       minCoords=(minX,minY),
#                                       maxCoords=(maxX,maxY),)


p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

# x and y coordinates
x = mesh.N.x
y = mesh.N.y

# +

if show_plot and uw.mpi.size == 1:

    # plot the mesh
    import pyvista as pv

    pv.global_theme.window_size = [750, 750]
    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)

    pl.show(cpos="xy")
# -

# #### Set up the Darcy solver

# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh, p_soln, v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
darcy.constitutive_model = uw.constitutive_models.DiffusionModel


# #### Set up the hydraulic conductivity layout 

# +
# Groundwater pressure boundary condition on the bottom wall

initialPressure = -1.0 * y * max_pressure

# +
# set up two materials

from sympy import Piecewise, ceiling, Abs


# +
if benchmark_version == 1:
    ### swarm version
    swarm = uw.swarm.Swarm(mesh=mesh)
    material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=False)
    # k = uw.swarm.IndexSwarmVariable("k", swarm, indices=2)
    
    swarm.populate_petsc(fill_param=2)

    material.data[swarm.data[:, 1] >= interfaceY] = 0
    material.data[swarm.data[:, 1] < interfaceY] = 1

    mat_k = np.array([k1, k2])

    kFunc = mat_k[0] * material.sym[0] + mat_k[1] * material.sym[1]

else:
    #### The piecewise version
    kFunc = Piecewise((k1, y >= interfaceY), (k2, y < interfaceY), (1.0, True))
    
    # A smooth version
    # kFunc = k2 + (k1-k2) * (0.5 + 0.5 * sympy.tanh(100.0*(y-interfaceY)))
# -

# ### Add in the parameters for the Darcy solve
# - darcy.f = forcing term
# - darcy.s = bodyforce term
#
# <br />
# add some additional terms to the velocity projector in the Darcy solver

# +
darcy.f = 0.0

darcy.s = sympy.Matrix([0, 0]).T

darcy.constitutive_model.Parameters.diffusivity=kFunc


# set up boundary conditions
darcy.add_dirichlet_bc([0.0], "Top")
darcy.add_dirichlet_bc([-1.0 * minY * max_pressure], "Bottom")

# Zero pressure gradient at sides / base (implied bc)

darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-6
darcy._v_projector.smoothing = 1.0e-6
darcy._v_projector.add_dirichlet_bc(0.0, "Left",  [0])
darcy._v_projector.add_dirichlet_bc(0.0, "Right", [0])
# -

# ### Solve

# Solve times
darcy.solve()

# +
### Visualise the result

if show_plot and uw.mpi.size == 1:

    import pyvista as pv

    pv.global_theme.window_size = [1250, 750]
    pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)

    usol = v_soln.data.copy()

    pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, mesh.X.coords, mesh.N)
    pvmesh.point_data["K"] = uw.function.evaluate(kFunc, mesh.X.coords, mesh.N)
    # pvmesh.point_data["S"]  = uw.function.evaluate(sympy.log(v_soln.fn.dot(v_soln.fn)), mesh.data)

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    # point sources at cell centres

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]
    point_cloud = pv.PolyData(points[::3])

    v_vectors = np.zeros((mesh.data.shape[0], 3))
    v_vectors[:, 0:2] = uw.function.evaluate(v_soln.fn, mesh.data)
    pvmesh.point_data["V"] = v_vectors

    pvstream = pvmesh.streamlines_from_source(
        point_cloud,
        vectors="V",
        integrator_type=45,
        integration_direction="both",
        max_steps=1000,
        max_time=0.1,
        initial_step_length=0.001,
        max_step_length=0.01,
    )

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="P", use_transparency=False, opacity=1.0
    )

    pl.add_mesh(pvstream, line_width=10.0)

    pl.add_arrows(arrow_loc, arrow_length, mag=0.005, opacity=0.75)

    pl.show(cpos="xy")


# +
# set up interpolation coordinates
ycoords = np.linspace(minY + 0.01 * (maxY - minY), maxY - 0.01 * (maxY - minY), 100)
xcoords = np.full_like(ycoords, 0.5 * (minX + maxX))
xy_coords = np.column_stack([xcoords, ycoords])

pressure_interp = uw.function.evaluate(p_soln.sym[0], xy_coords).flatten()
# -


# #### Get analytical solution

# +
La = -1.0 * interfaceY
Lb = 1.0 + interfaceY
dP = max_pressure

S = 1
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
)

S = 0
Pa = (dP / Lb - S + k1 / k2 * S) / (1.0 / Lb + k1 / k2 / La)
pressure_analytic_noG = np.piecewise(
    ycoords,
    [ycoords >= -La, ycoords < -La],
    [lambda ycoords: -Pa * ycoords / La, lambda ycoords: Pa + (dP - Pa) * (-ycoords - La) / Lb],
)
# -

# ### Compare analytical and numerical solution

# +
import matplotlib.pyplot as plt

# %matplotlib inline

fig = plt.figure()
ax1 = fig.add_subplot(111, xlabel="Pressure", ylabel="Depth")
ax1.plot(pressure_interp, ycoords, linewidth=3, label="Numerical solution")
ax1.plot(pressure_analytic, ycoords, linewidth=3, linestyle="--", label="Analytic solution")
ax1.plot(pressure_analytic_noG, ycoords, linewidth=3, linestyle="--", label="Analytic (no gravity)")
ax1.grid("on")
ax1.legend()
# -
if not np.allclose(pressure_analytic_noG, pressure_interp, atol=1e-2):
    raise RuntimeError('Analytical and numerical solution not close')

np.allclose(pressure_analytic_noG, pressure_interp, atol=1e-2)


