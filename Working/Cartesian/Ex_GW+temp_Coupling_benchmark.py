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

# ### Temperature advection due to groundwater flow
#
# Underworld can efficiently solve the diffusion-advection equation and can set up as many simultaneous instances of it in a model as we would like. Heat-flow and groundwater flow are both described by this equation, so we can set these up and couple them, to model temperature advection by groundwater.
#
# **Temperature Diffusion-Advection Equation**
#
# $\frac{\partial T} {\partial t} = \kappa \nabla^2 T + G$
#
#
# where: $\kappa = \frac{k}{C \rho}$
#
# ($\kappa$ - thermal diffusivity, $T$ - temperature, $t$ - time, $G$ - groundwater advection term, $C$ - rock heat capacity, $\rho$ - rock density)
#
# **Groundwater Flow Equation**
#
# $ \nabla\cdot q = 0 $
#
# $ q = \kappa_H \left(\nabla H +  \rho_w g\right) $
# where: $\kappa_H = \frac{k_H}{\mu S}$
#
# In this particular example, we'll leave out the gravitational term ($\rho_w g$) and drive flow with just a pressure gradient.
#
# ($\kappa_H$ - hydraulic diffusivity, $H$ - groundwater pressure,  $k_H$ - hydraulic permeability, $\mu$ - water viscosity, $S$ - specific storage, $g$ - gravitational accelleration, $\rho_w$ - density of water)
#
# **Coupling**
#
# $G = -\nabla T \cdot \frac{\rho_w C_w}{\rho C} q$
#
# ($C_w$ - heat capacity of water)
#
# Coupling is implemented by handing an 'effective velocity field', $\frac{\rho_w C_w}{\rho C} q$ to the temperature diffusion-advection solver.
#

# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

options = PETSc.Options()
# -

# #### Set up the mesh and mesh vars

# +
minX, maxX =  0.0, 2.0
minY, maxY = -1.0, 0.0

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=0.05, qdegree=3, regular=False)

# mesh = uw.meshing.StructuredQuadBox(elementRes=(20,20),
#                                       minCoords=(minX,minY),
#                                       maxCoords=(maxX,maxY),)


t_soln = uw.discretisation.MeshVariable("T", mesh, 1, degree=3)
gw_p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=2)
gw_v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=1)

# x and y coordinates
x = mesh.N.x
y = mesh.N.y

# +

if uw.mpi.size == 1:

    # plot the mesh
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = 'ssaa'
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)

    pl.show(cpos="xy")
# -

# #### Set up the Darcy solver

# Create Darcy Solver
darcy = uw.systems.SteadyStateDarcy(mesh, gw_p_soln, gw_v_soln)
darcy.petsc_options.delValue("ksp_monitor")
darcy.petsc_options["snes_rtol"] = 1.0e-6  # Needs to be smaller than the contrast in properties
darcy.constitutive_model = uw.constitutive_models.DiffusionModel


# #### Set up the hydraulic conductivity layout 

# +
# Groundwater pressure boundary condition on the bottom wall
maxHydDiff = 1.

maxgwpressure = 1.


initialPressure   = -1.0 * y * maxgwpressure

initialTemperature = -1.0 * y

# +
# set up two materials

interfaceX = 1.

k1 = 1
k2 = 1e-2


#### The piecewise version
kFunc = sympy.Piecewise((k1, x <= interfaceX), (k2, x > interfaceX), (0.0, True))

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
darcy.add_dirichlet_bc([initialPressure], "Bottom")

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

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    v_soln = gw_v_soln
    p_soln = gw_p_soln

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln.data.copy()

    pvmesh.point_data["P"] = uw.function.evaluate(p_soln.fn, mesh.data, mesh.N)
    pvmesh.point_data["K"] = uw.function.evaluate(kFunc, mesh.data, mesh.N)
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

    pl.add_arrows(arrow_loc, arrow_length, mag=0.5, opacity=0.75)

    pl.show(cpos="xy")
# -


# ### Set up the steady state heat solver with a source term (from the GW flow)
#
# For a ground-water velocity $q$, this the following is an analytic solution to the 1D temperature diffusion-advection equation: $\frac{d^2 T}{dy^2} +  q'\frac{dT}{dy} = 0$
#
# where: $q' = \frac{\rho_w c_w}{\rho c} \frac{q}{\kappa}$
#
# The solution, assuming $H=0$ and $T=0$ at the surface and $H = H_{max}$ and $T=1$ at the base of the domain, is:
#
# $T(y) = \alpha \left(e^{q' y} - 1 \right)\ \ \ $ where: $\alpha = \left( e^{-q'}-1 \right)^{-1}$    (you can check it's actually a solution)
#
#
# We can use this to benchmark an effectively 1D part of the domain.

tempAdvDiff = uw.systems.Poisson(mesh, t_soln)

### set up initial temp field from 0 (top) to 1 (bottom)
with mesh.access(t_soln):
    t_soln.data[:,0] = uw.function.evalf(initialTemperature, t_soln.coords)

# #### Set up properties of the system

# +
#Hydraulic storage capacity
Storage = 1.

# coeff is equivalent to rho_water / rho_rock * c_water / c_rock
coeff = 1.

### diffusivity coeff
thermalDiff = 1.
# -

tempAdvDiff.add_dirichlet_bc(0, 'Top')
tempAdvDiff.add_dirichlet_bc(1, 'Bottom')


tempAdvDiff.constitutive_model = uw.constitutive_models.DiffusionModel

tempAdvDiff.constitutive_model.Parameters.diffusivity = thermalDiff

tempAdvDiff.f = -1*coeff*v_soln.sym.dot(t_soln.gradient())

# #### Solve

tempAdvDiff.solve()

# +
### Visualise the result

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    v_soln = gw_v_soln
    p_soln = gw_p_soln

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1250, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    with mesh.access():
        usol = v_soln.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.sym, mesh.data, mesh.N)
    pvmesh.point_data["K"] = uw.function.evaluate(kFunc, mesh.data, mesh.N)
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
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=1.0
    )

    pl.add_mesh(pvstream, line_width=10.0)

    pl.add_arrows(arrow_loc, arrow_length, mag=0.05, opacity=0.75)

    pl.show(cpos="xy")
# -

# #### Check the solution with the analytical version

# +
sample_y = np.arange(-0.01, -1, -0.01)
sample_x = 0.5 + np.zeros_like(sample_y)

coords0 = np.vstack([sample_x, sample_y]).T
coords1 = np.vstack([sample_x+1.5, sample_y]).T


# -

def analyticsol(y):
    qp = coeff * maxHydDiff / thermalDiff * Storage * (maxgwpressure)
    
    if qp == 0.:
        return -y
    else:
        return (sympy.exp(qp*y) - 1.) / (sympy.exp(-qp) -1. )


# +
n = 101
arrY = np.linspace(-1.,0.,n)
arrT = np.zeros(n)

### Can also be run in parallel (?)

import matplotlib.pyplot as plt

if uw.mpi.rank == 0:
    plt.clf()

for xpos in [0.,1.,2.]:
    arrT = np.zeros(n)
    arrT = uw.function.evalf(t_soln.sym, np.vstack([np.zeros_like(arrY)+xpos,arrY]).T)

    if uw.mpi.rank == 0:
        plt.plot(arrT,arrY,label="x=%i" %xpos)


# Analytic Solution
if uw.mpi.rank == 0:
    arrY = np.linspace(-1.,0.,10)
    arrAnalytic = np.zeros(10)
    for i in range(10):
        arrAnalytic[i] = analyticsol(arrY[i])

    plt.scatter(arrAnalytic,arrY,label="Analytic Advection",lw=0,c="Blue")
    plt.scatter(-1. * arrY,arrY, label="Analytic Non-Advect",lw=0,c="Black")

    plt.legend(loc=0,frameon=False)
    plt.xlabel('Temperature')
    plt.ylabel('Depth')

    plt.xlim(0,1)
    plt.ylim(-1,0)
    plt.title('Temperature Profiles')
    if uw.mpi.size == 1:
        plt.show()

    # plt.savefig("Geotherms.pdf")
# -

# x = 0, where the temp field is altered by the gw flow due to the high hydraulic conductivity <br>
# x = 1, the interface between the high (x < 1) and low (x > 1) hydraulic conductivity (and the flow field) <br>
# x = 2, the area with low hydrauilic conductivity


