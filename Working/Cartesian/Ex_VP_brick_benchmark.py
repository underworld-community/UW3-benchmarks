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

# # The brick benchmark
#
# As outlined in [Kaus, 2010](http://jupiter.ethz.ch/~kausb/k10.pdf) and [Glerum et al., 2018](https://se.copernicus.org/articles/9/267/2018/se-9-267-2018.pdf)
#
# [UWGeodynamics (UW2) version](https://github.com/underworldcode/underworld2/blob/master/docs/UWGeodynamics/benchmarks/Kaus_BrickBenchmark-Compression.ipynb)

import petsc4py
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy
import gmsh
import os

from underworld3 import timing
timing.reset()
timing.start()

# +
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

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

# +
xmin, xmax = 0., ndim(40*u.kilometer)
ymin, ymax = 0., ndim(10*u.kilometer)

## set brick height and length
BrickHeight = nd(400.*u.meter)
BrickLength = nd(800.*u.meter)

### set the res in x and y
resx = 160
resy =  40

### add material index
BrickIndex = 0
BGIndex    = 1

# +
# mesh = uw.meshing.StructuredQuadBox(elementRes =(int(resx),int(resy)),
#                                     minCoords=(xmin,ymin), 
#                                     maxCoords=(xmax,ymax))

mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin,ymin), 
                                         maxCoords=(xmax,ymax), cellSize=0.05)

# +
swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.SwarmVariable(
    "M", swarm, num_components=1, proxy_continuous=False, proxy_degree=0)

mat = uw.swarm.IndexSwarmVariable(
    'mat', swarm, indices=2, proxy_degree=0)

swarm.populate(fill_param=2)
# -

v_soln = uw.discretisation.MeshVariable(r"U", mesh, mesh.dim, degree=2)
p_soln = uw.discretisation.MeshVariable(r"P", mesh, 1, degree=1, continuous=True)
p_null = uw.discretisation.MeshVariable(r"P2", mesh, 1, degree=1, continuous=True)

edot = uw.discretisation.MeshVariable(
    r"\dot\varepsilon", mesh, 1, degree=1, continuous=True
)
visc = uw.discretisation.MeshVariable(r"\eta", mesh, 1, degree=1, continuous=True)
stress = uw.discretisation.MeshVariable(r"\sigma", mesh, 1, degree=1, continuous=True)

# + [markdown] magic_args="[markdown]"
# This is how we extract cell data from the mesh. We can map it to the swarm data structure and use this to
# build material properties that depend on cell type.
# -

for i in [material, mat]:
        with swarm.access(i):
            i.data[:] = BGIndex
            i.data[(swarm.data[:,1] <= BrickHeight) & 
                  (swarm.data[:,0] >= (((xmax - xmin) / 2.) - (BrickLength / 2.)) ) & 
                  (swarm.data[:,0] <= (((xmax - xmin) / 2.) + (BrickLength / 2.)) )] = BrickIndex

# check the mesh if in a notebook / serial

if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh.vtk('tempMesh.vtk')
    pvmesh = pv.read('tempMesh.vtk')

    pl = pv.Plotter()

    # points = np.zeros((mesh._centroids.shape[0], 3))
    # points[:, 0] = mesh._centroids[:, 0]
    # points[:, 1] = mesh._centroids[:, 1]

    with swarm.access():
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()

    pvmesh.point_data["eta"] = uw.function.evaluate(
        material.sym[0], mesh.data, mesh.N
    )

    # pl.add_mesh(
    #     pvmesh,
    #     cmap="coolwarm",
    #     edge_color="Black",
    #     show_edges=True,
    #     use_transparency=False,
    #     opacity=0.5,
    # )
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )
    pl.add_mesh(pvmesh, "Black", "wireframe")

    pl.show(cpos="xy")


# ### Check that this mesh can be solved for a simple, linear problem

# Create Stokes object

stokes = uw.systems.Stokes(
    mesh,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
    verbose=False,
)


# +
# Set solve options here (or remove default values
stokes.petsc_options["ksp_monitor"] = None

# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1e-4
# stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres" # gmres here for bulletproof
stokes.petsc_options[
    "fieldsplit_pressure_pc_type"
] = "gamg"  # can use gasm / gamg / lu here
stokes.petsc_options[
    "fieldsplit_pressure_pc_gasm_type"
] = "basic"  # can use gasm / gamg / lu here
stokes.petsc_options[
    "fieldsplit_pressure_pc_gamg_type"
] = "classical"  # can use gasm / gamg / lu here
stokes.petsc_options["fieldsplit_pressure_pc_gamg_classical_type"] = "direct"
stokes.petsc_options["fieldsplit_pressure_pc_gamg_esteig_ksp_type"] = "cg"

stokes.petsc_options['pc_type'] = 'lu'

# -


viscosity_L = nd(1e20*u.pascal*u.second) * mat.sym[0] + \
              nd(1e25*u.pascal*u.second) * mat.sym[1] 

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.1

# Velocity boundary conditions
vel = nd(2e-11 * u.meter / u.second)

stokes.add_dirichlet_bc(vel, "Left", 0)
stokes.add_dirichlet_bc(0, "Left", 1)
stokes.add_dirichlet_bc(-vel, "Right", 0)
stokes.add_dirichlet_bc(0, "Right", 1)
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))
# stokes.add_dirichlet_bc((0.0,), "Top", (1,))


nd_gravity = nd(9.81*u.meter/u.second**2)
nd_density = nd(2.7e3*u.kilogram/u.meter**3)

stokes.bodyforce = sympy.Matrix([0, -1*nd_gravity*nd_density])


# +
x, y = mesh.X

res = 0.1
hw = 1000.0 / res
surface_defn_fn = sympy.exp(-((y - 0) ** 2) * hw)
base_defn_fn = sympy.exp(-((y + 1) ** 2) * hw)
edges_fn = sympy.exp(-((x - 2) ** 2) / 0.025) + sympy.exp(-((x + 2) ** 2) / 0.025)
# stokes.bodyforce -= 10000.0 * surface_defn_fn * v_soln.sym[1] * mesh.CoordinateSystem.unit_j
# -

# This is a strategy to obtain integrals over the surface (etc)


def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral


strain_rate_calc = uw.systems.Projection(mesh, edot)
strain_rate_calc.uw_function = stokes._Einv2
strain_rate_calc.smoothing = 1.0e-3

viscosity_calc = uw.systems.Projection(mesh, visc)
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
viscosity_calc.smoothing = 1.0e-3

stress_calc = uw.systems.Projection(mesh, stress)
S = stokes.stress_deviator
stress_calc.uw_function = (
    sympy.simplify(sympy.sqrt(((S**2).trace()) / 2)) - p_soln.sym[0]
)
stress_calc.smoothing = 1.0e-3


def plotFig():
    import numpy as np
    import pyvista as pv
    import vtk
    
    viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    stress_calc.uw_function = (
        2 * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
    )

    strain_rate_calc.solve()
    viscosity_calc.solve()
    stress_calc.solve()

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [1050, 500]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, 1.0]

    mesh.vtk("tmp_notch.vtk")
    pvmesh = pv.read("tmp_notch.vtk")

    points = np.zeros((mesh._centroids.shape[0], 3))
    points[:, 0] = mesh._centroids[:, 0]
    points[:, 1] = mesh._centroids[:, 1]

    pvmesh.point_data["sfn"] = uw.function.evaluate(
        surface_defn_fn, mesh.data, mesh.N
    )
    pvmesh.point_data["pres"] = uw.function.evaluate(p_soln.sym[0], mesh.data)
    pvmesh.point_data["str"] = uw.function.evaluate(stress.sym[0], mesh.data)
    # pvmesh.point_data["tauy"] = uw.function.evaluate(tau_y, mesh.data, mesh.N)
    
    # pvmesh.point_data["edot"] = uw.function.evaluate(edot.sym[0], mesh.data)
    # pvmesh.point_data["eta"] = uw.function.evaluate(visc.sym[0], mesh.data)
    
    with mesh.access(visc, edot):
        pvmesh.point_data["edot"] = edot.data[:]
        pvmesh.point_data["eta"]  = visc.data[:]

    with mesh.access():
        usol = v_soln.data.copy()

    arrow_loc = np.zeros((v_soln.coords.shape[0], 3))
    arrow_loc[:, 0:2] = v_soln.coords[...]

    arrow_length = np.zeros((v_soln.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    point_cloud = pv.PolyData(points)

    with swarm.access():
        point_cloud.point_data["M"] = uw.function.evaluate(material.sym[0], mesh._centroids, mesh.N)

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.03, opacity=0.75)

    pl.add_mesh(
        pvmesh,
        cmap="RdYlGn",
        scalars="edot",
        edge_color="Grey",
        show_edges=True,
        use_transparency=False,
        # clim=[0.1, 1.5],
        opacity=1.0,
        log_scale=True
    )
    
    pl.add_scalar_bar('edot', interactive=True, vertical=False,
                           title_font_size=35,
                           label_font_size=30,
                           outline=True, fmt='%10.5f')

    # pl.add_points(
    #     point_cloud,
    #     cmap="coolwarm",
    #     render_points_as_spheres=False,
    #     point_size=5,
    #     opacity=0.1,
    # )


    pl.show(cpos="xy")

# +
# First, we solve the linear problem
stokes.petsc_options["snes_atol"] = 1.0e-6
stokes.petsc_options["snes_rtol"] = 1.0e-6
stokes.petsc_options["snes_max_it"] = 500

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)
stokes.constitutive_model.Parameters.viscosity = viscosity_L
stokes.saddle_preconditioner = 1 / viscosity_L
stokes.penalty = 0.1



stokes.solve(zero_init_guess=True)

if uw.mpi.rank == 0:
    print("Linear solve complete", flush=True)
# -


if uw.mpi.size ==1:
    plotFig()

tau_y = nd(1e8*u.pascal)
viscosity_Y = (tau_y / (2 * stokes._Einv2 + 1.0e-18))

# + jupyter={"outputs_hidden": true}
visc_bg    = 1 / ((1/nd(1e25*u.pascal*u.second)) + (1./viscosity_Y))
visc_brick = nd(1e20*u.pascal*u.second)

viscosity = visc_brick   * mat.sym[0] + \
            visc_bg      * mat.sym[1]

stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.saddle_preconditioner = 1 / viscosity
stokes.solve(zero_init_guess=False)
# -


if uw.mpi.size ==1:
    plotFig()

# +
# visc_top = 1 / ((1/nd(1e24*u.pascal*u.second)) + (1./viscosity_Y))

visc_bg = sympy.Min(nd(1e25*u.pascal*u.second), viscosity_Y)

visc_brick = nd(1e20*u.pascal*u.second)

viscosity = visc_brick * mat.sym[0] + \
            visc_bg    * mat.sym[1]

stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.saddle_preconditioner = 1 / viscosity
stokes.solve(zero_init_guess=False)

# -

if uw.mpi.size ==1:
    plotFig()
