# # The Bunge et al. mantle convection experiments
#
# [Recreated from the aspect documentaiton](https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/cookbooks/bunge_et_al_mantle_convection/doc/bunge_et_al_mantle_convection.html#the-bunge-et-al-mantle-convection-experiments)

# +
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np

import sympy

import os

from underworld3.utilities import generateXdmf

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()

# +
outputPath = './output/mantleConvection/'


if uw.mpi.rank == 0:
    # checking if the directory demo_folder 
    # exist or not.
    if not os.path.exists(outputPath):

        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(outputPath)

# +
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# +
### set reference values
refLength    = 3480e3
refDensity   = 4.5e3
refGravity   = 10.
refViscosity = 1e22
T_0          = 1060
T_1          = 3450
alpha        = 2.5e-5


refPressure  = refDensity * refGravity * refLength
refTime      = refViscosity / refPressure

bodyforce    = refDensity  * u.kilogram / u.metre**3 * refGravity * u.meter / u.second**2

### create unit registry
KL = refLength * u.meter
Kt = refTime   * u.second
KM = bodyforce * KL**2 * Kt**2
KT = (T_1 - T_0)  * u.kelvin

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients["[temperature]"]= KT
scaling_coefficients

# +
### values for the system

rI = nd(3480*u.kilometer)
rO = nd(6370*u.kilometer)

# UM_visc = nd(6e22 * u.pascal * u.second)
# LM_visc = nd(6e22 * u.pascal * u.second)

UM_visc = nd(1.7e24 * u.pascal * u.second)
LM_visc = nd(1.7e24 * u.pascal * u.second)

T_surf = nd(T_0 * u.kelvin)
T_cmb  = nd(T_1 * u.kelvin)


# +
meshball = uw.meshing.Annulus(radiusInner=rI, radiusOuter=rO, cellSize=0.075, degree=1, qdegree=3)


r, th = meshball.CoordinateSystem.R
x, y = meshball.CoordinateSystem.X


# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True
    pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
    pv.global_theme.camera["position"] = [0.0, 0.0, -5.0]

    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()

# +
v_soln       = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln       = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
T_soln       = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
density_proj = uw.discretisation.MeshVariable("rho", meshball, 1, degree=1)
T0           = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)
timeField      = uw.discretisation.MeshVariable("time", meshball, 1, degree=1)




# +
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
stokes.petsc_options.delValue("ksp_monitor")

# +
# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshball,
    u_Field=T_soln,
    V_Field=v_soln,
    solver_name="adv_diff")

# -

nd_density = nd(refDensity  * u.kilogram / u.metre**3)
nd_gravity = nd(refGravity * u.meter / u.second**2)

nd_alpha = nd(alpha * (1/u.kelvin))

# +
### density = rho0 * (1 + (beta * deltaP) - (alpha * deltaT))

T_density = nd_density * (1 - (nd_alpha * (T_soln.sym[0] - T_surf)))


nodal_rho_calc = uw.systems.Projection(meshball, density_proj)
nodal_rho_calc.uw_function = T_density
nodal_rho_calc.smoothing = 0.
nodal_rho_calc.petsc_options.delValue("ksp_monitor")

def updateFields(time):
    ### density
    nodal_rho_calc.uw_function = T_density
    nodal_rho_calc.solve(_force_setup=True)
    ### time
    with meshball.access(timeField):
        timeField.data[:,0] = dim(time, u.year).m


# -

# Constant visc
stokes.constitutive_model.Parameters.viscosity = UM_visc

# +
A=100
B=75
C=50
D=25

pi=np.pi

Ri=rI
Ro=rO
Ti=T_1 # nd(T_1*u.kelvin)
To=T_0 #nd(T_0*u.kelvin)


# +
initial_T = (r-Ri)/(Ro-Ri)*(To-Ti)+Ti + A*sympy.sin(7*th) + B*sympy.sin(13*th) + C*sympy.cos(0.123*th+pi/3) + D*sympy.cos(0.456*th+pi/6)


with meshball.access(T_soln):
    T_soln.data[:,0] = nd(uw.function.evaluate(initial_T, T_soln.coords, meshball.N) * u.kelvin)
    
# -


stokes.bodyforce = (meshball.rvec * T_density * nd_gravity * 1e6)

# +
# check the mesh if in a notebook / serial

if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    # pv.start_xvfb()

    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")

    with meshball.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(T_soln.sym[0], meshball.data)

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -

# Velocity boundary conditions
stokes.add_dirichlet_bc((0.0, 0.0), "Upper", (0, 1))
stokes.add_dirichlet_bc((0.0, 0.0), "Lower", (0, 1))

# +
# Create adv_diff object

# Set some things
nd_k = nd(1e-6 * u.meter**2/u.second)



adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
adv_diff.constitutive_model.Parameters.diffusivity = nd_k
adv_diff.theta = 0.5


# -
##### fix temp at top and bottom 
adv_diff.add_dirichlet_bc(T_cmb, "Lower")
adv_diff.add_dirichlet_bc(T_surf, "Upper")

# Check the diffusion part of the solve converges
adv_diff.petsc_options["ksp_monitor"] = None
adv_diff.petsc_options["monitor"] = None


adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 1

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")

    with meshball.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(T_soln.sym[0], meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=1e6)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def saveData(step, outputPath, time):
    
    ### update projections first
    updateFields(time)
    
    ### save mesh vars
    fname = f"{outputPath}mesh_{'step_'}{step:02d}.h5"
    xfname = f"{outputPath}mesh_{'step_'}{step:02d}.xmf"
    viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE,  comm=PETSc.COMM_WORLD)

    viewer(meshball.dm)

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    viewer(stokes.u._gvec)         # add velocity
    viewer(stokes.p._gvec)         # add pressure
    viewer(T_soln._gvec)           # add temperature
    viewer(density_proj._gvec)   # add density
    # viewer(materialField._gvec)    # add material projection
    viewer(timeField._gvec)        # add time
    viewer.destroy()              
    generateXdmf(fname, xfname)
    
    # ### save all swarm variables attached to DM
    # x_swarm_fname = f"{outputPath}swarm_{'step_'}{step:02d}.xmf"
    # swarm.dm.viewXDMF(x_swarm_fname)


step = 0
time = 0.
time_dim = 0.

# +
# Convection model / update in time

expt_name = "output/Cylinder_Ra1e6i"

while step < 500:
    
    if step % 20 == 0:
        saveData(step=step, outputPath=outputPath, time = time)

    stokes.solve()
    delta_t = adv_diff.estimate_dt()
    adv_diff.solve(timestep=delta_t)

    # stats then loop
    tstats = T_soln.stats()

    if uw.mpi.rank == 0:
        print(f"\n\nTimestep: {step}, time: {dim(time, u.year)}\n\n")
        
    step += 1
    time += delta_t
    
    time_dim = dim(time, u.year).m
        




# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")

    with meshball.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(T_soln.sym[0] - T0.sym[0], meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=0.1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -


