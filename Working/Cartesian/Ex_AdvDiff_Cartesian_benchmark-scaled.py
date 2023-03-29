# # Constant viscosity convection, Cartesian domain (benchmark)
#
#
#
# This example solves 2D dimensionless isoviscous thermal convection with a Rayleigh number, for comparison with the [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
#
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.

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
# -


##### Set some things
res= 32 ### x and y res of box

# +
outputPath = './output/boxConvection-scaled/'


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
outerRadius  = 6370e3
innerRadius  = 3480e3
refLength    = (outerRadius - innerRadius) ### thickness of mantle
refGravity   = 10.
mu           = 1e22
T_0          = 1060
T_1          = 3450
dT           = T_1 - T_0
alpha        = 2.5e-5
kappa        = 1e-6
rho0         = 4.5e3

refVelocity  = 1e-11 

rho_surf   = rho0 * (1 - (alpha * T_0))
rho_bot    = rho0 * (1  - (alpha * dT))
dRho       = rho_surf - rho_bot

refDensity   = rho0

refPressure  = refDensity * refGravity * refLength

refTime      = refLength / refVelocity
# refTime      = mu / refPressure



refViscosity = mu * u.pascal * u.second

bodyforce    = refDensity  * u.kilogram / u.metre**3 * refGravity * u.meter / u.second**2


Ra_number = (alpha * refDensity * refGravity * (T_1 - T_0) * refLength**3) / (kappa*mu)




KL = refLength * u.meter
KT = (T_1 - T_0)  * u.kelvin
# Kt = refTime   * u.second
# Kt = 200000.*u.year
Kt = refTime * u.second


# KM = bodyforce * KL**2 * Kt**2
KM = refViscosity * KL * Kt



### create unit registry
scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients["[temperature]"]= KT
scaling_coefficients
# -

nd(rho_surf * u.kilogram / u.metre**3 )*nd(9.81* u.meter / u.second**2)

nd(rho_bot * u.kilogram / u.metre**3 )*nd(9.81* u.meter / u.second**2)

# +
boxLength = nd(refLength*u.meter)
boxHeight = nd(refLength*u.meter)
tempMin   = nd(T_0*u.kelvin)
tempMax   = nd(T_1*u.kelvin)



viscosity = nd(1e22 * u.pascal*u.second)
# -

viscosity

# +
# meshbox = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0, regular=True, qdegree=2
# )

meshbox = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res,res))





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
    pv.global_theme.show_edges = True
    pv.global_theme.axes.show = True

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")
    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, edge_color="Black", show_edges=True)

    pl.show(cpos="xy")

# +
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=1)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=1)


timeField    = uw.discretisation.MeshVariable("time", meshbox, 1, degree=1)
density_proj = uw.discretisation.MeshVariable("rho", meshbox, 1, degree=1)


# +
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

# Set solve options here (or remove default values
# stokes.petsc_options.getAll()
# stokes.petsc_options.delValue("ksp_monitor")

# stokes.petsc_options["snes_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_pressure_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_velocity_ksp_monitor"] = None
# stokes.petsc_options["fieldsplit_pressure_ksp_rtol"] = 1.0e-6
# stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1.0e-2
# stokes.petsc_options.delValue("pc_use_amat")


stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity=viscosity
# stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))

# +
# Create a density structure / buoyancy force
# gravity will vary linearly from zero at the centre
# of the sphere to (say) 1 at the surface

import sympy

# Some useful coordinate stuff

x, y = meshbox.X


# +
# Create adv_diff object



adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = nd(kappa * u.meter**2/u.second)



adv_diff.theta = 0.5

##### fix temp at top and bottom 
adv_diff.add_dirichlet_bc(nd(T_0*u.kelvin), "Top")
adv_diff.add_dirichlet_bc(nd(T_1*u.kelvin), "Bottom")
# -


pertStrength = 0.1
deltaTemp = tempMax - tempMin

# +
import math, sympy
with meshbox.access(t_soln, t_0):
    t_soln.data[:] = 0.
    t_0.data[:] = 0.

for index, coord in enumerate(meshbox.data):
    # print(index, coord)
    pertCoeff = math.cos( math.pi * coord[0]/boxLength ) * math.sin( math.pi * coord[1]/boxLength )
    with meshbox.access(t_soln):
        t_soln.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
        t_soln.data[index] = max(tempMin, min(tempMax, t_soln.data[index]))
        

with meshbox.access(t_soln, t_0):
    t_0.data[:,0] = t_soln.data[:,0]
    
# -


def plotFig(var='T'):    
    import numpy as np
    import pyvista as pv
    import vtk

    meshbox.vtk("tmp_box_mesh.vtk")
    pvmesh = pv.read("tmp_box_mesh.vtk")

    with meshbox.access(t_0, t_soln):
        usol = stokes.u.data.copy()
        
        pvmesh.point_data["T0"] =t_0.data # uw.function.evaluate(t_0.sym[0], meshbox.data)

        pvmesh.point_data["T"] = t_soln.data # uw.function.evaluate(t_soln.sym[0], meshbox.data)
    
        pvmesh.point_data["dT"] =  pvmesh.point_data["T"] - pvmesh.point_data["T0"]

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars=var, use_transparency=False, opacity=0.5
    )

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.0001)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# +
meshbox.vtk("tmp_box_mesh.vtk")
pvmesh = pv.read("tmp_box_mesh.vtk")

with meshbox.access(t_0, t_soln):
    usol = stokes.u.data.copy()

    pvmesh.point_data["T0"] =t_0.data # uw.function.evaluate(t_0.sym[0], meshbox.data)

    pvmesh.point_data["T"] = t_soln.data # uw.function.evaluate(t_soln.sym[0], meshbox.data)

    pvmesh.point_data["dT"] =  pvmesh.point_data["T"] - pvmesh.point_data["T0"]
# -

plotFig(var='T')

plotFig(var='T0')

# +
nd_density = nd(rho0  * u.kilogram / u.metre**3)


nd_gravity = nd(refGravity * u.meter / u.second**2)

nd_alpha = nd(alpha * (1/u.kelvin))


T_density = nd_density * (1. - (nd_alpha * (t_soln.sym[0] - tempMin)))

# +
nodal_rho_calc = uw.systems.Projection(meshbox, density_proj)
nodal_rho_calc.uw_function = T_density
nodal_rho_calc.smoothing = 0.
nodal_rho_calc.petsc_options.delValue("ksp_monitor")

def updateFields(time):
    ### density
    nodal_rho_calc.uw_function = T_density
    nodal_rho_calc.solve(_force_setup=True)
    ### time
    with meshbox.access(timeField):
        timeField.data[:,0] = dim(time, u.year).m


# -

#### buoyancy_force = rho0 * (1 + (beta * deltaP) - (alpha * deltaT)) * gravity
buoyancy_force =  (T_density * -1 * nd_gravity) #(1 * (1. - (1 * (t_soln.sym[0] - tempMin)))) * -1
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5


# +
def v_rms():
    v_rms = math.sqrt(uw.maths.Integral(meshbox, v_soln.fn.dot(v_soln.fn)).evaluate())
    return v_rms

print(f'initial v_rms = {v_rms()}')


# -

def saveData(step, outputPath, time):
    
    ### update projections first
    updateFields(time)
    
    ### save mesh vars
    fname = f"{outputPath}mesh_{'step_'}{step:02d}.h5"
    xfname = f"{outputPath}mesh_{'step_'}{step:02d}.xmf"
    viewer = PETSc.ViewerHDF5().createHDF5(fname, mode=PETSc.Viewer.Mode.WRITE,  comm=PETSc.COMM_WORLD)

    viewer(meshbox.dm)

    ### add mesh vars to viewer to save as one h5/xdmf file. Has to be a PETSc object (?)
    viewer(v_soln._gvec)         # add velocity
    viewer(p_soln._gvec)         # add pressure
    viewer(t_soln._gvec)           # add temperature
    viewer(density_proj._gvec)   # add density
    # viewer(materialField._gvec)    # add material projection
    viewer(timeField._gvec)        # add time
    viewer.destroy()              
    generateXdmf(fname, xfname)
    
    # ### save all swarm variables attached to DM
    # x_swarm_fname = f"{outputPath}swarm_{'step_'}{step:02d}.xmf"
    # swarm.dm.viewXDMF(x_swarm_fname)


# +
step = 0
time = 0.

nsteps = 11

timeVal =  np.zeros(nsteps)*np.nan
vrmsVal =  np.zeros(nsteps)*np.nan

# +
#### Convection model / update in time

"""
There is a strange interaction here between the solvers if the zero_guess is set to False
"""

# expt_name = "output/Ra1e6"

while step < nsteps:
    
    vrmsVal[step] = v_rms()
    timeVal[step] = time
    
    ### get t stats
    tstats = t_soln.stats()
    
    time_dim = dim(time, u.megayear)
    
    if uw.mpi.rank == 0:
        print(f"\n\nTimestep: {step}, time: {time_dim}\n\n")
        
        print(f't_rms = {t_soln.stats()[6]}, v_rms = {v_rms()}\n\n')
    
    if step % 5 == 0:
        saveData(step=step, outputPath=outputPath, time = time)
    

    stokes.solve(zero_init_guess=True)
    delta_t = 0.5 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)



    step   += 1
    time   += delta_t

# savefile = "{}_ts_{}.h5".format(expt_name,step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)

# -


plotFig(var='T')

dim(time, u.megayear)


