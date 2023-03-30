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
### FS - free slip top, no slip base
### NS - no slip top and base
boundaryConditions = 'NS'

outputPath = f'./output/mantleConvection-NS-VP/'
# -

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
outerRadius   = 6370e3
internalRadius= (6370e3 - 660e3) ### UM - LM transition
innerRadius   = 3480e3
refLength     = (outerRadius - innerRadius) ### thickness of mantle
g             = 10.
mu            = 1e22
T_0           = 1060
T_1           = 3450
dT            = T_1 - T_0
alpha         = 2.5e-5
kappa         = 1e-6
rho0          = 4.5e3
refVelocity   = (1.0 * u.centimeter / u.year).to(u.meter/u.second).m

rho_surf   = rho0 * (1 - (alpha * T_0))
rho_bot    = rho0 * (1  - (alpha * dT))
dRho       = rho_surf - rho_bot

lithoPressure  = rho0 * g * refLength

refTime      = refLength / refVelocity
# refTime      = mu / lithoPressure

refViscosity = mu * u.pascal * u.second

# bodyforce    = rho0  * u.kilogram / u.metre**3 * g * u.meter / u.second**2


Ra_number = (alpha * rho0 * g * (T_1 - T_0) * refLength**3) / (kappa*mu)




KL = refLength * u.meter
KT = (T_1 - T_0)  * u.kelvin
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

Ra_number

# +
### values for the system

rI = nd(innerRadius*u.meter)
rInt = nd(internalRadius*u.meter)
rO = nd(outerRadius*u.meter)



T_surf = nd(T_0 * u.kelvin)
T_cmb  = nd(T_1 * u.kelvin)
# -


res = 0.075

# +
# meshball = uw.meshing.Annulus_internalBoundary(radiusInner=rI, radiusInternal=rInt, radiusOuter=rO, cellSize=res, degree=1, qdegree=2)
meshball = uw.meshing.AnnulusInternalBoundary(radiusOuter=rO, radiusInternal=rInt, radiusInner=rI, cellSize=res, cellSize_Outer=res, qdegree=3)



r, th = meshball.CoordinateSystem.R
x, y = meshball.CoordinateSystem.X
ra, th = meshball.CoordinateSystem.xR
# -


for i in range(meshball.dm.getNumLabels()):
    name = meshball.dm.getLabelName(i)
    print("label name = %s" % name, "\tlabel size = %d" % meshball.dm.getLabelSize(name))

v_soln       = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln       = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
T_soln       = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
T0           = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)
timeField    = uw.discretisation.MeshVariable("time", meshball, 1, degree=1)
density_proj = uw.discretisation.MeshVariable("rho", meshball, 1, degree=1)
visc         = uw.discretisation.MeshVariable(r"\eta", meshball, 1, degree=1, continuous=True)
meshr        = uw.discretisation.MeshVariable(r"r", meshball, 1, degree=1)

# +
# check the mesh if in a notebook / serial


if uw.mpi.size == 1:
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.anti_aliasing = 'ssaa'
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
# Create Stokes object

stokes = Stokes(meshball, velocityField=v_soln, pressureField=p_soln, solver_name="stokes")

stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshball.dim)
stokes.constitutive_model.Parameters.viscosity = sympy.sympify(1)



# +
# #### Create the advDiff solver

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshball,
    u_Field=T_soln,
    V_Field=v_soln,
    solver_name="adv_diff")
# -


swarm = uw.swarm.Swarm(mesh=meshball)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2, proxy_continuous=True)
swarm.populate(fill_param=1)


with swarm.access(material):
    material.data[:] = 0
    material.data[np.sqrt(swarm.data[:,0]**2 + swarm.data[:,1]**2) <= rInt] = 1

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
    
    with swarm.access(material):
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]
        
        point_cloud = pv.PolyData(points)
        
        point_cloud.point_data["M"] = material.data.copy()
        point_cloud.point_data["depth"] = (rO - np.sqrt(material.swarm.data[:,0]**2 + material.swarm.data[:,1]**2))*refLength
        
    pl.add_points(
        point_cloud,
        scalars='depth',
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()

# +
nd_density = nd(rho0  * u.kilogram / u.metre**3)


nd_gravity = nd(g * u.meter / u.second**2)

nd_alpha = nd(alpha * (1/u.kelvin))


T_density = nd_density * (1. - (nd_alpha * (T_soln.sym[0] - T_surf)))

# +
### Temp dependent density
### density = rho0 * (1 + (beta * deltaP) - (alpha * deltaT))

T_density = nd_density * (1. - (nd_alpha * (T_soln.sym[0] - T_surf)))

# +
nodal_rho_calc = uw.systems.Projection(meshball, density_proj)
nodal_rho_calc.uw_function = T_density
nodal_rho_calc.smoothing = 0.
nodal_rho_calc.petsc_options.delValue("ksp_monitor")

viscosity_calc = uw.systems.Projection(meshball, visc)
viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
viscosity_calc.smoothing = 1.0e-3

def updateFields(time):
    ### density
    nodal_rho_calc.uw_function = T_density
    nodal_rho_calc.solve(_force_setup=True)
    ### viscosity
    viscosity_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    viscosity_calc.smoothing = 1.0e-3
    viscosity_calc.solve(_force_setup=True)
    
    ### time
    with meshball.access(timeField):
        timeField.data[:,0] = dim(time, u.year).m


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
    
    T0 = nd(uw.function.evaluate(initial_T, T_soln.coords, meshball.N) * u.kelvin)
    
rho_0 = uw.function.evaluate(T_density, T_soln.coords, meshball.N) 

with meshball.access(meshr):
    meshr.data[:, 0] = uw.function.evaluate(sympy.sqrt(x**2 + y**2), meshball.data, meshball.N)  # cf radius_fn which is 0->1
    


# +
radius_fn = sympy.sqrt(meshball.rvec.dot(meshball.rvec))  # normalise by outer radius if not 1.0
unit_rvec = meshball.X / (radius_fn)


buoyancy_force = (T_density * -1*nd_gravity) *  unit_rvec

if boundaryConditions == 'FS':

    # Some useful coordinate stuff

    x, y = meshball.CoordinateSystem.X
    ra, th = meshball.CoordinateSystem.xR

    hw = 1000.0 / res
    
    surface_fn = sympy.exp(-(((meshr.sym[0] - rO) / rO) ** 2) * hw)
    
    base_fn = sympy.exp(-(((meshr.sym[0] - rI) / rO) ** 2) * hw)

    free_slip_penalty_upper = v_soln.sym.dot(unit_rvec) * unit_rvec * surface_fn
    free_slip_penalty_lower = v_soln.sym.dot(unit_rvec) * unit_rvec * base_fn
    
    ### Buoyancy force RHS plus free slip surface enforcement
    ptf = 1e7
    penalty_terms_upper = ptf * free_slip_penalty_upper
    penalty_terms_lower = ptf * free_slip_penalty_lower

    stokes.bodyforce = buoyancy_force  - penalty_terms_upper # - penalty_terms_lower
    
    stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Lower.name, (0, 1))
    
else:
    stokes.bodyforce = buoyancy_force
    
    # Velocity boundary conditions
    stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Upper.name, (0, 1))
    stokes.add_dirichlet_bc((0.0, 0.0), meshball.boundaries.Lower.name, (0, 1))

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

    with swarm.access('M'):
        points = np.zeros((swarm.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm.particle_coordinates.data[:, 0]
        points[:, 1] = swarm.particle_coordinates.data[:, 1]
        
        point_cloud = pv.PolyData(points)
        
        point_cloud.point_data["M"] = material.data.copy()
        
    pl.add_points(point_cloud, render_points_as_spheres=False, point_size=10, opacity=0.1)
    
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
# +
# Create adv_diff object

# Set some things
nd_k = nd(kappa * u.meter**2/u.second)



adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
adv_diff.constitutive_model.Parameters.diffusivity = nd_k
adv_diff.theta = 0.5


# nd_IH    = nd(0.02*u.microwatt/u.meter**3) # nd(1e-12 * u.watt/u.kilogram)
# adv_diff.f = nd_IH


# -
##### fix temp at top and bottom 
adv_diff.add_dirichlet_bc(T_cmb, meshball.boundaries.Lower.name)
adv_diff.add_dirichlet_bc(T_surf, meshball.boundaries.Upper.name)


def plotFig(var='T',arrowSize=500):    
    import numpy as np
    import pyvista as pv
    import vtk

    meshball.vtk("ignore_meshball.vtk")
    pvmesh = pv.read("ignore_meshball.vtk")

    with meshball.access():
        usol = stokes.u.data.copy()
        pvmesh.point_data["T0"] = nd(uw.function.evaluate(initial_T,  meshball.data, meshball.N) * u.kelvin)
        
        pvmesh.point_data["eta"] = visc.data[:]

    pvmesh.point_data["T"] = uw.function.evaluate(T_soln.sym[0], meshball.data)
    
    
    
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

    pl.add_arrows(arrow_loc, arrow_length, mag=arrowSize)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


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
    viewer(density_proj._gvec)     # add density
    viewer(visc._gvec)             # add viscosity
    viewer(timeField._gvec)        # add time
    viewer.destroy()              
    generateXdmf(fname, xfname)
    
    swarm.save_checkpoint(outputPath=outputPath, swarmName='swarm', swarmVars=[material], index=step, time=round(dim(time, u.megayear).m, 2))
    
    # ### save all swarm variables attached to DM
    # x_swarm_fname = f"{outputPath}swarm_{'step_'}{step:02d}.xmf"
    # swarm.dm.viewXDMF(x_swarm_fname)


step = 0
time = 0.
time_dim = 0.

# +
# with meshball.access(v_soln):
#     print(v_soln.data[np.where(uw.function.evaluate(surface_fn, meshball.data) == 1 )])

# +
UM_visc = nd(1e22*u.pascal*u.second)

LM_visc = nd(1e24*u.pascal*u.second)
# -

viscosity = sympy.Piecewise((LM_visc, sympy.sqrt(x**2+y**2) <= rInt), (UM_visc, True))

# +
# Constant visc
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.petsc_options['pc_type']   = 'lu'
stokes.tolerance                  = 1e-4
stokes.petsc_options["snes_atol"] = 1.0e-6
stokes.petsc_options["snes_rtol"] = 1.0e-8

stokes.solve(zero_init_guess=True)

# +
stokes.petsc_options["snes_max_it"] = 500

stokes.tolerance                   = 1e-4
stokes.petsc_options["snes_atol"] = 1.0e-6
stokes.petsc_options["snes_rtol"] = 1.0e-8

# +
with swarm.access():
    depth = (rO - np.sqrt(swarm.data[:,0]**2 + swarm.data[:,1]**2))

tau_y = nd(1e9*u.pascal)

# # + jupyter={"outputs_hidden": true}
# viscosity_L = 999.0 * material.sym[0] + 1.0
viscosity_Y = (tau_y / (2 * stokes._Einv2 + 1.0e-18))
# viscosity = 1 / ((1 / viscosity_Y) + (1 / viscosity_L))

UM_visc = 1 / ((1/nd(1e24*u.pascal*u.second)) + (1./viscosity_Y))

UM_visc_lim = sympy.Max(nd(1e22*u.pascal*u.second), sympy.Min(UM_visc, nd(1e25*u.pascal*u.second)) )


LM_visc_lim = 30 * nd(1e22*u.pascal*u.second)

### assign material viscosity
# viscosity = UM_visc_lim * material.sym[0] + LM_visc_lim * material.sym[1]

### assign viscosity based on location
viscosity = sympy.Piecewise((LM_visc, sympy.sqrt(x**2+y**2) <= rInt), (UM_visc, True))

# NL viscoplasitc viscosity
stokes.constitutive_model.Parameters.viscosity = viscosity
# -

# Check the diffusion part of the solve converges
adv_diff.petsc_options["ksp_monitor"] = None
adv_diff.petsc_options["monitor"] = None
adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 1

# +
# Convection model / update in time

while step < 31:
    
    time_dim = dim(time, u.megayear)

    if uw.mpi.rank == 0:
        print(f"\n\nTimestep: {step}, time: {time_dim}\n\n")
    
    if step % 5 == 0:
        saveData(step=step, outputPath=outputPath, time = time)
        


    stokes.solve(zero_init_guess=False)
    delta_t = adv_diff.estimate_dt()
    adv_diff.solve(timestep=delta_t)
    
    swarm.advection(v_soln.sym, delta_t=delta_t) 

    # stats then loop
    tstats = T_soln.stats()
        
    step += 1
    time += delta_t
# -


plotFig(var='eta', arrowSize=0.1)

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
    
    with swarm1.access(mat):
        points = np.zeros((swarm1.particle_coordinates.data.shape[0], 3))
        points[:, 0] = swarm1.particle_coordinates.data[:, 0]
        points[:, 1] = swarm1.particle_coordinates.data[:, 1]
        
        point_cloud = pv.PolyData(points)
        
        point_cloud.point_data["M"] = mat.data.copy()
        point_cloud.point_data["depth"] = (rO - np.sqrt(mat.swarm.data[:,0]**2 + mat.swarm.data[:,1]**2))*refLength
        
    pl.add_points(
        point_cloud,
        scalars='M',
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )

    # pl.add_mesh(pvmesh,'Black', 'wireframe', opacity=0.5)
    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False, opacity=0.5)

    pl.show()


