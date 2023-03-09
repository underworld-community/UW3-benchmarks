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

# options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
# options["dm_plex_check_all"] = None
# options.getAll()

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
refViscosity = 1.7e24
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

UM_visc = nd(1.7e24 * u.pascal * u.second)
LM_visc = nd(1.7e24 * u.pascal * u.second)

T_surf = nd(T_0 * u.kelvin)
T_cmb  = nd(T_1 * u.kelvin)


# +
meshball = uw.meshing.Annulus(radiusInner=rI, radiusOuter=rO, cellSize=0.07, degree=1, qdegree=3)


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
# -

v_soln = uw.discretisation.MeshVariable("U", meshball, meshball.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshball, 1, degree=1)
T_soln = uw.discretisation.MeshVariable("T", meshball, 1, degree=3)
T0 = uw.discretisation.MeshVariable("T0", meshball, 1, degree=3)


swarm = uw.swarm.Swarm(mesh=meshball)
T1 = uw.swarm.SwarmVariable("Tminus1", swarm, 1)
X1 = uw.swarm.SwarmVariable("Xminus1", swarm, 2)
swarm.populate(fill_param=3)


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

T_dep_density = nd_density * (1 - (nd_alpha * (T_soln.sym[0] - T_surf)))
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
Ti=T_1
To=T_0


# +
initial_T = (r-Ri)/(Ro-Ri)*(To-Ti)+Ti + A*sympy.sin(7*th) + B*sympy.sin(13*th) + C*sympy.cos(0.123*th+pi/3) + D*sympy.cos(0.456*th+pi/6)


with meshball.access(T_soln):
    T_soln.data[:,0] = uw.function.evaluate(initial_T, T_soln.coords, meshball.N)
    


# +
# stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity
# stokes.bodyforce = sympy.Matrix([(th * T_dep_density * nd_density), 0.])

Rayleigh = 1e6

stokes.bodyforce = sympy.Matrix([Rayleigh * initial_T, 0])
# -





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
k = nd(1e-6 * u.meter**2/u.second)



adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.theta = 0.5

### fix temp at top and bottom 
adv_diff.add_dirichlet_bc(T_cmb, "Lower")
adv_diff.add_dirichlet_bc(T_surf, "Upper")
# +

# check the stokes solve converges
stokes.solve()

# +
# Check the diffusion part of the solve converges
adv_diff.petsc_options["ksp_monitor"] = None
adv_diff.petsc_options["monitor"] = None

adv_diff.solve(timestep=0.00001 * stokes.estimate_dt())


# +
# diff = uw.systems.Poisson(meshball, u_Field=t_soln, solver_name="diff_only")

# diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshball.dim)
# diff.constitutive_model.material_properties = adv_diff.constitutive_model.Parameters(diffusivity=1)
# diff.solve()


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

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.sym[0], meshball.data)
    pvmesh.point_data["Ts"] = uw.function.evaluate(adv_diff._u_star.sym[0], meshball.data)
    pvmesh.point_data["dT"] = uw.function.evaluate(t_soln.sym[0] - adv_diff._u_star.sym[0], meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter(window_size=(750, 750))

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    pl.add_arrows(arrow_loc, arrow_length, mag=0.0005)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    # pl.add_points(pdata)

    pl.show(cpos="xy")
# -


pvmesh.point_data["Ts"].min()

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

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.sym[0] - t_0.sym[0], meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    # pl.add_mesh(pvmesh,'Black', 'wireframe')

    pl.add_mesh(
        pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="T", use_transparency=False, opacity=0.5
    )

    # pl.add_arrows(arrow_loc, arrow_length, mag=0.025)

    # pl.add_points(pdata)

    pl.show(cpos="xy")


# -


def plot_T_mesh(filename):

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
        pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

        pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

        points = np.zeros((t_soln.coords.shape[0], 3))
        points[:, 0] = t_soln.coords[:, 0]
        points[:, 1] = t_soln.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshball.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        with meshball.access():
            usol = stokes.u.data.copy()

        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshball.data)

        arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
        arrow_loc[:, 0:2] = stokes.u.coords[...]

        arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
        arrow_length[:, 0:2] = usol[...]

        pl = pv.Plotter()

        pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)

        pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.66)

        pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("mag")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
        # pl.show()


# +
# Convection model / update in time

expt_name = "output/Cylinder_Ra1e6i"

for step in range(0, 25):

    stokes.solve()
    delta_t = 5.0 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(step, delta_t))
    #         print(tstats)

    #     plot_T_mesh(filename="{}_step_{}".format(expt_name,step))

    savefile = "{}_{}_iter.h5".format(expt_name, step)
    meshball.save(savefile)
    v_soln.save(savefile)
    t_soln.save(savefile)
    meshball.generate_xdmf(savefile)

# -


# savefile = "output_conv/convection_cylinder.h5".format(step)
# meshball.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshball.generate_xdmf(savefile)


# +


if uw.mpi.size == 1:

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "pythreejs"
    pv.global_theme.smooth_shading = True

    pv.start_xvfb()

    pvmesh = meshball.mesh2pyvista(elementType=vtk.VTK_TRIANGLE)

    points = np.zeros((t_soln.coords.shape[0], 3))
    points[:, 0] = t_soln.coords[:, 0]
    points[:, 1] = t_soln.coords[:, 1]

    point_cloud = pv.PolyData(points)

    with meshball.access():
        point_cloud.point_data["T"] = t_soln.data.copy()

    with meshball.access():
        usol = stokes.u.data.copy()

    pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshball.data)

    arrow_loc = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_loc[:, 0:2] = stokes.u.coords[...]

    arrow_length = np.zeros((stokes.u.coords.shape[0], 3))
    arrow_length[:, 0:2] = usol[...]

    pl = pv.Plotter()

    pl.add_arrows(arrow_loc, arrow_length, mag=0.00002, opacity=0.75)
    # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

    pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=True, point_size=7.5, opacity=0.25)

    pl.add_mesh(pvmesh, "Black", "wireframe", opacity=0.75)

    pl.show(cpos="xy")
