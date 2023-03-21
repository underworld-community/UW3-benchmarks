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


# +
##### Set some things
Ra = 1e4 #### Rayleigh number

k = 1.0 #### diffusivity

res= 64 ### x and y res of box

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

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
# -

v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=1)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=1)


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
adv_diff.constitutive_model.Parameters.diffusivity = k



adv_diff.theta = 0.5
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
    


# +
# check the mesh if in a notebook / serial

def plotFig():
    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 250]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True

        meshbox.vtk("tmp_box_mesh.vtk")
        pvmesh = pv.read("tmp_box_mesh.vtk")

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

        pvmesh.point_data["V"] = velocity / 10

        points = np.zeros((t_soln.coords.shape[0], 3))
        points[:, 0] = t_soln.coords[:, 0]
        points[:, 1] = t_soln.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshbox.access():
            point_cloud.point_data["Tp"] = t_soln.data.copy()

        # point sources at cell centres

        cpoints = np.zeros((meshbox._centroids.shape[0] // 4, 3))
        cpoints[:, 0] = meshbox._centroids[::4, 0]
        cpoints[:, 1] = meshbox._centroids[::4, 1]

        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=2,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=1000,
            surface_streamlines=True,
        )
        with meshbox.access(t_soln):
            pvmesh.point_data["T"] = t_soln.data[:]#uw.function.evaluate(t_soln.fn, meshbox.data)

        pl = pv.Plotter()

        # pl.add_mesh(pvmesh,'Gray', 'wireframe')

        # pl.add_mesh(
        #     pvmesh, cmap="coolwarm", edge_color="Black",
        #     show_edges=True, scalars="T", use_transparency=False, opacity=0.5,
        # )

        pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=True, point_size=10, opacity=0.33)

        # pl.add_mesh(pvstream, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")
        pvmesh.clear_data()
        pvmesh.clear_point_data()
        
plotFig()
# -

#### buoyancy_force = rho0 * (1 + (beta * deltaP) - (alpha * deltaT)) * gravity
buoyancy_force = (1 * (1. - (1 * (t_soln.sym[0] - tempMin)))) * -1
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5


# +
def v_rms():
    v_rms = math.sqrt(uw.maths.Integral(meshbox, v_soln.fn.dot(v_soln.fn)).evaluate())
    return v_rms

print(f'initial v_rms = {v_rms()}')


# -

def plot_T_mesh(filename):

    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [750, 750]
        pv.global_theme.antialiasing = True
        pv.global_theme.jupyter_backend = "pythreejs"
        pv.global_theme.smooth_shading = True
        pv.global_theme.camera["viewup"] = [0.0, 1.0, 0.0]
        pv.global_theme.camera["position"] = [0.0, 0.0, 5.0]

        meshbox.vtk("tmp_box_mesh.vtk")
        pvmesh = pv.read("tmp_box_mesh.vtk")

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_soln.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_soln.sym[1], meshbox.data)

        pvmesh.point_data["V"] = velocity / 333
        pvmesh.point_data["T"] = uw.function.evaluate(t_soln.fn, meshbox.data)

        # point sources at cell centres

        cpoints = np.zeros((meshbox._centroids.shape[0] // 4, 3))
        cpoints[:, 0] = meshbox._centroids[::4, 0]
        cpoints[:, 1] = meshbox._centroids[::4, 1]
        cpoint_cloud = pv.PolyData(cpoints)

        pvstream = pvmesh.streamlines_from_source(
            cpoint_cloud,
            vectors="V",
            integrator_type=45,
            integration_direction="forward",
            compute_vorticity=False,
            max_steps=25,
            surface_streamlines=True,
        )

        points = np.zeros((t_soln.coords.shape[0], 3))
        points[:, 0] = t_soln.coords[:, 0]
        points[:, 1] = t_soln.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshbox.access():
            point_cloud.point_data["T"] = t_soln.data.copy()

        pl = pv.Plotter()

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm",
            edge_color="Gray",
            show_edges=True,
            scalars="T",
            use_transparency=False,
            opacity=0.5,
        )

        pl.add_points(point_cloud, cmap="coolwarm", render_points_as_spheres=False, point_size=10, opacity=0.5)

        pl.add_mesh(pvstream, opacity=0.4)

        pl.remove_scalar_bar("T")
        pl.remove_scalar_bar("V")

        pl.screenshot(filename="{}.png".format(filename), window_size=(1280, 1280), return_img=False)
        # pl.show()
        pl.close()

        pvmesh.clear_data()
        pvmesh.clear_point_data()

        pv.close_all()


# +
t_step = 0
time = 0.

nsteps = 10

timeVal =  np.zeros(nsteps)*np.nan
vrmsVal =  np.zeros(nsteps)*np.nan

# +
#### Convection model / update in time

"""
There is a strange interaction here between the solvers if the zero_guess is set to False
"""

# expt_name = "output/Ra1e6"

while t_step < nsteps:
    vrmsVal[t_step] = v_rms()
    timeVal[t_step] = time

    stokes.solve(zero_init_guess=True)
    delta_t = 0.5 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))
        
        print(f't_rms = {t_soln.stats()[6]}, v_rms = {v_rms()}')

    # if t_step % 5 == 0:
    #     plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1
    time   += delta_t

# savefile = "{}_ts_{}.h5".format(expt_name,step)
# meshbox.save(savefile)
# v_soln.save(savefile)
# t_soln.save(savefile)
# meshbox.generate_xdmf(savefile)

# -


plotFig()


