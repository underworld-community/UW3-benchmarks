# %% [markdown]
# # Constant viscosity convection, Cartesian domain (benchmark)
# 
# 
# 
# This example solves 2D dimensionless isoviscous thermal convection with a Rayleigh number, for comparison with the [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
# 
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import numpy as np
import sympy

# %%
##### Set some things
Ra = 1e4 #### Rayleigh number

k = 1.0 #### diffusivity

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

res= 16     ### x and y res of box

# parameters needed when a high res model is run after a low res model

do_rerun = True # if this is set to True, a low resolution model is first run followed by a high resolution model


# %%
# meshbox = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32.0, regular=True, qdegree=2
# )

meshbox = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res,res))


# %%
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

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=1)
t_0 = uw.discretisation.MeshVariable("T0", meshbox, 1, degree=1)

# additional variable for the gradient
dTdZ = uw.discretisation.MeshVariable(r"\partial T/ \partial \Z", # FIXME: Z should not be a function of x, y, z 
                                      meshbox, 
                                      1, 
                                      degree = 1) 

# variable containing stress in the z direction
sigma_zz = uw.discretisation.MeshVariable(r"\sigma_{zz}",  
                                        meshbox, 
                                        1, degree=1)

# %%
x, y = meshbox.X

# gradient of temp field 
t_soln_grad = uw.systems.Projection(meshbox, dTdZ)
delT = t_soln.sym.diff(y)[0]
t_soln_grad.uw_function = delT
t_soln_grad.smoothing = 1.0e-3
t_soln_grad.petsc_options.delValue("ksp_monitor")

# %%
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)

#stokes.petsc_options["ksp_rtol"]  = 1e-5 # reduce tolerance to increase speed

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

# %%
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

adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

# %%
pertStrength = 0.1
deltaTemp = tempMax - tempMin

# %%
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


# %%
# check the mesh if in a notebook / serial

def plotFig(meshbox = meshbox, v_soln = v_soln, t_soln = t_soln, dTdZ = dTdZ, is_grad = False): 
    """
    is_grad - set flag to True if you want to plot the gradient of the temp field along z-direction 
    """
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
            if is_grad:
                point_cloud.point_data["Tp"] = dTdZ.data.copy()
            else:
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
            if is_grad:
                pvmesh.point_data["T"] = dTdZ.data
            else:
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

# %%
#### buoyancy_force = rho0 * (1 + (beta * deltaP) - (alpha * deltaT)) * gravity
# buoyancy_force = (1 * (1. - (1 * (t_soln.sym[0] - tempMin)))) * -1

buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %%
adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5

# %%
def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = math.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

print(f'initial v_rms = {v_rms()}')

# %%
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

# %%
t_step = 0
time = 0.

if do_rerun: 
    nsteps = 400
else: 
    nsteps = 2100

timeVal =  np.zeros(nsteps)*np.nan
vrmsVal =  np.zeros(nsteps)*np.nan

# %%
#### Convection model / update in time

"""
There is a strange interaction here between the solvers if the zero_guess is set to False
"""

while t_step < nsteps:
    vrmsVal[t_step] = v_rms()
    timeVal[t_step] = time

    stokes.solve(zero_init_guess=True) # originally True
    delta_t = 0.5 * stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False) # originally False

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))
        
        print(f't_rms = {t_soln.stats()[6]}, v_rms = {v_rms()}')

    # if t_step % 5 == 0:
    #     plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

    t_step += 1
    time   += delta_t


# %%
import os 

# save final mesh variables in the run 

os.makedirs("../meshes", exist_ok = True)

expt_name = "Ra1e4_res" + str(res)
savefile = "{}_ts{}.h5".format(expt_name,t_step)
meshbox.save(savefile)
v_soln.save(savefile)
t_soln.save(savefile)
meshbox.generate_xdmf(savefile)

# %%
import matplotlib.pyplot as plt

# plot how v_rms is evolving through time

fig,ax = plt.subplots(dpi = 100)
color = "black"
ax.set_xlabel("Step")
ax.set_ylabel(r"$v_{rms}$", color = color)
ax.plot(vrmsVal, color = color)
ax.tick_params(axis = "y", labelcolor = color)
ax.set_xlim([0, 1000])


# %%
# Calculate benchmark values
print("RMS velocity at the final time step is {}.".format(vrmsVal[-1]))

# %%
# update the gradient values 
t_soln_grad.solve()

# %% [markdown]
# ### Create a high resolution model and run it 
# Note that this is an optional step

# %%
# set-up mesh, variables, and solvers

if do_rerun:

    res2 = 32

    if res2 <= res: 
        print("Warning: second model resolution is not higher than the first model's...")

    # set-up mesh
    meshbox_hr = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res2,res2))

    # set-up mesh variables
    v_soln_hr = uw.discretisation.MeshVariable("U", meshbox_hr, meshbox_hr.dim, degree=2)
    p_soln_hr = uw.discretisation.MeshVariable("P", meshbox_hr, 1, degree=1)
    t_soln_hr = uw.discretisation.MeshVariable("T", meshbox_hr, 1, degree=1)
    t_0_hr    = uw.discretisation.MeshVariable("T0", meshbox_hr, 1, degree=1)

    # additional variable for the gradient
    dTdZ_hr   = uw.discretisation.MeshVariable(r"\partial T/ \partial \Z", # FIXME: Z should not be a function of x, y, z 
                                            meshbox_hr, 
                                            1, 
                                            degree = 1) 

    # variable containing stress in the z direction
    sigma_zz_hr = uw.discretisation.MeshVariable(r"\sigma_{zz}",  
                                            meshbox_hr, 
                                            1, 
                                            degree=1)

    x, y = meshbox_hr.X

    # gradient of temp field 
    t_soln_grad_hr = uw.systems.Projection(meshbox_hr, dTdZ_hr)
    delT = t_soln_hr.sym.diff(y)[0]
    t_soln_grad_hr.uw_function = delT
    t_soln_grad_hr.smoothing = 1.0e-3
    t_soln_grad_hr.petsc_options.delValue("ksp_monitor")

    ##### set-up the solvers #####
    # create stokes object
    stokes_hr = Stokes(
                        meshbox_hr,
                        velocityField=v_soln_hr,
                        pressureField=p_soln_hr,
                        solver_name="stokes_hr"
                    )
    stokes_hr.petsc_options["ksp_rtol"]  = 1e-5 # reduce tolerance to increase speed

    stokes_hr.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox_hr.dim)
    stokes_hr.constitutive_model.Parameters.viscosity = viscosity

    # Free-slip boundary conditions
    stokes_hr.add_dirichlet_bc((0.0,), "Left", (0,))
    stokes_hr.add_dirichlet_bc((0.0,), "Right", (0,))
    stokes_hr.add_dirichlet_bc((0.0,), "Top", (1,))
    stokes_hr.add_dirichlet_bc((0.0,), "Bottom", (1,))

    # Create adv_diff object
    adv_diff_hr = uw.systems.AdvDiffusionSLCN(
                                                meshbox_hr,
                                                u_Field = t_soln_hr,
                                                V_Field = v_soln_hr,
                                                solver_name = "adv_diff_hr",
                                            )

    adv_diff_hr.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox_hr.dim)
    adv_diff_hr.constitutive_model.Parameters.diffusivity = k

    adv_diff_hr.theta = 0.5

    adv_diff_hr.add_dirichlet_bc(1.0, "Bottom")
    adv_diff_hr.add_dirichlet_bc(0.0, "Top")

# %%
# interpolate the variables from the low res model to the high res model
if do_rerun: 
    with meshbox_hr.access(v_soln_hr, t_soln_hr, p_soln_hr):
        
        t_soln_hr.data[:, 0] = uw.function.evaluate(t_soln.fn, t_soln_hr.coords)
        p_soln_hr.data[:, 0] = uw.function.evaluate(p_soln.fn, p_soln_hr.coords)

        #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        v_coords = v_soln_hr.coords
    
        cond = v_coords[:, 0] == boxLength      # v_x
        v_coords[cond, 0] = 0.999999*boxLength 
        
        cond = v_coords[:, 1] == boxHeight      # v_y
        v_coords[cond, 1] = 0.999999*boxHeight 

        v_soln_hr.data[:] = uw.function.evaluate(v_soln.fn, v_coords)

        # FIXME: still need to check what v_soln looks like - see if workaround works 

        # final set-up of other variables
        buoyancy_force = Ra * t_soln_hr.sym[0]
        stokes_hr.bodyforce = sympy.Matrix([0, buoyancy_force])

        adv_diff_hr.petsc_options["pc_gamg_agg_nsmooths"] = 5


# %%
plotFig(meshbox = meshbox_hr, v_soln = v_soln_hr, t_soln = t_soln_hr, dTdZ = dTdZ_hr, is_grad = False)

# %%
if do_rerun:
    # use the t_step and time from the low-res run

    t_step = nsteps     # remove this later
    time = timeVal[-1]  # remove later
    
    nsteps_hr = 1000 # test value of 50

    timeVal_hr = np.zeros(nsteps_hr)*np.nan
    vrmsVal_hr = np.zeros(nsteps_hr)*np.nan

    while t_step < nsteps + nsteps_hr:

        print(t_step)
        vrmsVal_hr[t_step - nsteps] = v_rms(meshbox_hr, v_soln_hr)
        timeVal_hr[t_step - nsteps] = time

        stokes_hr.solve(zero_init_guess=True) # originally True
        delta_t = 0.5 * stokes_hr.estimate_dt()
        adv_diff_hr.solve(timestep=delta_t, zero_init_guess=False) # originally False

        # stats then loop
        tstats = t_soln_hr.stats()

        if uw.mpi.rank == 0:
            print("Timestep {}, dt {}".format(t_step, delta_t))
            
            print(f't_rms = {t_soln.stats()[6]}, v_rms = {v_rms(meshbox_hr, v_soln_hr)}')

        # if t_step % 5 == 0:
        #     plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))

        t_step += 1
        time   += delta_t

# %%
vrmsVal_hr.shape

# %%

# plot how v_rms is evolving through time for both low res and high res
if do_rerun:
    fig,ax = plt.subplots(dpi = 100)

    ax.set_xlabel("Step")
    ax.set_ylabel(r"$v_{rms}$", color = "k")

    ax.plot(np.arange(nsteps), vrmsVal, color = "k", label = "12 x 12")
    ax.tick_params(axis = "y", labelcolor = "k")

    ax.plot(nsteps + np.arange(nsteps_hr), vrmsVal_hr, color = "C0", label = "32 x 32")
    ax.legend()
    #ax.tick_params(axis = "y", labelcolor = "C0")


    #ax.set_xlim([0, 700])




# %%
vrmsVal[-1] # low res   

# %%
vrmsVal_hr[-1] # high res

# %%
# function for calculating the surface integral 
# from Ex_Shear_Band_Notch_Benchmark
 
def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral

# %%
# define upper and lower surface functions
x, z = meshbox.X

up_surface_defn_fn = sympy.exp(-1e4*((z - 1)**2)) # at z = 1
lw_surface_defn_fn = sympy.exp(-1e4*((z)**2)) # at z = 0

# display(up_surface_defn_fn)
# display(lw_surface_defn_fn)

# %%
# calculate the surface integral for both upper and lower surfaces

up_int = surface_integral(meshbox, dTdZ.sym[0], up_surface_defn_fn)
lw_int = surface_integral(meshbox, t_soln.sym[0], lw_surface_defn_fn)

Nu = -up_int/lw_int

print("Calculated value of Nu: {}".format(Nu))

# %%
# calculate q values which depend onn the temperature gradient fields

q1 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ.sym[0], np.array([[0., 0.999999*boxHeight]]))[0]
q2 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ.sym[0], np.array([[0.999999*boxLength, 0.999999*boxHeight]]))[0]
q3 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ.sym[0], np.array([[0.999999*boxLength, 0.]]))[0]
q4 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ.sym[0], np.array([[0., 0.]]))[0]

if(uw.mpi.rank==0):
    print('Rayleigh number = {0:.1e}'.format(Ra))
    print('q1 = {0:.3f}; q2 = {1:.3f}'.format(q1, q2))
    print('q3 = {0:.3f}; q4 = {1:.3f}'.format(q3, q4))

# %%
# projection for the stress in the zz direction
x, y = meshbox.X

stress_calc = uw.systems.Projection(meshbox, sigma_zz)
stress_calc.uw_function = stokes.stress[1, 1]
stress_calc.smoothing = 1.0e-3
stress_calc.petsc_options.delValue("ksp_monitor")

# %%
stress_calc.solve()

# %%
# subtract the average value for the benchmark 

mean_sigma_zz_top = -surface_integral(meshbox, 
                                     sigma_zz.sym[0], 
                                     up_surface_defn_fn)/boxLength


# %%
print(mean_sigma_zz_top)
print(boxLength)

# %%
# Set parameters in SI units

grav = 10        # m.s^-2
height = 1.e6    # m 
rho  = 4.0e3     # g.m^-3
kappa  = 1.0e-6  # m^2.s^-1

eta0=1.e23



def calculate_topography(coord): # only coord has local scope

    sigma_zz_top = -uw.function.evaluate(sigma_zz.sym[0], coord) - mean_sigma_zz_top
    
    # dimensionalise 
    dim_sigma_zz_top  = ((eta0 * kappa) / (height**2)) * sigma_zz_top
    topography = dim_sigma_zz_top / (rho * grav)
    
    return topography



# %%
e1 = calculate_topography(np.array([[0, 0.999999*boxHeight]]))
e2 = calculate_topography(np.array([[0.999999*boxLength, 0.999999*boxHeight]]))

with meshbox.access():
    cond = meshbox.data[:, 1] == meshbox.data[:, 1].max()
    up_surface_coords = meshbox.data[cond]
    up_surface_coords[:, 1] = 0.999999*up_surface_coords[:, 1]

abs_topo = abs(calculate_topography(up_surface_coords))

min_abs_topo_coord = up_surface_coords[np.where(abs_topo == abs_topo.min())[0]].flatten()

print(e1, e2, min_abs_topo_coord)


