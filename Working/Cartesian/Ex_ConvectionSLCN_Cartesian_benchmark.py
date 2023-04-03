# %% [markdown]
# # Constant viscosity convection, Cartesian domain (benchmark)
# 
# 
# 
# This example solves 2D dimensionless isoviscous thermal convection with a Rayleigh number, for comparison with the [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
# 
# We set up a v, p, T system in which we will solve for a steady-state T field in response to thermal boundary conditions and then use the steady-state T field to compute a stokes flow in response.
# 
# There are two options:
# 1. The user can run a single model by setting do_rerun flag to False.
# 2. The user can run a low resolution model until it reaches a steady state, copy the physical fields into a second model with a higher resolution, then run the second model reaches its steady state.
# This is done by setting do_rerun to True and the parameters accordingly.

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import os 
import numpy as np
import sympy

# %% [markdown]
# ### Set parameters to use 

# %%
Ra = 1e4 #### Rayleigh number

k = 1.0 #### diffusivity

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

res= 16             ### x and y res of box
nsteps = 1000        ### maximum number of time steps to run the first model 
epsilon_lr = 1e-8   ### criteria for early stopping; relative change of the Vrms in between iterations  

##########
# parameters needed when a high res model is run after a low res model
##########
do_rerun = True 
res2 = 64
nsteps_hr = 1000     ### maximum number of time steps to run the second model      
epsilon_hr = 1e-8   ### criteria for early stopping


# %% [markdown]
# ### Create mesh and variables

# %%
# meshbox = uw.meshing.UnstructuredSimplexBox(
#                                                 minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight), cellSize=1.0 /res, regular=True, qdegree=2
#                                         )

meshbox = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res,res))


# %%
# visualise the mesh if in a notebook / serial

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


# ### some projection objects to calculate the
# x, y = meshbox.X

# t_soln_grad = uw.systems.Projection(meshbox, dTdZ)
# delT = t_soln.sym.diff(y)[0]
# t_soln_grad.uw_function = delT
# t_soln_grad.smoothing = 1.0e-3
# t_soln_grad.petsc_options.delValue("ksp_monitor")

# %% [markdown]
# ### System set-up 
# Create solvers and set boundary conditions

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

#stokes.petsc_options["snes_rtol"] = 1.0e-6
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


#### buoyancy_force = rho0 * (1 + (beta * deltaP) - (alpha * deltaT)) * gravity
# buoyancy_force = (1 * (1. - (1 * (t_soln.sym[0] - tempMin)))) * -1
buoyancy_force = Ra * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

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

# Dirichlet boundary conditions for temperature
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5

# %% [markdown]
# ### Set initial temperature field 
# 
# The initial temperature field is set to a sinusoidal perturbation. 

# %%
import math, sympy

pertStrength = 0.1
deltaTemp = tempMax - tempMin

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


# %% [markdown]
# ### Some plotting and analysis tools 

# %%
# check the mesh if in a notebook / serial
# allows you to visualise the mesh with initial perturbation

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

# %% [markdown]
# #### RMS velocity
# The root mean squared velocity, $v_{rms}$, is defined as: 
# 
# 
# \begin{aligned}
# v_{rms}  =  \sqrt{ \frac{ \int_V (\mathbf{v}.\mathbf{v}) dV } {\int_V dV} }
# \end{aligned}
# 
# where $\bf{v}$ denotes the velocity field and $V$ is the volume of the box.

# %%
# underworld3 function for calculating the rms velocity 

def v_rms(mesh = meshbox, v_solution = v_soln): 
    # v_soln must be a variable of mesh
    v_rms = math.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

#print(f'initial v_rms = {v_rms()}')

# %% [markdown]
# ### Main simulation loop

# %%
t_step = 0
time = 0.

timeVal =  np.zeros(nsteps)*np.nan
vrmsVal =  np.zeros(nsteps)*np.nan

# %%
#### Convection model / update in time


# NOTE: There is a strange interaction here between the solvers if the zero_guess is set to False


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

    if t_step > 1 and abs((vrmsVal[t_step] - vrmsVal[t_step - 1])/vrmsVal[t_step]) < epsilon_lr:
        break

    t_step += 1
    time   += delta_t


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

# plot how v_rms is evolving through time for both low res and high res
fig,ax = plt.subplots(dpi = 100)

# low resolution model
ax.hlines(42.865, 0, 1000, linestyle = "--", linewidth = 0.5, color = "gray", label = r"Benchmark $v_{rms}$")
ax.plot(np.arange((~np.isnan(vrmsVal)).sum()), 
        vrmsVal[~np.isnan(vrmsVal)], 
        color = "k", 
        label = str(res) + " x " + str(res))

ax.legend()
ax.set_xlabel("Time step")
ax.set_ylabel(r"$v_{rms}$", color = "k")

ax.set_xlim([0, 1000])
ax.set_ylim([0, 100])

# %%
# Calculate benchmark values
if uw.mpi.rank == 0:
    print("RMS velocity at the final time step is {}.".format(v_rms()))

# %% [markdown]
# ### Create a high resolution model and run it 
# NOTE: this is an optional step

# %%
# set-up mesh, variables, and solvers

if do_rerun:

    # if res2 <= res: 
    #     print("Warning: second model resolution is not higher than the first model's...")

    ##### Setting up of mesh and variables #####
    # set-up mesh
    meshbox_hr = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res2,res2))

    # set-up mesh variables
    v_soln_hr = uw.discretisation.MeshVariable("U_2", meshbox_hr, meshbox_hr.dim, degree=2)
    p_soln_hr = uw.discretisation.MeshVariable("P_2", meshbox_hr, 1, degree=1)
    t_soln_hr = uw.discretisation.MeshVariable("T_2", meshbox_hr, 1, degree=1)
    t_0_hr    = uw.discretisation.MeshVariable("T0_2", meshbox_hr, 1, degree=1)

    # additional variable for the gradient
    dTdZ_hr   = uw.discretisation.MeshVariable(r"\partial T_2/ \partial \Z", # FIXME: Z should not be a function of x, y, z 
                                            meshbox_hr, 
                                            1, 
                                            degree = 1) 

    # variable containing stress in the z direction
    sigma_zz_hr = uw.discretisation.MeshVariable(r"\sigma_{zz,2}",  
                                            meshbox_hr, 
                                            1, 
                                            degree=1)

    ##### set-up the solvers and boundary conditions #####
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

# %% [markdown]
# ### Copy the p, T, v fields from the low res to the high res model

# %%
# NOTE: interpolation using uw.fuction.evaluate does not work
#       work-around is to do nearest neighbor interpolation for mesh variables 

# inputs:
# hi-res mesh variable
# low-res mesh variable
# hr_mesh

def nearest_neigh_workaround(lr_mesh, hr_mesh, lr_mesh_var, hr_mesh_var):
    '''
    Inputs:
    lr_mesh - low resolution mesh
    hr_mesh - high resolution mesh 
    lr_mesh_var - low resolution mesh variable
    hr_mesh_var - high resolution mesh variable
    # NOTE: do not vary anything in the input variables within the function as it will modify the original variable 
    '''

    with hr_mesh.access() and lr_mesh.access():
        
        data = np.zeros_like(hr_mesh_var.data)

        for i, coord in enumerate(hr_mesh_var.coords):
            
            # find coord in lr_mesh_var that is closes to the current coordinate
            dist = (coord[0] - lr_mesh_var.coords[:, 0])**2 + (coord[1] - lr_mesh_var.coords[:, 1])**2
            
            # index in the lr_mesh_var that is closest
            closest_idx = np.where(dist == dist.min())[0][0]

            data[i] = lr_mesh_var.data[closest_idx]

    return data


# %%
# interpolate the variables from the low res model to the high res model
if do_rerun: 
    from copy import deepcopy

    with meshbox_hr.access(v_soln_hr, t_soln_hr, p_soln_hr):
        
        # use nearest-neighbord workaround
        t_soln_hr.data[:] = nearest_neigh_workaround(meshbox, meshbox_hr, t_soln, t_soln_hr)
        p_soln_hr.data[:] = nearest_neigh_workaround(meshbox, meshbox_hr, p_soln, p_soln_hr)
        v_soln_hr.data[:] = nearest_neigh_workaround(meshbox, meshbox_hr, v_soln, v_soln_hr)

        # NOTE: something is weird when using the uw.function.evaluate
        # t_soln_hr.data[:, 0] = uw.function.evaluate(t_soln.sym[0], t_soln_hr.coords)
        # p_soln_hr.data[:, 0] = uw.function.evaluate(p_soln.sym[0], p_soln_hr.coords)

        # #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        # v_coords = deepcopy(v_soln_hr.coords)
    
        # cond = v_coords[:, 0] == boxLength      # v_x
        # v_coords[cond, 0] = 0.999999*boxLength 
        
        # cond = v_coords[:, 1] == boxHeight      # v_y
        # v_coords[cond, 1] = 0.999999*boxHeight 

        # v_soln_hr.data[:] = uw.function.evaluate(v_soln.fn, v_coords)

        # final set-up of other variables
        buoyancy_force = Ra * t_soln_hr.sym[0]
        stokes_hr.bodyforce = sympy.Matrix([0, buoyancy_force])

        adv_diff_hr.petsc_options["pc_gamg_agg_nsmooths"] = 5


# %%
# Final temperature field solved by the low resolution model
plotFig(meshbox = meshbox, v_soln = v_soln, t_soln = t_soln, dTdZ = dTdZ, is_grad = False)

# %%
# Field above interpolated into a high resolution mesh
plotFig(meshbox = meshbox_hr, v_soln = v_soln_hr, t_soln = t_soln_hr, dTdZ = dTdZ_hr, is_grad = False)

# %% [markdown]
# ### Main simulation loop for the high resolution model

# %%
if do_rerun:
    # use the t_step and time from the low-res run

    t_step = 0     # better to set this counter to zero
    time = timeVal[-1]  # remove later

    timeVal_hr = np.zeros(nsteps_hr)*np.nan
    vrmsVal_hr = np.zeros(nsteps_hr)*np.nan

    while t_step < nsteps_hr:

        #print(t_step)
        vrmsVal_hr[t_step] = v_rms(meshbox_hr, v_soln_hr)
        timeVal_hr[t_step] = time

        stokes_hr.solve(zero_init_guess = True)                     # it appears that solver converges whether it is True or False 
        delta_t = 0.5 * stokes_hr.estimate_dt()
        adv_diff_hr.solve(timestep=delta_t, zero_init_guess=False) # originally False

        # stats then loop
        tstats = t_soln_hr.stats()

        if uw.mpi.rank == 0:
            print("Timestep {}, dt {}".format(t_step, delta_t))
            
            print(f't_rms = {t_soln.stats()[6]}, v_rms = {v_rms(meshbox_hr, v_soln_hr)}')

        # if t_step % 5 == 0:
        #     plot_T_mesh(filename="{}_step_{}".format(expt_name, t_step))
        if t_step > 1 and abs((vrmsVal_hr[t_step] - vrmsVal_hr[t_step - 1])/vrmsVal_hr[t_step]) < epsilon_hr:
            break

        t_step += 1
        time   += delta_t

# %%

# plot how v_rms is evolving through time for both low res and high res
if do_rerun:
    fig,ax = plt.subplots(dpi = 100)

   # low resolution model
    ax.hlines(42.865, 0, 1000, linestyle = "--", linewidth = 0.5, color = "gray", label = r"Benchmark $v_{rms}$")
    ax.plot(np.arange((~np.isnan(vrmsVal)).sum()), 
            vrmsVal[~np.isnan(vrmsVal)], 
            color = "k", 
            label = str(res) + " x " + str(res))
    
    # high resolution model
    ax.plot(vrmsVal[~np.isnan(vrmsVal)].shape[0] + np.arange((~np.isnan(vrmsVal_hr)).sum()), 
            vrmsVal_hr[~np.isnan(vrmsVal_hr)], 
            linestyle = "--",
            color = "k", 
            label = str(res2) + " x " + str(res2)) # fix later
    
    ax.legend()
    ax.set_xlabel("Time step")
    ax.set_ylabel(r"$v_{rms}$", color = "k")

    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 100])




# %%
# figure of the high resolution model reaching a steady state
plotFig(meshbox = meshbox_hr, v_soln = v_soln_hr, t_soln = t_soln_hr, dTdZ = dTdZ_hr, is_grad = False)

# %% [markdown]
# ### Post-run analysis
# 
# **Benchmark values**
# The loop above outputs $v_{rms}$ as a general statistic for the system. For further comparison, the benchmark values for the RMS velocity, $v_{rms}$, Nusselt number, $Nu$, and non-dimensional gradients at the cell corners, $q_1$ and $q_2$, are shown below for different Rayleigh numbers. All benchmark values shown below were determined in Blankenbach *et al.* 1989 by extroplation of numerical results. 
# 
# | $Ra$ | $v_{rms}$ | $Nu$ | $q_1$ | $q_2$ |
# | ------------- |:-------------:|:-----:|:-----:|:-----:|
# | 10$^4$ | 42.865 | 4.884 | 8.059 | 0.589 |
# | 10$^5$ | 193.215 | 10.535 | 19.079 | 0.723 |
# | 10$^6$ | 833.990 | 21.972 | 45.964 | 0.877 |

# %%
# set-up variables to use in calculating the benchmark values 

if do_rerun: # re-run with higher resolution mesh
    meshbox_use = meshbox_hr
    dTdZ_use = dTdZ_hr 
    t_soln_use = t_soln_hr
    sigma_zz_use = sigma_zz_hr
    sigma_zz_fn = stokes_hr.stress[1, 1]
else: # 
    meshbox_use = meshbox 
    dTdZ_use = dTdZ 
    t_soln_use = t_soln
    sigma_zz_use = sigma_zz

    sigma_zz_fn = stokes.stress[1, 1]

### define the Projection object for calculating the temperature gradient
# define upper and lower surface functions
x, z = meshbox_use.X

# gradient of temp field 
t_soln_grad = uw.systems.Projection(meshbox_use, dTdZ_use)
delT = t_soln_use.sym.diff(z)[0]
t_soln_grad.uw_function = delT
t_soln_grad.smoothing = 1.0e-3
t_soln_grad.petsc_options.delValue("ksp_monitor")

t_soln_grad.solve()

# %% [markdown]
# ### Calculate the $Nu$ value
# The Nusselt number is defined as: 
# 
# \begin{aligned}
# Nu  =   -h\frac{ \int_{0}^{l} \partial_{z}T(x, z = h) dx} {\int_{0}^{l} T(x, z = 0) dx} 
# \end{aligned}

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
up_surface_defn_fn = sympy.exp(-1e6*((z - 1)**2)) # at z = 1
lw_surface_defn_fn = sympy.exp(-1e6*((z)**2)) # at z = 0

# display(up_surface_defn_fn)
# display(lw_surface_defn_fn)

up_int = surface_integral(meshbox_use, dTdZ_use.sym[0], up_surface_defn_fn)
lw_int = surface_integral(meshbox_use, t_soln_use.sym[0], lw_surface_defn_fn)

Nu = -up_int/lw_int

if uw.mpi.rank == 0:
    print("Calculated value of Nu: {}".format(Nu))

# %% [markdown]
# ### Calculate the non-dimensional gradients at the cell corners, $q_i$
# The non-dimensional temperature gradient at the cell corner, $q_i$, is defined as: 
# \begin{aligned}
# q  =  \frac{-h}{\Delta T} \left( \frac{\partial T}{\partial z} \right)
# \end{aligned}
#    
# Note that these values depend on the non-dimensional temperature gradient in the vertical direction, $\frac{\partial T}{\partial z}$.
# These gradients are evaluated at the following points:
# 
# $q_1$ at $x=0$, $z=h$; $q_2$ at $x=l$, $z=h$;
# 
# $q_3$ at $x=l$, $z=0$; $q_4$ at $x=0$, $z=0$.   

# %%
# calculate q values which depend on the temperature gradient fields

q1 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ_use.sym[0], np.array([[0., 0.999999*boxHeight]]))[0]
q2 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ_use.sym[0], np.array([[0.999999*boxLength, 0.999999*boxHeight]]))[0]
q3 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ_use.sym[0], np.array([[0.999999*boxLength, 0.]]))[0]
q4 = -(boxHeight/(tempMax - tempMin))*uw.function.evaluate(dTdZ_use.sym[0], np.array([[0., 0.]]))[0]

if uw.mpi.rank == 0:
    print('Rayleigh number = {0:.1e}'.format(Ra))
    print('q1 = {0:.3f}; q2 = {1:.3f}'.format(q1, q2))
    print('q3 = {0:.3f}; q4 = {1:.3f}'.format(q3, q4))

# %% [markdown]
# ### Calculate the stress for comparison with benchmark value
# 
# The stress field for whole box in dimensionless units (King 2009) is:
# \begin{equation}
# \tau_{ij} = \eta \frac{1}{2} \left[ \frac{\partial v_j}{\partial x_i} + \frac{\partial v_i}{\partial x_j}\right].
# \end{equation}
# For vertical normal stress it becomes:
# \begin{equation}
# \tau_{zz} = \eta \frac{1}{2} \left[ \frac{\partial v_z}{\partial z} + \frac{\partial v_z}{\partial z}\right] = \eta \frac{\partial v_z}{\partial z}.
# \end{equation}
# This is calculated below.

# %%
# projection for the stress in the zz direction
x, y = meshbox_use.X

stress_calc = uw.systems.Projection(meshbox_use, sigma_zz_use)
stress_calc.uw_function = sigma_zz_fn
stress_calc.smoothing = 1.0e-3
stress_calc.petsc_options.delValue("ksp_monitor")
stress_calc.solve()

# %% [markdown]
# The vertical normal stress is dimensionalised as: 
# 
# $$
#     \sigma_{t} = \frac{\eta_0 \kappa}{\rho g h^2}\tau _{zz} \left( x, z=h\right)
# $$
# 
# where all constants are defined below. 
# 
# Finally, we calculate the topography defined using $h = \sigma_{top} / (\rho g)$. The topography of the top boundary calculated in the left and right corners as given in Table 9 of Blankenbach et al 1989 are:
# 
# | $Ra$          |    $\xi_1$  | $\xi_2$  |  $x$ ($\xi = 0$) |
# | ------------- |:-----------:|:--------:|:--------------:|
# | 10$^4$  | 2254.02   | -2903.23  | 0.539372          |
# | 10$^5$  | 1460.99   | -2004.20  | 0.529330          |
# | 10$^6$  | 931.96   | -1283.80  | 0.506490          |

# %%
# subtract the average value for the benchmark since the mean is set to zero 

mean_sigma_zz_top = -surface_integral(meshbox_use, 
                                     sigma_zz_use.sym[0], 
                                     up_surface_defn_fn)/boxLength


# %%
# Set parameters in SI units

grav = 10        # m.s^-2
height = 1.e6    # m 
rho  = 4.0e3     # g.m^-3
kappa  = 1.0e-6  # m^2.s^-1

eta0=1.e23

def calculate_topography(coord): # only coord has local scope

    sigma_zz_top = -uw.function.evaluate(sigma_zz_use.sym[0], coord) - mean_sigma_zz_top
    
    # dimensionalise 
    dim_sigma_zz_top  = ((eta0 * kappa) / (height**2)) * sigma_zz_top
    topography = dim_sigma_zz_top / (rho * grav)
    
    return topography



# %%
# topography at the top corners 
e1 = calculate_topography(np.array([[0, 0.999999*boxHeight]]))
e2 = calculate_topography(np.array([[0.999999*boxLength, 0.999999*boxHeight]]))

# calculate the x-coordinate with zero stress
with meshbox_use.access():
    cond = meshbox_use.data[:, 1] == meshbox_use.data[:, 1].max()
    up_surface_coords = meshbox_use.data[cond]
    up_surface_coords[:, 1] = 0.999999*up_surface_coords[:, 1]

abs_topo = abs(calculate_topography(up_surface_coords))

min_abs_topo_coord = up_surface_coords[np.where(abs_topo == abs_topo.min())[0]].flatten()

if uw.mpi.rank == 0:
    print("Topography [x=0], [x=max] = {0:.2f}, {1:.2f}".format(e1[0], e2[0]))
    print("x where topo = 0 is at {0:.6f}".format(min_abs_topo_coord[0]))


