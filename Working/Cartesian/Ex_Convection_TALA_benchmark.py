# %% [markdown]
# # King / Blankenbach Benchmark Case 1
# 
# ## Isoviscous thermal convection with TALA formulation.
# 
# Two-dimensional, compressible, bottom heated, steady isoviscous thermal convection in a 1 x 1 box, see case 1 of [King et al. (2009)](https://doi.org/10.1111/j.1365-246X.2009.04413.x) / [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
# 
# Keywords: Truncated Anelastic Liquid Approximation, TALA, Convection
# 

# %%
import petsc4py
from petsc4py import PETSc

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import os 
import numpy as np
import sympy
from sympy.vector import gradient, divergence, dot
from copy import deepcopy 

# %% [markdown]
# ### Set parameters to use 

# %%
Di = 0.5 # dissipation factor 
Ra = 1e4 #### Rayleigh number

# other non-dimensional parameters
alphaBar = 1.      # this is set to 1 in King et al. 
gammaR = 1.        # Gruneisen parameter
cp = 1             # set to 1 for now
Ts = 0.091         # non-dimensionalized surface temp (273 K/3000 K as per Davies et al)
rho0  = 1          # non-dimensionalized surface density 

k = 1.0 #### diffusivity

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.
deltaTemp = tempMax - tempMin
viscosity = 1

#### run configuration
tol = 1e-5              ### solver tolerance
res= 96                 ### x and y res of box
nsteps = 2             ### maximum number of time steps to run the first model 
epsilon_lr = 1e-8       ### criteria for early stopping; relative change of the Nusselt number in between iterations  
use_checkpoint = True   ### if set to True, use T, p, v fields close to steady-state provided in repo
                        ### will also neglect output config below if set to True

##########
# parameters needed for saving checkpoints
# will be ignored if use_checkpoint set to True
outdir = "/Users/jcgraciosa/Documents/codes/uw3-dev/TALA_test" 
outfile = outdir + "/convection_16"
save_every = 10
prev_res = 16           ### if infile is not None, then this should be set to the previous model resolution
infile = outfile        ### set infile to a value if there's a checkpoint from a previous run that you want to start from
##########

if use_checkpoint:
    prev_res = 96 
    infile = "./UW3-benchmarks/Data/TALA_benchmark/TALA_96" 
    #infile = "/Users/jgra0019/Documents/codes/uw3-dev/benchmarks/UW3-benchmarks/Data/TALA_benchmark/TALA_96"

if not use_checkpoint: 
    if uw.mpi.rank == 0:
        os.makedirs(outdir, exist_ok = True)

# %% [markdown]
# ### Create mesh and variables

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords=(0.0, 0.0), 
                                                maxCoords=(boxLength, boxHeight), 
                                                cellSize=1.0 /res, 
                                                regular=False, 
                                                qdegree = 3
                                        )

#meshbox = uw.meshing.StructuredQuadBox(minCoords=(0.0, 0.0), maxCoords=(boxLength, boxHeight),  elementRes=(res,res))


# %% [markdown]
# ## Reference values 
# The temperature, T, pressure, p, and density are expressed as a sum of a reference state and a departure from this state:
# 
# \begin{aligned}
# T = \bar T + T'\space \; \; \; p = \bar p + p'  \; \; \;  \rho = \bar \rho (\bar T, \bar p) + \rho'    
# \end{aligned}
# 
# Where the overbarred quantities are the reference states, and the primed quantities are the perturbations. The reference states are temporally static and varies with depth, z.  

# %%
# the non-dimensionalized reference states are defined here
# In King et al and Davies et al, the reference states are set such that z = 0 is the top surface 
# and z = 1 is the lower surface
# For our case, z = 0 is the bottom, while z = 1 is the top. Because of this, we use 1 - z as coordinates
# for the reference values 

x, z = meshbox.X
rhoBar     = rho0*sympy.exp(Di*(1 - z)/gammaR)        
temp_bar    = Ts*(sympy.exp(Di*(1 - z)) - 1)            # Ts is the non-dimensionalized surface temperature  

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
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)

# additional variable for the gradient
dTdZ = uw.discretisation.MeshVariable(r"\partial T/ \partial \Z", 
                                      meshbox, 
                                      1, 
                                      degree = 3) 

# projection object to calculate the gradient along Z
dTdZ_calc = uw.systems.Projection(meshbox, dTdZ)
dTdZ_calc.uw_function = t_soln.sym.diff(z)[0]
dTdZ_calc.smoothing = 1.0e-3
dTdZ_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
# ### System set-up (Stokes)
# In the Truncated Anelastic Liquid Approximation, the conservation of mass is expressed as: 
# \begin{aligned}
# \nabla \cdot (\bar \rho \vec u) = \nabla \cdot \vec u + \frac{1}{\bar \rho} \vec u \nabla \bar \rho = 0.
# \end{aligned}
# While the conservation of momentum is given as: 
# \begin{aligned}
# \nabla \cdot [\eta (\nabla \vec u + \nabla \vec u^T - \frac {2}{3}\nabla \cdot \vec u \bold I)] - \nabla p' - Ra \bar \rho \bar g \bar \alpha (T - \bar T)= 0.
# \end{aligned}
# The deviatoric strain tensor, $\tau$, is: 
# \begin{aligned}
# \tau = 2 \eta \dot \epsilon = \eta (\nabla \vec u + \nabla \vec u^T - \frac {2}{3}\nabla \cdot \vec u \bold I) 
# \end{aligned}
# 
#  

# %%
# Create Stokes object

stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
    solver_name="stokes",
)
''' set-up petsc options here '''
stokes.petsc_options["snes_type"] = "ksponly"
stokes.tolerance = tol
stokes.petsc_options["snes_max_it"] = 500

# additional term in constitutive equation? 
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(meshbox.dim)
stokes.constitutive_model.Parameters.viscosity=viscosity
stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((0.0,), "Left", (0,))
stokes.add_dirichlet_bc((0.0,), "Right", (0,))
stokes.add_dirichlet_bc((0.0,), "Top", (1,))
stokes.add_dirichlet_bc((0.0,), "Bottom", (1,))

# NOTE: we set z = 0 to be the top and z = 1 to be bottom
buoyancy_force = Ra * rhoBar * alphaBar * (t_soln.sym[0] - temp_bar) # directed downwards toward z = 0 (surface)
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# add terms into the conservation of mass according to the formulation of TALA compressible convection
stokes.PF0 = sympy.Matrix([dot(v_soln.fn, gradient(rhoBar))/rhoBar])
stokes.UF1 = (-2/3)*divergence(v_soln.fn) * sympy.eye(meshbox.dim)

stokes.stokes_problem_description() 

# %%
# check the equations for the Stokes system
display(stokes._p_f0) # LHS of c.o. mass 
display(stokes._u_f1) # LHS of c.o. momentum
display(stokes._u_f0) # RHS of c.o. momentum / buoyancy force

# %% [markdown]
# ### System set-up (Advection-Diffusion)
# In the TALA formulation, the conservation of energy in terms of the total temperature, T, is expressed as:
# 
# \begin{aligned}
# \frac{DT}{Dt} - \frac{Di \bar \alpha}{\bar c_p}\vec g \cdot \vec u (T + T_s) = \frac {1} {\bar \rho \bar c_p} \nabla \cdot (\bar k \nabla T) + \frac{Di}{Ra} \frac{1}{\bar \rho \bar c_p} \phi
# \end{aligned}
#  

# %%
# Create adv_diff object

adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_Field=v_soln,
    solver_name="adv_diff",
)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(meshbox.dim)
adv_diff.constitutive_model.Parameters.diffusivity = (k/(rhoBar*cp))
adv_diff.tolerance = tol

# add source terms based needed for the EBA case

#viscous dissipation term
comp_strainrate = stokes.strainrate -(2/3)*divergence(v_soln.fn) * sympy.eye(meshbox.dim)
visc_diss = 2*viscosity*comp_strainrate.norm()**2

# adiabatic term
adiab_heat = (Di*alphaBar/cp)*v_soln.sym[1]*(t_soln.sym[0] + Ts)

adv_diff.f = (Di/Ra)*(1/(rhoBar*cp))*visc_diss - adiab_heat #- term3 

# Dirichlet boundary conditions for temperature
# note that that Top is now the source of heat - deep in the mantle
# z = 0 is the surface 
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.adv_diff_slcn_problem_description() # need to run this? 

# %%
# comp_strainrate = stokes.strainrate -(2/3)*divergence(v_soln.fn) * sympy.eye(meshbox.dim)
# sympy.simplify(comp_strainrate.norm()**2)

# %% [markdown]
# ### Set initial temperature field 
# 
# The initial temperature field is set to a sinusoidal perturbation. 

# %%
import math, sympy

if infile is None:
    pertStrength = 0.1
    deltaTemp = tempMax - tempMin

    with meshbox.access(t_soln):
        t_soln.data[:] = 0.

    with meshbox.access(t_soln):
        for index, coord in enumerate(t_soln.coords):
            # print(index, coord)
            pertCoeff = math.cos( math.pi * coord[0]/boxLength ) * math.sin( math.pi * coord[1]/boxLength )
        
            t_soln.data[index] = tempMin + deltaTemp*(boxHeight - coord[1]) + pertStrength * pertCoeff
            t_soln.data[index] = max(tempMin, min(tempMax, t_soln.data[index]))
else:
    meshbox_prev = uw.meshing.UnstructuredSimplexBox(
                                                            minCoords=(0.0, 0.0), 
                                                            maxCoords=(boxLength, boxHeight), 
                                                            cellSize=1.0/prev_res,
                                                            qdegree = 3,
                                                            regular = False
                                                        )
    
    # T should have high degree for it to converge
    v_soln_prev = uw.discretisation.MeshVariable("U2", meshbox_prev, meshbox_prev.dim, degree=2) # degree = 2
    p_soln_prev = uw.discretisation.MeshVariable("P2", meshbox_prev, 1, degree=1) # degree = 1
    t_soln_prev = uw.discretisation.MeshVariable("T2", meshbox_prev, 1, degree=3) # degree = 3

    v_soln_prev.read_from_vertex_checkpoint(infile + ".U.0.h5", data_name="U")
    p_soln_prev.read_from_vertex_checkpoint(infile + ".P.0.h5", data_name="P")
    t_soln_prev.read_from_vertex_checkpoint(infile + ".T.0.h5", data_name="T")

    with meshbox.access(v_soln, t_soln, p_soln):    
        t_soln.data[:, 0] = uw.function.evaluate(t_soln_prev.sym[0], t_soln.coords)
        p_soln.data[:, 0] = uw.function.evaluate(p_soln_prev.sym[0], p_soln.coords)

        #for velocity, encounters errors when trying to interpolate in the non-zero boundaries of the mesh variables 
        v_coords = deepcopy(v_soln.coords)

        v_soln.data[:] = uw.function.evaluate(v_soln_prev.fn, v_coords)

    del meshbox_prev
    del v_soln_prev
    del p_soln_prev
    del t_soln_prev

# %%
with meshbox.access():
    print()

# %% [markdown]
# ### Some plotting and analysis tools 

# %%
# check the mesh if in a notebook / serial
# allows you to visualise the mesh and the mesh variable
def plotFig(meshbox, s_field, v_field, s_field_name, save_fname = None, with_arrows = False, cmap = "coolwarm"): 
    """
    s_field - scalar field - corresponds to colors
    v_field - vector field - usually the velocity - 2 components
    """
    if uw.mpi.size == 1:

        import numpy as np
        import pyvista as pv
        import vtk

        pv.global_theme.background = "white"
        pv.global_theme.window_size = [500, 500]
        pv.global_theme.anti_aliasing = None #"ssaa", "msaa", "fxaa", or None
        #pv.global_theme.jupyter_backend = "panel"
        pv.global_theme.smooth_shading = True

        meshbox.vtk("tmp_box_mesh.vtk")
        pvmesh = pv.read("tmp_box_mesh.vtk")

        velocity = np.zeros((meshbox.data.shape[0], 3))
        velocity[:, 0] = uw.function.evaluate(v_field.sym[0], meshbox.data)
        velocity[:, 1] = uw.function.evaluate(v_field.sym[1], meshbox.data)

        #pvmesh.point_data["V"] = velocity / 10

        points = np.zeros((s_field.coords.shape[0], 3))
        points[:, 0] = s_field.coords[:, 0]
        points[:, 1] = s_field.coords[:, 1]

        point_cloud = pv.PolyData(points)

        with meshbox.access():
            point_cloud.point_data[s_field_name] = uw.function.evaluate(s_field.fn, points[:, 0:2])

        skip = 2
        num_row = len(meshbox._centroids[::skip, 0])

        cpoints = np.zeros((num_row, 3))
        cpoints[:, 0] = meshbox._centroids[::skip, 0]
        cpoints[:, 1] = meshbox._centroids[::skip, 1]

        cpoint_cloud = pv.PolyData(cpoints)

        # pvstream = pvmesh.streamlines_from_source(
        #     cpoint_cloud,
        #     vectors="V",
        #     integrator_type=2,
        #     integration_direction="forward",
        #     compute_vorticity=False,
        #     max_steps=1000,
        #     surface_streamlines=True,
        # )
 
        pl = pv.Plotter()

        with meshbox.access():
            skip = 2
        
            num_row = len(v_field.coords[::skip, 0:2])

            arrow_loc = np.zeros((num_row, 3))
            arrow_loc[:, 0:2] = v_field.coords[::skip, 0:2]

            arrow_length = np.zeros((num_row, 3))
            arrow_length[:, 0] = v_field.data[::skip, 0]
            arrow_length[:, 1] = v_field.data[::skip, 1]

        pl = pv.Plotter()

        #pl.add_mesh(pvmesh,'Gray', 'wireframe')

        pl.add_mesh(
            pvmesh, cmap=cmap, edge_color="Black",
            show_edges=True, use_transparency=False, opacity=0.1,
        )

      
        if with_arrows:
            pl.add_arrows(arrow_loc, arrow_length, mag=0.04, opacity=0.8)
        else:
            pl.add_points(point_cloud, cmap=cmap, point_size=18, opacity=0.8)


        # pl.add_mesh(pvstream, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy", jupyter_backend = "panel")

        if save_fname is not None:
            #pl.save_graphic(save_fname, dpi = 300)
            pl.image_scale = 3
            pl.screenshot(save_fname) 

        pvmesh.clear_data()
        pvmesh.clear_point_data()
        
        
plotFig(meshbox, t_soln, v_soln, "T", save_fname = None, with_arrows = False, cmap = "coolwarm")

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
# #### Surface integrals
# Since there is no uw3 function yet to calculate the surface integral, we define one.  \
# The surface integral of a function, $f_i(\mathbf{x})$, is approximated as:  
# 
# \begin{aligned}
# F_i = \int_V f_i(\mathbf{x}) S(\mathbf{x})  dV  
# \end{aligned}
# 
# With $S(\mathbf{x})$ defined as an un-normalized Gaussian function with the maximum at $z = a$  - the surface we want to evaluate the integral in (e.g. z = 1 for surface integral at the top surface):
# 
# \begin{aligned}
# S(\mathbf{x}) = exp \left( \frac{-(z-a)^2}{2\sigma ^2} \right)
# \end{aligned}
# 
# In addition, the full-width at half maximum is set to 1/res so the standard deviation, $\sigma$ is calculated as: 
# 
# \begin{aligned}
# \sigma = \frac{1}{2}\frac{1}{\sqrt{ 2 log 2}}\frac{1}{res} 
# \end{aligned}

# %%
# function for calculating the surface integral 
def surface_integral(mesh, uw_function, mask_fn):

    calculator = uw.maths.Integral(mesh, uw_function * mask_fn)
    value = calculator.evaluate()

    calculator.fn = mask_fn
    norm = calculator.evaluate()

    integral = value / norm

    return integral

''' set-up surface expressions for calculating Nu number '''
sdev = 0.5*(1/math.sqrt(2*math.log(2)))*(1/res) 

up_surface_defn_fn = sympy.exp(-((z - 1)**2)/(2*sdev**2)) # at z = 1
lw_surface_defn_fn = sympy.exp(-(z**2)/(2*sdev**2)) # at z = 0

# %%
# functions for calculating the viscous dissipation and adiabatic heating integrals 
# used for checking since they should be equal
visc_diss_int_calc = uw.maths.Integral(meshbox, visc_diss)
adiab_heat_int_calc = uw.maths.Integral(meshbox, adiab_heat)

# %% [markdown]
# ### Main simulation loop

# %%
t_step = 0
time = 0.

timeVal =  np.zeros(nsteps)*np.nan
vrmsVal =  np.zeros(nsteps)*np.nan
NuVal =  np.zeros(nsteps)*np.nan
viscDissVal = np.zeros(nsteps)*np.nan
adiabHeatVal = np.zeros(nsteps)*np.nan

# %%
#### Convection model / update in time
# NOTE: There is a strange interaction here between the solvers if the zero_guess is set to False
while t_step < nsteps:
    vrmsVal[t_step] = v_rms()
    timeVal[t_step] = time

    stokes.solve(zero_init_guess=True) # originally True
    delta_t = stokes.estimate_dt()
    adv_diff.solve(timestep=delta_t, zero_init_guess=False) # originally False

    # calculate Nusselt number
    # for this case, top surface is set to 1, while bottom is set to 0
    dTdZ_calc.solve()
    up_int = surface_integral(meshbox, dTdZ.sym[0], up_surface_defn_fn)
    lw_int = surface_integral(meshbox, t_soln.sym[0], lw_surface_defn_fn)

    Nu = -up_int/lw_int

    NuVal[t_step] = -up_int/lw_int

    # calculate the integrals of viscous dissipation and adiabatic heating
    viscDissVal[t_step] = visc_diss_int_calc.evaluate()
    adiabHeatVal[t_step] = adiab_heat_int_calc.evaluate()

    # stats then loop
    tstats = t_soln.stats()

    if uw.mpi.rank == 0:
        print("Timestep {}, dt {}".format(t_step, delta_t))
        print(f't_rms = {t_soln.stats()[6]}, v_rms = {vrmsVal[t_step]}, Nu = {NuVal[t_step]}')

        ''' save mesh variables together with mesh '''
        if t_step % save_every == 0 and not use_checkpoint:
            print("Saving checkpoint for time step: ", t_step)
            meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln, dTdZ], index=0)

    # early stopping criterion
    if t_step > 1 and abs((NuVal[t_step] - NuVal[t_step - 1])/NuVal[t_step]) < epsilon_lr:
        break

    t_step += 1
    time   += delta_t

# save final mesh variables in the run 
if not use_checkpoint:
    os.makedirs("../TALA_meshes", exist_ok = True)
    expt_name = "TALA_Ra1e4_res" + str(res)
    savefile = "{}_ts{}.h5".format(expt_name,t_step)
    # save final mesh variables in the run 
    meshbox.write_timestep_xdmf(filename = outfile, meshVars=[v_soln, p_soln, t_soln, dTdZ, sigma_zz], index=0)

# %%
# Calculate some benchmark values
if uw.mpi.rank == 0:
    print("RMS velocity at the final time step is {}.".format(v_rms()))
    print("Nusselt number at the final time step is {}.".format(Nu))


