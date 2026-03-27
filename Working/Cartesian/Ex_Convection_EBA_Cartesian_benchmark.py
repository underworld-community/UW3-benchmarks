# %% [markdown]
# # King / Blankenbach Benchmark Case 1
#
# ## Isoviscous thermal convection with EBA formulation.
#
# Two-dimensional, incompressible, bottom heated, steady isoviscous thermal convection in a 1 x 1 box following the Extended Boussinesq Approximation formulation. See case 1 of [King et al. (2009)](https://doi.org/10.1111/j.1365-246X.2009.04413.x) / [Blankenbach et al. (1989) benchmark](https://academic.oup.com/gji/article/98/1/23/622167).
#
# Extended Boussinesq Approximation, EBA, Stokes flow formulation.
#

# %%
import sys
import petsc4py
petsc4py.init(sys.argv)

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3 import function

import os
import numpy as np
import sympy

params = uw.Params(
    uw_Ra      = 1e4,
    uw_res     = 16,
    uw_n_steps = 5,
    uw_infile  = None,
)

Ra     = params.uw_Ra
res    = params.uw_res
nsteps = params.uw_n_steps
infile = params.uw_infile

show_plot = False

# %% [markdown]
# ### Set parameters to use

# %%
Di = 0.5        # dissipation factor
alphaBar = 1.   # set to 1 in King et al.
cp = 1          # set to 1 for now
Ts = 0.091      # non-dimensionalized surface temperature

k = 1.0         # diffusivity

boxLength = 1.0
boxHeight = 1.0
tempMin   = 0.
tempMax   = 1.

viscosity = 1

tol = 1e-6
epsilon_lr = 1e-8   # criteria for early stopping; relative change of Vrms between iterations
max_its = 100

##########
# parameters needed for saving checkpoints
outdir     = "./output/EBA_test"
outfile    = "EBA_checkpoint"
save_every = 10

if uw.mpi.rank == 0:
    os.makedirs(outdir, exist_ok=True)

# %% [markdown]
# ### Create mesh and variables

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(boxLength, boxHeight),
    cellSize=1.0 / res,
    qdegree=3,
    regular=False
)

# %%
v_soln = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree=2)
p_soln = uw.discretisation.MeshVariable("P", meshbox, 1, degree=1)
t_soln = uw.discretisation.MeshVariable("T", meshbox, 1, degree=3)

# additional variable for the gradient
dTdZ = uw.discretisation.MeshVariable(r"\partial T/ \partial \Z",
                                      meshbox,
                                      1,
                                      degree=3)

x, z = meshbox.X

# projection object to calculate the gradient along Z
dTdZ_calc = uw.systems.Projection(meshbox, dTdZ)
dTdZ_calc.uw_function = t_soln.sym.diff(z)[0]
dTdZ_calc.smoothing = 1.0e-3
dTdZ_calc.petsc_options.delValue("ksp_monitor")

# %% [markdown]
# ### System set-up (Stokes)
#
# In the Extended Boussinesq Approximation, the conservation of mass is expressed as:
# \begin{aligned}
# \nabla \cdot \vec u = 0.
# \end{aligned}
# While the conservation of momentum is given as:
# \begin{aligned}
# \nabla \cdot [\eta (\nabla \vec u + \nabla \vec u^T)] - \nabla p - Ra g \bar \alpha T= 0.
# \end{aligned}

# %%
stokes = Stokes(
    meshbox,
    velocityField=v_soln,
    pressureField=p_soln,
)

stokes.tolerance = tol
stokes.petsc_options["snes_max_it"] = max_its
stokes.petsc_options["ksp_max_it"] = max_its
stokes.petsc_options["snes_converged_reason"] = None

stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.viscosity = viscosity
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
stokes.saddle_preconditioner = 1.0 / viscosity

# Free-slip boundary conditions
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Bottom")
stokes.add_dirichlet_bc((sympy.oo, 0.0), "Top")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Left")
stokes.add_dirichlet_bc((0.0, sympy.oo), "Right")

buoyancy_force = Ra * alphaBar * t_soln.sym[0]
stokes.bodyforce = sympy.Matrix([0, buoyancy_force])

# %% [markdown]
# ### System set-up (Advection-Diffusion)
# In the EBA formulation, the conservation of energy in terms of the total temperature, T, is expressed as:
#
# \begin{aligned}
# \frac{DT}{Dt} - \frac{Di \bar \alpha}{\bar c_p}\vec g \cdot \vec u T = \frac {1} {\bar c_p} \nabla \cdot (\bar k \nabla T) + \frac{Di}{Ra} \frac{1}{\bar c_p} \phi
# \end{aligned}

# %%
adv_diff = uw.systems.AdvDiffusionSLCN(
    meshbox,
    u_Field=t_soln,
    V_fn=v_soln,
)

adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
adv_diff.constitutive_model.Parameters.diffusivity = k / cp

# viscous dissipation term
visc_diss = 2 * viscosity * stokes.strainrate.norm()**2

# adiabatic heating term
adiab_heat = (Di * alphaBar / cp) * v_soln.sym[1] * (t_soln.sym[0] + Ts)

# add EBA source terms
adv_diff.f = (Di / (Ra * cp)) * visc_diss - adiab_heat

adv_diff.theta = 0.5

# Dirichlet boundary conditions for temperature
adv_diff.add_dirichlet_bc(1.0, "Bottom")
adv_diff.add_dirichlet_bc(0.0, "Top")

adv_diff.petsc_options["pc_gamg_agg_nsmooths"] = 5
adv_diff.petsc_options["snes_converged_reason"] = None

adv_diff.tolerance = tol
adv_diff.petsc_options["snes_max_it"] = max_its
adv_diff.petsc_options["ksp_max_it"] = max_its

# %% [markdown]
# ### Set initial temperature field
#
# The initial temperature field is set to a sinusoidal perturbation.

# %%
if infile is None:
    pertStrength = 0.1

    T_init = (tempMin + (tempMax - tempMin) * (boxHeight - z)
              + pertStrength * sympy.cos(sympy.pi * x / boxLength) * sympy.sin(sympy.pi * z / boxLength))

    t_soln.data[:] = 0.
    t_soln.data[:, 0] = np.clip(uw.function.evaluate(T_init, t_soln.coords).squeeze(), tempMin, tempMax)

else:
    t_soln.read_timestep(data_filename=infile,
                         data_name="T",
                         index=0,
                         outputPath=outdir)
    uw.pprint("Reading t_soln from checkpoint file ...")
    uw.pprint(f"min/max temp: {t_soln.data.min():.4f} / {t_soln.data.max():.4f}", proc="all")

# %% [markdown]
# ### Some plotting and analysis tools

# %%
def v_rms(mesh=meshbox, v_solution=v_soln):
    # v_soln must be a variable of mesh
    v_rms = np.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

# integrals for checking energy balance (viscous dissipation should equal adiabatic heating at steady state)
visc_diss_int_calc  = uw.maths.Integral(meshbox, (Di / Ra) * visc_diss)
adiab_heat_int_calc = uw.maths.Integral(meshbox, adiab_heat)

# %% [markdown]
# ### Main simulation loop

# %%
t_step = 0
time = 0.

timeVal      = np.zeros(nsteps) * np.nan
vrmsVal      = np.zeros(nsteps) * np.nan
NuVal        = np.zeros(nsteps) * np.nan
viscDissVal  = np.zeros(nsteps) * np.nan
adiabHeatVal = np.zeros(nsteps) * np.nan

uw.pprint("Initial solve")

# %%
#### Convection model / update in time

while t_step < nsteps:
    vrmsVal[t_step] = v_rms()
    timeVal[t_step] = time

    stokes.solve(zero_init_guess=True)

    t_adv   = adv_diff.estimate_dt()
    delta_t = 5 * t_adv
    adv_diff.solve(timestep=delta_t, zero_init_guess=False)

    # calculate Nusselt number
    dTdZ_calc.solve()
    up_int = uw.maths.BdIntegral(meshbox, dTdZ.sym[0], "Top").evaluate()
    lw_int = uw.maths.BdIntegral(meshbox, t_soln.sym[0], "Bottom").evaluate()
    NuVal[t_step] = -1. * up_int / lw_int

    # calculate energy balance diagnostics
    viscDissVal[t_step]  = visc_diss_int_calc.evaluate()
    adiabHeatVal[t_step] = adiab_heat_int_calc.evaluate()

    uw.pprint(f"Timestep {t_step}, dt {delta_t}, v_rms {vrmsVal[t_step]}, Nu {NuVal[t_step]}")
    uw.pprint(f"  visc_diss={viscDissVal[t_step]:.4f}, adiab_heat={adiabHeatVal[t_step]:.4f}")
    uw.pprint(f"  min/max temp: {t_soln.data.min():.4f} / {t_soln.data.max():.4f}", proc="all", flush=True)

    ''' save mesh variables together with mesh '''
    if t_step % save_every == 0 and t_step > 0:
        uw.pprint(f"Saving checkpoint for time step: {t_step}")
        meshbox.write_timestep(outfile, outputPath=outdir, meshVars=[t_soln, v_soln, p_soln], index=0)

    # early stopping criterion
    if t_step > 1 and abs((vrmsVal[t_step] - vrmsVal[t_step - 1]) / vrmsVal[t_step - 1]) < epsilon_lr:
        uw.pprint("Stopping criterion reached ...")
        break

    t_step += 1
    time   += delta_t

# save final mesh variables in the run
uw.pprint("End of loop reached ...")
uw.pprint(f"min/max temp: {t_soln.data.min():.4f} / {t_soln.data.max():.4f}", proc="all", flush=True)

meshbox.write_timestep(outfile, outputPath=outdir, meshVars=[t_soln, v_soln, p_soln], index=0)
