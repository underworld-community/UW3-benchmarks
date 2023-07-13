# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# ## Viscous fingering model
#
# Based on Darcy flow and advection-diffusion of two fluids with varying viscosity.
#
# From [Guy Simpson - Practical Finite Element Modeling in Earth Science using Matlab (2017)](https://www.wiley.com/en-au/Practical+Finite+Element+Modeling+in+Earth+Science+using+Matlab-p-9781119248620)
#
# - Section 10.2 of the book
#
# - Modified to test temperature-dependent viscosity on fingering
#
# #### Darcy velocity:
# $$ u = - \frac{k}{\mu_c}\nabla p$$
#
# ### viscosity:
# $$ \mu_T = \mu_0 + e^{-\beta (T-T_0)} $$
#
# #### Advection-diffusion of material:
# $$ \varphi \frac{\delta c}{\delta t} + \nabla(uc) = \nabla(\kappa\nabla c)  $$
#
#
#
# ##### Model physical parameters:
#
#
# | paramter | symbol  | value  | units  |   |
# |---|---|---|---|---|
# | x |  | $$10$$  | $$m$$  |   |
# | y  |  | $$10$$  | $$m$$  |   |
# | permeability  | $$k$$ | $$10^{-15}$$  | $$m^2$$  |   |
# | porosity  | $$\varphi$$ | $$0.2$$ |   |   |
# | diffusivity  | $$\kappa$$  | $$10^{-6}$$  | $$m^2 s^{-1}$$  |   |
# | viscosity  | $$\eta_{0}$$ | $$10^{2}$$  | $$Pa s$$  |   |
# | pressure  | $$p$$  | $$10^{5}$$  | $$Pa$$  |   |
#

# +
from petsc4py import PETSc
import underworld3 as uw
import numpy as np
import sympy

from scipy.interpolate import griddata, interp1d

import matplotlib.pyplot as plt

import os

options = PETSc.Options()

# +
res = 100


outputDir = f'./output/viscousFingering_temperature_example_res={res}/'

if uw.mpi.rank==0:
    ### create folder if not run before
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

# +
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim, nd = uw.scaling.non_dimensionalise, uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

boxLength     = 10 ### length and height of box in meters
g             = 9.81
eta           = 1e-4
kappa         = 1e-6 ### m^2/s
perm          = 30*(1e-3*u.darcy).to(u.meter**2).m ### 30mD, 1mD = 1e-15  #1e-13 ### m^2
porosity      = 0.2
T_0           = 273.15
T_1           = 1573.15
dT            = T_1 - T_0


refTime        = perm / kappa

refViscosity   = eta * u.pascal * u.second



# KL = boxLength     * u.meter
KL = np.sqrt(perm) * u.meter
KT = dT            * u.kelvin
Kt = refTime       * u.second
KM = refViscosity  * KL * Kt



### create unit registry
scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
scaling_coefficients["[temperature]"]= KT
scaling_coefficients

# +
minX, maxX = 0, nd(10*u.meter)
minY, maxY = 0, nd(10*u.meter)

# mesh = uw.meshing.UnstructuredSimplexBox(
#     minCoords=(minX, minY), maxCoords=(maxX, maxY), cellSize=0.02, qdegree=3)

mesh = uw.meshing.StructuredQuadBox(elementRes=(res,res),
                                      minCoords=(minX,minY),
                                      maxCoords=(maxX,maxY), qdegree=3 )

## define our pressure, velocity and material on the mesh. 
p_soln = uw.discretisation.MeshVariable("P", mesh, 1, degree=3) ## scalar
v_soln = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2) ## vector (2d) 
temp   = uw.discretisation.MeshVariable("T", mesh, 1, degree=3) ## scalar
visc   = uw.discretisation.MeshVariable("\eta", mesh, 1, degree=1) ## scalar

# x and y coordinates
x = mesh.N.x
y = mesh.N.y
# -

## print out what the mesh looks like
if uw.mpi.size == 1:

    # plot the mesh
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")

    pl = pv.Plotter()

    pl.add_mesh(
        pvmesh,
        cmap="coolwarm",
        edge_color="Black",
        show_edges=True,
        use_transparency=False,
    )

    pl.show(cpos="xy")

# +
swarm = uw.swarm.Swarm(mesh=mesh) ## create a swarm on the mesh

## material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
## create a swarm variable to keep track of the material in space
temp_swarm = swarm.add_variable(name='T', size=1, proxy_degree=temp.degree) 

swarm.populate(fill_param=temp.degree)
# -

# Create Darcy Solver on the mesh
darcy = uw.systems.SteadyStateDarcy(mesh, u_Field=p_soln, v_Field=v_soln)
darcy.petsc_options.delValue("ksp_monitor")

darcy.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)

# - The Darcy equation relates the gradients in pressure to the flux of material (velocity).
# - Physically, then, the velocity (vector) is proportional to the spatial derivative of the pressure.
# - The Darcy solver caculates the darcy_flux as the flux of the flow parameter minus any bodyforce term.
# - Although the flow parameter is termed 'diffusivity', any equation can be used to represent the flow parameter. 
# - $\frac{k}{\mu_c}$ is the parameter (which are combined and entered into the model as the darcy.constitutive_model.Parameters.diffusivity parameter) that relates the pressure gradients to the resultant velocities.
#
#
# In the cells bellow we define the viscosity, which is material depedent and non-linear. From the viscosity we define the flow term in our darcy consitutive equation.

# ##### Create a symbolic equation using sympy

# +
perm_sym = sympy.symbols('k')
eta_sym  = sympy.symbols('\eta_c')
flow_term_symbolic = perm_sym/eta_sym

darcy.constitutive_model.Parameters.diffusivity = flow_term_symbolic

-darcy.darcy_flux
# -

# Which is the equivient of:
# $$ u = - \frac{k}{\eta_c}\nabla p$$

# ##### Replace symbolic representation with actual values

# #### [Temp dependent viscosity](https://npg.copernicus.org/articles/10/545/2003/npg-10-545-2003.pdf)
#
# - µ = µ0 exp[−β(T − T0)]
# - with β = B/T0
# - and µ0 = µA exp(B/T0)

# +
eta_0 = nd(refViscosity)

### use the mesh var to map composition to viscosity
# eta_fn = (mat.sym[0]/eta_s**0.25+(1-mat.sym[0])/eta_o**0.25)**(-4)

### use the swarm var to map composition to viscosity
eta_fn = eta_0 * sympy.exp(-11*( (temp_swarm.sym[0]-nd(T_0*u.kelvin)) ) )

nd_perm = nd(perm*u.meter**2)

flow_term_fn = nd_perm / eta_fn

darcy.constitutive_model.Parameters.diffusivity = flow_term_fn

-darcy.darcy_flux


# -

# ##### Create projection of viscosity field

def proj_viscosity():
    nodal_visc_calc = uw.systems.Projection(mesh, visc)
    nodal_visc_calc.uw_function = eta_fn
    nodal_visc_calc.smoothing = 1e-3
    nodal_visc_calc.petsc_options.delValue("ksp_monitor")
    nodal_visc_calc.solve(_force_setup=True,)


# #### setup an advection diffusion solver
# - This is the advection-diffusion of the material based on the velocity that is determined by the Darcy flow and the diffusion of the material $<M>$.
# - We give this a term u_Star_fun to keep track of the history of the material $(<M>)$ as the model is diffusing the material as the oil and solvent mix.

# +
adv_diff = uw.systems.AdvDiffusionSwarm(mesh=mesh, u_Field=temp, V_Field=v_soln, u_Star_fn=temp_swarm.sym)

adv_diff.constitutive_model = uw.systems.constitutive_models.DiffusionModel(mesh.dim)
# -

# We have the advection diffusion equation for a concentration $c$ of the material $<M>$ being advected by a velocity field $v_{sol}$ with and diffusing with a diffusion constant of $k$. The consitutive equation for the relation between the velocity and concentration is:
# \begin{equation}
#     u = k \frac{d c}{d x}
# \end{equation}
# This is of the form of the diffusionModel with a diffusion constant $k$

adv_diff.constitutive_model.Parameters.diffusivity = nd(1e-6*u.meter**2/u.second)

# ### Random material distribution along the interface

# +
np.random.seed(100)

x0 = nd(2.5*u.meter)
dx = mesh.get_min_radius()

### on the mesh

with mesh.access(temp):
    
    temp.data[temp.coords[:,0]  < x0] = nd(1573.15* u.kelvin)
    temp.data[temp.coords[:,0] >= x0] = nd(273.15  *u.kelvin)
    
    randomT = np.random.random(temp.coords[:,0][(temp.coords[:,0] > (x0-dx)) & (temp.coords[:,0] < (x0+dx))].shape[0])
   
    temp.data[:,0][(temp.coords[:,0] > (x0-dx)) & (temp.coords[:,0] < (x0+dx))] = (nd(T_0*u.kelvin) + randomT[:,])
    
### on the swarm

with swarm.access(temp_swarm):
    temp_swarm.data[:,0] = temp.rbf_interpolate(new_coords=temp_swarm.swarm.data, nnn=1)[:,0]


    

# -


if uw.mpi.size == 1:

    # plot the mesh
    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = "white"
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = 'ssaa'
    pv.global_theme.jupyter_backend = "panel"
    pv.global_theme.smooth_shading = True

    mesh.vtk("tmp_mesh.vtk")
    pvmesh = pv.read("tmp_mesh.vtk")
    
    
    pvmesh['T'] = temp.rbf_interpolate(mesh.data) #uw.function.evaluate(mat.sym[0], mesh.data)
    
    with swarm.access(temp_swarm):
        points = np.zeros((temp_swarm.swarm.data.shape[0], 3))
        points[:, 0] = temp_swarm.swarm.data[:, 0]
        points[:, 1] = temp_swarm.swarm.data[:, 1]
    
        
        
        point_cloud = pv.PolyData(points)
        
        point_cloud.point_data["T"] = temp_swarm.data.copy()
        

    pl = pv.Plotter()

    pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, use_transparency=False)
    
    pl.add_points(
        point_cloud,
        cmap="coolwarm",
        render_points_as_spheres=False,
        point_size=10,
        opacity=0.66,
    )

    pl.show(cpos="xy")

# ### Set up other aspects of the model
# - Add in the pressure boundary conditions for the Darcy solver
# - Remove additional terms from the Darcy solver
# - Add in the scalar boundary conditions (material) for the advection-diffusion solver

# +
p0_nd = nd(0.1e6*u.pascal)


# p_dx = p0_nd * (1 - mesh.X[0])

# with mesh.access(p_soln):
#     p_soln.data[:,0] = uw.function.evaluate(p_dx, p_soln.coords, mesh.N)

# +
## Make sure additional terms are set to zero
darcy.f = 0.0
darcy.s = sympy.Matrix([0, 0]).T

### set up boundary conditions for the Darcy solver
darcy.add_dirichlet_bc(p0_nd, "Left")
darcy.add_dirichlet_bc(0.0, "Right")

### set up boundary conditions for the adv diffusion solver
adv_diff.add_dirichlet_bc(nd(T_1*u.kelvin), 'Left')
adv_diff.add_dirichlet_bc(nd(T_0*u.kelvin), 'Right')
# +
darcy.petsc_options["snes_atol"]    = 1.0e-6  # Needs to be smaller than the contrast in properties
adv_diff.petsc_options["snes_atol"] = 1.0e-6

darcy.petsc_options["snes_rtol"]    = 1.0e-12  # Needs to be smaller than the contrast in properties
adv_diff.petsc_options["snes_rtol"] = 1.0e-12

darcy._v_projector.petsc_options["snes_rtol"] = 1.0e-12
darcy._v_projector.petsc_options["snes_atol"] = 1.0e-6


darcy._v_projector.petsc_options["ksp_rtol"] = 1.0e-12
# darcy._v_projector.petsc_options["ksp_atol"] = 1.0e-6

# -

adv_diff._u_star_projector.petsc_options["snes_rtol"] = 1.0e-12
adv_diff._u_star_projector.petsc_options["snes_atol"] = 1.0e-6

# ### Solve loop

time = 0
step = 0
# finish_time = 0.01*u.year
nstep = 100

# +

while step < nstep:
    
    if uw.mpi.rank == 0:
        print(f'\n\nstep: {step}, time: {dim(time, u.year)}\n\n')
        
    
        
    
    if step % 5 == 0:
        proj_viscosity()
        mesh.petsc_save_checkpoint(index=step, meshVars=[temp, p_soln, v_soln, visc], outputPath=outputDir)
        swarm.petsc_save_checkpoint(swarmName='swarm', index=step, outputPath=outputDir)
    
    
    

    ### get the Darcy velociy from the darcy solve
    darcy.solve()
    
    ## Divide by the porosity to get the actual velocity
    with mesh.access(v_soln):
        v_soln.data[:,] /= porosity
        
    
        
    ### estimate dt from the adv_diff solver
    dt = adv_diff.estimate_dt()

    ### do the advection-diffusion
    adv_diff.solve(timestep=dt)
    
    
    ### update swarm / swarm variables
    with swarm.access(temp_swarm):
        temp_swarm.data[:,0] = temp.rbf_interpolate(new_coords=temp_swarm.swarm.data, nnn=1)[:,0]
        # temp_swarm.data[:,0] = uw.function.evaluate(temp.sym, swarm.particle_coordinates.data)
    
    ### advect the swarm
    swarm.advection(V_fn=v_soln.sym, delta_t=dt, order=v_soln.degree)
    

    step += 1
    
    time += dt
    


