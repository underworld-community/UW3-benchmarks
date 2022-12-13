# %% [markdown]
# # The brick experiment
#
# Test of visco-plastic rheology from [Glerum et al., 2018](https://se.copernicus.org/articles/9/267/2018/se-9-267-2018.pdf): 'The brick experiment'
# - should produce shear bands at top of Brick that has a certain angle, which is dependent on the friction coefficient
# - [UWGeodynamics (UW2) version](https://github.com/underworldcode/underworld2/blob/master/docs/UWGeodynamics/benchmarks/Kaus_BrickBenchmark-Compression.ipynb)
#
#

# %% [markdown]
# Working:
# - Non-linear yielding based on SR
#
# Issues:
# - Currently produces broad region of yielding
# - Issue with stokes._Einv2 (SR_2ndInv), produces weird values depending on the plasticity formulation

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from sympy import Piecewise, Abs, Min, Max
from mpi4py import MPI


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


# %%
options = PETSc.Options()
# options["help"] = None
# options["pc_type"]  = "svd"
#options["ksp_rtol"] =  1.0e-6
#options["ksp_atol"] =  1.0e-6
#options["ksp_monitor"] = None
# options["snes_type"]  = "fas"
#options["snes_type"]="ksponly"
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None

# options["snes_view"]=None
# options["snes_test_jacobian"] = None
# options["snes_rtol"] = 1.0e-2  # set this low to force single SNES it. 
# options["snes_max_it"] = 3

# options["mat_type"]="aij"

# options["ksp_type"]="preonly"
# options["pc_type"] = "lu"
# options["pc_factor_mat_solver_type"] = "mumps"

sys = PETSc.Sys()
sys.pushErrorHandler("traceback")

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# %%
### reference values
length         = 10.                       #* u.kilometer
kappa          = 1e-6                      #* u.meter**2/u.second
g              = 9.81                      #* u.meter/u.second**2
v              = 1                         # u.centimeter/u.year, velocity in cm/yr
alpha          = 3.0e-5                    # *1./u.kelvin
tempMax        = 1573.15                   # * u.kelvin
tempMin        = 273.15                    #* u.kelvin
rho0           = 2700.0                    #* u.kilogram / u.metre**3#  * 9.81 * u.meter / u.second**2
R              = 8.3145                    # [J/(K.mol)], gas constant

# %%
lengthScale   = length  * u.kilometer
surfaceTemp   = tempMin * u.degK
baseModelTemp = tempMax * u.degK
bodyforce     = rho0    * u.kilogram / u.metre**3 * g * u.meter / u.second**2

half_rate     = v * u.centimeter / u.year

KL = lengthScale.to_base_units()
Kt = (KL / half_rate).to_base_units()
KM = (bodyforce * KL**2 * Kt**2).to_base_units()
KT = (baseModelTemp - surfaceTemp).to_base_units()

# %%
scaling_coefficients                  = uw.scaling.get_coefficients()

scaling_coefficients["[length]"]      = KL.to_base_units()
scaling_coefficients["[time]"]        = Kt.to_base_units()
scaling_coefficients["[mass]"]        = KM.to_base_units()
scaling_coefficients["[temperature]"] = KT.to_base_units()


scaling_coefficients

# %%
### fundamental values
ref_length    = uw.scaling.dimensionalise(1., u.meter).magnitude

ref_length_km = uw.scaling.dimensionalise(1., u.kilometer).magnitude

ref_density   =  uw.scaling.dimensionalise(1., u.kilogram/u.meter**3).magnitude

ref_gravity   = uw.scaling.dimensionalise(1., u.meter/u.second**2).magnitude

ref_temp      = uw.scaling.dimensionalise(1., u.kelvin).magnitude

ref_velocity  = uw.scaling.dimensionalise(1., u.meter/u.second).magnitude

### derived values
ref_time      = ref_length / ref_velocity

ref_pressure  = ref_density * ref_gravity * ref_length

ref_stress    = ref_pressure

ref_viscosity = ref_pressure * ref_time

### Key ND values
ND_diffusivity = kappa        / (ref_length**2/ref_time)
ND_gravity     = g            / ref_gravity

# %%
if uw.mpi.rank == 0:
    print(f'time scaling: {ref_time/(60*60*24*365.25*1e6)} [Myr]')
    print(f'pressure scaling: {ref_pressure/1e6} [MPa]')
    print(f'viscosity scaling: {ref_viscosity} [Pa s]')
    print(f'velocity scaling: {ref_velocity*(1e2*60*60*24*365.25)} [cm/yr]')
    print(f'length scaling: {ref_length/1e3} [km]')

# %%
## Set the resolution.
res = 20

# %%
## set box properties

x_length   = 40e3 / ref_length
y_length   = 10e3 / ref_length

xmin, xmax = 0., x_length
ymin, ymax = 0., y_length

# %%
## set brick height and length
BrickHeight = (400. / ref_length)
BrickLength = (800. / ref_length)

# %%
### set global and material properties
minVisc    = 1e20 / ref_viscosity
maxVisc    = 1e25 / ref_viscosity


### define some index values
materialBG    = 0
materialBrick = 1

### BG material properties
densityBG   = 2.7e3 / ref_density       ### kg/m**3
viscoistyBG = maxVisc                   
cohesionBG  = 40e6  / ref_stress        ### Pa
FA_BG       = 0.                        ### degrees


### Brick material properties
densityBrick    = 2.7e3/ref_density     ### kg/m**3
viscosityBrick  = 1e20 /ref_viscosity   ### Pa s


''' Gauss point count, creates the number of particles in each cell '''
swarmGPC = 4

# %%
mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=(1.0 / res), regular=False)


# mesh = uw.meshing.StructuredQuadBox(elementRes=((res*4), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax))


# %%
# Create Stokes object

v      = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p      = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
lithoP = uw.discretisation.MeshVariable("lithoP", mesh, 1, degree=1)

strain_rate_inv2 = uw.discretisation.MeshVariable("eps", mesh, 1, degree=1)
dev_stress_inv2 = uw.discretisation.MeshVariable("tau", mesh, 1, degree=1)
node_viscosity = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1)



# %%
stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)

# %%
nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
# nodal_strain_rate_inv2.add_dirichlet_bc(1.0, "Left", 0)
# nodal_strain_rate_inv2.add_dirichlet_bc(-1.0, "Right", 0)

nodal_strain_rate_inv2 = uw.systems.Projection(mesh, strain_rate_inv2)
nodal_strain_rate_inv2.uw_function = stokes._Einv2
# nodal_strain_rate_inv2.smoothing = 1.0e-3
nodal_strain_rate_inv2.petsc_options.delValue("ksp_monitor")

nodal_visc_calc = uw.systems.Projection(mesh, node_viscosity)
nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
nodal_visc_calc.smoothing = 1.0e-3
nodal_visc_calc.petsc_options.delValue("ksp_monitor")


nodal_tau_inv2 = uw.systems.Projection(mesh, dev_stress_inv2)
nodal_tau_inv2.uw_function = 2. * stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
nodal_tau_inv2.smoothing = 1.0e-3
nodal_tau_inv2.petsc_options.delValue("ksp_monitor")



# %%
def updateFields():

    nodal_strain_rate_inv2.solve()

    nodal_visc_calc.uw_function = stokes.constitutive_model.Parameters.viscosity
    nodal_visc_calc.solve(_force_setup=True)

    nodal_tau_inv2.uw_function = stokes.constitutive_model.Parameters.viscosity * stokes._Einv2
    nodal_tau_inv2.solve(_force_setup=True)


# %%
def SZAnalysis():
    ### horizontle profile across the model
    ### x coords to sample
    sample_x = np.arange(mesh.data[:,0].min(),mesh.data[:,0].max(), mesh.get_min_radius()) ### get horizontal profile


    sample_points0 = np.empty((sample_x.shape[0], 2))
    sample_points0[:,0] = sample_x
    sample_points0[:,1] = np.zeros_like(sample_x) + (5*BrickHeight) ### get the shear zone angle close to the Brick

    sample_points1 = np.empty((sample_x.shape[0], 2))
    sample_points1[:,0] = sample_x
    sample_points1[:,1] =  np.zeros_like(sample_x) + (10*BrickHeight) ### get the shear zone angle close to the Brick

    SR_profile0 = uw.function.evaluate(strain_rate_inv2.fn, sample_points0)
    SR_profile1 = uw.function.evaluate(strain_rate_inv2.fn, sample_points1)

    Visc_profile0 = uw.function.evaluate(node_viscosity.fn, sample_points0)
    Visc_profile1 = uw.function.evaluate(node_viscosity.fn, sample_points1)

    if uw.mpi.size ==1:
        ''' only plot if model is run locally '''   
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter1d

        plt.figure(1)
        plt.plot(sample_points0[:,0], Visc_profile0*ref_viscosity, ls=":")
        plt.plot(sample_points1[:,0], Visc_profile1*ref_viscosity, ls='-.')
        
        plt.plot(sample_points0[:,0], 40e6/(2*(SR_profile0/ref_time)), ls=":")
        plt.plot(sample_points1[:,0], 40e6/(2*(SR_profile1/ref_time)), ls="-.")
        
        
        plt.title('Visosity profile')
        plt.show()
        
        
        plt.figure(2)
        plt.plot(sample_points0[:,0], SR_profile0/ref_time, ls=":")
        x0 = sample_points0[:,0][SR_profile0 == SR_profile0[(sample_points0[:,0] < 2.)].max()]
        y0 = SR_profile0[SR_profile0 == SR_profile0[(sample_points0[:,0] < 2.)].max()]

        plt.scatter(x0,y0/ref_time)

        plt.plot(sample_points1[:,0], SR_profile1/ref_time)

        x1 = sample_points1[:,0][SR_profile1 == SR_profile1[(sample_points1[:,0] < 2.)].max()]
        y1 = SR_profile1[SR_profile1 == SR_profile1[(sample_points1[:,0] < 2.)].max()]

        plt.scatter(x1,y1/ref_time)

        plt.title('SR profile')
        plt.show()


        delta_x = (x0 - x1)
        delta_y = (sample_points1[:,1].max() - sample_points0[:,1].max())
        theta   = (np.rad2deg(np.arctan(delta_y/delta_x)))

        print(f'Shear zone angle: {theta[0]} [degree]') 
        
        
        

# %%
''' free slip bottom '''
stokes.add_dirichlet_bc(sympy.Matrix([0, 0]), "Bottom", (1,))

''' open top '''
# # stokes.add_dirichlet_bc(0.0,  "Top", 1)


''' velocity sides '''
stokes.add_dirichlet_bc(sympy.Matrix([ 1, 0]), "Left", (0,))
stokes.add_dirichlet_bc(sympy.Matrix([-1, 0]), "Right", (0,))


# %%
swarm = uw.swarm.Swarm(mesh=mesh)
material = swarm.add_variable(name="materialVariable", num_components=1, dtype=PETSc.IntType) # uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=swarmGPC)

# %%
with swarm.access(material):
    material.data[...] = materialBG
    
    material.data[(swarm.data[:,1] <= BrickHeight) & 
                  (swarm.data[:,0] >= (((xmax - xmin) / 2.) - (BrickLength / 2.)) ) & 
                  (swarm.data[:,0] <= (((xmax - xmin) / 2.) + (BrickLength / 2.)) )] = materialBrick

# %%
with mesh.access(lithoP):
    lithoP.data[:,0] = densityBG * ND_gravity * mesh.data[:,1] ### rho * g * h
    
    if uw.mpi.size ==1:
        print(f'Lithostatic pressure at base: {(lithoP.data[:,0].max() * ref_stress)/1e6} [MPa]')


# %%
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

        # pv.start_xvfb()

        mesh.vtk("tmpMsh.vtk")
        pvmesh = pv.read("tmpMsh.vtk")

        # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

        with mesh.access():
            vsol = v.data.copy()
            # pvmesh.point_data["Vmag"] = np.linalg.norm(v.data[:,], axis=1) # uw.function.evaluate(sympy.sqrt(v_soln.fn.dot(v_soln.fn)), mesh1.data)
            # pvmesh.point_data["P"] = p.data[:,0] #uw.function.evaluate(p_soln.fn, mesh1.data)
            pvmesh.point_data["SR"] = strain_rate_inv2.data[:,0]
            pvmesh.point_data["Str"] = dev_stress_inv2.data[:,0]
            pvmesh.point_data["Visc"] = node_viscosity.data[:,0]
            # pvmesh.point_data["SR"] = uw.function.evaluate(strain_rate_inv2.fn, mesh.data)
            # pvmesh.point_data["Str"] = uw.function.evaluate(dev_stress_inv2.fn, mesh.data)
            # pvmesh.point_data["Visc"] = uw.function.evaluate(node_viscosity.fn, mesh.data)

        with swarm.access():
            points = np.zeros((swarm.data.shape[0], 3))
            points[:, 0] = swarm.data[:, 0]
            points[:, 1] = swarm.data[:, 1]
            points[:, 2] = 0.0

        point_cloud = pv.PolyData(points)

        with swarm.access():
            point_cloud.point_data["M"] = material.data.copy()

        arrow_loc = np.zeros((v.coords.shape[0], 3))
        arrow_loc[:, 0:2] = v.coords[...]

        arrow_length = np.zeros((v.coords.shape[0], 3))
        arrow_length[:, 0:2] = vsol[...]

        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        # pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="rho",
        #                 use_transparency=False, opacity=0.95)

        # pl.add_mesh(pvmesh, cmap="coolwarm", edge_color="Black", show_edges=True, scalars="S",
        #               use_transparency=False, opacity=0.5)

        pl.add_mesh(
            point_cloud,
            cmap="coolwarm",
            edge_color="Black",
            show_edges=False,
            scalars="M",
            use_transparency=False,
            opacity=0.9,
        )

        pl.add_mesh(
            pvmesh,
            cmap="coolwarm_r",
            edge_color="Black",
            show_edges=False,
            scalars="Visc",
            use_transparency=False,
            log_scale=True,
            opacity=0.5)

        pl.add_arrows(arrow_loc, arrow_length, mag=0.01, opacity=0.5)
        # pl.add_arrows(arrow_loc2, arrow_length2, mag=1.0e-1)

        # pl.add_points(pdata)

        pl.show(cpos="xy")
        # pl.save_graphic("outPut.pdf") 

# %%
# mat_density = np.array([densityBG, densityBrick])

# density = mat_density[0] * material.sym[0] + mat_density[1] * material.sym[1]


density = Piecewise( (densityBG, Abs(material.sym[0] - materialBG) < 0.5),
                     (densityBrick,                            True ))

stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density])

# %%
plotFig()

# %%
### linear solve
### linear viscosity
stokes.penalty = 0.

viscosityL = Piecewise( (1., Abs(material.sym[0] - materialBG) < 0.5),
                        (0.1,                                True ))

stokes.constitutive_model.Parameters.viscosity = viscosityL
# stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity
stokes.solve(zero_init_guess=True)

# %%
updateFields()
    
SZAnalysis()

# %%
# viscosityL = mat_viscosityL[0] * material.sym[0] + mat_viscosityL[1] * material.sym[1]

viscosityL = Piecewise( (viscoistyBG, Abs(material.sym[0] - materialBG) < 0.5),
                        (viscosityBrick,                                True ))



### linear viscosity
stokes.constitutive_model.Parameters.viscosity = viscosityL
# stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity
stokes.solve(zero_init_guess=False)


# %%
updateFields()
    
SZAnalysis()

# %%
# sigmaBG = 40e6/ref_stress ### Pa
sigmaBG = ((40e6/ref_stress)*np.cos(np.deg2rad(FA_BG))) + (lithoP.fn * np.sin(np.deg2rad(FA_BG)))

### Solves but gives wrong viscosity, SR and visc are correlated (Should be anticorrelated).
plasticity = sigmaBG / (2. * (stokes._Einv2+1.0e-18))  # sympy.Min((sigmaBG / (2. * (strainRate_2ndInvariant+1.0e-18)), viscoistyBG)


NLBGVisc = Min(plasticity, viscoistyBG)



# %%
# viscosityNL = mat_viscosityNL[0] * material.sym[0] + mat_viscosityNL[1] * material.sym[1]



viscosityNL  = Piecewise( (NLBGVisc,   Abs(material.sym[0] - materialBG) < 0.5),
                         (viscosityBrick,                         True ))


# %%


### NL viscosity
stokes.constitutive_model.Parameters.viscosity = viscosityNL
# stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity
### NL solve
stokes.solve(zero_init_guess=False)

# %%
updateFields()
    
SZAnalysis()

# %%
plotFig()

# %%

# %%
