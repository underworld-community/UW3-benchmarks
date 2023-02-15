# %% [markdown]
# # Stokes sinker - sinking block
#
# Sinking block benchmark as outlined in:
#
# - [Gerya, T.V. and Yuen, D.A., 2003. Characteristics-based marker-in-cell method with conservative finite-differences schemes for modeling geological flows with strongly variable transport properties. Physics of the Earth and Planetary Interiors, 140(4), pp.293-318.](http://jupiter.ethz.ch/~tgerya/reprints/2003_PEPI_method.pdf)
#
# - Only value to change is: **viscBlock**
#
# - utilises the UW scaling module to convert from dimensional to non-dimensional values
#
# - [ASPECT benchmark](https://aspect-documentation.readthedocs.io/en/latest/user/benchmarks/benchmarks/sinking_block/doc/sinking_block.html)

# %%
from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
options = PETSc.Options()


# %%
sys = PETSc.Sys()
sys.pushErrorHandler("traceback")


options["snes_converged_reason"] = None
options["snes_monitor_short"] = None

# %%
# Set the resolution, a structured quad box of 51x51 is used in the paper
res = 51

### plot figs
if uw.mpi.size == 1:
    render = True
else:
    render = False

# %%
## number of steps
nsteps = 31

## swarm gauss point count (particle distribution)
swarmGPC = 3

# %% [markdown]
# #### Value to change:

# %%
# viscosity of block in Pa S (the value is ND later), reference is 1e21
viscBlock = 1e21

# %% [markdown]
# #### Set up scaling of model

# %%
# import unit registry to make it easy to convert between units
u = uw.scaling.units

### make scaling easier
ndim = uw.scaling.non_dimensionalise
dim  = uw.scaling.dimensionalise 

# %%
### set reference values
refLength    = 500e3
refDensity   = 3.3e3
refGravity   = 9.81
refVelocity  = (1*u.centimeter/u.year).to(u.meter/u.second).m ### 1 cm/yr in m/s
refViscosity = 1e21
refPressure  = refDensity * refGravity * refLength
refTime      = refViscosity / refPressure

bodyforce    = refDensity  * u.kilogram / u.metre**3 * refGravity * u.meter / u.second**2

# %%
### create unit registry
KL = refLength * u.meter
Kt = refTime   * u.second
KM = bodyforce * KL**2 * Kt**2

scaling_coefficients                    = uw.scaling.get_coefficients()
scaling_coefficients["[length]"] = KL
scaling_coefficients["[time]"] = Kt
scaling_coefficients["[mass]"]= KM
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

ref_time_Myr = dim(1, u.megayear).m

ref_pressure  = ref_density * ref_gravity * ref_length

ref_stress    = ref_pressure

ref_viscosity = ref_pressure * ref_time

### Key ND values
ND_gravity     = 9.81    / ref_gravity

# %% [markdown]
# ND key values

# %%
## Set constants for the viscosity and density of the sinker.
viscBG        = 1e21/ref_viscosity
viscBlock     = viscBlock/ref_viscosity

BGIndex = 0

## set density of blocks
densityBG     = 3.2e3/ref_density
densityBlock  = 3.3e3/ref_density

BlockIndex = 1

# %% [markdown]
# Set up dimensions of model and sinking block

# %%
xmin, xmax = 0., ndim(500*u.kilometer)
ymin, ymax = 0., ndim(500*u.kilometer)

# %%
# Set the box min and max coords
boxCentre_x, boxCentre_y = ndim(250.0*u.kilometer), ndim(375.0*u.kilometer)

box_xmin, box_xmax = boxCentre_x-ndim(50*u.kilometer), boxCentre_x+ndim(50*u.kilometer)
box_ymin, box_ymax = boxCentre_y-ndim(50*u.kilometer), boxCentre_y+ndim(50*u.kilometer)

# location of tracer at bottom of sinker
x_pos = box_xmax - ((box_xmax - box_xmin)/2.)
y_pos = box_ymin

### add a tracer
tracer = np.zeros(shape=(1,2))
tracer[:,0] = x_pos
tracer[:,1] = y_pos

# %% [markdown]
# ### Create mesh

# %%
# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,0.0), 
#                                               maxCoords=(1.0,1.0), 
#                                               cellSize=1.0/res, 
#                                               regular=True)

# mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(xmin, ymin), maxCoords=(xmax, ymax), cellSize=1.0 / res, regular=False)


mesh = uw.meshing.StructuredQuadBox(elementRes =(int(res),int(res)),
                                    minCoords=(xmin,ymin), 
                                    maxCoords=(xmax,ymax))


# %% [markdown]
# ### Create Stokes object

# %%


v = uw.discretisation.MeshVariable('U',    mesh,  mesh.dim, degree=2 )
p = uw.discretisation.MeshVariable('P',    mesh, 1, degree=1 )

stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p )
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(mesh.dim)

# %%
#### No slip
sol_vel = sympy.Matrix([0.,0.])

# stokes.add_dirichlet_bc( sol_vel, ["Left", "Right"],  [0,1] )  # left/right: components, function, markers
# stokes.add_dirichlet_bc( sol_vel, ["Top", "Bottom"],  [0,1] )  # top/bottom: components, function, markers 

### free slip
stokes.add_dirichlet_bc( sol_vel, ["Left", "Right"],  [0] )  # left/right: components, function, markers
stokes.add_dirichlet_bc( sol_vel, ["Top", "Bottom"],  [1] )  # top/bottom: components, function, markers 





# %% [markdown]
# ### Setup swarm

# %%
swarm     = uw.swarm.Swarm(mesh=mesh)
material  = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=swarmGPC)

# %%
with swarm.access(material):
    material.data[:] = BGIndex
    material.data[(swarm.data[:,0] >= box_xmin) & (swarm.data[:,0] <= box_xmax) &
                  (swarm.data[:,1] >= box_ymin) & (swarm.data[:,1] <= box_ymax)] = BlockIndex


# %% [markdown]
# ### Add a passive tracer(s)

# %%
passiveSwarm = uw.swarm.Swarm(mesh)

passiveSwarm.add_particles_with_coordinates(tracer)

# %%
with passiveSwarm.access(passiveSwarm.particle_coordinates):
    print(passiveSwarm.particle_coordinates.data[:])


# %%
def globalPassiveSwarmCoords(swarm, bcast=True, rootProc=0):
    '''
    Distribute passive swarm coordinate data to all CPUs (bcast = True) or the rootProc, (bcast = False)
    
    Used for the analysis of coordinates of swarm that may move between processors
    
    '''
    
    comm = uw.mpi.comm
    rank = uw.mpi.rank
    size = uw.mpi.size
    

    with swarm.access():
        if len(swarm.data) > 0:
            x_local = np.ascontiguousarray(swarm.data[:,0].copy())
            y_local = np.ascontiguousarray(swarm.data[:,1].copy())
            if swarm.data.shape[1] == 3:
                z_local = np.ascontiguousarray(swarm.data[:,2].copy())
            else:
                z_local = np.zeros_like(swarm.data[:,0])*np.nan
                
        else:
            x_local = np.array([np.nan], dtype='float64')
            y_local = np.array([np.nan], dtype='float64')
            z_local = np.array([np.nan], dtype='float64')
            
            
            
    ### Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.gather(len(x_local), root=rootProc))
    
    
    if rank == rootProc:
        x_global = np.zeros((sum(sendcounts)), dtype='float64')
        y_global = np.zeros((sum(sendcounts)), dtype='float64')
        z_global = np.zeros((sum(sendcounts)), dtype='float64')
    else:
        x_global = None
        y_global = None
        z_global = None
        

    comm.barrier()

    ## gather x values, can't do them together
    comm.Gatherv(sendbuf=x_local, recvbuf=(x_global, sendcounts), root=rootProc)
    ## gather y values
    comm.Gatherv(sendbuf=y_local, recvbuf=(y_global, sendcounts), root=rootProc)

    ## gather z values
    comm.Gatherv(sendbuf=z_local, recvbuf=(z_global, sendcounts), root=rootProc)
    
    comm.barrier()
    
    def sortCoords():
        ## Put back into combined array
        Coords = np.zeros(shape=(len(x_global),3))*np.nan
        Coords[:,0] = x_global
        Coords[:,1] = y_global
        Coords[:,2] = z_global
        
        comm.barrier()

        ### remove rows with NaN
        Coords = Coords[~np.isnan(Coords[:,0])]
        ### remove cols with NaN
        Coords = Coords[:, ~np.isnan(Coords).all(axis=0)]
        
        comm.barrier()
        
        return Coords
    
    if bcast == True:
        #### make swarm coords available on all processors
        x_global = comm.bcast(x_global, root=rootProc)
        y_global = comm.bcast(y_global, root=rootProc)
        z_global = comm.bcast(z_global, root=rootProc)
        
        comm.barrier()
        
        Coords = sortCoords()
        
        comm.barrier()
           
    else:
        ### swarm coords only available on root processor
        if rank == rootProc:
            Coords = sortCoords()
            
        comm.barrier()
            
    return Coords


# %% [markdown]
# #### Set up density and viscosity of materials

# %%
mat_density = np.array([densityBG,densityBlock])

density = mat_density[0] * material.sym[0] + \
          mat_density[1] * material.sym[1]

# %%
mat_viscosity = np.array([viscBG, viscBlock])

viscosityMat = mat_viscosity[0] * material.sym[0] + \
               mat_viscosity[1] * material.sym[1] 


# %% [markdown]
# ### Create figure function

# %%
def plot_mat():

    import numpy as np
    import pyvista as pv
    import vtk

    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [750, 750]
    pv.global_theme.antialiasing = True
    pv.global_theme.jupyter_backend = 'panel'
    pv.global_theme.smooth_shading = True


    mesh.vtk("tempMsh.vtk")
    pvmesh = pv.read("tempMsh.vtk") 

    with swarm.access():
        points = np.zeros((swarm.data.shape[0],3))
        points[:,0] = swarm.data[:,0]
        points[:,1] = swarm.data[:,1]
        points[:,2] = 0.0

    point_cloud = pv.PolyData(points)
    
    ### create point cloud for passive tracers
    with passiveSwarm.access():
        passiveCloud = pv.PolyData(np.vstack((passiveSwarm.data[:,0],passiveSwarm.data[:,1], np.zeros(len(passiveSwarm.data)))).T)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
        



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)
    
    ### add points of passive tracers
    pl.add_mesh(passiveCloud, color='black', show_edges=True,
                    use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")
    
if render == True & uw.mpi.size==1:
    plot_mat()

# %% [markdown]
# ### Initial linear solve

# %%
### linear solve
stokes.constitutive_model.Parameters.viscosity = ndim(ref_viscosity * u.pascal*u.second)
# stokes.saddle_preconditioner = 1.0 / stokes.constitutive_model.Parameters.viscosity
stokes.bodyforce = sympy.Matrix([0, -1 * ND_gravity * density])


# %%
stokes.constitutive_model.Parameters.viscosity

# %%
stokes.solve()

# %%
### add in material-based viscosity
stokes.constitutive_model.Parameters.viscosity = viscosityMat

# %%
stokes.saddle_preconditioner = 1 / viscosityMat
stokes.penalty = 0.1

# %%
# stokes.petsc_options.view()
options["snes_converged_reason"] = None
options["snes_monitor_short"] = None
options["snes_test_jacobian"] = None
options["snes_test_jacobian_view"] = None
# stokes.petsc_options['snes_test_jacobian'] = None
# stokes.petsc_options['snes_test_jacobian_view'] = None

# %%
# Set solve options here (or remove default values
stokes.petsc_options["ksp_monitor"] = None

stokes.tolerance = 1.0e-4
stokes.petsc_options["snes_atol"] = 1e-4

stokes.petsc_options["fieldsplit_velocity_ksp_rtol"] = 1e-4
stokes.petsc_options["fieldsplit_pressure_ksp_type"] = "gmres" # gmres here for bulletproof

stokes.petsc_options[
    "fieldsplit_pressure_pc_type"
] = "gamg"  # can use gasm / gamg / lu here

stokes.petsc_options[
    "fieldsplit_pressure_pc_gasm_type"
] = "basic"  # can use gasm / gamg / lu here

stokes.petsc_options[
    "fieldsplit_pressure_pc_gamg_type"
] = "classical"  # can use gasm / gamg / lu here

stokes.petsc_options["fieldsplit_pressure_pc_gamg_classical_type"] = "direct"

# # stokes.petsc_options["fieldsplit_velocity_pc_gamg_agg_nsmooths"] = 5
# # stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
# # stokes.petsc_options["fieldsplit_pressure_mg_levels_ksp_converged_maxits"] = None


# # Fast: preonly plus gasm / gamg / mumps
# # Robust: gmres plus gasm / gamg / mumps

# stokes.petsc_options["fieldsplit_velocity_pc_type"] = "gamg"
# # stokes.petsc_options["fieldsplit_velocity_pc_gasm_type"] = "basic" # can use gasm / gamg / lu here

# stokes.petsc_options["fieldsplit_velocity_pc_gamg_agg_nsmooths"] = 2
# stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 3

# stokes.petsc_options["fieldsplit_velocity_pc_gamg_esteig_ksp_type"] = "cg"

stokes.petsc_options["fieldsplit_pressure_pc_gamg_esteig_ksp_type"] = "cg"

# %%
stokes.bodyforce

# %%
stokes.constitutive_model.Parameters.viscosity

# %%
step   = 0
time   = 0.

# %%
tSinker = np.zeros(nsteps)*np.nan
ySinker = np.zeros(nsteps)*np.nan

# %% [markdown]
# ### Solver loop for multiple iterations

# %%
while step < nsteps:
    ### Get the position of the sinking ball
    PTdata = globalPassiveSwarmCoords(swarm=passiveSwarm)
    ymin = PTdata[:,1].min()
        
    ySinker[step] = ymin
    tSinker[step] = time
    
    ### print some stuff    
    if uw.mpi.rank==0:
        print(f"\n\nStep: {str(step).rjust(3)}, time: {dim(time, u.megayear).m:6.2f} Myr, tracer:  {dim(ymin, u.kilometer).m:6.2f} km \n\n")
    
    ### solve stokes 
    stokes.solve()
    ### estimate dt
    dt = stokes.estimate_dt()
    
    # print('start advection of particles')

    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False)
    
    # print('finished advecting particle swarm')
    
    passiveSwarm.advection(stokes.u.sym, dt, corrector=False)
    
    # print('finished advecting passive tracers')
    
        
    step+=1
    time+=dt
    
    # print('finished solver loop')

# %% [markdown]
# #### Check the results against the benchmark 

# %%
### remove nan values, if any. Convert to km and Myr
ySinker = dim(ySinker[~np.isnan(ySinker)], u.kilometer)
tSinker = dim(tSinker[~np.isnan(tSinker)], u.megayear)

if uw.mpi.rank==0:
    print('Initial position: t = {0:.3f}, y = {1:.3f}'.format(tSinker[0], ySinker[0]))
    print('Final position:   t = {0:.3f}, y = {1:.3f}'.format(tSinker[-1], ySinker[-1]))


    UWvelocity = ((ySinker[0] - ySinker[-1]) / (tSinker[-1] - tSinker[0])).to(u.meter/u.second).m
    print(f'Velocity:         v = {UWvelocity} m/s')

    
if uw.mpi.size==1:
        
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(tSinker.m, ySinker.m) 
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Sinker position [km]')

# %% [markdown]
# ##### compare values against published results
#
#
# - The marker, representing the velocity calculated from the UW model, should fit along the curved line. 
# - These velocities are taken from _Gerya (2010), Introduction to numerical modelling (2nd Ed), page 345_, but show the same model referenced in the paper above

# %%
if uw.mpi.rank==0:
    from scipy.interpolate import interp1d
    
    #### col 0 is log10( visc_block / visc_BG ), col 1 is block velocity, m/s
    paperData = np.array([(-6.01810758939326, 1.3776991077026654e-9),
                            (-5.014458950015076, 1.3792676876049961e-9),
                            (-4.018123543216514, 1.3794412652019993e-9),
                            (-3.021737084183539, 1.3740399388011341e-9),
                            (-2.0104944249364634, 1.346341549020515e-9),
                            (-1.0053652707603105, 1.1862379129846573e-9),
                            (-0.005609364256097038, 8.128929227244664e-10),
                            (0.993865754958847, 4.702099044525527e-10),
                            (2.005950776073732, 3.505255987071023e-10),
                            (3.0024521026341358, 3.3258073831103253e-10),
                            (4.006139031188129, 3.2996814021496194e-10),
                            (5.00247443798669, 3.301417178119651e-10),
                            (6.013474599120308, 3.289241220212219e-10)])

    ### some errors from sampling, so rounding are used
    visc_ratio    = np.round(paperData[:,0])
    paperVelocity = np.round(paperData[:,1], 11)


    f = interp1d(visc_ratio, paperVelocity, kind='cubic')

    x = np.arange(-6,6, 0.01)
    
    
    
    print(f'\n\n\n Is UW3 close to benchmark?: {np.isclose(paperVelocity[visc_ratio == round(np.log10(viscBlock/viscBG))][0],UWvelocity, atol=1e-11)} \n\n\n')
    
    

    if uw.mpi.size==1:
        import matplotlib.pyplot as plt
        plt.title('check benchmark')
        plt.plot(x, f(x), label='benchmark velocity curve', c='k')
        plt.scatter(np.log10(viscBlock/viscBG), UWvelocity, label='model velocity', c='red', marker='x')
        plt.legend()

# %%
if render == True & uw.mpi.size==1:
    plot_mat()

# %%
