# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Linear stokes sinker
#
# The stokes sinker is a benchmark to test sinking velocities determined by the stokes solver.
#
# Two materials are used, one for the background and one for the sinking ball.
#
# Stokes velocity is calculated and compared with the velocity from the UW model to benchmark the stokes solver
#
# Other model benchmarks:
# - [Aspect benchmarks](https://aspect-documentation.readthedocs.io/en/latest/user/cookbooks/cookbooks/stokes/doc/stokes.html)
# - [UW2 example](https://github.com/underworldcode/underworld2/blob/master/docs/examples/04_StokesSinker.ipynb)
#

from petsc4py import PETSc
import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
from mpi4py import MPI
import os


import pyvista as pv
import vtk

# +
#### visualisation within script
render = True

#### save output
save_output = False

# +
### stokes tolerance
tol = 1e-7

### number of timesteps
nstep = 10

### resolution of model
res = 64

# +
#### output folder name
outputPath = f"output/sinker_eta10_rho10/"

if uw.mpi.rank==0:      
    ### create folder if not run before
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
# -

# Set size and position of dense sphere.
sphereRadius = 0.1
sphereCentre = (0., 0.7)

# define some names for our index
materialLightIndex = 0
materialHeavyIndex = 1

# Set constants for the viscosity of each material
viscBG     =  1
viscSphere =  10

# set density of the different materials
densityBG     =  1
densitySphere =  10

gravity = 1

# +
# location of tracer at bottom of sinker
x_pos = sphereCentre[0]
y_pos = sphereCentre[1]

tracer_coord = np.vstack([x_pos, (y_pos-sphereRadius)]).T


# -

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
    
    # ### create point cloud for passive tracers
    # with passiveSwarm.access():
    #     passiveCloud = pv.PolyData(np.vstack((passiveSwarm.data[:,0],passiveSwarm.data[:,1], np.zeros(len(passiveSwarm.data)))).T)


    with swarm.access():
        point_cloud.point_data["M"] = material.data.copy()
        



    pl = pv.Plotter(notebook=True)

    pl.add_mesh(pvmesh,'Black', 'wireframe')

    # pl.add_points(point_cloud, color="Black",
    #                   render_points_as_spheres=False,
    #                   point_size=2.5, opacity=0.75)       



    pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="M",
                        use_transparency=False, opacity=0.95)
    
    # ### add points of passive tracers
    # pl.add_mesh(passiveCloud, color='black', show_edges=True,
    #                 use_transparency=False, opacity=0.95)



    pl.show(cpos="xy")


# +

mesh = uw.meshing.StructuredQuadBox(minCoords=(-1, 0.0), maxCoords=(1.0, 1.0),  elementRes=(res,res), qdegree=3)

v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)


stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.systems.constitutive_models.ViscousFlowModel(v)


### free slip BC
stokes.add_dirichlet_bc( (0.,0.), 'Left',   (0) ) # left/right: function, boundaries, components
stokes.add_dirichlet_bc( (0.,0.), 'Right',  (0) )

stokes.add_dirichlet_bc( (0.,0.), 'Top',    (1) )
stokes.add_dirichlet_bc( (0.,0.), 'Bottom', (1) )# top/bottom: function, boundaries, components 


swarm = uw.swarm.Swarm(mesh=mesh)
material = uw.swarm.IndexSwarmVariable("M", swarm, indices=2)
swarm.populate(fill_param=mesh.qdegree)

with swarm.access(material):
    material.data[...] = materialLightIndex

    cx, cy, r, m = sphereCentre[0], sphereCentre[1], sphereRadius, materialHeavyIndex
    inside = (swarm.data[:, 0] - cx) ** 2 + (swarm.data[:, 1] - cy) ** 2 < r**2
    material.data[inside] = m

tracer = uw.swarm.Swarm(mesh=mesh)
tracer.add_particles_with_coordinates(tracer_coord)

if uw.mpi.size==1:
    plot_mat()


density_fn = material.createMask([densityBG, densitySphere])

stokes.bodyforce = sympy.Matrix([0, -1 * gravity * density_fn])


viscosity_fn = material.createMask([viscBG, viscSphere])

stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn


stokes.tolerance = tol

if uw.mpi.size == 1:
    stokes.petsc_options['pc_type'] = 'lu'


# +
def v_rms(mesh, v_solution): 
    # v_soln must be a variable of mesh
    v_rms = np.sqrt(uw.maths.Integral(mesh, v_solution.fn.dot(v_solution.fn)).evaluate())
    return v_rms

v_rms(mesh, v)

# +
tSinker = np.zeros(nstep)*np.nan
ySinker = np.zeros(nstep)*np.nan

vrms    = np.zeros(nstep)*np.nan

# -

step = 0
time = 0.

# +
while step < nstep:

    stokes.solve()

    
    with tracer.access():
        ySinker[step] = tracer.data[:,1][0]
    
    tSinker[step] = time
    vrms[step]    = v_rms(mesh, v)

    if uw.mpi.rank == 0:
        print('\n\nstep = {0:6d}; time = {1:.3e}; v_rms = {2:.3e}; height = {3:.3e}\n\n'
              .format(step,time,vrms[step],ySinker[step]))


    ### estimate dt
    dt = stokes.estimate_dt()


    ### advect the swarm
    swarm.advection(stokes.u.sym, dt, corrector=False)
    
    tracer.advection(stokes.u.sym, dt, corrector=False)


    step += 1
    time += dt




    

# +
print(v_rms(mesh, v))

if step==10 and not np.isclose(v_rms(mesh, v), 1.0135e-02, rtol=1e-3):
    raise RuntimeError("The Velocity RMS is not close")

# +

stokesSink_vel = (2/9)*(((densitySphere-densityBG)*(sphereRadius**2)*gravity)/viscBG)
stokesSink_vel
# -

if uw.mpi.rank==0:
    print('Initial position: t = {0:.3f}, y = {1:.3f}'.format(tSinker[0], ySinker[0]))
    print('Final position:   t = {0:.3f}, y = {1:.3f}'.format(tSinker[nstep-1], ySinker[nstep-1]))

    # uw.utils.matplotlib_inline()
    import matplotlib.pyplot as pyplot
    fig = pyplot.figure()
    fig.set_size_inches(12, 6)
    ax = fig.add_subplot(1,1,1)
    ax.plot(tSinker, ySinker, label='UW results') 
    ax.plot(tSinker, 0.6-(tSinker*stokesSink_vel), label='numerical solution', ls=':')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Sinker position')


