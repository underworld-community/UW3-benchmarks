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

# # Diffusion using the generic solver
#

# +
import underworld3 as uw
import numpy as np
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
# -
# #### Setup the mesh params

# +
# Set the resolution.
res = 64

Tdegree = 4

### diffusivity constant
k = 1

mesh_qdegree = Tdegree
mesh_qdegree

# +
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0

tmin, tmax = 0.5, 1
# -

# #### Set up the mesh

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=mesh_qdegree)


# #### Create mesh vars

# Create mesh vars
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=Tdegree)
T_star = uw.discretisation.MeshVariable("T_star", mesh, 1, degree=Tdegree)

# #### Create the solver
# using the SNES scalar so nothing is pre-defined

diffusion = uw.systems.SNES_Scalar(mesh, T)

# ##### Set up solver parameters

diffusion.constitutive_model = uw.constitutive_models.DiffusionModel
diffusion.constitutive_model.Parameters.diffusivity = k

### fix temp of top and bottom walls
diffusion.add_dirichlet_bc(tmin, "Bottom")
diffusion.add_dirichlet_bc(tmin, "Top")


with mesh.access(T, T_star):
    T.data[...] = tmin

    T.data[(T.coords[:,1] > 0.4) & (T.coords[:,1] < 0.6)] = tmax
    T_star.data[:,0] = T.data[:,0]


# %%
def plot_fig():
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
        # mesh.vtk("ignore_periodic_mesh.vtk")
        pvmesh = pv.read("tmpMsh.vtk")

        # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

        with mesh.access():
            pvmesh["T"] =  uw.function.evaluate(T.sym[0], mesh.data)


        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        with mesh.access(T):
            points = np.zeros((T.coords.shape[0],3))
            points[:,0] = T.coords[:,0]
            points[:,1] = T.coords[:,1]
            points[:,2] = 0.0

        point_cloud = pv.PolyData(points)

        with mesh.access():
            point_cloud.point_data["T"] = T.data[:,0]


        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
                        use_transparency=False, opacity=0.95)


        pl.show(cpos="xy")

        # return vsol

plot_fig()

# +
diffusion.petsc_options['snes_rtol'] = 1e-12
diffusion.petsc_options['snes_atol'] = 1e-6

### see the SNES output
diffusion.petsc_options["snes_converged_reason"] = None


# -

def diffusion_1D(sample_points, T0, diffusivity, time_1D):
    x = sample_points
    T = T0
    k = diffusivity
    time = time_1D

    dx = sample_points[1] - sample_points[0]

    dt_dif = (dx**2 / k)

    dt = 0.5 * dt_dif


    if time > 0:

        """ determine number of its """
        nts = math.ceil(time / dt)
    
        """ get dt of 1D model """
        final_dt = time / nts

    
        for i in range(nts):
            qT = -k * np.diff(T) / dx
            dTdt = -np.diff(qT) / dx
            T[1:-1] += dTdt * final_dt

    

    return T

# +
### y coords to sample
sample_y = np.arange(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), 0.1*mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y) + 0.5  ### centre of the box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y
# -

T_orig = uw.function.evalf(T.sym[0], sample_points)

# +
## SNES scalar equation - works
### Set up the F0 and F1 terms
dt = mesh.get_min_radius()**2 / k

diffusion.F0 += (T.sym - T_star.sym) / dt
flux_star = diffusion.constitutive_model.flux

theta = 0.5
diffusion.F1 += (theta*diffusion.constitutive_model.flux.T) + ((1-theta) * flux_star.T)
# -

step = 0
model_time = 0.

if uw.mpi.size == 1:
    plt.plot(sample_points[:,1], T_orig)
while step < 11:

    ### update terms before solve
    dt = mesh.get_min_radius()**2 / k

    if uw.mpi.rank == 0:
        print(f'step: {step}, time: {model_time}\n\n')

        

    
    diffusion.solve()
    ### get the updated temp profile
    T_new = uw.function.evalf(T.sym[0], sample_points)
    
    if uw.mpi.size == 1:
        plt.plot(sample_points[:,1], T_new)
    
    ### update the flux history in case it's non-linear
    flux_star = diffusion.constitutive_model.flux
    ### update the unknown history variable
    with mesh.access(T, T_star):
        T_star.data[:,0] = T.data[:,0]




    step += 1
    model_time += dt

# #### Numerical 1D solution to compare against

T_1D = diffusion_1D(sample_points=sample_points[:,1], T0=T_orig.copy(), diffusivity=k, time_1D=model_time)
T_UW = uw.function.evalf(T.sym[0], sample_points)
if uw.mpi.size == 1:
    plt.plot(sample_points[:,1], T_1D, label='1D')
    plt.plot(sample_points[:,1], T_UW, label='UW', ls=':', c='k')
    plt.legend()

if uw.mpi.size == 1:
    ### Check the result is close
    np.allclose(T_1D, T_UW, rtol=0.01)


