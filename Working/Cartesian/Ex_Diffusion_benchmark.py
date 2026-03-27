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
import sys
import petsc4py
petsc4py.init(sys.argv)
import underworld3 as uw
import numpy as np
import math

if uw.mpi.size == 1:
    import matplotlib.pyplot as plt
# -
# #### Setup the mesh params

# +
params = uw.Params(
    uw_res      = 64,
    uw_t_degree = 4,
    uw_kappa    = 1.0,
    uw_tmin     = 0.5,
    uw_tmax     = 1.0,
    uw_n_steps  = 11,
)

res      = params.uw_res
Tdegree  = params.uw_t_degree
k        = params.uw_kappa
tmin     = params.uw_tmin
tmax     = params.uw_tmax
nsteps   = params.uw_n_steps

show_plot = False

mesh_qdegree = Tdegree

# +
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.0
# -

# #### Set up the mesh

mesh = uw.meshing.StructuredQuadBox(
    elementRes=(int(res), int(res)), minCoords=(xmin, ymin), maxCoords=(xmax, ymax), qdegree=mesh_qdegree)


# #### Create mesh vars

# Create mesh vars
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=Tdegree)

# #### Create the solver
diffusion = uw.systems.Diffusion(mesh, T, theta=0.5)

# ##### Set up solver parameters

diffusion.constitutive_model = uw.constitutive_models.DiffusionModel
diffusion.constitutive_model.Parameters.diffusivity = k

### fix temp of top and bottom walls
diffusion.add_dirichlet_bc(tmin, "Bottom")
diffusion.add_dirichlet_bc(tmin, "Top")


T.data[...] = tmin

T.data[(T.coords[:,1] > 0.4) & (T.coords[:,1] < 0.6)] = tmax


# %%
def plot_fig():
    if uw.mpi.size == 1:

        import pyvista as pv

        pv.global_theme.window_size = [750, 250]
        pvmesh = uw.visualisation.mesh_to_pv_mesh(mesh)

        # pvmesh.point_data["S"]  = uw.function.evaluate(s_soln.fn, meshbox.data)

        pvmesh["T"] =  uw.function.evaluate(T.sym[0], mesh.data)


        pl = pv.Plotter()

        pl.add_mesh(pvmesh, "Black", "wireframe")

        points = np.zeros((T.coords.shape[0],3))
        points[:,0] = T.coords[:,0]
        points[:,1] = T.coords[:,1]
        points[:,2] = 0.0

        point_cloud = pv.PolyData(points)

        point_cloud.point_data["T"] = T.data[:,0]


        pl.add_mesh(point_cloud, cmap="coolwarm", edge_color="Black", show_edges=False, scalars="T",
                        use_transparency=False, opacity=0.95)


        pl.show(cpos="xy")

        # return vsol

if show_plot:
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
    mesh.X.coords[:, 1].min(), mesh.X.coords[:, 1].max(), 0.1*mesh.get_min_radius()
)  ### Vertical profile

### x coords to sample
sample_x = np.zeros_like(sample_y) + 0.5  ### centre of the box

sample_points = np.empty((sample_x.shape[0], 2))
sample_points[:, 0] = sample_x
sample_points[:, 1] = sample_y
# -

T_orig = uw.function.evaluate(T.sym[0], sample_points).flatten()

dt = mesh.get_min_radius()**2 / k

step = 0
model_time = 0.

if uw.mpi.size == 1:
    plt.plot(sample_points[:,1], T_orig)
while step < nsteps:

    ### update terms before solve
    dt = mesh.get_min_radius()**2 / k

    if uw.mpi.rank == 0:
        print(f'step: {step}, time: {model_time}\n\n')

        

    
    diffusion.solve(timestep=dt, zero_init_guess=False)
    ### get the updated temp profile
    T_new = uw.function.evaluate(T.sym[0], sample_points).flatten()

    if uw.mpi.size == 1:
        plt.plot(sample_points[:,1], T_new)




    step += 1
    model_time += dt

# #### Numerical 1D solution to compare against

T_1D = diffusion_1D(sample_points=sample_points[:,1], T0=T_orig.copy(), diffusivity=k, time_1D=model_time)
T_UW = uw.function.evaluate(T.sym[0], sample_points).flatten()
if uw.mpi.size == 1:
    plt.plot(sample_points[:,1], T_1D, label='1D')
    plt.plot(sample_points[:,1], T_UW, label='UW', ls=':', c='k')
    plt.legend()

if uw.mpi.size == 1:
    ### Check the result is close
    np.allclose(T_1D, T_UW, rtol=0.01)


