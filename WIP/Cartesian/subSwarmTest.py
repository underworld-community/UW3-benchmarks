#!/usr/bin/env python
# coding: utf-8
# %%
import underworld3 as uw
import numpy as np

mesh = uw.meshing.StructuredQuadBox(elementRes =(int(32),int(32)),
                                    minCoords=(0.,0.), 
                                    maxCoords=(1.,1.))




# %%
### add a tracer
tracer = np.zeros(shape=(1,4))
tracer[:,0] = 0.2
tracer[:,1] = 0.5



# %%
passiveSwarm0 = uw.swarm.Swarm(mesh)

passiveSwarm0.dm.finalizeFieldRegister()

passiveSwarm0.dm.addNPoints(npoints=len(tracer))

passiveSwarm0.dm.setPointCoordinates(tracer)

with passiveSwarm0.access(passiveSwarm0):
    swarm0 = (passiveSwarm0.data)

# %%
swarm0

# %%
passiveSwarm1 = uw.swarm.Swarm(mesh)

passiveSwarm1.add_particles_with_coordinates(tracer)

with passiveSwarm1.access(passiveSwarm1.particle_coordinates):
    swarm1 = (passiveSwarm1.data)

# %%
swarm1

# %%
np.allclose(swarm0, swarm1, tracer)

# %%
