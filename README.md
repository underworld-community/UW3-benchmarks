# A community repository for Underworld3 benchmark models.

Try them on mybinder now [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/julesghub/UW3-benchmarks/dev) 

 or download the Underworld3 image at https://hub.docker.com/repository/docker/julesg/underworld3/

To run the image locally we recommend using either podman or docker, either via the desktop-GUI or command line application.

### Desktop GUI applications
 - podman - https://podman-desktop.io/
 - docker - https://www.docker.com/get-started/

### Command line usage
 - podman - https://podman.io/docs/installation
 - docker - https://docs.docker.com/get-docker/

Some ways of running the container via command line. NOTE: In examples `podamn` can be swapped for `docker`

```bash
# create a persistent I/O volume to transfer data
podman volume create uw3_vol
# run image with jupyterlab on localhost:9999
podman run -p 9999:8888 -v uw3_vol:/home/jovyan/vol_space underworld3:0.9

# alternative run with an interactive bash prompt (no jupyterlab) and extra volume mount.
podman run --rm -it \
            -p 9999:8888 \
            -v uw3_vol:/home/jovyan/vol_space \
            -v ${HOME}:/home/jovyan/host \
            bash
```

Docker management
```bash
# see your containers and delete them
podman ps -a
podman rm <container_name>

# see your images and delete them
podman images -a
podman rmi underworld3:0.9
```
