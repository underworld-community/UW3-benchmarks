ARG BASE_IMAGE=ghcr.io/pyvista/pyvista:latest
FROM ${BASE_IMAGE} as base

USER root

RUN apt-get update -qq \
&&  apt-get upgrade -y \
&&  DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        ca-certificates \
        vim \
        libocct-visualization-dev libocct-data-exchange-dev libocct-draw-dev \
        tcl8.6-dev \
        libxcursor-dev \
        libxft2 \
        libxinerama1 \
        libgmsh-dev \
        libfreetype6-dev \
        xvfb \
        libgl1-mesa-glx \
&&  apt-get clean \
&&  rm -rf /var/lib/apt/lists/*

USER ${NB_UID}
ENV PETSC_DIR /home/jovyan/opt/petsc
WORKDIR /home/jovyan/

# get mamba requirements
RUN mamba install --yes git \
            compilers \
            mpich-mpicc \
            mpich-mpicxx \
            mpich-mpifort \
            openssh \
            binutils \
            make \
            cython \ 
            mpi4py \
            sympy \
            cmake \
&& mamba install --yes petsc --only-deps

RUN wget https://gitlab.com/lmoresi/petsc/-/archive/main/petsc-main.tar.gz --no-check-certificate \
&& tar -xf petsc-main.tar.gz \
&& cd petsc-main \
&& PETSC_DIR=`pwd` ./configure --prefix=/home/jovyan/opt/petsc --with-debugging=0 \
        LDFLAGS=$LDFLAGS CXXFLAGS=$CXXFLAGS \
        AR=$AR RANLIB=$RANLIB \
        --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3" --FOPTFLAGS="-g -O3" \
        --with-batch=1 \
        --with-shared-libraries=1 \
        --with-cxx-dialect=C++11 \
        --with-mpi=1 \
        --with-petsc4py=1 \
        --with-make-np=4 \
        --with-hdf5=1 \
        --with-mumps=1 \
        --with-parmetis=1 \
        --with-metis=1 \
        --with-hypre=1 \
        --with-scalapack=1 \
        --with-superlu=1 \
        --download-superlu_dist=1 \
        --download-eigen=1 \
        --download-triangle=1 \
        --download-ctetgen=1 \
        --download-zlib=1 \
        --useThreads=0 \
&&  make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt all \
&&  make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install \
&&  rm -rf /home/jovyan/petsc-main

# enable PETSC env vars
ENV PETSC_ARCH=arch-linux-c-opt
ENV PYTHONPATH=${PETSC_DIR}/lib:${PYTHONPATH}

# install the rest of the dependencies
RUN CC=mpicc HDF5_MPI="ON" HDF5_DIR=/opt/conda pip install h5py --no-cache-dir \
&&  pip install jupytext gmsh xxhash pint typeguard --no-cache-dir

# install underworld3 
RUN git clone --branch development --depth 1 https://github.com/underworldcode/underworld3.git /home/jovyan/uw3 \
&& cd /home/jovyan/uw3 \
&& pip install -e .

WORKDIR /home/jovyan/
